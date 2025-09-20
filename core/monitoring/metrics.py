# backend/core/monitoring/metrics.py
import time
import psutil
import torch
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
from .logger import structured_logger


class MetricsCollector:
    """Collects and stores system and application metrics"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            import os

            AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
            cache_dir = Path(AI_CACHE_ROOT) / "outputs" / "multi-modal-lab"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "metrics.db"

        self.db_path = db_path
        self._init_db()

        # In-memory metrics for real-time tracking
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "errors": 0}
        )

    def _init_db(self):
        """Initialize metrics database"""
        with sqlite3.connect(self.db_path) as conn:
            # System metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_used_gb REAL,
                    memory_total_gb REAL,
                    memory_percent REAL,
                    disk_used_gb REAL,
                    disk_total_gb REAL,
                    disk_percent REAL,
                    gpu_metrics TEXT
                )
            """
            )

            # API metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_metrics (
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time_ms REAL,
                    error_message TEXT
                )
            """
            )

            # Task metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_metrics (
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    task_type TEXT,
                    task_id TEXT,
                    status TEXT,
                    execution_time_ms REAL,
                    queue_time_ms REAL,
                    worker_name TEXT,
                    error_message TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_timestamp ON task_metrics(timestamp)"
            )

    def record_system_metrics(self):
        """Record current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # GPU metrics
            gpu_metrics = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f"gpu_{i}"] = {
                        "memory_used": torch.cuda.memory_allocated(i) / 1024**3,
                        "memory_total": torch.cuda.get_device_properties(i).total_memory
                        / 1024**3,
                        "temperature": (
                            torch.cuda.temperature(i)
                            if hasattr(torch.cuda, "temperature")
                            else None
                        ),
                    }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO system_metrics
                    (cpu_percent, memory_used_gb, memory_total_gb, memory_percent,
                     disk_used_gb, disk_total_gb, disk_percent, gpu_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cpu_percent,
                        memory.used / 1024**3,
                        memory.total / 1024**3,
                        memory.percent,
                        disk.used / 1024**3,
                        disk.total / 1024**3,
                        (disk.used / disk.total) * 100,
                        json.dumps(gpu_metrics),
                    ),
                )

        except Exception as e:
            structured_logger.error(f"Failed to record system metrics: {e}")

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        error_message: str = None,
    ):
        """Record API request metrics"""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO api_metrics
                    (endpoint, method, status_code, response_time_ms, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (endpoint, method, status_code, response_time_ms, error_message),
                )

            # Update in-memory stats
            self.request_times.append(response_time_ms)
            endpoint_key = f"{method} {endpoint}"
            self.endpoint_stats[endpoint_key]["count"] += 1
            self.endpoint_stats[endpoint_key]["total_time"] += response_time_ms

            if status_code >= 400:
                self.endpoint_stats[endpoint_key]["errors"] += 1
                self.error_counts[status_code] += 1

        except Exception as e:
            structured_logger.error(f"Failed to record API metrics: {e}")

    def record_task_metrics(
        self,
        task_type: str,
        task_id: str,
        status: str,
        execution_time_ms: float,
        queue_time_ms: float = None,
        worker_name: str = None,
        error_message: str = None,
    ):
        """Record task execution metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO task_metrics
                    (task_type, task_id, status, execution_time_ms,
                     queue_time_ms, worker_name, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task_type,
                        task_id,
                        status,
                        execution_time_ms,
                        queue_time_ms,
                        worker_name,
                        error_message,
                    ),
                )

        except Exception as e:
            structured_logger.error(f"Failed to record task metrics: {e}")

    async def get_task_metrics(self) -> Dict[str, Any]:
        """Get aggregated task metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get task counts by status
                cursor = conn.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM task_metrics
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY status
                """
                )
                status_counts = dict(cursor.fetchall())

                # Get average execution time
                cursor = conn.execute(
                    """
                    SELECT AVG(execution_time_ms) as avg_time
                    FROM task_metrics
                    WHERE timestamp > datetime('now', '-1 hour')
                    AND status = 'SUCCESS'
                """
                )
                avg_time = cursor.fetchone()[0] or 0

                # Get tasks per minute
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) * 60.0 / 60 as tpm
                    FROM task_metrics
                    WHERE timestamp > datetime('now', '-1 hour')
                """
                )
                tasks_per_minute = cursor.fetchone()[0] or 0

                total_tasks = sum(status_counts.values())
                failed_tasks = status_counts.get("FAILURE", 0)
                error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0

                return {
                    "total_tasks": total_tasks,
                    "completed_tasks": status_counts.get("SUCCESS", 0),
                    "failed_tasks": failed_tasks,
                    "pending_tasks": status_counts.get("PENDING", 0),
                    "avg_execution_time": avg_time,
                    "tasks_per_minute": tasks_per_minute,
                    "error_rate": error_rate,
                }

        except Exception as e:
            structured_logger.error(f"Failed to get task metrics: {e}")
            return {
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "pending_tasks": 0,
                "avg_execution_time": 0,
                "tasks_per_minute": 0,
                "error_rate": 0,
            }

    async def generate_performance_report(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # API metrics
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as total_requests,
                           AVG(response_time_ms) as avg_response_time,
                           SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as error_rate
                    FROM api_metrics
                    WHERE timestamp BETWEEN ? AND ?
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                api_stats = cursor.fetchone()

                # Top endpoints by request count
                cursor = conn.execute(
                    """
                    SELECT endpoint, method, COUNT(*) as requests
                    FROM api_metrics
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY endpoint, method
                    ORDER BY requests DESC
                    LIMIT 10
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                top_endpoints = [
                    {"endpoint": f"{row[1]} {row[0]}", "requests": row[2]}
                    for row in cursor.fetchall()
                ]

                # Slowest endpoints
                cursor = conn.execute(
                    """
                    SELECT endpoint, method, AVG(response_time_ms) as avg_time
                    FROM api_metrics
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY endpoint, method
                    HAVING COUNT(*) >= 5
                    ORDER BY avg_time DESC
                    LIMIT 10
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                slowest_endpoints = [
                    {"endpoint": f"{row[1]} {row[0]}", "avg_time_ms": row[2]}
                    for row in cursor.fetchall()
                ]

                # System resource peaks
                cursor = conn.execute(
                    """
                    SELECT MAX(cpu_percent) as peak_cpu,
                           MAX(memory_percent) as peak_memory
                    FROM system_metrics
                    WHERE timestamp BETWEEN ? AND ?
                """,
                    (start_time.isoformat(), end_time.isoformat()),
                )

                resource_peaks = cursor.fetchone()

                return {
                    "total_requests": api_stats[0] or 0,
                    "avg_response_time": api_stats[1] or 0,
                    "error_rate": api_stats[2] or 0,
                    "peak_cpu_usage": resource_peaks[0] or 0,
                    "peak_memory_usage": resource_peaks[1] or 0,
                    "top_endpoints": top_endpoints,
                    "slowest_endpoints": slowest_endpoints,
                }

        except Exception as e:
            structured_logger.error(f"Failed to generate performance report: {e}")
            return {}
