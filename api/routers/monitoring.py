# api/routers/monitoring.py
"""
System Monitoring Router
"""

import logging
import psutil
import torch
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from schemas.monitoring import (
    SystemMetrics,
    TaskMetrics,
    WorkerStatus,
    PerformanceReport,
    AlertStatus,
)

from core.performance.monitor import get_performance_monitor
from core.utils import get_model_manager, get_cache_manager
from core.monitoring.metrics import MetricsCollector
from core.monitoring.logger import structured_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])
metrics_collector = MetricsCollector()


# Response models
class SystemMetricsResponse(BaseModel):
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_metrics: Dict[int, Dict[str, Any]]
    network_io: Dict[str, int]
    process_metrics: Dict[str, Any]


class PerformanceReportResponse(BaseModel):
    timestamp: float
    system_metrics: Dict[str, Any]
    request_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    active_requests: int
    health_status: str


class ModelMemoryResponse(BaseModel):
    tracked_models: int
    tracked_memory_mb: float
    gpu_allocated_mb: float
    gpu_cached_mb: float
    gpu_free_mb: float
    gpu_total_mb: float
    models: Dict[str, Dict[str, Any]]


class CacheStatsResponse(BaseModel):
    cache_root: str
    total_size_mb: float
    file_cache: Dict[str, Any]
    model_cache: Dict[str, Any]
    result_cache: Dict[str, Any]


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get current system metrics"""
    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU metrics (if available)
        gpu_metrics = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_metrics[f"gpu_{i}"] = {
                    "memory_used": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "memory_total": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,
                    "utilization": (
                        torch.cuda.utilization(i)
                        if hasattr(torch.cuda, "utilization")
                        else 0
                    ),
                }

        # Disk usage
        disk = psutil.disk_usage("/")

        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory.used / 1024**3,
            memory_total_gb=memory.total / 1024**3,
            memory_percent=memory.percent,
            disk_used_gb=disk.used / 1024**3,
            disk_total_gb=disk.total / 1024**3,
            disk_percent=(disk.used / disk.total) * 100,
            gpu_metrics=gpu_metrics,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=TaskMetrics)
async def get_task_metrics():
    """Get task execution metrics"""
    try:
        metrics = await metrics_collector.get_task_metrics()

        return TaskMetrics(
            timestamp=datetime.utcnow(),
            total_tasks=metrics["total_tasks"],
            completed_tasks=metrics["completed_tasks"],
            failed_tasks=metrics["failed_tasks"],
            pending_tasks=metrics["pending_tasks"],
            avg_execution_time=metrics["avg_execution_time"],
            tasks_per_minute=metrics["tasks_per_minute"],
            error_rate=metrics["error_rate"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers", response_model=List[WorkerStatus])
async def get_worker_status():
    """Get Celery worker status"""
    try:
        from celery import current_app

        # Get active workers
        inspect = current_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()

        workers = []
        for worker_name, worker_stats in (stats or {}).items():
            worker_active_tasks = len(active_tasks.get(worker_name, []))

            workers.append(
                WorkerStatus(
                    name=worker_name,
                    status="online",
                    active_tasks=worker_active_tasks,
                    processed_tasks=worker_stats.get("total", {}).get(
                        "worker.processed", 0
                    ),
                    last_heartbeat=datetime.utcnow(),
                    queue_size=worker_stats.get("pool", {}).get("max-concurrency", 0),
                )
            )

        return workers

    except Exception as e:
        # Return empty list if Celery not available
        structured_logger.warning(f"Failed to get worker status: {e}")
        return []


@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "components": {},
        }

        # Check Redis connection
        try:
            from workers.celery_app import redis_client

            redis_client.ping()
            health_status["components"]["redis"] = "healthy"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check GPU availability
        if torch.cuda.is_available():
            health_status["components"][
                "gpu"
            ] = f"healthy ({torch.cuda.device_count()} devices)"
        else:
            health_status["components"]["gpu"] = "unavailable"

        # Check disk space
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 90:
            health_status["components"]["disk"] = f"warning: {disk_percent:.1f}% used"
            health_status["status"] = "degraded"
        else:
            health_status["components"]["disk"] = f"healthy ({disk_percent:.1f}% used)"

        return health_status

    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/performance", response_model=PerformanceReport)
async def get_performance_report(hours: int = 24):
    """Get performance report for the last N hours"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        report = await metrics_collector.generate_performance_report(
            start_time=start_time, end_time=end_time
        )

        return PerformanceReport(
            time_range=f"{start_time.isoformat()} to {end_time.isoformat()}",
            total_requests=report["total_requests"],
            avg_response_time=report["avg_response_time"],
            error_rate=report["error_rate"],
            peak_cpu_usage=report["peak_cpu_usage"],
            peak_memory_usage=report["peak_memory_usage"],
            top_endpoints=report["top_endpoints"],
            slowest_endpoints=report["slowest_endpoints"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory", response_model=ModelMemoryResponse)
async def get_memory_stats():
    """Get model memory usage statistics"""
    try:
        model_manager = get_model_manager()
        stats = model_manager.get_memory_stats()

        return ModelMemoryResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {e}")


@router.get("/cache", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache usage statistics"""
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_cache_stats()

        return CacheStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {e}")


@router.post("/cache/cleanup")
async def cleanup_cache(target_size_gb: Optional[float] = None):
    """Cleanup cache to target size"""
    try:
        cache_manager = get_cache_manager()
        result = cache_manager.cleanup_cache(target_size_gb)

        return {
            "success": True,
            "cleanup_performed": result["cleaned"],
            "method": result.get("method"),
            "removed_count": result.get("removed_count", 0),
            "removed_size_mb": result.get("removed_size_mb", 0),
            "final_stats": result["stats"],
        }

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {e}")


@router.get("/alerts", response_model=List[AlertStatus])
async def get_active_alerts():
    """Get current system alerts"""
    try:
        alerts = []

        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            alerts.append(
                AlertStatus(
                    id="high_memory",
                    level="warning",
                    message=f"High memory usage: {memory.percent:.1f}%",
                    timestamp=datetime.utcnow(),
                )
            )

        # Check GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                used = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                usage_percent = (used / total) * 100

                if usage_percent > 90:
                    alerts.append(
                        AlertStatus(
                            id=f"high_gpu_memory_{i}",
                            level="critical",
                            message=f"GPU {i} memory critical: {usage_percent:.1f}%",
                            timestamp=datetime.utcnow(),
                        )
                    )

        # Check failed tasks
        task_metrics = await metrics_collector.get_task_metrics()
        if task_metrics["error_rate"] > 0.1:  # 10% error rate
            alerts.append(
                AlertStatus(
                    id="high_task_error_rate",
                    level="warning",
                    message=f"High task error rate: {task_metrics['error_rate']:.1%}",
                    timestamp=datetime.utcnow(),
                )
            )

        return alerts

    except Exception as e:
        structured_logger.error(f"Failed to get alerts: {e}")
        return []


# Example usage:
"""
# Get system metrics
GET /api/v1/monitoring/metrics
{
  "timestamp": "2025-01-01T00:00:00Z",
  "cpu_percent": 45.2,
  "memory_used_gb": 12.5,
  "memory_total_gb": 32.0,
  "gpu_metrics": {
    "gpu_0": {
      "memory_used": 8.2,
      "memory_total": 24.0,
      "utilization": 75
    }
  }
}

# Health check
GET /api/v1/monitoring/health
{
  "status": "healthy",
  "components": {
    "redis": "healthy",
    "gpu": "healthy (1 devices)",
    "disk": "healthy (65.2% used)"
  }
}
"""
