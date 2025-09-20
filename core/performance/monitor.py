# core/performance/monitor.py
"""
Performance Monitoring System
Real-time system metrics, GPU usage, memory tracking and performance profiling
"""

import time
import psutil
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from pathlib import Path
import torch
import numpy as np

from ..config import get_config
from ..exceptions import PerformanceError

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    network_io: Dict[str, int] = field(default_factory=dict)
    process_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Individual request performance metrics"""

    request_id: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    cpu_time: float = 0.0
    gpu_time: float = 0.0
    status: str = "running"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GPUMonitor:
    """GPU monitoring utilities"""

    def __init__(self):
        self.available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.available else 0

    def get_gpu_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get GPU metrics for all available devices"""
        if not self.available:
            return {}

        metrics = {}

        for device_id in range(self.device_count):
            try:
                # Memory information
                mem_info = torch.cuda.mem_get_info(device_id)
                mem_free, mem_total = mem_info
                mem_used = mem_total - mem_free

                # Device properties
                props = torch.cuda.get_device_properties(device_id)

                metrics[device_id] = {
                    "name": props.name,
                    "memory_used_mb": mem_used / 1024**2,
                    "memory_total_mb": mem_total / 1024**2,
                    "memory_percent": (mem_used / mem_total) * 100,
                    "memory_free_mb": mem_free / 1024**2,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multiprocessor_count,
                    "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
                }

                # Try to get temperature and utilization (requires nvidia-ml-py)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    metrics[device_id]["temperature_c"] = temp

                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[device_id]["gpu_utilization_percent"] = util.gpu
                    metrics[device_id]["memory_utilization_percent"] = util.memory

                    # Power
                    power = (
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )  # Convert to watts
                    metrics[device_id]["power_usage_w"] = power

                except ImportError:
                    logger.debug("pynvml not available, limited GPU metrics")
                except Exception as e:
                    logger.debug(f"Failed to get detailed GPU metrics: {e}")

            except Exception as e:
                logger.warning(f"Failed to get metrics for GPU {device_id}: {e}")
                metrics[device_id] = {"error": str(e)}

        return metrics


class SystemMonitor:
    """System resource monitoring"""

    def __init__(self):
        self.gpu_monitor = GPUMonitor()

    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage("/")

            # Network I/O
            net_io = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }

            # Process metrics for current process
            process = psutil.Process()
            process_metrics = {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss_mb": process.memory_info().rss / 1024**2,
                "memory_vms_mb": process.memory_info().vms / 1024**2,
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
            }

            # GPU metrics
            gpu_metrics = self.gpu_monitor.get_gpu_metrics()

            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / 1024**3,
                memory_available_gb=memory.available / 1024**3,
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / 1024**3,
                gpu_metrics=gpu_metrics,
                network_io=network_metrics,
                process_metrics=process_metrics,
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise PerformanceError(f"System metrics collection failed: {e}")


class PerformanceProfiler:
    """Request-level performance profiling"""

    def __init__(self):
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: deque = deque(maxlen=1000)
        self.lock = threading.Lock()

    def start_request(
        self, request_id: str, endpoint: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RequestMetrics:
        """Start profiling a request"""
        with self.lock:
            metrics = RequestMetrics(
                request_id=request_id,
                endpoint=endpoint,
                start_time=time.time(),
                metadata=metadata or {},
            )

            # Record initial memory state
            if torch.cuda.is_available():
                metrics.peak_gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2

            process = psutil.Process()
            metrics.peak_memory_mb = process.memory_info().rss / 1024**2

            self.active_requests[request_id] = metrics
            return metrics

    def update_request(self, request_id: str, **kwargs):
        """Update request metrics"""
        with self.lock:
            if request_id in self.active_requests:
                metrics = self.active_requests[request_id]

                # Update peak memory usage
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    metrics.peak_gpu_memory_mb = max(
                        metrics.peak_gpu_memory_mb, current_gpu_memory
                    )

                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024**2
                metrics.peak_memory_mb = max(metrics.peak_memory_mb, current_memory)

                # Update other fields
                for key, value in kwargs.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
                    else:
                        metrics.metadata[key] = value

    def end_request(
        self, request_id: str, status: str = "completed", error: Optional[str] = None
    ) -> Optional[RequestMetrics]:
        """End request profiling"""
        with self.lock:
            if request_id not in self.active_requests:
                return None

            metrics = self.active_requests.pop(request_id)
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.status = status
            metrics.error = error

            self.completed_requests.append(metrics)
            return metrics

    def get_active_requests(self) -> List[RequestMetrics]:
        """Get currently active requests"""
        with self.lock:
            return list(self.active_requests.values())

    def get_completed_requests(self, limit: int = 100) -> List[RequestMetrics]:
        """Get recent completed requests"""
        with self.lock:
            return list(self.completed_requests)[-limit:]

    def get_performance_stats(
        self, endpoint: Optional[str] = None, time_window: int = 3600
    ) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - time_window

            # Filter requests
            requests = [
                r
                for r in self.completed_requests
                if r.end_time
                and r.end_time >= cutoff_time
                and (not endpoint or r.endpoint == endpoint)
            ]

            if not requests:
                return {"error": "No requests found in time window"}

            # Calculate statistics
            durations = [r.duration for r in requests if r.duration]
            memory_usage = [r.peak_memory_mb for r in requests]
            gpu_memory_usage = [
                r.peak_gpu_memory_mb for r in requests if r.peak_gpu_memory_mb > 0
            ]

            stats = {
                "total_requests": len(requests),
                "successful_requests": len(
                    [r for r in requests if r.status == "completed"]
                ),
                "failed_requests": len([r for r in requests if r.status == "failed"]),
                "average_duration": np.mean(durations) if durations else 0,
                "median_duration": np.median(durations) if durations else 0,
                "p95_duration": np.percentile(durations, 95) if durations else 0,
                "p99_duration": np.percentile(durations, 99) if durations else 0,
                "min_duration": np.min(durations) if durations else 0,
                "max_duration": np.max(durations) if durations else 0,
                "average_memory_mb": np.mean(memory_usage) if memory_usage else 0,
                "peak_memory_mb": np.max(memory_usage) if memory_usage else 0,
                "requests_per_minute": len(requests) / (time_window / 60),
                "error_rate": (
                    len([r for r in requests if r.status == "failed"]) / len(requests)
                    if requests
                    else 0
                ),
            }

            if gpu_memory_usage:
                stats.update(
                    {
                        "average_gpu_memory_mb": np.mean(gpu_memory_usage),
                        "peak_gpu_memory_mb": np.max(gpu_memory_usage),
                    }
                )

            return stats


class PerformanceMonitor:
    """Main performance monitoring system"""

    def __init__(self):
        self.config = get_config()
        self.system_monitor = SystemMonitor()
        self.profiler = PerformanceProfiler()

        # Metrics history
        self.metrics_history: deque = deque(
            maxlen=1440
        )  # 24 hours at 1-minute intervals
        self.alerts: List[Dict[str, Any]] = []

        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 90,
            "memory_percent": 85,
            "gpu_memory_percent": 90,
            "disk_usage_percent": 95,
            "response_time_p95": 30.0,  # seconds
            "error_rate": 0.1,  # 10%
        }

    def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"✅ Performance monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ Performance monitoring stopped")

    def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self.system_monitor.get_system_metrics()
                self.metrics_history.append(metrics)

                # Check thresholds and generate alerts
                self._check_thresholds(metrics)

                # Log metrics periodically
                if len(self.metrics_history) % 5 == 0:  # Every 5 intervals
                    self._log_metrics_summary(metrics)

            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")

            time.sleep(interval)

    def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts"""
        alerts = []

        # CPU usage
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    "value": metrics.cpu_percent,
                    "threshold": self.thresholds["cpu_percent"],
                }
            )

        # Memory usage
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                    "value": metrics.memory_percent,
                    "threshold": self.thresholds["memory_percent"],
                }
            )

        # GPU memory usage
        for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
            if "memory_percent" in gpu_metrics:
                if (
                    gpu_metrics["memory_percent"]
                    > self.thresholds["gpu_memory_percent"]
                ):
                    alerts.append(
                        {
                            "type": "high_gpu_memory",
                            "severity": "warning",
                            "message": f"High GPU {gpu_id} memory: {gpu_metrics['memory_percent']:.1f}%",
                            "value": gpu_metrics["memory_percent"],
                            "threshold": self.thresholds["gpu_memory_percent"],
                            "gpu_id": gpu_id,
                        }
                    )

        # Disk usage
        if metrics.disk_usage_percent > self.thresholds["disk_usage_percent"]:
            alerts.append(
                {
                    "type": "high_disk_usage",
                    "severity": "critical",
                    "message": f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                    "value": metrics.disk_usage_percent,
                    "threshold": self.thresholds["disk_usage_percent"],
                }
            )

        # Add timestamps and store alerts
        for alert in alerts:
            alert["timestamp"] = metrics.timestamp
            self.alerts.append(alert)
            logger.warning(f"🚨 Performance Alert: {alert['message']}")

        # Keep only recent alerts
        cutoff_time = time.time() - 3600  # 1 hour
        self.alerts = [a for a in self.alerts if a["timestamp"] > cutoff_time]

    def _log_metrics_summary(self, metrics: SystemMetrics):
        """Log performance metrics summary"""
        gpu_summary = ""
        if metrics.gpu_metrics:
            gpu_info = []
            for gpu_id, gpu_data in metrics.gpu_metrics.items():
                if "memory_percent" in gpu_data:
                    gpu_info.append(f"GPU{gpu_id}: {gpu_data['memory_percent']:.1f}%")
            gpu_summary = f" | {', '.join(gpu_info)}" if gpu_info else ""

        logger.info(
            f"📊 Performance: CPU {metrics.cpu_percent:.1f}% | "
            f"RAM {metrics.memory_percent:.1f}% | "
            f"Disk {metrics.disk_usage_percent:.1f}%{gpu_summary}"
        )

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self.system_monitor.get_system_metrics()

    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = self.get_current_metrics()
        request_stats = self.profiler.get_performance_stats(time_window=3600)
        recent_alerts = [a for a in self.alerts if a["timestamp"] > time.time() - 3600]

        return {
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "memory_used_gb": current_metrics.memory_used_gb,
                "disk_usage_percent": current_metrics.disk_usage_percent,
                "gpu_metrics": current_metrics.gpu_metrics,
            },
            "request_metrics": request_stats,
            "alerts": recent_alerts,
            "active_requests": len(self.profiler.get_active_requests()),
            "health_status": (
                "healthy"
                if len(recent_alerts) == 0
                else (
                    "degraded"
                    if any(a["severity"] == "critical" for a in recent_alerts)
                    else "warning"
                )
            ),
        }


# Global instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
