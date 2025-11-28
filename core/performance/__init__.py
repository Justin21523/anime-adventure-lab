"""Performance utilities package."""

import torch

from .monitor import (
    PerformanceMonitor,
    SystemMonitor,
    PerformanceProfiler,
    GPUMonitor,
    SystemMetrics,
    RequestMetrics,
)

_performance_monitor = None


def gpu_available() -> bool:
    """Return whether CUDA GPU is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def get_performance_monitor() -> PerformanceMonitor:
    """Global singleton for performance monitoring."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


__all__ = [
    "gpu_available",
    "PerformanceMonitor",
    "SystemMonitor",
    "PerformanceProfiler",
    "GPUMonitor",
    "SystemMetrics",
    "RequestMetrics",
    "get_performance_monitor",
]
