"""Performance utilities with lazy optional model dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    **{
        name: ("core.performance.monitor", name)
        for name in [
            "PerformanceMonitor",
            "SystemMonitor",
            "PerformanceProfiler",
            "GPUMonitor",
            "SystemMetrics",
            "RequestMetrics",
        ]
    },
    "CacheManager": ("core.performance.cache_manager", "CacheManager"),
    "CacheConfig": ("core.performance.cache_manager", "CacheConfig"),
    **{
        name: ("core.performance.quantization", name)
        for name in [
            "QuantizationManager",
            "QuantizationConfig",
            "QuantizationMode",
            "create_quantization_config",
            "get_quantization_manager",
        ]
    },
    **{
        name: ("core.performance.batch_optimizer", name)
        for name in ["BatchProcessor", "BatchConfig", "BatchStrategy", "BatchMetrics"]
    },
}

_performance_monitor: Any = None
__all__ = [*_EXPORTS, "gpu_available", "get_performance_monitor"]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_performance_monitor():
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = __getattr__("PerformanceMonitor")()
    return _performance_monitor
