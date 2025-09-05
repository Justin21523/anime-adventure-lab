# api/routers/monitoring.py
"""
System Monitoring Router
"""

import logging
import psutil
import torch
from fastapi import APIRouter, HTTPException
from schemas.monitoring import MonitoringResponse, SystemMetrics

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/monitoring/system", response_model=MonitoringResponse)
async def get_system_monitoring():
    """Get system monitoring data"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        gpu_usage = None
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_usage = 45.0  # Mock GPU usage
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB

        metrics = SystemMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_used_gb=gpu_memory,
            disk_usage_percent=(disk.used / disk.total) * 100,
        )

        return MonitoringResponse(  # type: ignore
            metrics=metrics,
            model_stats={"models_loaded": 2, "memory_usage_gb": 3.5},
            api_stats={
                "requests_per_minute": 12,
                "average_response_time_ms": 250,
                "error_rate_percent": 2.1,
            },
            cache_stats={"cache_hit_rate_percent": 85.5, "cache_size_mb": 1024},
        )

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise HTTPException(500, f"Failed to get monitoring data: {str(e)}")
