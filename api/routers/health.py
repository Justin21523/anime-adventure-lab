# api/routers/health.py
"""
Health Check Router
Provides system health and status information
"""
from __future__ import annotations
import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
import sys
import platform

from ..dependencies import get_cache, get_settings
from core.performance import gpu_available

router = APIRouter(tags=["health"])  # no prefix; keep path clean via main.py


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    system: Dict[str, Any]
    cache: Dict[str, Any]
    config: Dict[str, Any]


class SystemInfo(BaseModel):
    """System information model"""

    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_free_gb: float
    python_version: str


# Store app start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(cache=Depends(get_cache), settings=Depends(get_settings)):
    """
    Comprehensive health information including GPU, memory, and cache status.
    """
    uptime = time.time() - _start_time

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory_percent": mem.percent,
        "memory_available_gb": round(mem.available / 1024**3, 2),
        "memory_total_gb": round(mem.total / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_total_gb": round(disk.total / 1024**3, 2),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "gpu_available": gpu_available(),
    }

    # Cache summary (shared_cache provides a summary method)
    try:
        cs = cache.get_summary()
        cache_info = {
            "root": cs["cache_root"],
            "directories_created": len(cs["directories"]),
            "gpu_info": cs.get("gpu_info", {}),
        }
    except Exception as e:
        cache_info = {"error": str(e)}

    # Config summary (if your config has a .get_summary)
    try:
        cfg = getattr(settings, "get_summary", None)
        summary = cfg() if callable(cfg) else {}
        config_info = {
            "app": summary.get("app", {}),  # type: ignore
            "features": summary.get("features", {}),  # type: ignore
            "model": summary.get("model", {}),  # type: ignore
            "api": summary.get("api", {}),  # type: ignore
        }
    except Exception as e:
        config_info = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        uptime_seconds=round(uptime, 2),
        version="0.1.0",
        system=system_info,
        cache=cache_info,
        config=config_info,
    )


@router.get("/ready")
async def readiness_check(cache=Depends(get_cache), settings=Depends(get_settings)):
    """
    Readiness probe for Kubernetes/Docker.
    Returns 200 if essential components are available.
    """
    issues = []
    ready = True

    try:
        _ = cache.get_path  # callable attribute presence
    except Exception:
        ready = False
        issues.append("Shared cache not initialized")

    if settings is None:
        ready = False
        issues.append("Configuration not loaded")

    return {
        "status": "ready" if ready else "not_ready",
        "issues": issues,
        "timestamp": datetime.now(),
    }


@router.get("/metrics")
async def metrics_endpoint(request: Request):
    """
    Prometheus-style metrics endpoint
    Basic metrics for monitoring and alerting
    """
    current_time = time.time()
    uptime = current_time - _start_time

    # System metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    # GPU metrics (if available)
    gpu_metrics = {}
    try:
        if hasattr(request.app.state, "cache"):
            cache = request.app.state.cache
            gpu_info = cache.get_gpu_info()
            if gpu_info["cuda_available"] and gpu_info.get("memory_info"):
                mem = gpu_info["memory_info"]
                gpu_metrics = {
                    "gpu_memory_allocated_gb": mem.get("allocated_gb", 0),
                    "gpu_memory_total_gb": mem.get("total_gb", 0),
                    "gpu_memory_utilization": mem.get("allocated_gb", 0)
                    / max(mem.get("total_gb", 1), 1),
                }
    except Exception:
        pass

    metrics = {
        "saga_forge_uptime_seconds": uptime,
        "saga_forge_cpu_usage_percent": cpu_percent,
        "saga_forge_memory_usage_percent": memory.percent,
        "saga_forge_memory_available_bytes": memory.available,
        **gpu_metrics,
    }

    return metrics
