# api/routers/health.py
"""
Health Check Router
Provides system health and status information
"""

import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request
from pydantic import BaseModel
import sys
import platform

router = APIRouter()


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


@router.get("/healthz", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Get system health status
    Returns comprehensive health information including GPU, memory, and cache status
    """
    current_time = time.time()
    uptime = current_time - _start_time

    # Get system info
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": memory.percent,
        "memory_available_gb": round(memory.available / 1024**3, 2),
        "memory_total_gb": round(memory.total / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_total_gb": round(disk.total / 1024**3, 2),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),  # Windows / Linux / Darwin
        "platform_release": platform.release(),  # 版本（可選）
        "platform_machine": platform.machine(),  # 架構（可選）
    }

    # Get cache info from app state
    cache_info = {}
    config_info = {}

    try:
        if hasattr(request.app.state, "cache"):
            cache = request.app.state.cache
            cache_summary = cache.get_summary()
            cache_info = {
                "root": cache_summary["cache_root"],
                "gpu_available": cache_summary["gpu_info"]["cuda_available"],
                "gpu_count": cache_summary["gpu_info"]["device_count"],
                "gpu_memory": cache_summary["gpu_info"].get("memory_info", {}),
                "directories_created": len(cache_summary["directories"]),
            }
    except Exception as e:
        cache_info = {"error": str(e)}

    try:
        if hasattr(request.app.state, "config"):
            config = request.app.state.config
            config_summary = config.get_summary()
            config_info = {
                "app_name": config_summary.get("app", {}).get("name", "SagaForge"),
                "features_enabled": config_summary.get("features", {}),
                "model_config": config_summary.get("model", {}),
                "api_debug": config_summary.get("api", {}).get("debug", False),
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
async def readiness_check(request: Request):
    """
    Readiness probe for Kubernetes/Docker
    Returns 200 if service is ready to accept requests
    """
    # Check if essential components are initialized
    ready = True
    issues = []

    if not hasattr(request.app.state, "cache"):
        ready = False
        issues.append("Shared cache not initialized")

    if not hasattr(request.app.state, "config"):
        ready = False
        issues.append("Configuration not loaded")

    if ready:
        return {"status": "ready", "timestamp": datetime.now()}
    else:
        return {"status": "not_ready", "issues": issues, "timestamp": datetime.now()}


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
