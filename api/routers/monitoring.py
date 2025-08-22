# api/routers/monitoring.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import psutil
import torch
import time
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import json
from pathlib import Path

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.config import get_settings
from core.database import get_db_session
from core.performance.memory_manager import MemoryManager, MemoryConfig
from core.performance.cache_manager import CacheManager, CacheConfig
from workers.celery_app import celery_app

# Global instances
memory_manager = None
cache_manager = None


class HealthStatus(BaseModel):
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_info: Optional[Dict[str, Any]] = None
    active_workers: int
    queue_depth: int
    uptime_seconds: float


# Global startup time for uptime calculation
startup_time = time.time()


def get_managers():
    global memory_manager, cache_manager
    if memory_manager is None:
        memory_manager = MemoryManager(MemoryConfig())
    if cache_manager is None:
        cache_manager = CacheManager(CacheConfig())
    return memory_manager, cache_manager


@router.get("/models")
def get_loaded_models():
    """Get information about currently loaded models"""
    mm, _ = get_managers()

    models_info = {}
    for model_key, model in mm.loaded_models.items():
        try:
            model_info = {
                "loaded_at": datetime.now().isoformat(),  # Would track this properly
                "device": str(getattr(model, "device", "unknown")),
                "dtype": str(getattr(model, "dtype", "unknown")),
                "memory_usage": mm.memory_usage.get(model_key, {}),
            }

            # Try to get model size
            if hasattr(model, "num_parameters"):
                model_info["parameters"] = model.num_parameters()
            elif hasattr(model, "get_memory_footprint"):
                model_info["memory_footprint"] = model.get_memory_footprint()

            models_info[model_key] = model_info

        except Exception as e:
            models_info[model_key] = {"error": str(e)}

    return {
        "timestamp": datetime.now().isoformat(),
        "loaded_models": models_info,
        "total_loaded": len(models_info),
    }


@router.post("/cleanup")
def force_cleanup():
    """Force memory cleanup"""
    mm, cm = get_managers()

    initial_memory = mm.get_memory_info()

    # Cleanup cache
    cm.cleanup_expired()

    # Cleanup models
    mm.cleanup_all()

    final_memory = mm.get_memory_info()

    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "memory_before": initial_memory,
        "memory_after": final_memory,
        "gpu_memory_freed_gb": (
            initial_memory.get("gpu_allocated_gb", 0)
            - final_memory.get("gpu_allocated_gb", 0)
        ),
    }


@router.post("/models/{model_key}/unload")
def unload_model(model_key: str):
    """Unload specific model"""
    mm, _ = get_managers()

    if model_key not in mm.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")

    initial_memory = mm.get_memory_info()
    mm.unload_model(model_key)
    final_memory = mm.get_memory_info()

    return {
        "status": "unloaded",
        "model_key": model_key,
        "timestamp": datetime.now().isoformat(),
        "memory_freed_gb": (
            initial_memory.get("gpu_allocated_gb", 0)
            - final_memory.get("gpu_allocated_gb", 0)
        ),
    }


@router.get("/cache/stats")
def get_cache_detailed_stats():
    """Get detailed cache statistics"""
    _, cm = get_managers()

    stats = cm.get_cache_stats()

    # Add breakdown by cache type if possible
    cache_breakdown = {}

    try:
        cache_dir = Path(cm.config.disk_cache_dir)

        # Count by prefix
        for prefix in ["emb:", "img:", "kv:"]:
            files = list(cache_dir.glob(f"{prefix}*.json")) + list(
                cache_dir.glob(f"{prefix}*.pkl")
            )
            cache_breakdown[prefix.rstrip(":")] = {
                "files": len(files),
                "size_mb": sum(f.stat().st_size for f in files) / 1024**2,
            }

    except Exception as e:
        cache_breakdown = {"error": str(e)}

    return {
        "timestamp": datetime.now().isoformat(),
        "overview": stats,
        "breakdown": cache_breakdown,
    }


@router.post("/cache/cleanup")
def cleanup_cache():
    """Clean up expired cache entries"""
    _, cm = get_managers()

    initial_stats = cm.get_cache_stats()
    cm.cleanup_expired()
    final_stats = cm.get_cache_stats()

    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "files_before": initial_stats.get("disk_cache_files", 0),
        "files_after": final_stats.get("disk_cache_files", 0),
        "size_freed_mb": (
            initial_stats.get("disk_cache_size_mb", 0)
            - final_stats.get("disk_cache_size_mb", 0)
        ),
    }


@router.get("/performance/recommendations")
def get_performance_recommendations():
    """Get performance optimization recommendations"""
    mm, cm = get_managers()

    memory_info = mm.get_memory_info()
    cache_stats = cm.get_cache_stats()

    recommendations = []

    # Memory recommendations
    gpu_usage_percent = (
        memory_info.get("gpu_allocated_gb", 0)
        / memory_info.get("gpu_total_gb", 8)
        * 100
    )

    if gpu_usage_percent > 80:
        recommendations.append(
            {
                "type": "memory",
                "priority": "high",
                "message": "GPU memory usage is high. Consider enabling CPU offload or reducing batch size.",
                "action": "Enable cpu_offload in memory config",
            }
        )

    if memory_info.get("memory_percent", 0) > 85:
        recommendations.append(
            {
                "type": "memory",
                "priority": "medium",
                "message": "RAM usage is high. Consider closing unused applications.",
                "action": "Free system RAM",
            }
        )

    # Cache recommendations
    if cache_stats.get("disk_cache_size_mb", 0) > 1000:
        recommendations.append(
            {
                "type": "cache",
                "priority": "low",
                "message": "Cache size is large. Consider cleanup to free disk space.",
                "action": "Run cache cleanup",
            }
        )

    if not cache_stats.get("redis_available", False):
        recommendations.append(
            {
                "type": "cache",
                "priority": "medium",
                "message": "Redis is not available. Performance may be degraded.",
                "action": "Start Redis server",
            }
        )

    # Performance recommendations
    if not mm.config.enable_xformers:
        recommendations.append(
            {
                "type": "performance",
                "priority": "medium",
                "message": "xformers is disabled. Enable for better memory efficiency.",
                "action": "Enable xformers in config",
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations,
        "total_recommendations": len(recommendations),
    }


def get_system_health():
    """Get overall system health status"""
    try:
        mm, cm = get_managers()

        memory_info = mm.get_memory_info()
        cache_stats = cm.get_cache_stats()

        # Determine health status
        health_status = "healthy"
        issues = []

        # Check memory usage
        if memory_info.get("memory_percent", 0) > 90:
            health_status = "warning"
            issues.append("High RAM usage")

        if (
            memory_info.get("gpu_allocated_gb", 0)
            > memory_info.get("gpu_total_gb", 8) * 0.9
        ):
            health_status = "warning"
            issues.append("High GPU memory usage")

        # Check disk space
        disk_usage = psutil.disk_usage("/")
        if disk_usage.percent > 90:
            health_status = "critical"
            issues.append("Low disk space")

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "memory": memory_info,
            "cache": cache_stats,
            "disk_usage_percent": disk_usage.percent,
        }

    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@router.get("/health", response_model=Dict[str, HealthStatus])
async def health_check(db: AsyncSession = Depends(get_db_session)):
    """Comprehensive health check for all services"""
    settings = get_settings()
    health_status = {}

    # Database health
    try:
        start_time = time.time()
        result = await db.execute("SELECT 1")
        latency = (time.time() - start_time) * 1000
        health_status["database"] = HealthStatus(
            service="database", status="healthy", latency_ms=latency
        )
    except Exception as e:
        health_status["database"] = HealthStatus(
            service="database", status="unhealthy", details={"error": str(e)}
        )

    # Redis health
    try:
        start_time = time.time()
        r = redis.from_url(settings.redis_url)
        r.ping()
        latency = (time.time() - start_time) * 1000
        health_status["redis"] = HealthStatus(
            service="redis", status="healthy", latency_ms=latency
        )
    except Exception as e:
        health_status["redis"] = HealthStatus(
            service="redis", status="unhealthy", details={"error": str(e)}
        )

    # GPU health
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_percent = (gpu_allocated / gpu_memory) * 100

            health_status["gpu"] = HealthStatus(
                service="gpu",
                status="healthy" if gpu_percent < 90 else "degraded",
                details={
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_used_gb": gpu_allocated / 1024**3,
                    "memory_total_gb": gpu_memory / 1024**3,
                    "memory_percent": gpu_percent,
                },
            )
        else:
            health_status["gpu"] = HealthStatus(
                service="gpu",
                status="unavailable",
                details={"message": "No CUDA devices available"},
            )
    except Exception as e:
        health_status["gpu"] = HealthStatus(
            service="gpu", status="unhealthy", details={"error": str(e)}
        )

    # Celery worker health
    try:
        active_workers = celery_app.control.inspect().active()
        worker_count = len(active_workers) if active_workers else 0

        health_status["workers"] = HealthStatus(
            service="workers",
            status="healthy" if worker_count > 0 else "degraded",
            details={
                "active_workers": worker_count,
                "worker_nodes": list(active_workers.keys()) if active_workers else [],
            },
        )
    except Exception as e:
        health_status["workers"] = HealthStatus(
            service="workers", status="unhealthy", details={"error": str(e)}
        )

    return health_status


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get detailed system metrics"""

    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # GPU metrics
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": [],
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)

            gpu_info["devices"].append(
                {
                    "id": i,
                    "name": props.name,
                    "memory_total_gb": props.total_memory / 1024**3,
                    "memory_allocated_gb": allocated / 1024**3,
                    "memory_cached_gb": cached / 1024**3,
                    "utilization_percent": (allocated / props.total_memory) * 100,
                }
            )

    # Celery queue metrics
    try:
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        active_count = (
            sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        )

        # Get queue length from Redis
        r = redis.from_url(get_settings().redis_url)
        queue_depth = r.llen("celery")
    except:
        active_count = 0
        queue_depth = 0

    uptime = time.time() - startup_time

    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        disk_percent=disk.percent,
        gpu_info=gpu_info,
        active_workers=active_count,
        queue_depth=queue_depth,
        uptime_seconds=uptime,
    )


@router.get("/tasks")
async def get_task_statistics():
    """Get Celery task statistics"""
    try:
        inspect = celery_app.control.inspect()

        # Active tasks
        active = inspect.active()
        active_tasks = []
        if active:
            for worker, tasks in active.items():
                for task in tasks:
                    active_tasks.append(
                        {
                            "worker": worker,
                            "task_id": task["id"],
                            "task_name": task["name"],
                            "args": task.get("args", []),
                            "kwargs": task.get("kwargs", {}),
                            "time_start": task.get("time_start"),
                        }
                    )

        # Scheduled tasks
        scheduled = inspect.scheduled()
        scheduled_tasks = []
        if scheduled:
            for worker, tasks in scheduled.items():
                for task in tasks:
                    scheduled_tasks.append(
                        {
                            "worker": worker,
                            "task_id": task["request"]["id"],
                            "task_name": task["request"]["task"],
                            "eta": task["eta"],
                        }
                    )

        # Worker stats
        stats = inspect.stats()
        worker_stats = []
        if stats:
            for worker, stat in stats.items():
                worker_stats.append(
                    {
                        "worker": worker,
                        "total_tasks": stat.get("total", {}),
                        "pool_processes": stat.get("pool", {}).get("processes"),
                        "rusage": stat.get("rusage"),
                    }
                )

        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "worker_stats": worker_stats,
            "summary": {
                "active_count": len(active_tasks),
                "scheduled_count": len(scheduled_tasks),
                "worker_count": len(worker_stats),
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get task statistics: {str(e)}"},
        )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {"message": f"Task {task_id} cancelled successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to cancel task: {str(e)}"}
        )


@router.get("/logs/{service}")
async def get_service_logs(service: str, lines: int = 100):
    """Get recent logs for a service (requires log aggregation setup)"""
    # This would typically read from a centralized logging system
    # For now, return a placeholder
    return {
        "service": service,
        "message": "Log retrieval not implemented yet. Use docker logs for now.",
        "command": f"docker logs --tail {lines} saga-{service}",
    }
