# api/routers/monitoring.py
"""Monitoring & performance router (lazy managers, graceful fallbacks)."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import torch

from core.config import get_config
from core.performance.memory_manager import MemoryManager, MemoryConfig
from core.performance.cache_manager import CacheManager, CacheConfig

# Celery is optional; import lazily in endpoints
try:
    from workers.celery_app import celery_app  # type: ignore
except Exception:  # pragma: no cover
    celery_app = None

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


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


# Lazy managers
_mm: Optional[MemoryManager] = None
_cm: Optional[CacheManager] = None


def _get_managers():
    global _mm, _cm
    if _mm is None:
        _mm = MemoryManager(MemoryConfig())
    if _cm is None:
        _cm = CacheManager(CacheConfig())
    return _mm, _cm


_start = time.time()


@router.get("/models")
def get_loaded_models():
    """List currently loaded models and memory footprints."""
    mm, _ = _get_managers()
    info: Dict[str, Any] = {}
    for key, model in mm.loaded_models.items():
        try:
            item = {
                "device": str(getattr(model, "device", "unknown")),
                "dtype": str(getattr(model, "dtype", "unknown")),
                "memory_usage": mm.memory_usage.get(key, {}),
            }
            if hasattr(model, "num_parameters"):
                item["parameters"] = model.num_parameters()
            elif hasattr(model, "get_memory_footprint"):
                item["memory_footprint"] = model.get_memory_footprint()
            info[key] = item
        except Exception as e:
            info[key] = {"error": str(e)}
    return {
        "timestamp": datetime.now().isoformat(),
        "loaded_models": info,
        "total_loaded": len(info),
    }


@router.post("/cleanup")
def force_cleanup():
    """Force memory+cache cleanup."""
    mm, cm = _get_managers()
    before = mm.get_memory_info()
    cm.cleanup_expired()
    mm.cleanup_all()
    after = mm.get_memory_info()
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "memory_before": before,
        "memory_after": after,
        "gpu_memory_freed_gb": before.get("gpu_allocated_gb", 0)
        - after.get("gpu_allocated_gb", 0),
    }


@router.post("/models/{model_key}/unload")
def unload_model(model_key: str):
    mm, _ = _get_managers()
    if model_key not in mm.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    before = mm.get_memory_info()
    mm.unload_model(model_key)
    after = mm.get_memory_info()
    return {
        "status": "unloaded",
        "model_key": model_key,
        "timestamp": datetime.now().isoformat(),
        "memory_freed_gb": before.get("gpu_allocated_gb", 0)
        - after.get("gpu_allocated_gb", 0),
    }


@router.get("/cache/stats")
def cache_stats():
    _, cm = _get_managers()
    stats = cm.get_cache_stats()
    breakdown: Dict[str, Any] = {}
    try:
        from pathlib import Path

        cache_dir = Path(cm.config.disk_cache_dir)
        for prefix in ("emb:", "img:", "kv:"):
            files = list(cache_dir.glob(f"{prefix}*.json")) + list(
                cache_dir.glob(f"{prefix}*.pkl")
            )
            breakdown[prefix.rstrip(":")] = {
                "files": len(files),
                "size_mb": sum(f.stat().st_size for f in files) / 1024**2,
            }
    except Exception as e:
        breakdown = {"error": str(e)}
    return {
        "timestamp": datetime.now().isoformat(),
        "overview": stats,
        "breakdown": breakdown,
    }


@router.post("/cache/cleanup")
def cache_cleanup():
    _, cm = _get_managers()
    before = cm.get_cache_stats()
    cm.cleanup_expired()
    after = cm.get_cache_stats()
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "files_before": before.get("disk_cache_files", 0),
        "files_after": after.get("disk_cache_files", 0),
        "size_freed_mb": before.get("disk_cache_size_mb", 0)
        - after.get("disk_cache_size_mb", 0),
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


@router.get("/performance/recommendations")
def performance_recommendations():
    mm, cm = _get_managers()
    recs: List[Dict[str, Any]] = []
    mem = mm.get_memory_info()
    cache = cm.get_cache_stats()

    gpu_total = max(mem.get("gpu_total_gb", 8), 1)
    gpu_pct = (mem.get("gpu_allocated_gb", 0) / gpu_total) * 100
    if gpu_pct > 80:
        recs.append(
            {
                "type": "memory",
                "priority": "high",
                "message": "High GPU memory usage.",
                "action": "Enable CPU offload or reduce batch size.",
            }
        )
    if mem.get("memory_percent", 0) > 85:
        recs.append(
            {
                "type": "memory",
                "priority": "medium",
                "message": "High RAM usage.",
                "action": "Free system RAM.",
            }
        )
    if cache.get("disk_cache_size_mb", 0) > 1000:
        recs.append(
            {
                "type": "cache",
                "priority": "low",
                "message": "Large disk cache.",
                "action": "Run cache cleanup.",
            }
        )
    if not cache.get("redis_available", False):
        recs.append(
            {
                "type": "cache",
                "priority": "medium",
                "message": "Redis not available.",
                "action": "Start Redis server.",
            }
        )
    if not mm.config.enable_xformers:
        recs.append(
            {
                "type": "performance",
                "priority": "medium",
                "message": "xformers disabled.",
                "action": "Enable xformers in config.",
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "recommendations": recs,
        "total_recommendations": len(recs),
    }


@router.get("/health", response_model=Dict[str, HealthStatus])
async def health_check():
    """Comprehensive health check for DB/Redis/GPU/Workers (best-effort)."""
    out: Dict[str, HealthStatus] = {}
    cfg = get_config()

    # Database (optional async engine)
    try:
        from core.database import get_db_session  # type: ignore

        async with get_db_session() as db:
            start = time.time()
            await db.execute("SELECT 1")
            out["database"] = HealthStatus(
                service="database",
                status="healthy",
                latency_ms=(time.time() - start) * 1000,
            )
    except Exception as e:
        out["database"] = HealthStatus(
            service="database", status="unhealthy", details={"error": str(e)}
        )

    # Redis
    try:
        import redis  # type: ignore

        start = time.time()
        r = redis.from_url(getattr(cfg, "redis_url", "redis://localhost:6379/0"))
        r.ping()
        out["redis"] = HealthStatus(
            service="redis", status="healthy", latency_ms=(time.time() - start) * 1000
        )
    except Exception as e:
        out["redis"] = HealthStatus(
            service="redis", status="unhealthy", details={"error": str(e)}
        )

    # GPU
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            alloc = torch.cuda.memory_allocated(0)
            out["gpu"] = HealthStatus(
                service="gpu",
                status=(
                    "healthy"
                    if alloc / max(props.total_memory, 1) < 0.9
                    else "degraded"
                ),
                details={
                    "device_name": props.name,
                    "memory_used_gb": alloc / 1024**3,
                    "memory_total_gb": props.total_memory / 1024**3,
                },
            )
        else:
            out["gpu"] = HealthStatus(
                service="gpu",
                status="unavailable",
                details={"message": "No CUDA devices"},
            )
    except Exception as e:
        out["gpu"] = HealthStatus(
            service="gpu", status="unhealthy", details={"error": str(e)}
        )

    # Celery workers
    try:
        if celery_app is None:
            raise RuntimeError("Celery not configured")
        insp = celery_app.control.inspect()
        active = insp.active() or {}
        out["workers"] = HealthStatus(
            service="workers",
            status="healthy" if len(active) > 0 else "degraded",
            details={
                "active_workers": len(active),
                "worker_nodes": list(active.keys()),
            },
        )
    except Exception as e:
        out["workers"] = HealthStatus(
            service="workers", status="unhealthy", details={"error": str(e)}
        )

    return out


@router.get("/metrics", response_model=SystemMetrics)
def system_metrics():
    """System metrics for dashboards/scrapers."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpu_info: Optional[Dict[str, Any]] = None
    if torch.cuda.is_available():
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "id": i,
                    "name": torch.cuda.get_device_properties(i).name,
                    "memory_total_gb": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                }
                for i in range(torch.cuda.device_count())
            ],
        }

    # Celery queue depth via Redis (best-effort)
    active_count = 0
    queue_depth = 0
    try:
        if celery_app is not None:
            insp = celery_app.control.inspect()
            active = insp.active() or {}
            active_count = sum(len(v) for v in active.values())
        import redis  # type: ignore

        r = redis.from_url(
            getattr(get_config(), "redis_url", "redis://localhost:6379/0")
        )
        queue_depth = int(r.llen("celery"))
    except Exception:
        pass

    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_percent=cpu,
        memory_percent=mem.percent,
        disk_percent=disk.percent,
        gpu_info=gpu_info,
        active_workers=active_count,
        queue_depth=queue_depth,
        uptime_seconds=time.time() - _start,
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
