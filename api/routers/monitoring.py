# api/routers/monitoring.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
import psutil
import torch
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.performance.memory_manager import MemoryManager, MemoryConfig
from core.performance.cache_manager import CacheManager, CacheConfig

# Global instances
memory_manager = None
cache_manager = None

def get_managers():
    global memory_manager, cache_manager
    if memory_manager is None:
        memory_manager = MemoryManager(MemoryConfig())
    if cache_manager is None:
        cache_manager = CacheManager(CacheConfig())
    return memory_manager, cache_manager

@router.get("/metrics")
def get_detailed_metrics():
    """Get detailed system metrics"""
    mm, cm = get_managers()

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # GPU metrics
    gpu_metrics = {}
    if torch.cuda.is_available():
        gpu_metrics = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "utilization_percent": None  # Would need nvidia-ml-py for this
        }

    # Cache metrics
    cache_stats = cm.get_cache_stats()

    # Memory manager stats
    memory_stats = mm.get_memory_info()

    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / 1024**3,
            "memory_used_gb": memory.used / 1024**3,
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / 1024**3,
            "disk_used_gb": disk.used / 1024**3,
            "disk_percent": disk.percent
        },
        "gpu": gpu_metrics,
        "cache": cache_stats,
        "memory_manager": memory_stats
    }

@router.get("/models")
def get_loaded_models():
    """Get information about currently loaded models"""
    mm, _ = get_managers()

    models_info = {}
    for model_key, model in mm.loaded_models.items():
        try:
            model_info = {
                "loaded_at": datetime.now().isoformat(),  # Would track this properly
                "device": str(getattr(model, 'device', 'unknown')),
                "dtype": str(getattr(model, 'dtype', 'unknown')),
                "memory_usage": mm.memory_usage.get(model_key, {})
            }

            # Try to get model size
            if hasattr(model, 'num_parameters'):
                model_info["parameters"] = model.num_parameters()
            elif hasattr(model, 'get_memory_footprint'):
                model_info["memory_footprint"] = model.get_memory_footprint()

            models_info[model_key] = model_info

        except Exception as e:
            models_info[model_key] = {"error": str(e)}

    return {
        "timestamp": datetime.now().isoformat(),
        "loaded_models": models_info,
        "total_loaded": len(models_info)
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
            initial_memory.get("gpu_allocated_gb", 0) -
            final_memory.get("gpu_allocated_gb", 0)
        )
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
            initial_memory.get("gpu_allocated_gb", 0) -
            final_memory.get("gpu_allocated_gb", 0)
        )
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
        for prefix in ['emb:', 'img:', 'kv:']:
            files = list(cache_dir.glob(f"{prefix}*.json")) + list(cache_dir.glob(f"{prefix}*.pkl"))
            cache_breakdown[prefix.rstrip(':')] = {
                "files": len(files),
                "size_mb": sum(f.stat().st_size for f in files) / 1024**2
            }

    except Exception as e:
        cache_breakdown = {"error": str(e)}

    return {
        "timestamp": datetime.now().isoformat(),
        "overview": stats,
        "breakdown": cache_breakdown
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
            initial_stats.get("disk_cache_size_mb", 0) -
            final_stats.get("disk_cache_size_mb", 0)
        )
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
        memory_info.get("gpu_allocated_gb", 0) /
        memory_info.get("gpu_total_gb", 8) * 100
    )

    if gpu_usage_percent > 80:
        recommendations.append({
            "type": "memory",
            "priority": "high",
            "message": "GPU memory usage is high. Consider enabling CPU offload or reducing batch size.",
            "action": "Enable cpu_offload in memory config"
        })

    if memory_info.get("memory_percent", 0) > 85:
        recommendations.append({
            "type": "memory",
            "priority": "medium",
            "message": "RAM usage is high. Consider closing unused applications.",
            "action": "Free system RAM"
        })

    # Cache recommendations
    if cache_stats.get("disk_cache_size_mb", 0) > 1000:
        recommendations.append({
            "type": "cache",
            "priority": "low",
            "message": "Cache size is large. Consider cleanup to free disk space.",
            "action": "Run cache cleanup"
        })

    if not cache_stats.get("redis_available", False):
        recommendations.append({
            "type": "cache",
            "priority": "medium",
            "message": "Redis is not available. Performance may be degraded.",
            "action": "Start Redis server"
        })

    # Performance recommendations
    if not mm.config.enable_xformers:
        recommendations.append({
            "type": "performance",
            "priority": "medium",
            "message": "xformers is disabled. Enable for better memory efficiency.",
            "action": "Enable xformers in config"
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    }get("/health")
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

        if memory_info.get("gpu_allocated_gb", 0) > memory_info.get("gpu_total_gb", 8) * 0.9:
            health_status = "warning"
            issues.append("High GPU memory usage")

        # Check disk space
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            health_status = "critical"
            issues.append("Low disk space")

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "memory": memory_info,
            "cache": cache_stats,
            "disk_usage_percent": disk_usage.percent
        }

    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
