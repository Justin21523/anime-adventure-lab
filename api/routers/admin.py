# api/routers/admin.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import psutil
import torch

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats")
async def get_system_stats() -> Dict[str, Any]:
    """Return basic system and GPU stats. Each probe is isolated to avoid cascading failures."""
    stats: Dict[str, Any] = {"system": {}, "gpu": {}}

    # system metrics
    try:
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }
    except Exception as e:
        stats["system"] = {
            "status": "basic_check_ok",
            "note": f"psutil unavailable: {e}",
        }

    # gpu metrics
    try:
        gpu_avail = torch.cuda.is_available()
        stats["gpu"]["available"] = gpu_avail
        stats["gpu"]["count"] = torch.cuda.device_count() if gpu_avail else 0
        if gpu_avail:
            stats["gpu"]["memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1024**3, 3
            )
            stats["gpu"]["memory_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1024**3, 3
            )
    except Exception as e:
        # Keep response shape stable even if torch isn't present
        stats["gpu"].update(
            {"available": False, "count": 0, "note": f"torch unavailable: {e}"}
        )

    return stats
