# api/routers/admin.py
"""
Admin Operations Router
"""

import logging
import os
import psutil
from fastapi import APIRouter, HTTPException, Depends
from api.dependencies import get_llm, get_vlm
from schemas.admin import AdminSystemInfoResponse, AdminModelControlRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/admin/system", response_model=AdminSystemInfoResponse)
async def get_system_info():
    """Get detailed system information"""
    try:
        from core.shared_cache import get_shared_cache

        cache = get_shared_cache()

        return AdminSystemInfoResponse(  # type: ignore
            cache_stats=cache.get_cache_stats(),
            loaded_models={
                "llm": get_llm().list_loaded_models(),
                "vlm": get_vlm().get_status(),
            },
            system_resources={
                "gpu_memory_gb": 0.0,
                "cpu_cores": os.cpu_count() or 1,
                "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            },
        )

    except Exception as e:
        raise HTTPException(500, f"Failed to get system info: {str(e)}")


@router.post("/admin/models/control")
async def control_models(request: AdminModelControlRequest):
    """Load/unload models"""
    try:
        if request.action == "unload_all":
            get_llm().unload_all()
            get_vlm().unload_models()
            return {"message": "All models unloaded"}
        else:
            return {"message": f"Action {request.action} not implemented"}

    except Exception as e:
        raise HTTPException(500, f"Model control failed: {str(e)}")
