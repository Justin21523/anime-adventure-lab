# api/routers/admin.py
"""
Admin Operations Router
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from core.llm.adapter import get_llm_adapter
from core.vlm.engine import get_vlm_engine
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
                "llm": get_llm_adapter().list_loaded_models(),
                "vlm": get_vlm_engine().get_status(),
            },
            system_resources={
                "gpu_memory_gb": 12.0,  # Mock data
                "cpu_cores": 8,
                "ram_gb": 32.0,
            },
        )

    except Exception as e:
        raise HTTPException(500, f"Failed to get system info: {str(e)}")


@router.post("/admin/models/control")
async def control_models(request: AdminModelControlRequest):
    """Load/unload models"""
    try:
        if request.action == "unload_all":
            get_llm_adapter().unload_all()
            get_vlm_engine().unload_models()
            return {"message": "All models unloaded"}
        else:
            return {"message": f"Action {request.action} not implemented"}

    except Exception as e:
        raise HTTPException(500, f"Model control failed: {str(e)}")
