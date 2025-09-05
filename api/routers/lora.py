# api/routers/lora.py
"""
LoRA Management Router
"""

import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from schemas.lora import (
    LoRAListResponse,
    LoRALoadRequest,
    LoRALoadResponse,
    LoRAInfoRequest,
    LoRAInfoResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/lora/list", response_model=LoRAListResponse)
async def list_lora_models():
    """List available LoRA models"""
    try:
        from core.shared_cache import get_shared_cache

        cache = get_shared_cache()
        lora_dir = Path(cache.cache_root) / "models" / "lora"

        lora_models = []
        if lora_dir.exists():
            for model_dir in lora_dir.iterdir():
                if model_dir.is_dir():
                    lora_models.append(
                        {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "loaded": False,  # Mock status
                        }
                    )

        return LoRAListResponse(models=lora_models, total_count=len(lora_models))  # type: ignore

    except Exception as e:
        raise HTTPException(500, f"Failed to list LoRA models: {str(e)}")


@router.post("/lora/load", response_model=LoRALoadResponse)
async def load_lora_model(request: LoRALoadRequest):
    """Load LoRA model"""
    try:
        # Mock LoRA loading
        return LoRALoadResponse(  # type: ignore
            model_name=request.model_name,
            loaded=True,
            message=f"LoRA {request.model_name} loaded successfully",
        )

    except Exception as e:
        raise HTTPException(500, f"LoRA loading failed: {str(e)}")
