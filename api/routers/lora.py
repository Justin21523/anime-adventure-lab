# api/routers/lora.py
"""
LoRA Management Router
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from schemas.lora import (
    LoRAListResponse,
    LoRALoadRequest,
    LoRALoadResponse,
    LoRAInfoRequest,
    LoRAInfoResponse,
)
from api.dependencies import get_lora, get_t2i
from core.train.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()
model_registry = ModelRegistry()


@router.get("/lora/list", response_model=LoRAListResponse)
async def list_lora_models(lora=Depends(get_lora)):
    """List available LoRA models"""
    try:
        loras = lora.list_available_loras() if hasattr(lora, "list_available_loras") else []
        models = [
            {
                "name": item.get("lora_id", item.get("name", "unknown")),
                "path": item.get("path", ""),
                "loaded": item.get("loaded", False),
                "rank": item.get("rank"),
                "alpha": item.get("alpha"),
            }
            for item in loras
        ]
        return LoRAListResponse(models=models, total_count=len(models))  # type: ignore

    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Failed to list LoRA models: {str(e)}")


@router.post("/lora/load", response_model=LoRALoadResponse)
async def load_lora_model(request: LoRALoadRequest, engine=Depends(get_t2i)):
    """Load LoRA model"""
    try:
        if not hasattr(engine, "lora_manager"):
            raise RuntimeError("LoRA manager not available")

        pipeline = getattr(engine, "current_pipeline", None)
        if pipeline is None:
            raise RuntimeError("No active diffusion pipeline to attach LoRA")

        loaded = engine.lora_manager.load_lora(  # type: ignore
            pipeline, request.model_name, request.scale
        )

        message = (
            f"LoRA {request.model_name} loaded"
            if loaded
            else f"LoRA {request.model_name} load failed"
        )
        if loaded:
            model_registry.add(
                request.model_name,
                {
                    "model_type": "lora",
                    "base_model": getattr(engine, "current_model_id", ""),
                    "model_path": str(
                        Path(engine.lora_manager.lora_cache_dir) / request.model_name
                    ),
                },
            )

        return LoRALoadResponse(  # type: ignore
            model_name=request.model_name,
            loaded=loaded,
            message=message,
        )

    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"LoRA loading failed: {str(e)}")


@router.post("/lora/info", response_model=LoRAInfoResponse)
async def lora_info(request: LoRAInfoRequest, lora=Depends(get_lora)):
    """Get LoRA metadata and any registered training info."""
    try:
        model_info: Any = {}
        training_info: Any = {}

        if hasattr(lora, "get_lora_info"):
            model_info = lora.get_lora_info(request.model_name) or {}
        registry_info = model_registry.get_model(request.model_name)
        if registry_info:
            training_info = registry_info

        return LoRAInfoResponse(  # type: ignore
            model_name=request.model_name,
            model_info=model_info or {},
            training_info=training_info or {},
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to fetch LoRA info: %s", e)
        raise HTTPException(500, f"Failed to fetch LoRA info: {str(e)}") from e
