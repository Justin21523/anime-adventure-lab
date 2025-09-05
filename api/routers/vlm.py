# api/routers/vlm.py
"""
VLM Management Router
"""

import logging
from fastapi import APIRouter, HTTPException
from core.vlm.engine import get_vlm_engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/vlm/status")
async def get_vlm_status():
    """Get VLM engine status"""
    try:
        vlm_engine = get_vlm_engine()
        return vlm_engine.get_status()

    except Exception as e:
        raise HTTPException(500, f"Failed to get VLM status: {str(e)}")


@router.post("/vlm/load/{model_type}")
async def load_vlm_model(model_type: str):
    """Load specific VLM model"""
    try:
        vlm_engine = get_vlm_engine()

        if model_type == "caption":
            vlm_engine.load_caption_model()
            return {"message": "Caption model loaded successfully"}
        elif model_type == "vqa":
            vlm_engine.load_vqa_model()
            return {"message": "VQA model loaded successfully"}
        else:
            raise HTTPException(400, f"Unknown model type: {model_type}")

    except Exception as e:
        raise HTTPException(500, f"Failed to load VLM model: {str(e)}")


@router.post("/vlm/unload")
async def unload_vlm_models():
    """Unload all VLM models"""
    try:
        vlm_engine = get_vlm_engine()
        vlm_engine.unload_models()
        return {"message": "All VLM models unloaded successfully"}

    except Exception as e:
        raise HTTPException(500, f"Failed to unload VLM models: {str(e)}")
