# api/routers/t2i.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.shared_cache import bootstrap_cache
from core.t2i.pipeline import t2i_pipeline

# Setup cache on module import
cache = bootstrap_cache()

router = APIRouter(prefix="/t2i", tags=["text-to-image"])


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(768, ge=256, le=1024, description="Image width")
    height: int = Field(768, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(
        25, ge=10, le=50, description="Number of inference steps"
    )
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    style_id: Optional[str] = Field(None, description="Style preset ID")
    scene_id: str = Field("scene_001", description="Scene identifier")
    image_type: str = Field("bg", description="Image type (bg/character/portrait)")


class GenerationResponse(BaseModel):
    success: bool
    image_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    current_model: Optional[str]
    available_styles: Dict[str, str]
    gpu_available: bool
    memory_info: Optional[Dict[str, Any]] = None


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """T2I service health check"""
    try:
        gpu_available = torch.cuda.is_available()
        memory_info = None

        if gpu_available:
            memory_info = {
                "allocated": torch.cuda.memory_allocated(0),
                "reserved": torch.cuda.memory_reserved(0),
                "max_allocated": torch.cuda.max_memory_allocated(0),
            }

        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "memory_info": memory_info,
            "current_model": t2i_pipeline.current_model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/models", response_model=ModelInfo)
async def get_model_info():
    """Get current model and available styles"""
    try:
        memory_info = None
        if torch.cuda.is_available():
            memory_info = {
                "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
            }

        return ModelInfo(
            current_model=t2i_pipeline.current_model,
            available_styles=t2i_pipeline.get_available_styles(),
            gpu_available=torch.cuda.is_available(),
            memory_info=memory_info,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate image from text prompt"""
    try:
        # Validate style if provided
        if (
            request.style_id
            and request.style_id not in t2i_pipeline.get_available_styles()
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Style '{request.style_id}' not found. Available: {list(t2i_pipeline.get_available_styles().keys())}",
            )

        # Generate image
        result = t2i_pipeline.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            style_id=request.style_id,
            scene_id=request.scene_id,
            image_type=request.image_type,
        )

        if result["success"]:
            return GenerationResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/load_model")
async def load_model(model_id: str = Query(..., description="Model ID to load")):
    """Load specific T2I model"""
    try:
        success = t2i_pipeline.load_model(model_id)
        if success:
            return {
                "success": True,
                "model": model_id,
                "message": "Model loaded successfully",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")


@router.post("/unload_model")
async def unload_model():
    """Unload current model to free VRAM"""
    try:
        t2i_pipeline.unload_model()
        return {"success": True, "message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model unloading error: {str(e)}")


@router.get("/styles")
async def list_styles():
    """List available style presets"""
    return {
        "styles": t2i_pipeline.get_available_styles(),
        "count": len(t2i_pipeline.style_presets),
    }
