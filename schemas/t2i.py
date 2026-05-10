# schemas/t2i.py
"""
Text-to-Image API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from .schemas_base import BaseRequest, BaseResponse, BaseParameters


class T2IParameters(BaseParameters):
    """Text-to-Image parameters"""

    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    steps: int = Field(20, ge=10, le=50, description="Inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed")
    model: str = Field("stable-diffusion-xl", description="T2I model to use")


class T2IRequest(BaseRequest):
    """Text-to-Image generation request"""

    prompt: str = Field(..., min_length=3, max_length=1000, description="Text prompt")
    parameters: Optional[T2IParameters] = Field(default_factory=T2IParameters)  # type: ignore


class T2IResponse(BaseResponse):
    """Text-to-Image generation response"""

    image_path: str = Field(..., description="Generated image path")
    prompt: str = Field(..., description="Original prompt")
    model_used: str = Field(..., description="Model used for generation")
    parameters: T2IParameters = Field(..., description="Parameters used")
    generation_info: Dict[str, Any] = Field(..., description="Generation details")


class T2ILoRAConfig(BaseModel):
    """LoRA config for generation (UI-friendly)."""

    name: str = Field(..., min_length=1, description="LoRA id/name")
    weight: float = Field(0.8, ge=0.0, le=2.0, description="LoRA weight/scale")


class T2IGenerateRequest(BaseModel):
    """UI-friendly generation request (matches frontend/react)."""

    prompt: str = Field(..., min_length=3, max_length=2000)
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(1024, ge=256, le=1024)
    height: int = Field(1024, ge=256, le=1024)
    num_inference_steps: int = Field(30, ge=1, le=80)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)
    num_images: int = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None, description="Optional random seed")
    loras: List[T2ILoRAConfig] = Field(default_factory=list)
    session_id: Optional[str] = Field(None, description="Optional story session id")
    model_id: Optional[str] = Field(None, description="Optional diffusion model id/path override")


class T2IGeneratedImage(BaseModel):
    image_url: str
    seed: Optional[int] = None
    prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class T2IGenerateResponse(BaseModel):
    images: List[T2IGeneratedImage]
    generation_time: float
    model_used: str
