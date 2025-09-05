# schemas/lora.py
"""
LoRA Management API Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .base import BaseResponse


class LoRAModel(BaseModel):
    """LoRA model information"""

    name: str = Field(..., description="LoRA model name")
    path: str = Field(..., description="Model path")
    loaded: bool = Field(..., description="Whether model is loaded")
    rank: Optional[int] = Field(None, description="LoRA rank")
    alpha: Optional[float] = Field(None, description="LoRA alpha")


class LoRAListResponse(BaseResponse):
    """LoRA models list response"""

    models: List[LoRAModel] = Field(..., description="Available LoRA models")
    total_count: int = Field(..., description="Total number of models")


class LoRALoadRequest(BaseModel):
    """LoRA model loading request"""

    model_name: str = Field(..., description="LoRA model name to load")
    scale: float = Field(1.0, ge=0.0, le=2.0, description="LoRA scale factor")


class LoRALoadResponse(BaseResponse):
    """LoRA model loading response"""

    model_name: str = Field(..., description="Loaded model name")
    loaded: bool = Field(..., description="Loading success")
    message: str = Field(..., description="Status message")


class LoRAInfoRequest(BaseModel):
    """LoRA model info request"""

    model_name: str = Field(..., description="LoRA model name")


class LoRAInfoResponse(BaseResponse):
    """LoRA model info response"""

    model_name: str = Field(..., description="Model name")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    training_info: Dict[str, Any] = Field(..., description="Training information")
