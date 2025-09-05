# schemas/controlnet.py
"""
ControlNet API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from .base import BaseRequest, BaseResponse, BaseParameters


class ControlNetParameters(BaseParameters):
    """ControlNet parameters"""

    controlnet_type: str = Field("pose", description="ControlNet type")
    conditioning_scale: float = Field(
        1.0, ge=0.0, le=2.0, description="Conditioning scale"
    )
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    steps: int = Field(20, ge=10, le=50, description="Inference steps")

    @field_validator("controlnet_type")
    def validate_controlnet_type(cls, v):
        allowed = ["pose", "depth", "canny", "openpose", "scribble"]
        if v not in allowed:
            raise ValueError(f"ControlNet type must be one of: {allowed}")
        return v


class ControlNetRequest(BaseRequest):
    """ControlNet generation request"""

    prompt: str = Field(..., min_length=3, description="Text prompt")
    # control_image uploaded as file
    parameters: Optional[ControlNetParameters] = Field(
        default_factory=ControlNetParameters  # type: ignore
    )


class ControlNetResponse(BaseResponse):
    """ControlNet generation response"""

    image_path: str = Field(..., description="Generated image path")
    prompt: str = Field(..., description="Original prompt")
    controlnet_type: str = Field(..., description="ControlNet type used")
    parameters: ControlNetParameters = Field(..., description="Parameters used")
