# schemas/caption.py
"""
Image Caption API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from .schemas_base import BaseRequest, BaseResponse, UsageInfo, BaseParameters


class CaptionParameters(BaseParameters):
    """Caption-specific parameters"""

    max_length: int = Field(50, ge=10, le=200, description="Maximum caption length")
    num_beams: int = Field(3, ge=1, le=10, description="Number of beams for generation")
    temperature: float = Field(
        0.7, ge=0.1, le=2.0, description="Generation temperature"
    )
    language: str = Field("en", description="Caption language")

    @field_validator("language")
    def validate_language(cls, v):
        if v not in ["en", "zh", "zh-TW", "zh-CN"]:
            raise ValueError("Language must be en/zh/zh-TW/zh-CN")
        return v


class CaptionRequest(BaseRequest):
    """Image caption generation request"""

    # Image will be uploaded as file, so not in schema
    parameters: Optional[CaptionParameters] = Field(default_factory=CaptionParameters)  # type: ignore


class CaptionResponse(BaseResponse):
    """Image caption generation response"""

    caption: str = Field(..., description="Generated caption")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Caption confidence")
    model_used: str = Field(..., description="Model used for generation")
    language: str = Field(..., description="Caption language")
    parameters: CaptionParameters = Field(..., description="Parameters used")
    image_info: Dict[str, Any] = Field(..., description="Image metadata")


class BatchCaptionRequest(BaseRequest):
    """Batch caption generation request"""

    max_length: int = Field(50, ge=10, le=200)
    num_beams: int = Field(3, ge=1, le=5)
    language: str = Field("en")

    # Batch settings
    batch_size: int = Field(4, ge=1, le=10, description="Processing batch size")

    @field_validator("language", mode="after")
    def validate_language(cls, v):
        if v not in ["en", "zh", "auto"]:
            raise ValueError("Language must be 'en', 'zh', or 'auto'")
        return v


class BatchCaptionResponse(BaseResponse):
    """Batch caption generation response"""

    results: List[Dict[str, Any]] = Field(..., description="Caption results")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    parameters: CaptionParameters = Field(..., description="Parameters used")
