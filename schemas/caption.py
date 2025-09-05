# schemas/caption.py
"""
Image Caption API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from .base import BaseRequest, BaseResponse, UsageInfo, BaseParameters


class CaptionParameters(BaseParameters):
    """Caption-specific parameters"""

    max_length: int = Field(50, ge=10, le=200, description="Maximum caption length")
    num_beams: int = Field(3, ge=1, le=5, description="Number of beams for generation")
    language: str = Field("en", description="Caption language (en/zh)")

    @field_validator("language", mode="after")
    def validate_language(cls, v):
        if v not in ["en", "zh", "auto"]:
            raise ValueError("Language must be 'en', 'zh', or 'auto'")
        return v


class CaptionRequest(BaseRequest):
    """Image caption generation request"""

    # Image will be uploaded as file, so not in schema
    parameters: Optional[CaptionParameters] = Field(default_factory=CaptionParameters)  # type: ignore


class CaptionResponse(BaseResponse):
    """Image caption generation response"""

    caption: str = Field(..., description="Generated image caption")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Caption confidence score"
    )
    model_used: str = Field(..., description="Model used for generation")
    language: str = Field(..., description="Detected/requested language")

    # Unified parameters field
    parameters: CaptionParameters = Field(
        ..., description="Parameters used for generation"
    )

    # Optional detailed info
    usage: Optional[UsageInfo] = Field(None, description="Resource usage information")
    image_stats: Optional[Dict[str, Any]] = Field(
        None, description="Image analysis statistics"
    )


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

    results: list[CaptionResponse] = Field(
        ..., description="Caption results for each image"
    )
    total_processed: int = Field(..., description="Total images processed")
    success_count: int = Field(..., description="Successfully processed images")
    error_count: int = Field(..., description="Failed images")

    # Batch performance
    total_time_ms: Optional[float] = Field(None, description="Total processing time")
    average_time_per_image_ms: Optional[float] = Field(
        None, description="Average time per image"
    )
