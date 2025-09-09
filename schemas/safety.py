# schemas/safety.py
"""
Safety Filter API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from .schemas_base import BaseRequest, BaseResponse


class SafetyCheckRequest(BaseRequest):
    """Safety check request"""

    content: str = Field(..., description="Content to check")
    content_type: str = Field("text", description="Type of content")

    @field_validator("content_type")
    def validate_content_type(cls, v):
        if v not in ["text", "image", "audio"]:
            raise ValueError("Content type must be text/image/audio")
        return v


class SafetyCheckResponse(BaseResponse):
    """Safety check response"""

    safe: bool = Field(..., description="Whether content is safe")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    detected_issues: List[str] = Field(..., description="List of detected issues")
    confidence: float = Field(..., description="Detection confidence")
    recommendations: List[str] = Field(..., description="Safety recommendations")
