# schemas/base.py
"""
Base Pydantic Models
Shared data structures across all API endpoints
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class BaseRequest(BaseModel):
    """Base request model with common fields"""

    request_id: Optional[str] = Field(
        None, description="Optional request ID for tracking"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        # Allow extra fields for flexibility
        extra = "allow"


class BaseParameters(BaseModel):
    """Base parameters for model inference"""

    max_length: int = Field(512, ge=10, le=2000, description="Maximum output length")
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class BaseResponse(BaseModel):
    """Base response model with common fields"""

    success: bool = Field(True, description="Request success status")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    parameters: Optional[BaseParameters] = Field(
        None, description="Inference parameters used"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional response metadata"
    )

    class Config:
        # Use enum values for serialization
        use_enum_values = True


class ErrorResponse(BaseModel):
    """Standard error response"""

    success: bool = Field(False)
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Model information structure"""

    name: str = Field(..., description="Model name/identifier")
    description: Optional[str] = Field(None, description="Model description")
    version: Optional[str] = Field(None, description="Model version")
    parameters_count: Optional[str] = Field(
        None, description="Parameter count (e.g., '7B')"
    )
    languages: List[str] = Field(
        default_factory=list, description="Supported languages"
    )
    loaded: bool = Field(False, description="Whether model is currently loaded")


class UsageInfo(BaseModel):
    """Token/resource usage information"""

    prompt_tokens: Optional[int] = Field(None, description="Input tokens used")
    completion_tokens: Optional[int] = Field(
        None, description="Output tokens generated"
    )
    total_tokens: Optional[int] = Field(None, description="Total tokens")

    # Compute metrics
    inference_time_ms: Optional[float] = Field(
        None, description="Inference time in milliseconds"
    )
    gpu_memory_used_mb: Optional[float] = Field(
        None, description="GPU memory used in MB"
    )


class PaginationRequest(BaseModel):
    """Pagination parameters for list endpoints"""

    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page (max 100)")

    @field_validator("page_size", mode="after")
    def validate_page_size(cls, v):
        if v > 100:
            raise ValueError("page_size cannot exceed 100")
        return v


class PaginationResponse(BaseModel):
    """Pagination metadata for list responses"""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_prev: bool = Field(..., description="Whether there's a previous page")
