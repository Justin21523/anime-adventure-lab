# schemas/batch.py
"""
Batch Processing API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Any, Optional
from .base import BaseRequest, BaseResponse, BaseParameters


class BatchParameters(BaseParameters):
    """Batch processing parameters"""

    batch_size: int = Field(4, ge=1, le=20, description="Processing batch size")
    max_retries: int = Field(2, ge=0, le=5, description="Max retries for failed items")
    priority: str = Field("normal", description="Job priority")

    @field_validator("priority")
    def validate_priority(cls, v):
        if v not in ["low", "normal", "high", "urgent"]:
            raise ValueError("Priority must be low/normal/high/urgent")
        return v


class BatchJobRequest(BaseRequest):
    """Batch job submission request"""

    job_type: str = Field(..., description="Type of batch job")
    items: List[Any] = Field(..., min_items=1, description="Items to process")  # type: ignore
    parameters: Optional[BatchParameters] = Field(default_factory=BatchParameters)  # type: ignore


class BatchJobResponse(BaseResponse):
    """Batch job submission response"""

    job_id: str = Field(..., description="Batch job identifier")
    status: str = Field(..., description="Job status")
    total_items: int = Field(..., description="Total items to process")
    estimated_time_minutes: float = Field(..., description="Estimated completion time")
    parameters: BatchParameters = Field(..., description="Parameters used")


class BatchStatusResponse(BaseResponse):
    """Batch job status response"""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Progress percentage"
    )
    items_completed: int = Field(..., description="Items completed")
    items_failed: int = Field(..., description="Items failed")
    estimated_remaining_minutes: float = Field(
        ..., description="Estimated remaining time"
    )
