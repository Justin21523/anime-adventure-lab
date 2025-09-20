# schemas/batch.py
"""
Batch Processing API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class BatchStatus(str, Enum):
    """Batch job status enumeration"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class BatchJobRequest(BaseModel):
    """Request model for submitting batch jobs"""

    job_type: str = Field(..., description="Type of batch job (caption, vqa, chat)")
    inputs: List[Any] = Field(..., description="List of inputs to process")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Job configuration"
    )
    priority: Optional[int] = Field(
        default=5, ge=1, le=10, description="Job priority (1=highest)"
    )

    class Config:
        schema_extra = {
            "example": {
                "job_type": "caption",
                "inputs": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                "config": {"max_length": 50, "num_beams": 3, "temperature": 0.7},
                "priority": 5,
            }
        }


class BatchJobResponse(BaseModel):
    """Response model for batch job operations"""

    job_id: str = Field(..., description="Unique job identifier")
    task_id: str = Field(..., description="Celery task identifier")
    status: BatchStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    total_items: int = Field(..., description="Total number of items to process")
    processed_items: int = Field(default=0, description="Number of items processed")
    failed_items: int = Field(default=0, description="Number of items that failed")
    results_path: Optional[str] = Field(None, description="Path to results file")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )


class BatchJobList(BaseModel):
    """Response model for listing batch jobs"""

    jobs: List[BatchJobResponse] = Field(..., description="List of batch jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")


class TaskProgress(BaseModel):
    """Model for task progress tracking"""

    task_id: str = Field(..., description="Task identifier")
    total_items: int = Field(..., description="Total items to process")
    processed_items: int = Field(..., description="Items processed so far")
    failed_items: int = Field(default=0, description="Items that failed")
    progress_percent: float = Field(..., description="Progress percentage")
    current_item: Optional[str] = Field(None, description="Currently processing item")
    start_time: datetime = Field(..., description="Task start time")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion"
    )
    partial_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Latest partial results"
    )
