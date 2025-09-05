# schemas/finetune.py
"""
Fine-tuning API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from .base import BaseRequest, BaseResponse


class FinetuneParameters(BaseModel):
    """Fine-tuning parameters"""

    learning_rate: float = Field(2e-4, gt=0.0, description="Learning rate")
    batch_size: int = Field(2, ge=1, le=16, description="Training batch size")
    epochs: int = Field(3, ge=1, le=10, description="Training epochs")
    lora_rank: int = Field(16, ge=4, le=64, description="LoRA rank")
    lora_alpha: float = Field(32.0, gt=0.0, description="LoRA alpha")


class FinetuneRequest(BaseRequest):
    """Fine-tuning job request"""

    model_name: str = Field(..., description="Base model to fine-tune")
    dataset_path: str = Field(..., description="Training dataset path")
    output_name: str = Field(..., description="Output model name")
    parameters: Optional[FinetuneParameters] = Field(default_factory=FinetuneParameters)  # type: ignore


class FinetuneResponse(BaseResponse):
    """Fine-tuning job response"""

    job_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Job status")
    estimated_time_hours: float = Field(..., description="Estimated training time")
    parameters: FinetuneParameters = Field(..., description="Training parameters")


class FinetuneStatusResponse(BaseResponse):
    """Fine-tuning status response"""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress_percentage: float = Field(..., description="Training progress")
    current_epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    current_loss: Optional[float] = Field(None, description="Current training loss")
    model_path: Optional[str] = Field(
        None, description="Output model path (when completed)"
    )
