# schemas/admin.py
"""
Admin API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Tuple, Any, Optional
from .base import BaseResponse


class AdminSystemInfoResponse(BaseResponse):
    """System information response"""

    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    loaded_models: Dict[str, Any] = Field(..., description="Currently loaded models")
    system_resources: Dict[str, Any] = Field(..., description="System resource usage")


class AdminModelControlRequest(BaseModel):
    """Model control request"""

    action: str = Field(..., description="Action to perform")
    model_name: Optional[str] = Field(None, description="Specific model name")

    @field_validator("action")
    def validate_action(cls, v):
        allowed_actions = ["load", "unload", "reload", "unload_all"]
        if v not in allowed_actions:
            raise ValueError(f"Action must be one of: {allowed_actions}")
        return v
