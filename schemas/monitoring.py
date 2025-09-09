# ===================================
# schemas/monitoring.py
"""
System Monitoring API Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from .schemas_base import BaseResponse


class SystemMetrics(BaseModel):
    """System performance metrics"""

    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU usage percentage")
    gpu_memory_used_gb: Optional[float] = Field(None, description="GPU memory used")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")


class MonitoringResponse(BaseResponse):
    """System monitoring response"""

    metrics: SystemMetrics = Field(..., description="Current system metrics")
    model_stats: Dict[str, Any] = Field(..., description="Model loading statistics")
    api_stats: Dict[str, Any] = Field(..., description="API request statistics")
    cache_stats: Dict[str, Any] = Field(..., description="Cache usage statistics")
