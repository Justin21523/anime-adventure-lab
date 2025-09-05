# schemas/export.py
"""
Export/Import API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from .base import BaseRequest, BaseResponse


class ExportRequest(BaseRequest):
    """Export data request"""

    export_type: str = Field(..., description="Type of export")
    format: str = Field("json", description="Export format")
    items: List[str] = Field(..., description="Items to export")

    @field_validator("export_type")
    def validate_export_type(cls, v):
        allowed = ["models", "sessions", "documents", "configurations"]
        if v not in allowed:
            raise ValueError(f"Export type must be one of: {allowed}")
        return v

    @field_validator("format")
    def validate_format(cls, v):
        if v not in ["json", "yaml", "csv", "zip"]:
            raise ValueError("Format must be json/yaml/csv/zip")
        return v


class ExportResponse(BaseResponse):
    """Export response"""

    export_id: str = Field(..., description="Export job identifier")
    file_path: str = Field(..., description="Exported file path")
    file_size_mb: float = Field(..., description="File size in MB")
    items_exported: int = Field(..., description="Number of items exported")
