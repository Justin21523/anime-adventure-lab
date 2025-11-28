# schemas/export.py
"""
Export/Import API Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from .schemas_base import BaseRequest, BaseResponse


class ExportRequest(BaseRequest):
    """Export data request"""

    export_type: str = Field(..., description="Type of export")
    format: str = Field("json", description="Export format")
    items: List[str] = Field(..., description="Items to export")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional export options"
    )

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
    results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Per-item export results"
    )
    warnings: Optional[List[str]] = Field(None, description="Non-fatal warnings")


class ConvertRequest(BaseRequest):
    """Format conversion request"""

    input_path: str = Field(..., description="Input file path")
    output_path: Optional[str] = Field(
        None, description="Optional output path (auto if omitted)"
    )
    source_format: str = Field(..., description="Source format")
    target_format: str = Field(..., description="Target format")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Conversion options"
    )


class ConvertResponse(BaseResponse):
    """Format conversion response"""

    input_path: str = Field(..., description="Input file")
    output_path: str = Field(..., description="Converted output path")
    source_format: str = Field(..., description="Source format")
    target_format: str = Field(..., description="Target format")
    details: Dict[str, Any] = Field(default_factory=dict, description="Extra details")
