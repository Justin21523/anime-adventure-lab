from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    item_id: str
    filename: str
    image_url: str
    image_path: Optional[str] = None
    caption: str = ""
    tags: List[str] = Field(default_factory=list)


class DatasetDetail(BaseModel):
    dataset_id: str
    world_id: str
    name: str
    dataset_path: str
    created_at: str
    updated_at: str
    total_images: int = 0
    has_metadata_jsonl: bool = False
    metadata_jsonl_path: Optional[str] = None
    items: List[DatasetItem] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    dataset_id: str
    world_id: str
    name: str
    dataset_path: str
    created_at: str
    updated_at: str
    total_images: int = 0
    has_metadata_jsonl: bool = False


class DatasetUploadResponse(BaseModel):
    success: bool = True
    dataset: DatasetDetail


class DatasetListResponse(BaseModel):
    success: bool = True
    world_id: str
    datasets: List[DatasetSummary] = Field(default_factory=list)


class DatasetItemUpdateRequest(BaseModel):
    caption: Optional[str] = None
    tags: Optional[List[str]] = None


class DatasetItemUpdateResponse(BaseModel):
    success: bool = True
    item: Dict[str, Any] = Field(default_factory=dict)


class DatasetBuildMetadataResponse(BaseModel):
    success: bool = True
    metadata_jsonl_path: str
    total_images: int


class DatasetCaptionJobResponse(BaseModel):
    success: bool = True
    job_id: str
    status: str = "queued"

