"""Queue request schemas (adapted for anime-adventure-lab)."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueueSubmitRequest(BaseModel):
    """Request to submit a task to the queue."""
    task_type: Literal["txt2img", "img2img", "upscale", "face_restore", "video"]
    payload: dict = Field(default_factory=dict)
    priority: Literal["low", "normal", "high", "critical"] = "normal"


class QueueStatusRequest(BaseModel):
    """Request to check task status."""
    task_id: str


class QueueCancelRequest(BaseModel):
    """Request to cancel a task."""
    task_id: str


class QueueTaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    priority: str
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class QueueStatsResponse(BaseModel):
    total: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
