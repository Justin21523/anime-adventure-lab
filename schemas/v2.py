from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class V2WorldCreate(BaseModel):
    world_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=200)
    pack: dict[str, Any] = Field(default_factory=dict)


class V2CapabilitiesResponse(BaseModel):
    api_version: str
    product: str
    persistence: str
    object_store: str
    auth_required: bool
    experimental_features_enabled: bool
    experimental_features: list[str]


class V2WorldResponse(BaseModel):
    world_id: str
    name: str
    pack: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime


class V2WorldUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    pack: dict[str, Any]


class V2StorySessionCreate(BaseModel):
    world_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    player_name: str = Field(min_length=1, max_length=120)
    persona_id: str | None = None
    runtime_preset_id: str | None = None


class V2StorySessionResponse(BaseModel):
    session_id: str
    world_id: str
    player_name: str
    persona_id: str | None
    runtime_preset_id: str | None
    state: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime


class V2StoryTurnCreate(BaseModel):
    player_input: str = Field(min_length=1, max_length=8000)
    choice_id: str | None = Field(default=None, max_length=120)
    use_agent: bool = False
    rag_mode: Literal["auto", "on", "off"] = "auto"
    include_image: bool = False


class V2Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    position: int
    excerpt: str
    score: float = Field(ge=0, le=1)


class V2StoryTurnResponse(BaseModel):
    turn_id: str
    session_id: str
    turn_number: int
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    player_input: str
    narrative: str
    choices: list[dict[str, Any]]
    citations: list[V2Citation]
    state_delta: dict[str, Any]
    trace: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class V2JobResponse(BaseModel):
    job_id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: int
    attempt_count: int
    session_id: str | None
    turn_id: str | None
    execution_id: str | None
    request_id: str | None
    dispatch_status: Literal["pending", "dispatched", "deferred"]
    lease_expires_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    replayed: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    created_at: datetime
    updated_at: datetime


class V2JobEventResponse(BaseModel):
    event_id: str
    job_id: str
    event_type: str
    from_status: str | None
    to_status: str | None
    progress: int
    attempt_count: int
    execution_id: str | None
    request_id: str | None
    actor: Literal["api", "worker", "scheduler", "admin"]
    details: dict[str, Any]
    occurred_at: datetime


class V2DocumentResponse(BaseModel):
    document_id: str
    world_id: str
    filename: str
    content_type: str | None
    checksum: str
    status: Literal["queued", "indexing", "ready", "failed"]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class V2DocumentUploadResponse(BaseModel):
    document: V2DocumentResponse
    job: V2JobResponse
    replayed: bool = False


class V2ReviewProposalCreate(BaseModel):
    world_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    session_id: str | None = None
    patch: dict[str, Any]
    reasoning: str | None = Field(default=None, max_length=4000)

    @field_validator("patch")
    @classmethod
    def validate_patch(cls, value: dict[str, Any]) -> dict[str, Any]:
        def depth(item: Any, level: int = 0) -> int:
            if not isinstance(item, dict) or not item:
                return level
            return max(depth(child, level + 1) for child in item.values())

        if not value:
            raise ValueError("patch cannot be empty")
        if "world_id" in value:
            raise ValueError("world_id is immutable")
        if depth(value) > 5:
            raise ValueError("patch exceeds maximum depth")
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16 * 1024:
            raise ValueError("patch exceeds 16 KiB")
        return value


class V2ReviewProposalResponse(BaseModel):
    proposal_id: str
    world_id: str
    session_id: str | None
    status: Literal["pending", "approved", "rejected"]
    patch: dict[str, Any]
    reasoning: str | None
    created_at: datetime
    reviewed_at: datetime | None


class V2ReviewApprovalResponse(BaseModel):
    proposal: V2ReviewProposalResponse
    world: V2WorldResponse


class V2ReconcileResponse(BaseModel):
    dispatched: list[str]
    deferred: list[str]
    failed: list[str]


class V2ServiceStatus(BaseModel):
    status: Literal["healthy", "degraded", "unavailable"]
    detail: str | None = None


class V2SystemStatusResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    api_version: str
    migration_revision: str | None
    services: dict[str, V2ServiceStatus]
    story_runtime: str
    rag_runtime: str
    worker_profile: str
    checked_at: datetime
