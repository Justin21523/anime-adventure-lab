from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # API/mock profile can run without persistence extras
    Vector = None  # type: ignore[assignment,misc]
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


JsonType = JSON
EmbeddingType = (
    Vector(1024).with_variant(JSON(), "sqlite") if Vector is not None else JSON()
)


class Base(DeclarativeBase):
    pass


class WorldRecord(Base):
    __tablename__ = "worlds"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    pack: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class StorySessionRecord(Base):
    __tablename__ = "story_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    world_id: Mapped[str] = mapped_column(
        ForeignKey("worlds.id", ondelete="RESTRICT"), index=True
    )
    player_name: Mapped[str] = mapped_column(String(120))
    persona_id: Mapped[str | None] = mapped_column(String(120))
    runtime_preset_id: Mapped[str | None] = mapped_column(String(120))
    state: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    turns: Mapped[list["StoryTurnRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class StoryTurnRecord(Base):
    __tablename__ = "story_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_number", name="uq_story_turn_number"),
        UniqueConstraint(
            "session_id", "idempotency_key", name="uq_story_turn_idempotency"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("story_sessions.id", ondelete="CASCADE"), index=True
    )
    turn_number: Mapped[int] = mapped_column(Integer)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    player_input: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    narrative: Mapped[str] = mapped_column(Text, default="")
    choices: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list)
    state_delta: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    trace: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    session: Mapped[StorySessionRecord] = relationship(back_populates="turns")


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("world_id", "checksum", name="uq_document_world_checksum"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    world_id: Mapped[str] = mapped_column(
        ForeignKey("worlds.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    object_key: Mapped[str] = mapped_column(String(1024), unique=True)
    content_type: Mapped[str | None] = mapped_column(String(200))
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JsonType, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_world_document", "world_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    world_id: Mapped[str] = mapped_column(String(64), index=True)
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JsonType, default=dict
    )


class JobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint(
            "kind", "session_id", "idempotency_key", name="uq_job_story_idempotency"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    kind: Mapped[str] = mapped_column(String(80), index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_sessions.id", ondelete="CASCADE"), index=True
    )
    turn_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_turns.id", ondelete="SET NULL"), index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    execution_id: Mapped[str | None] = mapped_column(String(128), index=True)
    request_id: Mapped[str | None] = mapped_column(String(128), index=True)
    dispatch_status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False, index=True
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    error_code: Mapped[str | None] = mapped_column(String(120))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class JobEventRecord(Base):
    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_occurred", "job_id", "occurred_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int | None] = mapped_column(Integer)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    execution_id: Mapped[str | None] = mapped_column(String(128))
    request_id: Mapped[str | None] = mapped_column(String(128))
    actor: Mapped[str] = mapped_column(String(32))
    details: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_sessions.id", ondelete="SET NULL"), index=True
    )
    bucket: Mapped[str] = mapped_column(String(120))
    object_key: Mapped[str] = mapped_column(String(1024), unique=True)
    content_type: Mapped[str] = mapped_column(String(200))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    checksum: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ReviewProposalRecord(Base):
    __tablename__ = "review_proposals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_id)
    world_id: Mapped[str] = mapped_column(
        ForeignKey("worlds.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_sessions.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    patch: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    reasoning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
