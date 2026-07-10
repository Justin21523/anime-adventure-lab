"""Create the Story-first Postgres and pgvector schema."""

from alembic import op
import sqlalchemy as sa

try:
    from pgvector.sqlalchemy import Vector
except ImportError:

    class Vector(sa.types.UserDefinedType):
        cache_ok = True

        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions

        def get_col_spec(self, **_kwargs) -> str:
            return f"vector({self.dimensions})"


revision = "20260710_01"
down_revision = None
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    ]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "worlds",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("pack", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
    )
    op.create_table(
        "story_sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "world_id",
            sa.String(64),
            sa.ForeignKey("worlds.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("player_name", sa.String(120), nullable=False),
        sa.Column("persona_id", sa.String(120)),
        sa.Column("runtime_preset_id", sa.String(120)),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
    )
    op.create_index("ix_story_sessions_world_id", "story_sessions", ["world_id"])

    op.create_table(
        "story_turns",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("story_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("player_input", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("narrative", sa.Text(), nullable=False, server_default=""),
        sa.Column("choices", sa.JSON(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("state_delta", sa.JSON(), nullable=False),
        sa.Column("trace", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("session_id", "turn_number", name="uq_story_turn_number"),
        sa.UniqueConstraint(
            "session_id", "idempotency_key", name="uq_story_turn_idempotency"
        ),
    )
    op.create_index("ix_story_turns_session_id", "story_turns", ["session_id"])
    op.create_index("ix_story_turns_status", "story_turns", ["status"])

    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "world_id",
            sa.String(64),
            sa.ForeignKey("worlds.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("content_type", sa.String(200)),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("world_id", "checksum", name="uq_document_world_checksum"),
    )
    op.create_index("ix_documents_world_id", "documents", ["world_id"])
    op.create_index("ix_documents_checksum", "documents", ["checksum"])
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("world_id", sa.String(64), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024)),
        sa.Column("metadata", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )
    op.create_index("ix_document_chunks_world_id", "document_chunks", ["world_id"])
    op.create_index(
        "ix_document_chunks_world_document",
        "document_chunks",
        ["world_id", "document_id"],
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(80), nullable=False),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("story_sessions.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "turn_id",
            sa.String(64),
            sa.ForeignKey("story_turns.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
        ),
        sa.Column("idempotency_key", sa.String(128)),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("execution_id", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON()),
        sa.Column("error_code", sa.String(120)),
        sa.Column("error_message", sa.Text()),
        *_timestamps(),
        sa.UniqueConstraint(
            "kind", "session_id", "idempotency_key", name="uq_job_story_idempotency"
        ),
    )
    for column in [
        "kind",
        "session_id",
        "turn_id",
        "document_id",
        "status",
        "execution_id",
        "lease_expires_at",
    ]:
        op.create_index(f"ix_jobs_{column}", "jobs", [column])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "job_id", sa.String(64), sa.ForeignKey("jobs.id", ondelete="SET NULL")
        ),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("story_sessions.id", ondelete="SET NULL"),
        ),
        sa.Column("bucket", sa.String(120), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("content_type", sa.String(200), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_artifacts_job_id", "artifacts", ["job_id"])
    op.create_index("ix_artifacts_session_id", "artifacts", ["session_id"])

    op.create_table(
        "review_proposals",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "world_id",
            sa.String(64),
            sa.ForeignKey("worlds.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("story_sessions.id", ondelete="SET NULL"),
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("patch", sa.JSON(), nullable=False),
        sa.Column("reasoning", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_review_proposals_world_id", "review_proposals", ["world_id"])
    op.create_index(
        "ix_review_proposals_session_id", "review_proposals", ["session_id"]
    )
    op.create_index("ix_review_proposals_status", "review_proposals", ["status"])


def downgrade() -> None:
    op.drop_table("review_proposals")
    op.drop_table("artifacts")
    op.drop_table("jobs")
    op.drop_index("ix_document_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("story_turns")
    op.drop_table("story_sessions")
    op.drop_table("worlds")
