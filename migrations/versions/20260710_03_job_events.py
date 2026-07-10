"""Add the durable job event audit trail."""

from alembic import op
import sqlalchemy as sa


revision = "20260710_03"
down_revision = "20260710_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(64),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("from_status", sa.String(32)),
        sa.Column("to_status", sa.String(32)),
        sa.Column("progress", sa.Integer()),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("execution_id", sa.String(128)),
        sa.Column("request_id", sa.String(128)),
        sa.Column("actor", sa.String(32), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])
    op.create_index("ix_job_events_event_type", "job_events", ["event_type"])
    op.create_index("ix_job_events_occurred_at", "job_events", ["occurred_at"])
    op.create_index(
        "ix_job_events_job_occurred", "job_events", ["job_id", "occurred_at"]
    )


def downgrade() -> None:
    op.drop_table("job_events")
