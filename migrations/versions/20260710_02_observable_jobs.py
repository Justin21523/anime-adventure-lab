"""Add observable job lifecycle fields."""

from alembic import op
import sqlalchemy as sa


revision = "20260710_02"
down_revision = "20260710_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("request_id", sa.String(128)))
    op.add_column(
        "jobs",
        sa.Column(
            "dispatch_status",
            sa.String(32),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column("jobs", sa.Column("started_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("finished_at", sa.DateTime(timezone=True)))
    op.create_index("ix_jobs_request_id", "jobs", ["request_id"])
    op.create_index("ix_jobs_dispatch_status", "jobs", ["dispatch_status"])


def downgrade() -> None:
    op.drop_index("ix_jobs_dispatch_status", table_name="jobs")
    op.drop_index("ix_jobs_request_id", table_name="jobs")
    op.drop_column("jobs", "finished_at")
    op.drop_column("jobs", "started_at")
    op.drop_column("jobs", "dispatch_status")
    op.drop_column("jobs", "request_id")
