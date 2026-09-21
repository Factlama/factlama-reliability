"""G5 storage foundation: jobs, idempotency_keys, evaluations, outbox_events

Revision ID: 0001
Revises:
Create Date: 2026-09-21

Mirrors `storage/postgres/tables.py` exactly; keep the two in lockstep by
hand (no autogenerate was run against a live database for this initial
revision).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(128), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("application_id", sa.String(128), nullable=False),
        sa.Column("route", sa.String(128), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("request_json", JSONB, nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("fencing_token", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("evaluation_id", sa.String(128), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_jobs_claimable", "jobs", ["state", "lease_expires_at", "available_at"])
    op.create_index("ix_jobs_tenant", "jobs", ["tenant_id"])

    op.create_table(
        "idempotency_keys",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("route", sa.String(128), primary_key=True),
        sa.Column("project_id", sa.String(128), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), primary_key=True),
        sa.Column("canonical_request_hash", sa.String(64), nullable=False),
        sa.Column("job_id", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "evaluations",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("evaluation_id", sa.String(128), primary_key=True),
        sa.Column("job_id", sa.String(128), nullable=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("application_id", sa.String(128), nullable=False),
        sa.Column("result_json", JSONB, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "outbox_events",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("event_id", sa.String(128), primary_key=True),
        sa.Column("evaluation_id", sa.String(128), nullable=False),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acked", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("event_id", name="uq_outbox_event_id_global"),
    )
    op.create_index("ix_outbox_claimable", "outbox_events", ["acked", "lease_expires_at"])


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("evaluations")
    op.drop_table("idempotency_keys")
    op.drop_index("ix_jobs_tenant", table_name="jobs")
    op.drop_index("ix_jobs_claimable", table_name="jobs")
    op.drop_table("jobs")
