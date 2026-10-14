"""ADR-020 execution-input storage: job_execution_inputs

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-13

Mirrors `storage/postgres/tables.py` exactly; keep the two in lockstep by
hand (no autogenerate was run against a live database for this revision).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "job_execution_inputs",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("job_id", sa.String(128), primary_key=True),
        sa.Column("key_id", sa.String(64), nullable=False),
        sa.Column("payload_version", sa.Integer, nullable=False),
        sa.Column("nonce", sa.LargeBinary, nullable=False),
        sa.Column("ciphertext", sa.LargeBinary, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_execution_inputs_expiry", "job_execution_inputs", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_job_execution_inputs_expiry", table_name="job_execution_inputs")
    op.drop_table("job_execution_inputs")
