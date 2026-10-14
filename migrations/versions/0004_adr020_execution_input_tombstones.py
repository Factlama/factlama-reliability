"""ADR-020 review A4 (2026-10-14): execution_input_tombstones

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14

Mirrors `storage/postgres/tables.py` exactly; keep the two in lockstep by
hand (no autogenerate was run against a live database for this revision).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "execution_input_tombstones",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("job_id", sa.String(128), primary_key=True),
        sa.Column("terminated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("execution_input_tombstones")
