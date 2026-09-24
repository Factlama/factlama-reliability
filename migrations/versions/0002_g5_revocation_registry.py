"""G5 revocation registry: revoked_evaluators

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25

Mirrors `storage/postgres/tables.py` exactly; keep the two in lockstep by
hand (no autogenerate was run against a live database for this revision).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "revoked_evaluators",
        sa.Column("provider_id", sa.String(256), primary_key=True),
        sa.Column("pinned_model_id", sa.String(512), primary_key=True),
        sa.Column("configuration_version", sa.String(128), primary_key=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("revoked_evaluators")
