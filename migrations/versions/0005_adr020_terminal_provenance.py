"""ADR-020 closure review P2 (2026-10-14): jobs.terminal_provenance_json

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-14

Mirrors `storage/postgres/tables.py` exactly; keep the two in lockstep by
hand (no autogenerate was run against a live database for this revision).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("terminal_provenance_json", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "terminal_provenance_json")
