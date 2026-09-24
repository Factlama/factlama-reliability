"""`PostgresRevocationRegistry`: the concrete `storage.registry.
RevocationRegistry` implementation for G5. Separate from
`PostgresStorageBackend` (`storage/postgres/backend.py`) since it backs a
different Protocol against a table with no relationship to the job/
idempotency/evaluation/outbox atomicity boundary ADR-019 SS2 defines for
that class.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.postgres.tables import revoked_evaluators


class PostgresRevocationRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def is_revoked(
        self, provider_id: str, pinned_model_id: str, configuration_version: str
    ) -> bool:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(revoked_evaluators.c.provider_id).where(
                    revoked_evaluators.c.provider_id == provider_id,
                    revoked_evaluators.c.pinned_model_id == pinned_model_id,
                    revoked_evaluators.c.configuration_version == configuration_version,
                )
            )
            return result.first() is not None

    async def revoke(
        self, provider_id: str, pinned_model_id: str, configuration_version: str
    ) -> None:
        # `on_conflict_do_nothing` makes a repeated revoke of the same
        # identity a no-op rather than a unique-violation error --
        # `storage.registry.RevocationRegistry.revoke`'s documented
        # idempotency, mirroring `core.qualification.revoke`'s same rule.
        stmt = (
            pg_insert(revoked_evaluators)
            .values(
                provider_id=provider_id,
                pinned_model_id=pinned_model_id,
                configuration_version=configuration_version,
                revoked_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_nothing(
                index_elements=["provider_id", "pinned_model_id", "configuration_version"]
            )
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
