"""ADR-019 SS3: "Unknown backend configuration... must fail startup/readiness
explicitly; never silently fall back to another database." No live Postgres
needed for this one -- `create_storage_engine` validates the URL scheme
before ever connecting."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.postgres.engine import StorageConfigurationError, check_ready, create_storage_engine


def test_rejects_a_non_asyncpg_url() -> None:
    with pytest.raises(StorageConfigurationError):
        create_storage_engine("postgresql://user:pass@localhost:5432/db")


def test_rejects_a_completely_different_scheme() -> None:
    with pytest.raises(StorageConfigurationError):
        create_storage_engine("sqlite:///local.db")


def test_accepts_a_well_formed_asyncpg_url() -> None:
    engine = create_storage_engine("postgresql+asyncpg://user:pass@localhost:5432/db")
    assert engine is not None


async def test_check_ready_succeeds_against_a_live_database(engine: AsyncEngine) -> None:
    await check_ready(engine)


async def test_check_ready_raises_against_an_unreachable_database() -> None:
    unreachable = create_storage_engine("postgresql+asyncpg://user:pass@localhost:1/db")
    with pytest.raises(OSError):
        await check_ready(unreachable)
    await unreachable.dispose()
