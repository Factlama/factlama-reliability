"""Every `worker.runner`/`worker.outbox_delivery` test constructs its own
`WorkerSettings`/`PostgresStorageBackend` directly, so `worker.bootstrap.run()`
-- engine construction, the startup readiness check, and wiring both loops
together -- is otherwise never actually executed by anything. This proves
it starts cleanly against a live Postgres container and keeps running
(rather than exiting immediately or raising) until cancelled, the same way
`tests/conformance/test_bootstrap_integration.py` proves `api/bootstrap.py`'s
lifespan.
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from worker.bootstrap import run


async def test_run_starts_and_keeps_running_until_cancelled(
    engine: AsyncEngine,  # ensures the schema exists before the worker connects
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = postgres_container.get_connection_url()
    _, _, rest = raw.partition("://")
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", f"postgresql+asyncpg://{rest}")
    monkeypatch.setenv("FACTLAMA_WORKER_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("FACTLAMA_WORKER_OUTBOX_POLL_INTERVAL_SECONDS", "0.05")

    with pytest.raises(TimeoutError):
        # A clean, still-running worker never returns on its own; timing
        # out (rather than raising some other error, e.g. a
        # StorageConfigurationError or a failed readiness check) is the
        # proof that startup succeeded and both loops are polling.
        await asyncio.wait_for(run(), timeout=0.5)
