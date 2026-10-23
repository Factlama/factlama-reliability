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
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)
from worker.bootstrap import run


def _asyncpg_url(container: PostgresContainer) -> str:
    raw = container.get_connection_url()
    _, _, rest = raw.partition("://")
    return f"postgresql+asyncpg://{rest}"


async def test_run_starts_and_keeps_running_until_cancelled(
    engine: AsyncEngine,  # ensures the schema exists before the worker connects
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_WORKER_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("FACTLAMA_WORKER_OUTBOX_POLL_INTERVAL_SECONDS", "0.05")

    with pytest.raises(TimeoutError):
        # A clean, still-running worker never returns on its own; timing
        # out (rather than raising some other error, e.g. a
        # StorageConfigurationError or a failed readiness check) is the
        # proof that startup succeeded and both loops are polling.
        await asyncio.wait_for(run(), timeout=0.5)


async def test_outbox_events_stay_unacked_when_observability_is_not_configured(
    backend: PostgresStorageBackend,
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1, `review-evidence/g6-2026-10-22/REVIEW.md`: the real bootstrap,
    with `FACTLAMA_OBSERVABILITY_INGRESS_URL` deliberately unset, must never
    acknowledge a committed outbox event -- acking one it never sent would
    make it permanently undeliverable once an operator does configure the
    ingress, since there is no way to un-ack a row later."""
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        "key-1",
        "hash-1",
        make_request(request_id="key-1"),
        default_deadline(),
    )
    claimed = await backend.claim_due_job("seed-worker", timedelta(minutes=5))
    assert claimed is not None
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-1", "eval-1"),
    )

    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_WORKER_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("FACTLAMA_WORKER_OUTBOX_POLL_INTERVAL_SECONDS", "0.05")
    # The loop's own failed claim already holds a (short-lived, but not
    # instantly expired) lease on this event -- force it negative so the
    # follow-up "checker" claim below sees it as already expired, the same
    # trick `tests/conformance/test_worker_outbox_delivery.py` uses.
    monkeypatch.setenv("FACTLAMA_WORKER_OUTBOX_LEASE_SECONDS", "-1")
    monkeypatch.delenv("FACTLAMA_OBSERVABILITY_INGRESS_URL", raising=False)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(run(), timeout=0.5)

    remaining = await backend.claim_outbox_batch("checker", timedelta(minutes=1), limit=10)
    assert [e.event_id for e in remaining] == ["event-1"]
