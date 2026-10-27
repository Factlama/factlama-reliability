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
import base64
import json
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from core.execution_storage import ExecutionStorageConfigurationError
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    TEST_EXECUTION_STORAGE_KEY,
    TEST_EXECUTION_STORAGE_KEY_ID,
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


async def test_run_refuses_to_start_when_a_declared_retired_key_is_still_configured(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-020 section 5's mandatory restore procedure, enforced
    mechanically: a worker must refuse to start -- never mind claim a job
    -- against a restored database while a key_id it has been told is
    retired is still present in its own configured keyring. Unlike
    `test_run_starts_and_keeps_running_until_cancelled` above, this must
    raise immediately rather than time out, proving the gate fires before
    the poll loop ever starts."""
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", TEST_EXECUTION_STORAGE_KEY_ID)

    with pytest.raises(ExecutionStorageConfigurationError, match=TEST_EXECUTION_STORAGE_KEY_ID):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_refuses_to_start_when_restore_mode_has_an_empty_manifest(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declaring `RESTORE_MODE=RESTORE` without naming any key to retire
    must not be a no-op way to "confirm" the restore procedure without
    doing anything."""
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.delenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", raising=False)

    with pytest.raises(ExecutionStorageConfigurationError, match="RESTORE_MODE=RESTORE"):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_refuses_to_start_with_an_invalid_restore_mode_value(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE must fail loudly
    at real worker startup, not silently fall back to NORMAL."""
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTOER")

    with pytest.raises(ExecutionStorageConfigurationError, match="RESTOER"):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_starts_successfully_once_declared_retired_keys_are_actually_removed(
    engine: AsyncEngine,
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive path this whole procedure exists to allow: an operator
    declares the restore, names the key that could have encrypted a now-
    stale row, and has actually rotated it out of the active keyring --
    startup must succeed and the worker must keep polling."""
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_WORKER_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("FACTLAMA_WORKER_OUTBOX_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    # Retiring a key_id that was never active in the first place is exactly
    # what "actually removed it" looks like from this gate's perspective --
    # the active keyring above only ever configured TEST_EXECUTION_STORAGE_KEY_ID.
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", "some-retired-key-from-the-backup"
    )

    with pytest.raises(TimeoutError):
        # Same proof as test_run_starts_and_keeps_running_until_cancelled:
        # a clean, still-running worker never returns on its own.
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
