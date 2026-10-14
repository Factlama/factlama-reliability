"""Tests for `worker.expiry_cleanup.run_execution_input_expiry_loop` against
a real Postgres-backed `StorageBackend` -- the independent sweep that
bounds `job_execution_inputs` retention even when a job is never claimed at
all (ADR-020 section 4's gap that terminal-only cleanup cannot close on its
own)."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.tables import job_execution_inputs
from tests.conformance.conftest import default_deadline, make_request, make_tenant
from worker.expiry_cleanup import run_execution_input_expiry_loop
from worker.settings import WorkerSettings


def _settings(sweep_interval_seconds: float = 0.01) -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.database_url = "unused-in-this-test"
    settings.worker_id = "expiry-sweep-test"
    settings.execution_input_sweep_interval_seconds = sweep_interval_seconds
    return settings


async def test_sweep_purges_an_expired_input_for_a_still_queued_job(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """The job is never claimed -- it stays QUEUED throughout. Only the
    independent sweep, not a terminal transition, removes the expired row."""
    tenant = make_tenant()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=past)
        )

    stop_event = asyncio.Event()

    async def _stop_after_one_sweep() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    await asyncio.gather(
        run_execution_input_expiry_loop(backend, _settings(), stop_event),
        _stop_after_one_sweep(),
    )

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).all()
    assert rows == []


async def test_sweep_does_not_touch_an_unexpired_input(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-2", "hash-2", make_request(), default_deadline()
    )

    stop_event = asyncio.Event()

    async def _stop_after_one_sweep() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    await asyncio.gather(
        run_execution_input_expiry_loop(backend, _settings(), stop_event),
        _stop_after_one_sweep(),
    )

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).all()
    assert len(rows) == 1


async def test_purge_logs_the_oldest_cleanup_lag_it_observed(
    backend: PostgresStorageBackend,
    engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ADR-020 section 4's cleanup-lag monitoring: the sweep must surface
    *how far behind* cleanup is, not just that it ran -- a 1-row purge
    whose row has been expired for days looks identical to a healthy sweep
    if only the count is observed."""
    tenant = make_tenant()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-lag-1", "hash-lag-1", make_request(), default_deadline()
    )
    long_expired = datetime.now(timezone.utc) - timedelta(hours=2)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=long_expired)
        )

    with caplog.at_level("INFO", logger="storage.postgres.backend"):
        purged = await backend.purge_expired_execution_inputs()

    assert purged == 1
    [record] = [r for r in caplog.records if "cleanup lag" in r.message]
    assert "purged 1 row" in record.message
    lag_seconds = float(record.message.rsplit(" ", 1)[-1].rstrip("s"))
    assert lag_seconds >= 7200.0


async def test_purge_returns_the_count_of_rows_removed(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    job_a = await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        "key-3",
        "hash-3",
        make_request(request_id="req-a"),
        default_deadline(),
    )
    job_b = await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        "key-4",
        "hash-4",
        make_request(request_id="req-b"),
        default_deadline(),
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id.in_([job_a.job_id, job_b.job_id]))
            .values(expires_at=past)
        )
    purged = await backend.purge_expired_execution_inputs()
    assert purged == 2
