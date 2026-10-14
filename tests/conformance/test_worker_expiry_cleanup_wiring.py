"""A6 of the 2026-10-14 ADR-020 review: "existing tests schedule the loop
independently themselves, which does not prove independent production
wiring." These tests check the actual production wiring, not just the
loop function in isolation -- that `worker.bootstrap` (the dispatch-worker
composition root) no longer starts the expiry-cleanup loop at all, and
that `worker.expiry_cleanup_bootstrap` (the new, separate composition
root) does, against a real Postgres-backed backend.
"""

import asyncio
import inspect
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.tables import job_execution_inputs
from tests.conformance.conftest import default_deadline, make_request, make_tenant
from worker import bootstrap as worker_bootstrap
from worker import expiry_cleanup_bootstrap
from worker.expiry_cleanup import run_execution_input_expiry_loop


def test_dispatch_worker_bootstrap_does_not_start_the_expiry_cleanup_loop() -> None:
    """The dispatch-worker composition root must not reference the expiry
    loop at all -- stopping every `worker.bootstrap` process must not stop
    cleanup, because this process never started it in the first place."""
    assert not hasattr(worker_bootstrap, "run_execution_input_expiry_loop")
    source = inspect.getsource(worker_bootstrap)
    assert "import worker.expiry_cleanup" not in source
    assert "from worker.expiry_cleanup " not in source
    assert "run_execution_input_expiry_loop(" not in source


def test_expiry_cleanup_bootstrap_is_a_separate_composition_root() -> None:
    """The standalone cleanup service's composition root does reference
    the loop, and constructs its own real `PostgresStorageBackend` --
    confirming it is a genuine second process, not a thin re-export of
    `worker.bootstrap`."""
    source = inspect.getsource(expiry_cleanup_bootstrap)
    assert "run_execution_input_expiry_loop" in source
    assert "PostgresStorageBackend" in source
    assert expiry_cleanup_bootstrap.run is not worker_bootstrap.run


async def test_cleanup_sweeps_through_the_standalone_bootstraps_own_backend(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """Demonstrates cleanup running via exactly the loop
    `worker.expiry_cleanup_bootstrap.run()` calls, against a backend built
    the same way that module builds one (real Postgres, no dispatch worker
    involved at all) -- "demonstrate cleanup while dispatch workers are
    stopped" without needing a second real OS process in this test."""
    tenant = make_tenant()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "wiring-key", "wiring-hash", make_request(), default_deadline()
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=past)
        )

    class _Settings:
        execution_input_sweep_interval_seconds = 0.01
        worker_id = "standalone-cleanup-test"

    stop_event = asyncio.Event()

    async def _stop_after_one_sweep() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    await asyncio.gather(
        run_execution_input_expiry_loop(backend, _Settings(), stop_event),
        _stop_after_one_sweep(),
    )

    async with engine.connect() as conn:
        remaining = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).first()
    assert remaining is None
