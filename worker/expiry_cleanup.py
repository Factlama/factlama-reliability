"""ADR-020 section 4: an independent expiry-cleanup loop for
`job_execution_inputs` rows -- runs on its own schedule, regardless of the
owning job's state, so retention stays bounded even when workers stop
entirely and a job never reaches a terminal state (the gap Phase 3's
terminal-transition atomic delete cannot close on its own: "if workers
stop, queued jobs do not become terminal; a job deadline limits execution
eligibility, it does not independently delete its input"). Mirrors
`worker/outbox_delivery.py`'s exact claim-loop/`_sleep_or_stop` shape.
"""

import asyncio
import contextlib

from storage.backend import StorageBackend
from worker.settings import WorkerSettings


async def run_execution_input_expiry_loop(
    backend: StorageBackend,
    settings: WorkerSettings,
    stop_event: asyncio.Event,
) -> None:
    """ADR-020 section 4: "Run expiry/cleanup independently of dispatch
    workers." Purges every `job_execution_inputs` row past its absolute
    `expires_at`, regardless of job state -- a queued job that never gets
    claimed still has its input's retention bounded by this loop alone, not
    by `claim_due_job`'s own expiry check (which only ever runs for a job
    that actually gets claimed).

    This loop never makes expired input *readable* -- `claim_due_job`'s own
    expiry check (Phase 3) is the real confidentiality backstop regardless
    of whether this sweep is currently healthy. A stalled sweep is a
    retention-bound/monitoring problem, not a confidentiality one (ADR-020
    section 4: "An unavailable cleanup process cannot make expired input
    readable, and an indefinite backlog is not a bounded-retention
    guarantee" -- the backlog-visibility half of that is the oldest-lag
    metric `PostgresStorageBackend.purge_expired_execution_inputs` itself
    logs on every sweep that finds work, which an operator monitors to
    catch exactly that "indefinite backlog" condition; logged at the
    backend layer, not here, so it still fires for any direct caller of
    that method, not only this loop).
    """
    while not stop_event.is_set():
        await backend.purge_expired_execution_inputs()
        await _sleep_or_stop(settings.execution_input_sweep_interval_seconds, stop_event)


async def _sleep_or_stop(seconds: float, stop_event: asyncio.Event) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
