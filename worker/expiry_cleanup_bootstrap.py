"""Composition root for a standalone execution-input expiry-cleanup
service (A6 of the 2026-10-14 ADR-020 review).

Before this fix, `worker.expiry_cleanup.run_execution_input_expiry_loop`'s
only production caller was `worker/bootstrap.py`'s own `asyncio.gather`,
sharing its process, event loop, and stop signal with dispatch -- stopping
every dispatch worker also stopped cleanup, and a slow synchronous call on
that same event loop could delay a sweep. "Existing tests schedule the
loop independently themselves" (the review's own words) proved nothing
about production wiring, since no production entrypoint ever ran it
without also running dispatch.

This module runs ONLY the expiry sweep, in its own process, independent of
every dispatch worker's lifecycle -- `python -m worker.expiry_cleanup_bootstrap`
alongside `python -m worker` as a second, separately deployed/scaled/
restarted process or container. Deploying only `python -m worker` no
longer provides any execution-input retention guarantee; it must be run
alongside at least one instance of this module. `claim_due_job`'s own
read-time expiry check (ADR-020 section 4; A1/A5 of this same review)
remains the real confidentiality backstop regardless of whether this
sweep is currently healthy -- an unavailable or lagging cleanup process
bounds retention more loosely, but can never make an already-expired
input readable.
"""

import asyncio
import contextlib
import logging
import signal

from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.engine import check_ready, close_engine, create_storage_engine
from worker.expiry_cleanup import run_execution_input_expiry_loop
from worker.settings import WorkerSettings

logger = logging.getLogger(__name__)


def _install_shutdown_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # NotImplementedError: signal handlers aren't supported on Windows'
        # default event loop -- this service is a POSIX/container target,
        # but failing to install a signal handler should not itself crash
        # startup on a platform where it just isn't available.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)


async def run() -> None:
    settings = WorkerSettings()
    engine = create_storage_engine(settings.database_url)
    # ADR-019 SS3: fail startup explicitly rather than sweeping against a
    # database that was never reachable in the first place.
    await check_ready(engine)
    backend = PostgresStorageBackend(
        engine, execution_storage_keyring=settings.execution_storage_keyring
    )

    stop_event = asyncio.Event()
    _install_shutdown_handlers(stop_event)

    logger.info("execution-input expiry-cleanup service %s starting", settings.worker_id)
    try:
        await run_execution_input_expiry_loop(backend, settings, stop_event)
    finally:
        await close_engine(engine)
        logger.info("execution-input expiry-cleanup service %s stopped", settings.worker_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
