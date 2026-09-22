"""Composition root for the worker process (ADR-019 SS3), mirroring
`api/bootstrap.py`: the one module in `worker/` allowed to import
`storage.postgres` directly. Constructs one `PostgresStorageBackend` and
one `Verifier` per process, validates the database is reachable before
polling for work, and stops both the job and outbox-delivery loops on
SIGTERM/SIGINT.
"""

import asyncio
import contextlib
import logging
import signal

from core.verifier import Verifier
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.engine import check_ready, close_engine, create_storage_engine
from worker.outbox_delivery import run_outbox_delivery_loop
from worker.runner import run_poll_loop
from worker.settings import WorkerSettings

logger = logging.getLogger(__name__)


def _install_shutdown_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # NotImplementedError: signal handlers aren't supported on Windows'
        # default event loop -- this worker is a POSIX/container target,
        # but failing to install a signal handler should not itself crash
        # startup on a platform where it just isn't available.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)


async def run() -> None:
    settings = WorkerSettings()
    engine = create_storage_engine(settings.database_url)
    # ADR-019 SS3: fail startup explicitly rather than polling against a
    # database that was never reachable in the first place.
    await check_ready(engine)
    backend = PostgresStorageBackend(engine)
    verifier = Verifier()

    stop_event = asyncio.Event()
    _install_shutdown_handlers(stop_event)

    logger.info("worker %s starting", settings.worker_id)
    try:
        await asyncio.gather(
            run_poll_loop(backend, verifier, settings, stop_event),
            run_outbox_delivery_loop(backend, settings, stop_event),
        )
    finally:
        await close_engine(engine)
        logger.info("worker %s stopped", settings.worker_id)
