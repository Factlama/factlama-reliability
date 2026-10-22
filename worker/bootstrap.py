"""Composition root for the worker process (ADR-019 SS3), mirroring
`api/bootstrap.py`: the one module in `worker/` allowed to import
`storage.postgres` directly. Constructs one `PostgresStorageBackend` and
one `Verifier` per process, validates the database is reachable before
polling for work, and stops both the job and outbox-delivery loops on
SIGTERM/SIGINT.

A6 of the 2026-10-14 ADR-020 review: this process does NOT run the
execution-input expiry-cleanup sweep -- that is
`worker/expiry_cleanup_bootstrap.py`'s own, separately deployed process,
precisely so stopping every instance of *this* process does not also stop
cleanup. A deployment running only this module has no execution-input
retention guarantee; see that module's docstring.
"""

import asyncio
import contextlib
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from core.verifier import Verifier
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.engine import check_ready, close_engine, create_storage_engine
from storage.postgres.registry import PostgresRevocationRegistry
from worker.outbox_delivery import DeliverFn, log_only_deliver, run_outbox_delivery_loop
from worker.outbox_http_delivery import build_http_deliver
from worker.runner import run_poll_loop
from worker.settings import WorkerSettings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _outbox_deliver(settings: WorkerSettings) -> AsyncIterator[DeliverFn]:
    """G6 handoff: a real HTTP `deliver` when Observability's ingress is
    configured, else the explicit, logged `log_only_deliver` stand-in --
    never a silent no-op either way."""
    if settings.observability_ingress_url is None:
        logger.warning(
            "FACTLAMA_OBSERVABILITY_INGRESS_URL is not set; outbox events will only be "
            "logged, not delivered to Observability"
        )
        yield log_only_deliver
        return
    async with httpx.AsyncClient(
        base_url=settings.observability_ingress_url,
        timeout=settings.observability_ingress_timeout_seconds,
    ) as client:
        yield build_http_deliver(client, settings)


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
    backend = PostgresStorageBackend(
        engine, execution_storage_keyring=settings.execution_storage_keyring
    )
    registry = PostgresRevocationRegistry(engine)
    verifier = Verifier()

    stop_event = asyncio.Event()
    _install_shutdown_handlers(stop_event)

    logger.info("worker %s starting", settings.worker_id)
    try:
        async with _outbox_deliver(settings) as deliver:
            await asyncio.gather(
                run_poll_loop(backend, verifier, settings, stop_event, registry),
                run_outbox_delivery_loop(backend, settings, stop_event, deliver=deliver),
            )
    finally:
        await close_engine(engine)
        logger.info("worker %s stopped", settings.worker_id)
