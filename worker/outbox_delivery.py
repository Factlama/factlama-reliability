"""Outbox delivery loop (G5): claims committed `OutboxEvent`s and
acknowledges them once delivered. No real consumer exists yet -- G6
(Observability's ingress) is not built -- so `deliver` defaults to a
log-only stand-in; a real G6 integration supplies its own async callable
here without changing this loop's claim/ack mechanics, which are already
proven against concurrent claimers by
`tests/conformance/test_delivery_semantics.py`.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta

from storage.backend import StorageBackend
from storage.models import OutboxEvent
from worker.settings import WorkerSettings

logger = logging.getLogger(__name__)

DeliverFn = Callable[[OutboxEvent], Awaitable[bool]]


async def log_only_deliver(event: OutboxEvent) -> bool:
    """The default `deliver` callable: logs the event and reports success.
    Stands in for G6's real ingress, which does not exist yet."""
    logger.info(
        "outbox event %s ready for delivery: tenant=%s evaluation=%s status=%s",
        event.event_id,
        event.tenant_id,
        event.evaluation_id,
        event.status,
    )
    return True


async def run_outbox_delivery_loop(
    backend: StorageBackend,
    settings: WorkerSettings,
    stop_event: asyncio.Event,
    deliver: DeliverFn = log_only_deliver,
) -> None:
    """CONTRACTS.md: "Delivery retries; duplicate does not double-count."
    A `deliver` callback that raises leaves the event un-acked, so a later
    claim (by this worker or another) redelivers it -- at-least-once, never
    silently dropped."""
    lease_duration = timedelta(seconds=settings.outbox_lease_seconds)
    while not stop_event.is_set():
        batch = await backend.claim_outbox_batch(
            settings.worker_id, lease_duration, settings.outbox_batch_size
        )
        if not batch:
            await _sleep_or_stop(settings.outbox_poll_interval_seconds, stop_event)
            continue
        for event in batch:
            try:
                delivered = await deliver(event)
            except Exception:
                logger.exception("outbox event %s: delivery callback raised", event.event_id)
                continue
            if delivered:
                await backend.ack_outbox(event.event_id, settings.worker_id)


async def _sleep_or_stop(seconds: float, stop_event: asyncio.Event) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
