"""Tests for `worker.outbox_delivery.run_outbox_delivery_loop` against a
real Postgres-backed `StorageBackend`."""

import asyncio
from datetime import timedelta

from storage.models import OutboxEvent
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)
from worker.outbox_delivery import run_outbox_delivery_loop
from worker.settings import WorkerSettings


def _settings(outbox_lease_seconds: float = 60.0) -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.database_url = "unused-in-this-test"
    settings.worker_id = "outbox-worker-test"
    settings.outbox_batch_size = 20
    settings.outbox_lease_seconds = outbox_lease_seconds
    settings.outbox_poll_interval_seconds = 0.01
    return settings


async def _commit_one(
    backend: PostgresStorageBackend, key: str, evaluation_id: str, event_id: str
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        key,
        f"hash-{key}",
        make_request(request_id=key),
        default_deadline(),
    )
    claimed = await backend.claim_due_job(f"worker-{key}", timedelta(minutes=5))
    assert claimed is not None
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result(evaluation_id),
        make_outbox_event(event_id, evaluation_id),
    )


async def test_delivery_loop_acks_delivered_events(backend: PostgresStorageBackend) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")

    delivered: list[OutboxEvent] = []

    async def _record(event: OutboxEvent) -> bool:
        delivered.append(event)
        return True

    stop_event = asyncio.Event()

    async def _stop_after_delivery() -> None:
        while not delivered:
            await asyncio.sleep(0.01)
        stop_event.set()

    await asyncio.gather(
        run_outbox_delivery_loop(backend, _settings(), stop_event, deliver=_record),
        _stop_after_delivery(),
    )

    assert [e.event_id for e in delivered] == ["event-1"]
    # Acked -- a fresh claim finds nothing left to deliver.
    remaining = await backend.claim_outbox_batch("checker", timedelta(minutes=1), limit=10)
    assert remaining == []


async def test_delivery_loop_leaves_a_failed_delivery_unacked(
    backend: PostgresStorageBackend,
) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")

    attempts = 0

    async def _fail_once(event: OutboxEvent) -> bool:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("simulated delivery failure")

    stop_event = asyncio.Event()

    async def _stop_after_one_attempt() -> None:
        while attempts == 0:
            await asyncio.sleep(0.01)
        stop_event.set()

    await asyncio.gather(
        run_outbox_delivery_loop(
            backend, _settings(outbox_lease_seconds=-1.0), stop_event, deliver=_fail_once
        ),
        _stop_after_one_attempt(),
    )

    # Not acked -- still claimable (at-least-once, CONTRACTS.md). The
    # delivery loop's own lease was already in the past
    # (outbox_lease_seconds=-1.0), so the event is immediately eligible
    # for redelivery without waiting out a real lease window.
    remaining = await backend.claim_outbox_batch("checker", timedelta(minutes=1), limit=10)
    assert [e.event_id for e in remaining] == ["event-1"]
