"""ADR-019 SS4.5: "outbox delivery is at least once with stable event IDs,
bounded leases/retries, and recovery after ambiguous acknowledgment." G5
exit evidence: "duplicate worker/event yields one logical result." This
producer has no real consumer yet (G6 does not exist); these tests exercise
`claim_outbox_batch`/`ack_outbox` directly rather than through Observability
ingress."""

import asyncio
from datetime import timedelta

from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


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


async def test_a_committed_event_is_claimable_for_delivery(backend: PostgresStorageBackend) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")

    batch = await backend.claim_outbox_batch("consumer-1", timedelta(minutes=1), limit=10)

    assert [e.event_id for e in batch] == ["event-1"]


async def test_a_leased_event_is_not_claimed_by_a_second_consumer(
    backend: PostgresStorageBackend,
) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")
    await backend.claim_outbox_batch("consumer-1", timedelta(minutes=5), limit=10)

    second = await backend.claim_outbox_batch("consumer-2", timedelta(minutes=5), limit=10)

    assert second == []


async def test_acked_event_is_never_redelivered(backend: PostgresStorageBackend) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")
    await backend.claim_outbox_batch("consumer-1", timedelta(minutes=5), limit=10)

    await backend.ack_outbox("event-1", "consumer-1")

    redelivered = await backend.claim_outbox_batch("consumer-2", timedelta(minutes=5), limit=10)
    assert redelivered == []


async def test_an_expired_delivery_lease_is_redelivered(backend: PostgresStorageBackend) -> None:
    await _commit_one(backend, "key-1", "eval-1", "event-1")
    await backend.claim_outbox_batch("consumer-1", timedelta(seconds=-1), limit=10)

    redelivered = await backend.claim_outbox_batch("consumer-2", timedelta(minutes=5), limit=10)
    assert [e.event_id for e in redelivered] == ["event-1"]


async def test_concurrent_consumers_never_claim_the_same_event(
    backend: PostgresStorageBackend,
) -> None:
    for i in range(6):
        await _commit_one(backend, f"key-{i}", f"eval-{i}", f"event-{i}")

    batches = await asyncio.gather(
        *(
            backend.claim_outbox_batch(f"consumer-{i}", timedelta(minutes=5), limit=1)
            for i in range(6)
        )
    )
    claimed_ids = [e.event_id for batch in batches for e in batch]
    assert len(claimed_ids) == 6
    assert len(set(claimed_ids)) == 6
