"""ADR-019 SS4.3: "concurrent claimers cannot both own the same lease;
after expiry/reclaim, stale workers cannot renew, update, cancel, or commit
using an obsolete token. A state filter alone is insufficient to fence an
old worker after a new worker has reclaimed the job." """

import asyncio
from datetime import timedelta

import pytest

from storage.errors import LeaseExpired
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


async def test_no_job_due_returns_none(backend: PostgresStorageBackend) -> None:
    assert await backend.claim_due_job("worker-1", timedelta(minutes=1)) is None


async def test_a_freshly_leased_job_is_not_reclaimed(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    first = await backend.claim_due_job("worker-1", timedelta(minutes=5))
    assert first is not None

    second = await backend.claim_due_job("worker-2", timedelta(minutes=5))
    assert second is None


async def test_an_expired_lease_is_reclaimed_with_a_higher_fencing_token(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    stale = await backend.claim_due_job("worker-1", timedelta(seconds=-1))
    assert stale is not None

    fresh = await backend.claim_due_job("worker-2", timedelta(minutes=5))
    assert fresh is not None
    assert fresh.job_id == stale.job_id
    assert fresh.fencing_token > stale.fencing_token


async def test_stale_worker_cannot_renew_after_reclaim(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    stale = await backend.claim_due_job("worker-1", timedelta(seconds=-1))
    assert stale is not None
    await backend.claim_due_job("worker-2", timedelta(minutes=5))

    with pytest.raises(LeaseExpired):
        await backend.renew_lease(stale.job_id, stale.fencing_token, timedelta(minutes=1))


async def test_stale_worker_cannot_commit_after_reclaim(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    stale = await backend.claim_due_job("worker-1", timedelta(seconds=-1))
    assert stale is not None
    fresh = await backend.claim_due_job("worker-2", timedelta(minutes=5))
    assert fresh is not None

    with pytest.raises(LeaseExpired):
        await backend.commit_evaluation_and_outbox(
            stale.job_id,
            stale.fencing_token,
            make_result("eval-stale"),
            make_outbox_event("event-stale", "eval-stale"),
        )

    # The current holder's own commit still succeeds.
    await backend.commit_evaluation_and_outbox(
        fresh.job_id,
        fresh.fencing_token,
        make_result("eval-fresh"),
        make_outbox_event("event-fresh", "eval-fresh"),
    )
    status = await backend.get_job_status(tenant, fresh.job_id)
    assert status is not None
    assert status.evaluation_id == "eval-fresh"


async def test_concurrent_claimers_never_share_a_lease(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    for i in range(5):
        await backend.submit_or_get_job(
            tenant,
            "verification-jobs",
            f"key-{i}",
            f"hash-{i}",
            make_request(request_id=f"req-{i}"),
            default_deadline(),
        )

    claims = await asyncio.gather(
        *(backend.claim_due_job(f"worker-{i}", timedelta(minutes=5)) for i in range(5))
    )
    claimed_job_ids = [c.job_id for c in claims if c is not None]
    assert len(claimed_job_ids) == 5
    assert len(set(claimed_job_ids)) == 5
