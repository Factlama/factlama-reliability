"""ADR-019 SS4.6: "cancellation races and terminal-state immutability hold
across restarts and ownership changes." `docs/async-evaluation.md`:
"Terminal states cannot be overwritten by late workers." A real process
restart is out of scope for a single test run; these tests instead assert
the durable invariants a restart must not be able to violate: a fencing
token that never resets, and a terminal state no later call can reopen."""

from datetime import datetime, timedelta, timezone

import pytest

from storage.errors import LeaseExpired
from storage.models import JobState
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


async def test_cancellation_is_recorded_and_visible_to_the_current_holder(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    claimed = await backend.claim_due_job("worker-1", timedelta(minutes=5))
    assert claimed is not None

    assert await backend.is_cancellation_requested(claimed.job_id) is False
    await backend.request_cancel(tenant, claimed.job_id)
    assert await backend.is_cancellation_requested(claimed.job_id) is True


async def test_a_succeeded_job_cannot_later_be_failed(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    claimed = await backend.claim_due_job("worker-1", timedelta(minutes=5))
    assert claimed is not None
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-1", "eval-1"),
    )

    with pytest.raises(LeaseExpired):
        await backend.fail_job_terminal(
            claimed.job_id, claimed.fencing_token, "TIMEOUT", "late failure after success"
        )


async def test_a_succeeded_job_cannot_later_be_released_for_retry(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    claimed = await backend.claim_due_job("worker-1", timedelta(minutes=5))
    assert claimed is not None
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-1", "eval-1"),
    )

    with pytest.raises(LeaseExpired):
        await backend.release_for_retry(
            claimed.job_id, claimed.fencing_token, datetime.now(timezone.utc)
        )


async def test_a_failed_job_cannot_later_be_committed_as_succeeded(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    claimed = await backend.claim_due_job("worker-1", timedelta(minutes=5))
    assert claimed is not None
    await backend.fail_job_terminal(
        claimed.job_id, claimed.fencing_token, "TIMEOUT", "exhausted retries"
    )

    with pytest.raises(LeaseExpired):
        await backend.commit_evaluation_and_outbox(
            claimed.job_id,
            claimed.fencing_token,
            make_result("eval-late"),
            make_outbox_event("event-late", "eval-late"),
        )

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.FAILED


async def test_fencing_token_strictly_increases_across_repeated_reclaims(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    tokens = []
    for i in range(4):
        claimed = await backend.claim_due_job(f"worker-{i}", timedelta(seconds=-1))
        assert claimed is not None
        tokens.append(claimed.fencing_token)

    assert tokens == sorted(tokens)
    assert len(set(tokens)) == len(tokens)
