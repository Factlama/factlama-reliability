"""ADR-019 SS4.4: "crashes around commit cannot expose a terminal
successful job without its evaluation/event, or publish an event for an
uncommitted result. Duplicate identical commits are retry-safe; conflicting
content under an immutable evaluation/event ID is rejected, not silently
treated as a no-op." """

from datetime import timedelta

import pytest

from schemas.tenancy import TenantContext
from storage.errors import EvaluationConflict
from storage.models import ClaimedJob, JobState
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


async def _claim_one(
    backend: PostgresStorageBackend, tenant: TenantContext, key: str, request_id: str
) -> ClaimedJob:
    await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        key,
        f"hash-{key}",
        make_request(request_id=request_id),
        default_deadline(),
    )
    claimed = await backend.claim_due_job(f"worker-{key}", timedelta(minutes=5))
    assert claimed is not None
    return claimed


async def test_commit_persists_evaluation_and_event_and_marks_job_succeeded(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    claimed = await _claim_one(backend, tenant, "key-1", "req-1")

    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-1", "eval-1"),
    )

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id == "eval-1"
    assert await backend.get_result(tenant, "eval-1") is not None


async def test_duplicate_identical_commit_is_retry_safe(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    claimed = await _claim_one(backend, tenant, "key-1", "req-1")
    result = make_result("eval-1")
    event = make_outbox_event("event-1", "eval-1")

    await backend.commit_evaluation_and_outbox(claimed.job_id, claimed.fencing_token, result, event)
    # A retried identical commit (e.g. the caller never saw the first
    # response) must succeed, not raise.
    await backend.commit_evaluation_and_outbox(claimed.job_id, claimed.fencing_token, result, event)


async def test_conflicting_content_under_the_same_evaluation_id_is_rejected(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    claimed = await _claim_one(backend, tenant, "key-1", "req-1")
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-1", "eval-1"),
    )

    different_result = make_result("eval-1", request_id="req-different")
    with pytest.raises(EvaluationConflict):
        await backend.commit_evaluation_and_outbox(
            claimed.job_id,
            claimed.fencing_token,
            different_result,
            make_outbox_event("event-other", "eval-1"),
        )


async def test_conflicting_event_id_reuse_rolls_back_the_whole_commit(
    backend: PostgresStorageBackend,
) -> None:
    """A shared `event_id` with different content must reject the whole
    atomic commit -- including the evaluation half that would otherwise
    have been inserted first in the same transaction (ADR-019 SS4.4's "no
    partial commit" guarantee)."""
    tenant = make_tenant()
    first = await _claim_one(backend, tenant, "key-1", "req-1")
    await backend.commit_evaluation_and_outbox(
        first.job_id,
        first.fencing_token,
        make_result("eval-1"),
        make_outbox_event("event-shared", "eval-1"),
    )

    second = await _claim_one(backend, tenant, "key-2", "req-2")
    with pytest.raises(EvaluationConflict):
        await backend.commit_evaluation_and_outbox(
            second.job_id,
            second.fencing_token,
            make_result("eval-2"),
            make_outbox_event("event-shared", "eval-2"),  # same event_id, different evaluation_id
        )

    # The rejected commit's evaluation half must not have leaked through
    # despite being inserted earlier in the same transaction.
    assert await backend.get_result(tenant, "eval-2") is None
    status = await backend.get_job_status(tenant, second.job_id)
    assert status is not None
    assert status.state != JobState.SUCCEEDED
