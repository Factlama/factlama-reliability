"""ADR-019 SS4.7 (the subset PR1 can exercise without PR4's capture-mode/
redaction machinery): "metadata-only capture, redaction-before-persistence,
retention/deletion, bounded reads, sanitized errors, lifecycle/health,
migration/upgrade, and backup/restore behavior remain intact." Full
capture-mode/redaction/retention behavior is REL-11/PR4's own scope; this
file only covers what the storage boundary itself already guarantees:
missing/inaccessible IDs fail safely, and reads are bounded by an explicit
limit rather than an unbounded scan."""

from datetime import timedelta

import pytest

from storage.errors import JobNotFound
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


async def test_missing_job_status_returns_none_not_an_error(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    assert await backend.get_job_status(tenant, "job_does_not_exist") is None


async def test_missing_result_returns_none_not_an_error(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    assert await backend.get_result(tenant, "eval_does_not_exist") is None


async def test_outbox_claim_never_exceeds_the_requested_limit(
    backend: PostgresStorageBackend,
) -> None:
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
        claimed = await backend.claim_due_job(f"worker-{i}", timedelta(minutes=5))
        assert claimed is not None
        await backend.commit_evaluation_and_outbox(
            claimed.job_id,
            claimed.fencing_token,
            make_result(f"eval-{i}"),
            make_outbox_event(f"event-{i}", f"eval-{i}"),
        )

    batch = await backend.claim_outbox_batch("consumer-1", timedelta(minutes=5), limit=2)

    assert len(batch) == 2


async def test_job_not_found_error_does_not_leak_a_different_tenants_data(
    backend: PostgresStorageBackend,
) -> None:
    owner = make_tenant(tenant_id="tenant-a")
    stranger = make_tenant(tenant_id="tenant-b")
    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    with pytest.raises(JobNotFound) as exc_info:
        await backend.request_cancel(stranger, job.job_id)

    assert "tenant-a" not in str(exc_info.value)
    assert "proj-a" not in str(exc_info.value)
