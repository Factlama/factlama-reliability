"""ADR-019 SS4.1: "tenant/project authorization is preserved at every
operation; cross-tenant reads, writes, claims, conflicts, and cursor reuse
cannot leak or mutate another tenant's data." """

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


async def test_job_status_invisible_to_other_tenant(backend: PostgresStorageBackend) -> None:
    owner = make_tenant(tenant_id="tenant-a")
    stranger = make_tenant(tenant_id="tenant-b")
    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    assert await backend.get_job_status(owner, job.job_id) is not None
    assert await backend.get_job_status(stranger, job.job_id) is None


async def test_job_status_invisible_outside_narrowed_project_scope(
    backend: PostgresStorageBackend,
) -> None:
    owner = make_tenant(tenant_id="tenant-a", project_id="proj-a", application_id="app-a")
    same_tenant_other_project = make_tenant(
        tenant_id="tenant-a", project_id="proj-b", application_id="app-a"
    )
    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    assert await backend.get_job_status(same_tenant_other_project, job.job_id) is None


async def test_cancel_on_other_tenants_job_fails_safely(backend: PostgresStorageBackend) -> None:
    owner = make_tenant(tenant_id="tenant-a")
    stranger = make_tenant(tenant_id="tenant-b")
    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    with pytest.raises(JobNotFound):
        await backend.request_cancel(stranger, job.job_id)


async def test_cancel_on_unknown_job_fails_safely(backend: PostgresStorageBackend) -> None:
    with pytest.raises(JobNotFound):
        await backend.request_cancel(make_tenant(), "job_does_not_exist")


async def test_result_invisible_to_other_tenant(backend: PostgresStorageBackend) -> None:
    owner = make_tenant(tenant_id="tenant-a")
    stranger = make_tenant(tenant_id="tenant-b")

    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )
    claimed = await backend.claim_due_job("worker-1", timedelta(minutes=1))
    assert claimed is not None
    result = make_result("eval-1", tenant_id="tenant-a")
    event = make_outbox_event("event-1", "eval-1", tenant_id="tenant-a")
    await backend.commit_evaluation_and_outbox(job.job_id, claimed.fencing_token, result, event)

    assert await backend.get_result(owner, "eval-1") is not None
    assert await backend.get_result(stranger, "eval-1") is None
