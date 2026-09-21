"""ADR-019 SS4.2: "concurrent matching submissions produce one logical job;
conflicting body hashes produce a typed conflict; interruption cannot leave
a job/mapping half-committed." Matches the G5 exit-evidence bullet: "Same
key/body returns same job, conflicting body 409."""

import asyncio

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.errors import JobConflict
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.tables import jobs
from tests.conformance.conftest import default_deadline, make_request, make_tenant


async def test_same_key_and_body_returns_the_same_job(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    request = make_request()
    deadline = default_deadline()

    first = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", request, deadline
    )
    second = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", request, deadline
    )

    assert first.job_id == second.job_id


async def test_conflicting_body_under_the_same_key_is_rejected(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    deadline = default_deadline()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", make_request(), deadline
    )

    with pytest.raises(JobConflict):
        await backend.submit_or_get_job(
            tenant, "verification-jobs", "key-1", "hash-2", make_request(), deadline
        )


async def test_different_projects_do_not_share_an_idempotency_key(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant(project_id="proj-a")
    deadline = default_deadline()
    job_a = await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        "key-1",
        "hash-1",
        make_request(project_id="proj-a"),
        deadline,
    )
    job_b = await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        "key-1",
        "hash-1",
        make_request(project_id="proj-b"),
        deadline,
    )

    assert job_a.job_id != job_b.job_id


async def test_concurrent_identical_submissions_produce_exactly_one_job(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request()
    deadline = default_deadline()

    results = await asyncio.gather(
        *(
            backend.submit_or_get_job(
                tenant, "verification-jobs", "key-race", "hash-race", request, deadline
            )
            for _ in range(8)
        )
    )

    job_ids = {job.job_id for job in results}
    assert len(job_ids) == 1

    async with engine.connect() as conn:
        count = (await conn.execute(select(func.count()).select_from(jobs))).scalar_one()
    assert count == 1
