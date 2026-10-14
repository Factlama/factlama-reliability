"""ADR-020: `DURABLE_ENCRYPTED` execution-input storage against real
PostgreSQL -- atomic insertion with the job, claim-time decryption,
absolute-expiry rejection, tamper detection, and atomic deletion on
terminal completion. `tests/test_execution_storage.py` covers the pure
crypto/keyring logic in isolation; this covers the storage-layer wiring
around it for real.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from schemas.policy import ExecutionStorageMode, ExecutionStoragePolicy, Policy
from storage.errors import ExecutionStorageNotAuthorized
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.tables import job_execution_inputs, jobs
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)


async def test_durable_encrypted_submission_stores_no_plaintext_in_jobs_row(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request(answer="PRIVATE_SENTINEL_ANSWER")
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-1", "hash-1", request, default_deadline()
    )
    async with engine.connect() as conn:
        row = (
            await conn.execute(select(jobs.c.request_json).where(jobs.c.job_id == job.job_id))
        ).one()
    assert "PRIVATE_SENTINEL_ANSWER" not in str(row.request_json)


async def test_durable_encrypted_submission_creates_an_opaque_execution_input_row(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request(answer="PRIVATE_SENTINEL_ANSWER")
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-2", "hash-2", request, default_deadline()
    )
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(job_execution_inputs.c.ciphertext, job_execution_inputs.c.expires_at).where(
                    job_execution_inputs.c.tenant_id == tenant.tenant_id,
                    job_execution_inputs.c.job_id == job.job_id,
                )
            )
        ).one()
    assert b"PRIVATE_SENTINEL_ANSWER" not in row.ciphertext
    assert row.expires_at > datetime.now(timezone.utc)


async def test_no_persistence_mode_is_rejected_before_any_insert(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request(
        policy=Policy(
            id="strict",
            execution_storage=ExecutionStoragePolicy(mode=ExecutionStorageMode.NO_PERSISTENCE),
        )
    )
    with pytest.raises(ExecutionStorageNotAuthorized):
        await backend.submit_or_get_job(
            tenant, "verification-jobs", "key-3", "hash-3", request, default_deadline()
        )
    async with engine.connect() as conn:
        rows = (await conn.execute(select(jobs.c.job_id))).all()
    assert rows == []


async def test_requesting_durable_encrypted_without_a_tenant_grant_is_rejected(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """A3 of the 2026-10-14 ADR-020 review: an earlier pass authorized
    DURABLE_ENCRYPTED whenever the *request* selected it, letting any
    authenticated caller grant itself durable storage just by changing its
    own request body. The request here asks for DURABLE_ENCRYPTED, but the
    tenant itself carries no grant (defaults to NO_PERSISTENCE) -- this must
    still be rejected, before any insert, exactly like the no-policy case
    above."""
    ungranted_tenant = make_tenant(
        tenant_id="tenant-ungranted", execution_storage_grant=ExecutionStorageMode.NO_PERSISTENCE
    )
    request = make_request(
        policy=Policy(
            id="self-authorizing",
            execution_storage=ExecutionStoragePolicy(mode=ExecutionStorageMode.DURABLE_ENCRYPTED),
        )
    )
    with pytest.raises(ExecutionStorageNotAuthorized):
        await backend.submit_or_get_job(
            ungranted_tenant, "verification-jobs", "key-3b", "hash-3b", request, default_deadline()
        )
    async with engine.connect() as conn:
        rows = (await conn.execute(select(jobs.c.job_id))).all()
    assert rows == []


async def test_two_tenants_with_different_grants_against_the_same_backend(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """The review's own suggested shape: one configured backend, two
    tenants, different trusted grants -- the granted tenant's identical
    request succeeds, the ungranted tenant's identical request is rejected.
    Proves the grant is resolved per-tenant, not as a backend-wide toggle."""
    granted = make_tenant(
        tenant_id="tenant-granted", execution_storage_grant=ExecutionStorageMode.DURABLE_ENCRYPTED
    )
    ungranted = make_tenant(
        tenant_id="tenant-ungranted-2", execution_storage_grant=ExecutionStorageMode.NO_PERSISTENCE
    )
    request = make_request(
        policy=Policy(
            id="shared-policy",
            execution_storage=ExecutionStoragePolicy(mode=ExecutionStorageMode.DURABLE_ENCRYPTED),
        )
    )
    job = await backend.submit_or_get_job(
        granted, "verification-jobs", "key-3c", "hash-3c", request, default_deadline()
    )
    assert job.tenant_id == "tenant-granted"
    with pytest.raises(ExecutionStorageNotAuthorized):
        await backend.submit_or_get_job(
            ungranted, "verification-jobs", "key-3d", "hash-3d", request, default_deadline()
        )


async def test_claim_before_expiry_decrypts_the_real_request(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    request = make_request(answer="The real, decrypted answer.")
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-4", "hash-4", request, default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None
    assert claimed.request.answer == "The real, decrypted answer."


async def test_claim_after_expiry_terminalizes_without_decrypting(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """A1 of the 2026-10-14 ADR-020 review: an expired input must not raise
    from inside the claim transaction (that rolled back the RUNNING
    transition too, leaving the job QUEUED and reselected forever) --
    `claim_due_job` now terminalizes the job directly and returns `None`."""
    tenant = make_tenant()
    request = make_request()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-5", "hash-5", request, default_deadline()
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=past)
        )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is None
    status = await backend.get_job_status(tenant, job.job_id)
    assert status is not None
    assert status.state.value == "FAILED"
    async with engine.connect() as conn:
        remaining = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).first()
    assert remaining is None


async def test_claim_with_tampered_ciphertext_terminalizes_not_fabricated_content(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """A1 of the 2026-10-14 ADR-020 review: same terminalize-and-return-None
    behavior as the expiry case above, for tampered ciphertext."""
    tenant = make_tenant()
    request = make_request()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-6", "hash-6", request, default_deadline()
    )
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                select(job_execution_inputs.c.ciphertext).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).one()
        tampered = bytearray(row.ciphertext)
        tampered[0] ^= 0xFF
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(ciphertext=bytes(tampered))
        )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is None
    status = await backend.get_job_status(tenant, job.job_id)
    assert status is not None
    assert status.state.value == "FAILED"


async def test_missing_execution_input_row_terminalizes(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    """A job somehow reaching QUEUED with no execution_input row at all
    (never possible through the normal submission path, but defensively
    checked) must terminalize loudly, not silently dispatch with no real
    content. A1 of the 2026-10-14 ADR-020 review: terminalize-and-return-
    None, same as the other two cases above."""
    tenant = make_tenant()
    request = make_request()
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-7", "hash-7", request, default_deadline()
    )
    async with engine.begin() as conn:
        await conn.execute(
            job_execution_inputs.delete().where(job_execution_inputs.c.job_id == job.job_id)
        )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is None
    status = await backend.get_job_status(tenant, job.job_id)
    assert status is not None
    assert status.state.value == "FAILED"


async def test_terminal_success_atomically_deletes_the_execution_input_row(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-8", "hash-8", request, default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None

    result = make_result("eval-8", tenant_id=tenant.tenant_id)
    event = make_outbox_event("event-8", "eval-8", tenant_id=tenant.tenant_id)
    await backend.commit_evaluation_and_outbox(claimed.job_id, claimed.fencing_token, result, event)

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == claimed.job_id
                )
            )
        ).all()
    assert rows == []


async def test_terminal_failure_atomically_deletes_the_execution_input_row(
    backend: PostgresStorageBackend, engine: AsyncEngine
) -> None:
    tenant = make_tenant()
    request = make_request()
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-9", "hash-9", request, default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None

    await backend.fail_job_terminal(
        claimed.job_id, claimed.fencing_token, "INTERNAL", "simulated terminal failure"
    )

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(job_execution_inputs.c.job_id).where(
                    job_execution_inputs.c.job_id == claimed.job_id
                )
            )
        ).all()
    assert rows == []
