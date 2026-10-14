import asyncio
import contextlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from core.verifier import Verifier
from schemas.claims import Claim
from schemas.evidence import Evidence
from storage.postgres.tables import job_execution_inputs, jobs
from tests.conformance.conftest import default_deadline, make_request, make_tenant
from tests.conformance.test_worker_runner import _fast_settings
from worker.runner import process_claimed_job, run_poll_loop

pytestmark = pytest.mark.asyncio


async def submit(backend, key="review", **kw):
    return await backend.submit_or_get_job(
        make_tenant(),
        "verification-jobs",
        key,
        "hash-" + key,
        make_request(**kw),
        default_deadline(),
    )


async def test_nested_metadata_does_not_persist_plaintext(backend, engine):
    req = make_request().model_copy(
        update={
            "evidence": [
                Evidence(
                    evidence_id="e1",
                    content="Private body",
                    metadata={"raw_context": "SECRET_NESTED_PAYLOAD"},
                )
            ]
        }
    )
    job = await backend.submit_or_get_job(
        make_tenant(), "verification-jobs", "nested", "hash", req, default_deadline()
    )
    async with engine.connect() as conn:
        payload = (
            await conn.execute(select(jobs.c.request_json).where(jobs.c.job_id == job.job_id))
        ).scalar_one()
    assert "SECRET_NESTED_PAYLOAD" not in json.dumps(payload), (
        "Nested evidence metadata is persisted as plaintext despite encrypted-input mode"
    )


async def test_expiry_does_not_poison_worker_queue(backend, registry, engine):
    bad = await submit(backend, "expired")
    good = await submit(backend, "healthy", request_id="healthy")
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == bad.job_id)
            .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
    await backend.purge_expired_execution_inputs()
    stop = asyncio.Event()

    async def stop_later():
        await asyncio.sleep(0.2)
        stop.set()

    error = None
    try:
        await asyncio.gather(
            run_poll_loop(backend, Verifier(), _fast_settings(), stop, registry), stop_later()
        )
    except Exception as exc:
        error = exc
        stop.set()
    assert error is None, f"One expired job crashes polling: {type(error).__name__}: {error}"
    status = await backend.get_job_status(make_tenant(), good.job_id)
    assert status.evaluation_id is not None


async def test_restored_preterminal_backup_cannot_revive_destroyed_input(backend, registry, engine):
    job = await submit(backend, "restore")
    async with engine.connect() as conn:
        job_snapshot = dict(
            (await conn.execute(select(jobs).where(jobs.c.job_id == job.job_id))).mappings().one()
        )
        input_snapshot = dict(
            (
                await conn.execute(
                    select(job_execution_inputs).where(job_execution_inputs.c.job_id == job.job_id)
                )
            )
            .mappings()
            .one()
        )
    claimed = await backend.claim_due_job("w", timedelta(minutes=1))
    await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)
    async with engine.begin() as conn:
        assert (
            await conn.execute(
                select(job_execution_inputs).where(job_execution_inputs.c.job_id == job.job_id)
            )
        ).first() is None
        # Restore preterminal operational state and its original ciphertext/expiry.
        await conn.execute(update(jobs).where(jobs.c.job_id == job.job_id).values(**job_snapshot))
        await conn.execute(job_execution_inputs.insert().values(**input_snapshot))
    revived = None
    with contextlib.suppress(Exception):
        revived = await backend.claim_due_job("restored-worker", timedelta(minutes=1))
    assert revived is None, (
        "Backup restore before original expiry decrypts input already deleted at terminal completion"
    )


async def test_claimed_input_expiry_blocks_late_dispatch(backend, registry, engine):
    req = make_request(answer="Paris is the capital of France.").model_copy(
        update={
            "claims": [Claim(claim_id="c1", text="Paris is the capital of France.")],
            "evidence": [Evidence(evidence_id="e1", content="Paris is the capital of France.")],
        }
    )
    job = await backend.submit_or_get_job(
        make_tenant(), "verification-jobs", "late", "hash", req, default_deadline()
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=datetime.now(timezone.utc) + timedelta(seconds=0.15))
        )
    claimed = await backend.claim_due_job("w", timedelta(minutes=1))
    await asyncio.sleep(0.2)
    await backend.purge_expired_execution_inputs()
    await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)
    status = await backend.get_job_status(make_tenant(), job.job_id)
    result = (
        await backend.get_result(make_tenant(), status.evaluation_id)
        if status.evaluation_id
        else None
    )
    assert result is None or result.verdict.value != "PASS", (
        "Worker dispatches and commits PASS after the input expired and was purged"
    )


async def test_input_expiry_cannot_outlive_job_deadline(backend, engine):
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    job = await backend.submit_or_get_job(
        make_tenant(), "verification-jobs", "ttl", "hash", make_request(), deadline
    )
    async with engine.connect() as conn:
        expiry = (
            await conn.execute(
                select(job_execution_inputs.c.expires_at).where(
                    job_execution_inputs.c.job_id == job.job_id
                )
            )
        ).scalar_one()
    assert expiry <= deadline, f"Input expiry {expiry} exceeds job deadline {deadline}"
