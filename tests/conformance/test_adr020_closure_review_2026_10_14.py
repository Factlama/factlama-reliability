import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from core.verifier import Verifier
from schemas.claims import Claim
from schemas.evidence import Evidence
from schemas.verification import Usage
from storage.postgres.tables import job_execution_inputs, metadata
from tests.conformance.conftest import default_deadline, make_request, make_tenant
from tests.conformance.test_worker_runner import _ScriptedProvider, _supported
from worker.runner import _commit

pytestmark = pytest.mark.asyncio

# The closure review's "full database restore revives a destroyed input"
# reproduction is not reproduced as a permanent test in this file: it is
# the one finding (P1) whose literal scenario -- restoring every
# application table, including the tombstone, with the keyring left
# unchanged -- cannot be closed from inside a single PostgreSQL instance
# by construction (restoring a tombstone table along with everything else
# restores its own absence too). The fix is an operational one (retire
# the key_id before bringing a restored database online), tested in
# `tests/conformance/test_execution_storage_key_management.py`'s
# `TestMandatoryKeyRetirementDefeatsFullDatabaseRestore` -- both that the
# procedure works when followed, and, kept permanent and intentionally
# red-if-ever-"fixed"-by-accident, that skipping it still does not close
# the gap. See that class's own docstring and REL-11's ledger entry.


async def submit(backend):
    return await backend.submit_or_get_job(
        make_tenant(), "verification-jobs", "review", "hash", make_request(), default_deadline()
    )


async def test_expiry_cleanup_terminalizes_jobs_without_dispatch(backend, engine):
    job = await submit(backend)
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
    await backend.purge_expired_execution_inputs()
    status = await backend.get_job_status(make_tenant(), job.job_id)
    assert status.state.value == "FAILED", (
        "Cleanup deletes input but leaves expired job QUEUED when dispatch workers are stopped"
    )


async def test_commit_expiry_preserves_completed_attempt_accounting(backend, registry, engine):
    tenant = make_tenant()
    req = make_request().model_copy(
        update={
            "claims": [Claim(claim_id="c1", text="Paris is the capital of France.")],
            "evidence": [Evidence(evidence_id="e1", content="Paris is the capital of France.")],
        }
    )
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "accounting", "hash", req, default_deadline()
    )
    claimed = await backend.claim_due_job("worker", timedelta(minutes=1))
    provider = _ScriptedProvider(
        [
            {
                "c1": _supported("e1").model_copy(
                    update={"usage": Usage(input_tokens=7, output_tokens=3, total_tokens=10)}
                )
            }
        ]
    )
    verifier = Verifier(model_provider=provider)
    cvs, attempts, cal, violations, budget = verifier.dispatch_claims(req.claims, req.evidence)
    started = datetime.now(timezone.utc)
    result = verifier.finalize(
        request=req,
        tenant_id=tenant.tenant_id,
        started_at=started,
        trace_id="t",
        span_id="s",
        claims=req.claims,
        claim_verifications=cvs,
        attempts=attempts,
        calibration_class=cal,
        judge_boundary_violations=violations,
        usage_budget_violation=budget,
    )
    assert result.usage_summary.total_tokens == 10
    async with engine.begin() as conn:
        await conn.execute(
            update(job_execution_inputs)
            .where(job_execution_inputs.c.job_id == job.job_id)
            .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
    await _commit(
        backend,
        registry,
        verifier,
        job.job_id,
        claimed.fencing_token,
        result,
        tenant_id=tenant.tenant_id,
        started_at=started,
        request=req,
        claim_count=1,
        trace_id="t",
        span_id="s",
    )
    status = await backend.get_job_status(tenant, job.job_id)
    assert status.state.value == "FAILED"
    async with engine.connect() as conn:
        stored = {
            table.name: [dict(row) for row in (await conn.execute(select(table))).mappings()]
            for table in metadata.sorted_tables
        }
    assert attempts[0].attempt_id in json.dumps(stored, default=str), (
        "Completed attempt and its known usage disappeared from all persisted tables"
    )
