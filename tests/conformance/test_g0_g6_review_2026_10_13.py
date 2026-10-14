import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from core import provider_identity
from core.verifier import Verifier
from schemas.claims import Claim
from schemas.evidence import Evidence
from schemas.verification import Usage
from storage.postgres.tables import jobs
from tests.conformance.conftest import make_tenant
from tests.conformance.test_worker_runner import (
    _fast_settings,
    _request,
    _ScriptedProvider,
    _submit_and_claim,
    _supported,
)
from worker.runner import _commit, process_claimed_job

pytestmark = pytest.mark.asyncio


def request():
    return _request(
        "review",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
    )


async def test_metadata_only_has_no_raw_content_in_job_after_completion(backend, registry, engine):
    tenant = make_tenant()
    claimed = await _submit_and_claim(backend, tenant, request(), "privacy")
    await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(jobs.c.state, jobs.c.request_json).where(jobs.c.job_id == claimed.job_id)
            )
        ).one()
    assert row.state == "SUCCEEDED"
    assert "Paris is the capital of France." not in json.dumps(row.request_json), (
        "Raw claim and evidence persist in the completed metadata-only job"
    )


async def test_job_deadline_bounds_dispatch(backend, registry):
    class SlowProvider(_ScriptedProvider):
        def evaluate(self, request, deadline, cancellation):
            time.sleep(0.15)
            return super().evaluate(request, deadline, cancellation)

    tenant = make_tenant()
    claimed = await _submit_and_claim(backend, tenant, request(), "deadline")
    claimed = replace(claimed, deadline=datetime.now(timezone.utc) + timedelta(seconds=0.05))
    provider = SlowProvider([{"c1": _supported("e1")}])
    await process_claimed_job(
        claimed, backend, Verifier(model_provider=provider), _fast_settings(), registry
    )
    status = await backend.get_job_status(tenant, claimed.job_id)
    result = (
        await backend.get_result(tenant, status.evaluation_id) if status.evaluation_id else None
    )
    assert result is None or result.verdict.value != "PASS", (
        "Job completed PASS after its fixed deadline"
    )


async def test_revocation_preserves_real_attempts_and_usage(backend, registry):
    tenant = make_tenant()
    claimed = await _submit_and_claim(backend, tenant, request(), "revoked")
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
    cvs, attempts, cal, violations, budget = verifier.dispatch_claims(
        claimed.request.claims, claimed.request.evidence
    )
    started = datetime.now(timezone.utc)
    result = verifier.finalize(
        request=claimed.request,
        tenant_id=tenant.tenant_id,
        started_at=started,
        trace_id="t",
        span_id="s",
        claims=claimed.request.claims,
        claim_verifications=cvs,
        attempts=attempts,
        calibration_class=cal,
        judge_boundary_violations=violations,
        usage_budget_violation=budget,
    )
    assert result.usage_summary.total_tokens == 10
    await registry.revoke(*provider_identity.registry_identity(provider))
    await _commit(
        backend,
        registry,
        verifier,
        claimed.job_id,
        claimed.fencing_token,
        result,
        tenant_id=tenant.tenant_id,
        started_at=started,
        request=claimed.request,
        claim_count=1,
        trace_id="t",
        span_id="s",
    )
    status = await backend.get_job_status(tenant, claimed.job_id)
    persisted = await backend.get_result(tenant, status.evaluation_id)
    assert persisted.abstention_reason.value == "REVOKED"
    assert persisted.usage_summary.total_tokens == 10, (
        "Revocation discarded completed dispatch usage"
    )
    assert attempts[0].attempt_id in {a.attempt_id for a in persisted.provenance.attempts}


async def test_duplicate_processing_yields_one_logical_result(backend, registry, engine):
    from storage.postgres.tables import evaluations

    tenant = make_tenant()
    claimed = await _submit_and_claim(backend, tenant, request(), "duplicate")
    verifier = Verifier()
    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)
    first = await backend.get_job_status(tenant, claimed.job_id)
    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)
    second = await backend.get_job_status(tenant, claimed.job_id)
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(evaluations.c.evaluation_id).where(evaluations.c.job_id == claimed.job_id)
            )
        ).all()
    assert len(rows) == 1, (
        f"Duplicate processing persisted {len(rows)} evaluations and changed {first.evaluation_id} to {second.evaluation_id}"
    )
