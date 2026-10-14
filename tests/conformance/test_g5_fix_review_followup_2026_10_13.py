import json

import pytest
from sqlalchemy import select

from core.verifier import Verifier
from schemas.policy import PolicyAction
from schemas.tools import ToolExecution
from schemas.verification import AbstentionReason
from storage.postgres.tables import jobs
from tests.conformance.conftest import make_tenant
from tests.conformance.test_g0_g6_review_2026_10_13 import request
from tests.conformance.test_worker_runner import _fast_settings, _submit_and_claim
from worker.runner import process_claimed_job

pytestmark = pytest.mark.asyncio


async def test_terminal_metadata_only_removes_tool_payloads(backend, registry, engine):
    req = request().model_copy(
        update={
            "tool_executions": [
                ToolExecution(
                    tool_execution_id="tool-1",
                    tool_name="lookup",
                    arguments={"query": "PRIVATE_QUERY_SENTINEL"},
                    result={"body": "PRIVATE_RESULT_SENTINEL"},
                )
            ]
        }
    )
    claimed = await _submit_and_claim(backend, make_tenant(), req, "tools")
    await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(jobs.c.state, jobs.c.request_json).where(jobs.c.job_id == claimed.job_id)
            )
        ).one()
    assert row.state == "SUCCEEDED"
    stored = json.dumps(row.request_json)
    assert "PRIVATE_QUERY_SENTINEL" not in stored and "PRIVATE_RESULT_SENTINEL" not in stored, (
        "Terminal METADATA_ONLY row still contains raw tool arguments/results"
    )


async def test_revocation_recomputes_policy_and_invalidates_scores():
    verifier = Verifier()
    result = verifier.verify(request(), tenant_id="tenant-a")
    assert result.verdict.value == "PASS"
    revoked = verifier.override_with_post_dispatch_abstention(
        result, AbstentionReason.REVOKED, request=request()
    )
    print(
        "revoked status:",
        revoked.status.value,
        "policy:",
        revoked.policy_action,
        "groundedness:",
        revoked.scores["groundedness"].model_dump(),
    )
    assert revoked.policy_action == PolicyAction.HUMAN_REVIEW, (
        "Revoked result still instructs caller to PASS"
    )
    assert revoked.scores["groundedness"].status.value == "UNAVAILABLE"
