"""End-to-end tests for `worker.runner.process_claimed_job` against a real
Postgres-backed `StorageBackend` (`tests/conformance/conftest.py`'s
`backend` fixture) and a real `core.verifier.Verifier` -- the only fake is
the judge provider's own per-claim verdicts, scripted to exercise G5's
bounded-retry and cross-attempt `DISPUTED` reconciliation (ADR-015).
"""

import asyncio
from datetime import datetime, timedelta, timezone

from core import provider_identity
from core.verifier import Verifier
from judges.port import (
    CancellationToken,
    JudgeError,
    JudgeErrorCode,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
)
from schemas.claims import Claim, ClaimVerdict, RationaleCode
from schemas.evidence import Evidence
from schemas.policy import Policy
from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest
from storage.models import ClaimedJob, JobState
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.registry import PostgresRevocationRegistry
from tests.conformance.conftest import default_deadline, make_tenant
from worker.runner import _commit, process_claimed_job, run_poll_loop
from worker.settings import WorkerSettings


class _ScriptedProvider(JudgeProvider):
    """Returns `rounds[N][claim_id]` on the (N+1)th call for that claim_id
    -- lets a test script exactly what each bounded-retry attempt sees,
    independent of claim dispatch order."""

    def __init__(self, rounds: list[dict[str, JudgeResult]]) -> None:
        self._rounds = rounds
        self._counts: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "scripted-test-provider"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        claim_id = request.claim.claim_id
        idx = self._counts.get(claim_id, 0)
        self._counts[claim_id] = idx + 1
        round_index = min(idx, len(self._rounds) - 1)
        return self._rounds[round_index][claim_id]


def _fast_settings(max_attempts: int = 2) -> WorkerSettings:
    """A `WorkerSettings` built without reading the environment (no
    `FACTLAMA_DATABASE_URL` needed here -- the backend fixture already owns
    its own connection) and with near-zero backoff so retry tests run fast."""
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.database_url = "unused-in-this-test"
    settings.worker_id = "worker-test"
    settings.lease_duration_seconds = 60.0
    settings.poll_interval_seconds = 0.01
    settings.max_attempts = max_attempts
    settings.backoff_base_seconds = 0.01
    settings.backoff_max_seconds = 0.02
    settings.backoff_jitter_fraction = 0.0
    settings.outbox_batch_size = 20
    settings.outbox_lease_seconds = 60.0
    settings.outbox_poll_interval_seconds = 0.01
    return settings


def _supported(evidence_id: str) -> JudgeResult:
    return JudgeResult(
        verdict=ClaimVerdict.SUPPORTED,
        evidence_ids=[evidence_id],
        rationale_code=RationaleCode.DIRECT_SUPPORT,
    )


def _contradicted() -> JudgeResult:
    return JudgeResult(
        verdict=ClaimVerdict.CONTRADICTED, rationale_code=RationaleCode.CONTRADICTION_DETECTED
    )


def _failure(code: JudgeErrorCode) -> JudgeResult:
    return JudgeResult(error=JudgeError(code=code, message=f"scripted {code.value}"))


def _request(
    request_id: str,
    claims: list[Claim],
    evidence: list[Evidence] | None = None,
    project_id: str = "proj-a",
    application_id: str = "app-a",
    policy: Policy | None = None,
) -> VerificationRequest:
    return VerificationRequest(
        request_id=request_id,
        project_id=project_id,
        application_id=application_id,
        answer="placeholder answer -- explicit claims are used unchanged",
        claims=claims,
        evidence=evidence or [],
        policy=policy,
    )


async def _submit_and_claim(
    backend: PostgresStorageBackend,
    tenant: TenantContext,
    request: VerificationRequest,
    key: str,
    deadline: datetime | None = None,
) -> ClaimedJob:
    await backend.submit_or_get_job(
        tenant, "verification-jobs", key, f"hash-{key}", request, deadline or default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None
    return claimed


async def test_single_successful_attempt_commits_completed_pass(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    tenant = make_tenant()
    request = _request(
        "req-1",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-1")

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "COMPLETED"
    assert result.verdict.value == "PASS"
    assert len(result.provenance.attempts) == 1


async def test_retryable_failure_then_success_commits_after_one_retry(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    tenant = make_tenant()
    request = _request(
        "req-2",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-2")

    provider = _ScriptedProvider(
        [
            {"c1": _failure(JudgeErrorCode.RATE_LIMIT)},
            {"c1": _supported("e1")},
        ]
    )
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(max_attempts=2), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "COMPLETED"
    assert result.verdict.value == "PASS"
    # One failed + one completed attempt -- the failed attempt is NOT
    # discarded (G5 exit evidence: "failed attempts remain in attempts[]").
    assert len(result.provenance.attempts) == 2
    outcomes = {a.outcome.value for a in result.provenance.attempts}
    assert outcomes == {"FAILED", "COMPLETED"}


async def test_disagreement_across_retry_attempts_yields_disputed(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """The core G5 scenario (ADR-015): claim c1's first dispatch is a
    retryable failure, forcing a full re-dispatch of *both* claims; claim
    c2 already succeeded on attempt 1 and gets a second, independent
    judgment on attempt 2 that disagrees -- DISPUTED, not a silent
    majority vote."""
    tenant = make_tenant()
    request = _request(
        "req-3",
        claims=[
            Claim(claim_id="c1", text="Paris is the capital of France."),
            Claim(claim_id="c2", text="The sky is blue."),
        ],
        evidence=[
            Evidence(evidence_id="e1", content="Paris is the capital of France."),
            Evidence(evidence_id="e2", content="The sky is blue."),
        ],
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-3")

    provider = _ScriptedProvider(
        [
            {"c1": _failure(JudgeErrorCode.RATE_LIMIT), "c2": _supported("e2")},
            {"c1": _supported("e1"), "c2": _contradicted()},
        ]
    )
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(max_attempts=2), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "DISPUTED"
    assert result.verdict.value == "DISPUTED"
    assert result.dispute_reason is not None
    assert result.dispute_reason.value == "JUDGE_DISAGREEMENT"

    claims_by_id = {c.claim_id: c for c in result.claims}
    assert claims_by_id["c1"].verdict == ClaimVerdict.SUPPORTED
    assert claims_by_id["c2"].verdict == ClaimVerdict.DISPUTED
    assert len(claims_by_id["c2"].contributing_judgments) == 2
    assert any(v.code == "JUDGE_DISAGREEMENT" for v in result.violations)


async def test_non_retryable_failure_commits_abstained_provider_failure(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    tenant = make_tenant()
    request = _request(
        "req-4", claims=[Claim(claim_id="c1", text="Paris is the capital of France.")]
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-4")

    provider = _ScriptedProvider([{"c1": _failure(JudgeErrorCode.CONFIGURATION)}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(max_attempts=3), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "PROVIDER_FAILURE"
    # Non-retryable -- exactly one attempt, no retry loop entered.
    assert len(result.provenance.attempts) == 1


async def test_cancellation_before_dispatch_fails_the_job(
    backend: PostgresStorageBackend, registry: PostgresRevocationRegistry
) -> None:
    tenant = make_tenant()
    request = _request("req-5", claims=[Claim(claim_id="c1", text="Paris is the capital.")])
    claimed = await _submit_and_claim(backend, tenant, request, "key-5")
    await backend.request_cancel(tenant, claimed.job_id)

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.FAILED


async def test_poll_loop_claims_and_processes_a_queued_job_then_stops(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    tenant = make_tenant()
    request = _request(
        "req-9",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
    )
    job = await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-9", "hash-9", request, default_deadline()
    )

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)
    settings = _fast_settings()
    stop_event = asyncio.Event()

    async def _stop_once_terminal() -> None:
        for _ in range(200):
            status = await backend.get_job_status(tenant, job.job_id)
            if status is not None and status.state != JobState.QUEUED:
                break
            await asyncio.sleep(0.01)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(
            run_poll_loop(backend, verifier, settings, stop_event, registry),
            _stop_once_terminal(),
        ),
        timeout=5.0,
    )

    status = await backend.get_job_status(tenant, job.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED


async def test_budget_exceeded_before_dispatch_commits_abstained_budget_exhausted(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """ADR-012's pre-dispatch claim-count cap -- gated before any judge
    call, so a `SUCCEEDED` job with zero attempts is the correct outcome,
    not a retry-worthy failure."""
    tenant = make_tenant()
    too_many_claims = [Claim(claim_id=f"c{i}", text=f"claim {i}") for i in range(60)]
    request = _request("req-6", claims=too_many_claims)
    claimed = await _submit_and_claim(backend, tenant, request, "key-6")

    provider = _ScriptedProvider([{}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "BUDGET_EXHAUSTED"
    assert len(result.provenance.attempts) == 1  # one synthetic gate-failure marker, no judge calls


async def test_noncompliant_provider_commits_abstained_no_compliant_provider(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """The scripted test provider declares no compliance tags
    (`JudgeProvider.compliance_tags` defaults to empty); a policy requiring
    one gates the job before any dispatch, same as the sync pipeline."""
    tenant = make_tenant()
    policy = Policy(id="strict-policy", required_provider_compliance=["hipaa"])
    request = _request(
        "req-7", claims=[Claim(claim_id="c1", text="Paris is the capital.")], policy=policy
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-7")

    provider = _ScriptedProvider([{}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "NO_COMPLIANT_PROVIDER"


async def test_deadline_already_past_before_any_dispatch_fails_terminal(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """A job claimed after its own overall deadline has already elapsed
    (e.g. a slow queue) must not be misreported as `NO_CHECKABLE_CLAIMS` --
    it is an operational timeout with real, non-empty claims."""
    tenant = make_tenant()
    request = _request("req-8", claims=[Claim(claim_id="c1", text="Paris is the capital.")])
    past_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    claimed = await _submit_and_claim(backend, tenant, request, "key-8", deadline=past_deadline)

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.FAILED


async def test_revoked_provider_abstains_before_dispatch_without_calling_the_judge(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """EXECUTION_PLAN.md's G5 gate: a baseline REVOKED abstention checked
    against G4's minimal registry entry before dispatch. No judge call
    happens at all -- a revoked provider must not be exercised, not merely
    have its result discarded."""
    tenant = make_tenant()
    request = _request("req-10", claims=[Claim(claim_id="c1", text="Paris is the capital.")])
    claimed = await _submit_and_claim(backend, tenant, request, "key-10")

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)
    await registry.revoke(*provider_identity.registry_identity(provider))

    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "REVOKED"
    assert len(result.provenance.attempts) == 1  # one synthetic gate-failure marker, no judge calls
    assert provider._counts == {}  # the scripted provider's evaluate() was never called


async def test_revocation_between_dispatch_and_commit_overrides_the_committed_result(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """evaluator-registry.md: REVOKED "is rechecked before result commit" --
    a provider revoked *after* the pre-dispatch check already passed (e.g.
    mid-flight during a bounded-retry sequence spanning real wall-clock
    time) must not have its already-computed factual verdict persisted.

    Exercises `_commit` directly rather than the whole `process_claimed_job`
    flow: revoking before the job is even claimed would also trip the
    pre-dispatch gate, which would not isolate this recheck's own behavior.
    A real completed PASS `VerificationResult` is built via
    `verifier.finalize()` first -- exactly as `process_claimed_job` would --
    then handed to `_commit` *after* the provider is revoked, proving the
    override happens at commit time, not only at the earlier gate.
    """
    tenant = make_tenant()
    request = _request(
        "req-11",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
    )
    claimed = await _submit_and_claim(backend, tenant, request, "key-11")

    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)
    assert not await registry.is_revoked(*provider_identity.registry_identity(provider))

    claim_verifications, attempts, calibration_class, violations, usage_violation = (
        verifier.dispatch_claims(claimed.request.claims, claimed.request.evidence)
    )
    started_at = datetime.now(timezone.utc)
    completed_result = verifier.finalize(
        request=claimed.request,
        tenant_id=tenant.tenant_id,
        started_at=started_at,
        trace_id="trace-11",
        span_id="span-11",
        claims=claimed.request.claims,
        claim_verifications=claim_verifications,
        attempts=attempts,
        calibration_class=calibration_class,
        judge_boundary_violations=violations,
        usage_budget_violation=usage_violation,
    )
    # Sanity check: a real factual verdict exists before the override below.
    assert completed_result.status.value == "COMPLETED"

    await registry.revoke(*provider_identity.registry_identity(provider))
    await _commit(
        backend,
        registry,
        verifier,
        claimed.job_id,
        claimed.fencing_token,
        completed_result,
        tenant_id=tenant.tenant_id,
        started_at=started_at,
        request=claimed.request,
        claim_count=len(claimed.request.claims),
        trace_id="trace-11",
        span_id="span-11",
    )

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.state == JobState.SUCCEEDED
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert result.status.value == "ABSTAINED"
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "REVOKED"


async def test_revoke_is_idempotent(registry: PostgresRevocationRegistry) -> None:
    identity = ("provider-x", "model-x@rev1", "0.1")
    await registry.revoke(*identity)
    await registry.revoke(*identity)  # must not raise a unique-violation
    assert await registry.is_revoked(*identity)


async def test_unregistered_identity_is_not_revoked(registry: PostgresRevocationRegistry) -> None:
    """Absence from the registry must never be conflated with revocation --
    G4's own scope leaves every current T0 adapter unregistered by
    default."""
    assert not await registry.is_revoked("some-provider", "some-model@rev1", "0.1")
