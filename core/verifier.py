"""Main verification module."""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from core import provider_identity
from core.budgets import (
    check_request_budget,
    check_usage_budget,
    check_usage_reservation,
    describe_cost_completeness,
    summarize_usage,
)
from core.claims import ClaimExtractor, EnhancedClaimExtractor
from core.compliance import is_provider_compliant
from core.evidence import EvidenceMapper, SimpleEvidenceMapper
from core.policy import PolicyEngine
from core.result import VerificationResultBuilder
from core.scoring import ScoringEngine, derive_calibration_class, determine_status
from judges.port import (
    CancellationToken,
    JudgeError,
    JudgeErrorCode,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
    apply_citation_support_check,
    detect_evidence_injection,
    validate_judge_result,
)
from judges.providers import RuleBasedProvider
from schemas.citation import Citation
from schemas.claims import (
    ClaimVerdict,
    ClaimVerification,
    ContributingJudgment,
    RationaleCode,
    TextStatus,
)
from schemas.evidence import Evidence
from schemas.instruction import Instruction, Priority
from schemas.policy import Policy
from schemas.tools import ToolExecution, ToolStatus
from schemas.verification import (
    AbstentionReason,
    Attempt,
    AttemptOutcome,
    OverallVerdict,
    Provenance,
    QualificationStatus,
    ResultStatus,
    Severity,
    Usage,
    VerificationRequest,
    VerificationResult,
    Violation,
)

logger = logging.getLogger(__name__)

_INSTRUCTION_SEVERITY = {
    Priority.CRITICAL: Severity.CRITICAL,
    Priority.HIGH: Severity.HIGH,
    Priority.MEDIUM: Severity.MEDIUM,
    Priority.LOW: Severity.LOW,
}

# Below this adherence score, an instruction is considered violated.
INSTRUCTION_ADHERENCE_THRESHOLD = 0.5

# Per-claim judge dispatch deadline. ADR-012's claim/evidence fan-out and
# usage/cost ceilings are enforced separately (core.budgets).
DEFAULT_JUDGE_TIMEOUT_SECONDS = 30.0

# Overall wall-clock ceiling for one request's entire judge-dispatch phase
# (every claim combined), distinct from DEFAULT_JUDGE_TIMEOUT_SECONDS, which
# only bounds a single claim's own call. F3 of the 2026-09-21 G0-G4
# validation report: a multi-claim request previously had no total deadline,
# so N claims each individually within budget could still run unbounded
# cumulative wall-clock time.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

EVALUATOR_ID = "reliability-verifier"
EVALUATOR_VERSION = "0.1"


class Verifier:
    """Main verification engine orchestrating the verification pipeline.

    The Verifier coordinates:
    1. Claim extraction from answers
    2. Evidence mapping to claims
    3. Claim verification via a bounded JudgeProvider call per claim
    4. Score calculation
    5. Policy evaluation
    6. Result compilation
    """

    def __init__(
        self,
        model_provider: JudgeProvider | None = None,
        claim_extractor: ClaimExtractor | None = None,
        evidence_mapper: EvidenceMapper | None = None,
        scoring_engine: ScoringEngine | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            model_provider: JudgeProvider for claim verification (defaults to RuleBasedProvider).
            claim_extractor: Claim extraction module (defaults to EnhancedClaimExtractor,
                which splits compound sentences into atomic clause-level claims).
            evidence_mapper: Evidence mapping module (defaults to SimpleEvidenceMapper).
            scoring_engine: Scoring engine (defaults to ScoringEngine).
            policy_engine: Policy evaluation engine (defaults to PolicyEngine).
        """
        self.model_provider = model_provider or RuleBasedProvider()
        self.claim_extractor = claim_extractor or EnhancedClaimExtractor()
        self.evidence_mapper = evidence_mapper or SimpleEvidenceMapper()
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.policy_engine = policy_engine or PolicyEngine()

    def verify(
        self, request: VerificationRequest, tenant_id: str = "default"
    ) -> VerificationResult:
        """Execute the full verification pipeline.

        Args:
            request: The verification request (never carries tenant identity).
            tenant_id: Trusted tenant identifier established at ingress by the
                caller -- never taken from `request` itself.

        Returns:
            VerificationResult with verdict, scores, and claims.
        """
        started_at = datetime.now(timezone.utc)
        trace_id = request.trace_id or f"trace_{uuid.uuid4().hex[:32]}"
        span_id = request.span_id or f"span_{uuid.uuid4().hex[:16]}"

        builder = VerificationResultBuilder(
            request_id=request.request_id,
            tenant_id=tenant_id,
            project_id=request.project_id,
            application_id=request.application_id,
            interaction_id=request.interaction_id,
        )
        builder.with_trace_id(trace_id).with_span_id(span_id)

        # F5 of the 2026-09-21 G0-G4 validation report: an exception raised
        # downstream of claim dispatch (e.g. in scoring or policy
        # evaluation) must not erase judge work already completed for this
        # request. Declared outside the try block so the except handler
        # below can still see whatever `dispatch_claims()` actually returned,
        # instead of discarding it in favor of one synthetic failure.
        claim_verifications: list[ClaimVerification] = []
        attempts: list[Attempt] = []

        try:
            policy = request.policy or Policy(id="default")

            claims = (
                list(request.claims) if request.claims else self._extract_claims(request.answer)
            )

            budget_violation = check_request_budget(len(claims), len(request.evidence))
            if budget_violation is not None:
                logger.warning(
                    "Request %s abstained before dispatch: %s",
                    request.request_id,
                    budget_violation,
                )
                return self.abstained_result(
                    tenant_id,
                    started_at,
                    request,
                    reason=AbstentionReason.BUDGET_EXHAUSTED,
                    error_label=JudgeErrorCode.BUDGET_EXHAUSTED.value,
                    claim_count=len(claims),
                    trace_id=trace_id,
                    span_id=span_id,
                )

            if not is_provider_compliant(
                self.model_provider.compliance_tags, policy.required_provider_compliance
            ):
                logger.warning(
                    "Request %s abstained before dispatch: provider '%s' (tags=%s) does not "
                    "satisfy policy's required_provider_compliance=%s",
                    request.request_id,
                    self.model_provider.name,
                    sorted(self.model_provider.compliance_tags),
                    policy.required_provider_compliance,
                )
                return self.abstained_result(
                    tenant_id,
                    started_at,
                    request,
                    reason=AbstentionReason.NO_COMPLIANT_PROVIDER,
                    error_label=JudgeErrorCode.NO_COMPLIANT_PROVIDER.value,
                    claim_count=len(claims),
                    trace_id=trace_id,
                    span_id=span_id,
                )

            (
                claim_verifications,
                attempts,
                calibration_class,
                judge_citation_violations,
                usage_budget_violation,
            ) = self.dispatch_claims(
                claims,
                request.evidence,
            )
            if usage_budget_violation is not None:
                logger.warning(
                    "Request %s's usage budget exceeded mid-dispatch: %s",
                    request.request_id,
                    usage_budget_violation,
                )

            return self.finalize(
                request=request,
                tenant_id=tenant_id,
                started_at=started_at,
                trace_id=trace_id,
                span_id=span_id,
                claims=claims,
                claim_verifications=claim_verifications,
                attempts=attempts,
                calibration_class=calibration_class,
                judge_boundary_violations=judge_citation_violations,
                usage_budget_violation=usage_budget_violation,
            )

        except Exception:
            logger.exception(
                "Unexpected pipeline failure during verification of request %s", request.request_id
            )
            completed_at = datetime.now(timezone.utc)
            configuration_version = (
                provider_identity.configuration_fingerprint(self.model_provider)
                or provider_identity.DEFAULT_CONFIGURATION_VERSION
            )
            resolved_pinned_version = provider_identity.pinned_model_version(self.model_provider)
            provider_model_id = provider_identity.model_id(self.model_provider)
            failure_attempt = Attempt(
                attempt_id=f"attempt_{uuid.uuid4().hex[:12]}",
                provider_id=self.model_provider.name,
                model_id=provider_model_id,
                pinned_model_version=resolved_pinned_version,
                configuration_version=configuration_version,
                qualification_status=QualificationStatus.UNQUALIFIED,
                calibration_class=derive_calibration_class(
                    evaluator_id=EVALUATOR_ID,
                    evaluator_version=EVALUATOR_VERSION,
                    provider_id=self.model_provider.name,
                    # Same fix as the main per-claim path below: use the
                    # composite (primary + secondary) identity, not the
                    # primary-only resolved revision.
                    pinned_model_id=(
                        provider_identity.full_pinned_model_id(self.model_provider)
                        or provider_model_id
                        or self.model_provider.name
                    ),
                    configuration_version=configuration_version,
                    qualification_status=QualificationStatus.UNQUALIFIED.value,
                ),
                outcome=AttemptOutcome.FAILED,
                error="CONFIGURATION",
                started_at=started_at,
                completed_at=completed_at,
            )
            # F5 of the 2026-09-21 G0-G4 validation report: `attempts` and
            # `claim_verifications` hold whatever real judge work completed
            # before this exception (declared outside the try block above) --
            # append the synthetic failure marker rather than replacing them,
            # so a later exception never erases earlier known usage/cost.
            all_attempts = [*attempts, failure_attempt]
            provenance = Provenance(
                evaluator_id=EVALUATOR_ID,
                evaluator_version=EVALUATOR_VERSION,
                policy_version=None,
                mode=request.mode,
                routing_profile_version=f"{request.mode.value.lower()}-0.1",
                started_at=started_at,
                completed_at=completed_at,
                attempts=all_attempts,
            )
            return (
                builder.with_status(ResultStatus.FAILED)
                .with_abstention_reason(AbstentionReason.PROVIDER_FAILURE)
                .with_verdict(OverallVerdict.ABSTAIN)
                .with_scores(self.scoring_engine.calculate_scores([], calibration_class=None))
                .with_claims(claim_verifications)
                .with_provenance(provenance)
                .with_usage_summary(summarize_usage(all_attempts))
                .with_metadata({"failure": "internal pipeline error"})
                .build()
            )

    def finalize(
        self,
        request: VerificationRequest,
        tenant_id: str,
        started_at: datetime,
        trace_id: str,
        span_id: str,
        claims: list,
        claim_verifications: list[ClaimVerification],
        attempts: list[Attempt],
        calibration_class: str | None,
        judge_boundary_violations: list[Violation],
        usage_budget_violation: str | None,
    ) -> VerificationResult:
        """Everything after per-claim judge dispatch: deterministic
        instruction/scope/citation/tool checks, scoring, aggregate
        status/verdict (including G5's `DISPUTED` precedence), policy
        evaluation, and result assembly.

        Split out of `verify()` (G5) so `worker.runner` can call it a
        second time with claim_verifications/attempts *merged* across a
        job's bounded retry attempts, reusing this exact scoring/policy
        logic instead of approximating it -- the only difference between a
        single synchronous call and a worker's reconciled result is what
        `claim_verifications`/`attempts` this method is handed.

        `judge_boundary_violations` is deliberately general, not
        citation-specific: `verify()` passes `dispatch_claims()`'s own
        citation-overlap/injection violations; `worker.runner` additionally
        folds in `worker.reconciliation`'s `JUDGE_DISAGREEMENT` violations
        for any claim two attempts disputed -- both are judge-boundary
        signals this method must not compute itself, only include.
        """
        builder = VerificationResultBuilder(
            request_id=request.request_id,
            tenant_id=tenant_id,
            project_id=request.project_id,
            application_id=request.application_id,
            interaction_id=request.interaction_id,
        )
        builder.with_trace_id(trace_id).with_span_id(span_id)

        policy = request.policy or Policy(id="default")

        instruction_violations = self._evaluate_instructions(request.answer, request.instructions)
        scope_violations = self._evaluate_scope(request.answer, policy)
        citation_support_score, citation_violations = self._evaluate_citations(
            request.citations, claim_verifications, request.evidence
        )
        tool_correctness_score, tool_violations = self._evaluate_tools(request.tool_executions)

        scores = self.scoring_engine.calculate_scores(
            claim_verifications=claim_verifications,
            calibration_class=calibration_class,
            citation_support_score=citation_support_score,
            tool_correctness_score=tool_correctness_score,
        )

        violations = (
            self._generate_violations(claim_verifications)
            + instruction_violations
            + scope_violations
            + citation_violations
            + judge_boundary_violations
            + tool_violations
        )

        status, verdict, abstention_reason, dispute_reason = determine_status(
            claims,
            claim_verifications,
            attempts,
            usage_budget_exceeded=usage_budget_violation is not None,
        )

        policy_action, policy_violations = self.policy_engine.evaluate(
            verdict=verdict,
            scores=scores,
            violations=violations,
            policy=request.policy,
            has_citations=bool(request.citations),
        )

        if request.policy:
            builder.with_policy_version(request.policy.version)

        completed_at = datetime.now(timezone.utc)
        provenance = Provenance(
            evaluator_id=EVALUATOR_ID,
            evaluator_version=EVALUATOR_VERSION,
            policy_version=request.policy.version if request.policy else None,
            mode=request.mode,
            routing_profile_version=f"{request.mode.value.lower()}-0.1",
            started_at=started_at,
            completed_at=completed_at,
            attempts=attempts,
        )

        metadata: dict[str, Any] = {
            "provider": self.model_provider.name,
            "claim_count": len(claims),
            "evidence_count": len(request.evidence),
            "instruction_count": len(request.instructions),
            "citation_count": len(request.citations),
            "tool_count": len(request.tool_executions),
        }
        if not request.claims:
            # Extraction mode (not explicit claims): record which
            # extractor/version produced these claim IDs/offsets
            # (claim-engine.md: "Record extractor ID/version in
            # provenance; changing segmentation may change scores").
            # `getattr` with a fallback, not a hard attribute access:
            # a caller-supplied extractor need only satisfy `extract()`
            # (see e.g. tests' duck-typed stubs), not this optional
            # identity contract.
            metadata["extractor_id"] = getattr(
                self.claim_extractor, "extractor_id", type(self.claim_extractor).__name__
            )
            metadata["extractor_version"] = getattr(
                self.claim_extractor, "extractor_version", "unknown"
            )

        cost_completeness = describe_cost_completeness(attempts)
        if cost_completeness is not None:
            # summarize_usage()'s cost is a known partial total whenever
            # an attempt's cost was UNAVAILABLE or measured in a
            # non-USD currency -- flag that explicitly so a caller
            # never mistakes "$X known" for "$X total" (see
            # describe_cost_completeness's own docstring).
            metadata["usage_cost_completeness"] = cost_completeness

        return (
            builder.with_status(status)
            .with_abstention_reason(abstention_reason)
            .with_dispute_reason(dispute_reason)
            .with_verdict(verdict)
            .with_scores(scores)
            .with_claims(claim_verifications)
            .with_violations(violations + policy_violations)
            .with_policy_action(policy_action)
            .with_provenance(provenance)
            .with_usage_summary(summarize_usage(attempts))
            .with_metadata(metadata)
            .build()
        )

    def abstained_result(
        self,
        tenant_id: str,
        started_at: datetime,
        request: VerificationRequest,
        reason: AbstentionReason,
        error_label: str,
        claim_count: int,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> VerificationResult:
        """Build a zero-dispatch abstained result for a pre-dispatch gate
        failure (budget or provider compliance) -- no judge call happened
        for any claim, so there is exactly one synthetic failed Attempt
        recording why, not one per claim.

        Public (G5): `verify()` calls this for its own pre-dispatch gates;
        `worker.runner` calls it identically for the same gates, checked
        once per job before any bounded-retry attempt (ADR-012's budget
        caps and ADR-004's compliance check are provider/request
        properties, not something a retry could change).
        """
        builder = VerificationResultBuilder(
            request_id=request.request_id,
            tenant_id=tenant_id,
            project_id=request.project_id,
            application_id=request.application_id,
            interaction_id=request.interaction_id,
        )
        builder.with_trace_id(trace_id).with_span_id(span_id)
        completed_at = datetime.now(timezone.utc)
        attempt = Attempt(
            attempt_id=f"attempt_{uuid.uuid4().hex[:12]}",
            provider_id=self.model_provider.name,
            configuration_version="0.1",
            qualification_status=QualificationStatus.UNQUALIFIED,
            calibration_class=derive_calibration_class(
                evaluator_id=EVALUATOR_ID,
                evaluator_version=EVALUATOR_VERSION,
                provider_id=self.model_provider.name,
                pinned_model_id=self.model_provider.name,
                configuration_version="0.1",
                qualification_status=QualificationStatus.UNQUALIFIED.value,
            ),
            outcome=AttemptOutcome.FAILED,
            error=error_label,
            started_at=started_at,
            completed_at=completed_at,
        )
        provenance = Provenance(
            evaluator_id=EVALUATOR_ID,
            evaluator_version=EVALUATOR_VERSION,
            policy_version=request.policy.version if request.policy else None,
            mode=request.mode,
            routing_profile_version=f"{request.mode.value.lower()}-0.1",
            started_at=started_at,
            completed_at=completed_at,
            attempts=[attempt],
        )
        return (
            builder.with_status(ResultStatus.ABSTAINED)
            .with_abstention_reason(reason)
            .with_verdict(OverallVerdict.ABSTAIN)
            .with_scores(self.scoring_engine.calculate_scores([], calibration_class=None))
            .with_provenance(provenance)
            .with_metadata(
                {
                    "provider": self.model_provider.name,
                    "claim_count": claim_count,
                    "evidence_count": len(request.evidence),
                }
            )
            .build()
        )

    def _extract_claims(self, answer: str) -> list:
        """Extract claims from the answer."""
        return self.claim_extractor.extract(answer)

    def dispatch_claims(
        self,
        claims: list,
        evidence: list[Evidence],
    ) -> tuple[list[ClaimVerification], list[Attempt], str | None, list[Violation], str | None]:
        """Verify all claims against evidence via one bounded JudgeProvider call each.

        Public (G5): `verify()` calls this once per request; `worker.runner`
        calls it once per bounded-retry attempt, feeding the accumulated
        claim_verifications/attempts from every attempt into `finalize()`
        (via `worker.reconciliation`) once retries are exhausted, instead of
        each attempt separately calling the full `verify()` pipeline (which
        would re-run policy/scoring per attempt instead of once on the
        merged result).

        Returns:
            Tuple of (claim verifications, judge attempts, calibration class
            shared by every attempt -- or None when there were no claims to
            dispatch at all -- violations raised by the judge-boundary
            citation checks below, and a budget violation reason if the
            request's cumulative usage/cost exceeded its per-request ceiling,
            or its overall wall-clock deadline (`DEFAULT_REQUEST_TIMEOUT_
            SECONDS`) elapsed, partway through -- either because a completed
            attempt's real usage pushed it over (`check_usage_budget`, after
            dispatch), dispatching one more call could not be reserved
            against what remains (`check_usage_reservation`, before
            dispatch), or the request's total time budget was already spent
            before the next claim could be dispatched -- in which case any
            remaining claims were never dispatched).
        """
        verifications: list[ClaimVerification] = []
        attempts: list[Attempt] = []
        citation_violations: list[Violation] = []
        calibration_class: str | None = None
        usage_budget_violation: str | None = None

        # F2 of the 2026-09-21 G0-G4 validation report: `self.model_provider.
        # name` alone cannot distinguish two same-model providers configured
        # with different tunable thresholds (e.g. two EmbeddingProviders at
        # support_threshold=0.70 vs 0.95) -- use the same shared identity
        # primitives the offline agreement harness uses instead. The
        # provider's own tunable state is fixed for the lifetime of this
        # Verifier call, so the fingerprint/model_id are computed once;
        # `pinned_model_version` (a vendor adapter's resolved model
        # revision) is re-read per attempt below since it only becomes
        # available once the model has actually loaded.
        provider_model_id = provider_identity.model_id(self.model_provider)
        provider_configuration_version = (
            provider_identity.configuration_fingerprint(self.model_provider)
            or provider_identity.DEFAULT_CONFIGURATION_VERSION
        )

        # F3 of the 2026-09-21 G0-G4 validation report: this bounds the
        # *whole* claim-dispatch loop's cumulative wall-clock time, not just
        # each individual claim's own DEFAULT_JUDGE_TIMEOUT_SECONDS -- a
        # request with many claims that each individually finish in time
        # could previously still run unbounded in total.
        request_deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_SECONDS

        for claim in claims:
            reservation_violation = check_usage_reservation(summarize_usage(attempts))
            if reservation_violation is not None:
                usage_budget_violation = reservation_violation
                logger.warning(
                    "Usage budget reservation exceeded before dispatching claim %s: %s",
                    claim.claim_id,
                    reservation_violation,
                )
                break

            now = time.monotonic()
            if now > request_deadline:
                usage_budget_violation = (
                    f"overall request deadline of {DEFAULT_REQUEST_TIMEOUT_SECONDS}s "
                    f"exceeded before dispatching claim {claim.claim_id}"
                )
                logger.warning(
                    "Request deadline exceeded before dispatching claim %s", claim.claim_id
                )
                break

            cancellation = CancellationToken()
            # Bounded by whichever is sooner: this claim's own per-claim
            # timeout, or what remains of the overall request deadline --
            # otherwise a claim dispatched just before the request deadline
            # could still run for another full DEFAULT_JUDGE_TIMEOUT_SECONDS.
            deadline = min(now + DEFAULT_JUDGE_TIMEOUT_SECONDS, request_deadline)
            judge_request = JudgeRequest(
                claim=claim,
                evidence=evidence,
                configuration_version=provider_configuration_version,
            )

            attempt_started = datetime.now(timezone.utc)
            try:
                judge_result = self.model_provider.evaluate(judge_request, deadline, cancellation)
            except Exception:
                # F5 of the 2026-09-21 G0-G4 validation report: an
                # unexpected exception from one claim's dispatch must not
                # discard every attempt already completed for earlier claims
                # in this same request -- normalize it into the same typed
                # dispatch-failure shape a provider's own JudgeError already
                # produces (handled by the `judge_result.error is not None`
                # branch below), instead of letting it propagate and lose
                # this loop's accumulated `verifications`/`attempts`.
                logger.exception(
                    "Judge provider '%s' raised unexpectedly evaluating claim %s",
                    self.model_provider.name,
                    claim.claim_id,
                )
                judge_result = JudgeResult(
                    error=JudgeError(
                        code=JudgeErrorCode.UNAVAILABLE,
                        message="judge provider raised an unexpected exception",
                    )
                )
            if judge_result.error is None and time.monotonic() > deadline:
                # F3 of the 2026-09-21 G0-G4 validation report: a provider
                # that returns a verdict *after* its own deadline must not
                # be trusted as a normal success -- `bounded_check()` inside
                # each T0 adapter only checks the deadline before its own
                # model work starts, not once that work has actually
                # finished, so a slow call could otherwise still land as
                # COMPLETED/PASS.
                logger.warning(
                    "Judge provider '%s' returned after its deadline for claim %s; "
                    "rejecting the late result",
                    self.model_provider.name,
                    claim.claim_id,
                )
                # R4 of the 2026-09-21 re-audit: the work already happened
                # and any usage/cost it incurred is real and known -- only
                # the verdict is untrusted, not the accounting. Preserve it
                # on the typed failure rather than discarding it.
                judge_result = JudgeResult(
                    error=JudgeError(
                        code=JudgeErrorCode.TIMEOUT,
                        message="provider returned a result after its deadline",
                    ),
                    usage=judge_result.usage,
                )
            # Read after `evaluate()`: a vendor adapter's resolved model
            # revision(s) are only available once its model(s) have actually
            # loaded (`judges/vendor_adapters.py`'s `resolved_revision`/
            # `secondary_resolved_revision`), which may first happen during
            # this very call.
            resolved_pinned_version = provider_identity.pinned_model_version(self.model_provider)
            # Re-audit finding (2026-09-21, runtime identity): this must be
            # `full_pinned_model_id()`, not bare `pinned_model_version()` --
            # the latter reports only the primary model's revision, so a
            # composite evaluator like NLIProvider (primary NLI model +
            # secondary relatedness model) could change its secondary
            # model's resolved revision without changing calibration_class,
            # silently violating the changed-model -> changed-class rule
            # `scripts/run_qualification.py`/`run_agreement_harness.py`
            # already uphold via this same shared primitive.
            calibration_class = derive_calibration_class(
                evaluator_id=EVALUATOR_ID,
                evaluator_version=EVALUATOR_VERSION,
                provider_id=self.model_provider.name,
                pinned_model_id=(
                    provider_identity.full_pinned_model_id(self.model_provider)
                    or provider_model_id
                    or self.model_provider.name
                ),
                configuration_version=judge_request.configuration_version,
                qualification_status=QualificationStatus.UNQUALIFIED.value,
            )
            # Reject a SUPPORTED verdict citing no evidence ID, or an ID absent
            # from this request (a hallucinated citation), before trusting it.
            judge_result = validate_judge_result(judge_result, judge_request)
            # Then run the conservative, non-model overlap/support check on
            # whatever real evidence was cited (CONTRACTS.md). Re-audit
            # follow-up (2026-09-21): this no longer overrides the verdict
            # itself -- see `apply_citation_support_check()`'s own docstring
            # for why zero recognized overlap alone must not be treated as
            # proof of a fabricated citation. `judge_result` is therefore
            # never reassigned here; only whether to flag is read back.
            _, citation_overlap_inconclusive = apply_citation_support_check(
                judge_result, judge_request
            )
            attempt_completed = datetime.now(timezone.utc)

            attempt_id = f"attempt_{uuid.uuid4().hex[:12]}"
            usage = judge_result.usage or Usage()

            if citation_overlap_inconclusive:
                # Not a verdict downgrade (see above) -- CITATION_OVERLAP_INCONCLUSIVE
                # is a distinct code from `core.policy`'s own CITATION_MISMATCH
                # (which flags a policy-level citation *requirement* violation, an
                # unrelated concern) so this can be routed to mandatory human
                # review (`core.policy`, ADR-013 pattern) without also forcing
                # review on every policy-level citation-requirement violation.
                citation_violations.append(
                    Violation(
                        code="CITATION_OVERLAP_INCONCLUSIVE",
                        severity=Severity.MEDIUM,
                        claim_ids=[claim.claim_id],
                        evidence_ids=judge_result.evidence_ids,
                        message=(
                            "Judge-cited evidence shares no recognized lexical/synonym/"
                            "morphological overlap with the claim; the SUPPORTED verdict "
                            "is not overridden (a legitimate paraphrase is not rejected "
                            "solely for lacking shared words -- CONTRACTS.md), but this "
                            "is routed to human review since this deterministic guard "
                            "cannot distinguish that case from a fabricated citation."
                        ),
                    )
                )

            injected_evidence_ids = detect_evidence_injection(judge_result, judge_request)
            if injected_evidence_ids:
                citation_violations.append(
                    Violation(
                        code="EVIDENCE_INJECTION_SUSPECTED",
                        severity=Severity.HIGH,
                        claim_ids=[claim.claim_id],
                        evidence_ids=injected_evidence_ids,
                        message=(
                            "Cited evidence contains judge-directive-style language; "
                            "the verdict above was reached independently of it, but the "
                            "attempt is flagged for review"
                        ),
                    )
                )

            if judge_result.error is None:
                verdict = judge_result.verdict
                assert verdict is not None  # verdict-xor-error invariant
                rationale_code = judge_result.rationale_code or RationaleCode.NO_SUPPORT
                verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        verdict=verdict,
                        evidence_ids=judge_result.evidence_ids,
                        rationale_code=rationale_code,
                        rationale=judge_result.reason,
                        text_status=TextStatus.AVAILABLE,
                        text=claim.text,
                        contributing_judgments=[
                            ContributingJudgment(
                                attempt_id=attempt_id,
                                verdict=verdict,
                                evidence_ids=judge_result.evidence_ids,
                                rationale_code=rationale_code,
                            )
                        ],
                        confidence=judge_result.confidence
                        if judge_result.confidence is not None
                        else 1.0,
                    )
                )
                attempts.append(
                    Attempt(
                        attempt_id=attempt_id,
                        provider_id=self.model_provider.name,
                        model_id=provider_model_id,
                        pinned_model_version=resolved_pinned_version,
                        configuration_version=judge_request.configuration_version,
                        qualification_status=QualificationStatus.UNQUALIFIED,
                        calibration_class=calibration_class,
                        outcome=AttemptOutcome.COMPLETED,
                        started_at=attempt_started,
                        completed_at=attempt_completed,
                        usage=usage,
                    )
                )
            else:
                # Provider/dispatch failure is never a factual UNSUPPORTED/FAIL.
                verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                        evidence_ids=[],
                        rationale_code=RationaleCode.PROVIDER_DISPATCH_FAILED,
                        rationale=f"Judge dispatch failed: {judge_result.error.code.value}",
                        text_status=TextStatus.AVAILABLE,
                        text=claim.text,
                        contributing_judgments=[
                            ContributingJudgment(
                                attempt_id=attempt_id,
                                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                                evidence_ids=[],
                                rationale_code=RationaleCode.PROVIDER_DISPATCH_FAILED,
                            )
                        ],
                        confidence=0.0,
                    )
                )
                attempts.append(
                    Attempt(
                        attempt_id=attempt_id,
                        provider_id=self.model_provider.name,
                        model_id=provider_model_id,
                        pinned_model_version=resolved_pinned_version,
                        configuration_version=judge_request.configuration_version,
                        qualification_status=QualificationStatus.UNQUALIFIED,
                        calibration_class=calibration_class,
                        outcome=AttemptOutcome.FAILED,
                        error=judge_result.error.code.value,
                        usage=usage,
                        started_at=attempt_started,
                        completed_at=attempt_completed,
                    )
                )

            usage_budget_violation = check_usage_budget(summarize_usage(attempts))
            if usage_budget_violation is not None:
                logger.warning(
                    "Usage budget exceeded after claim %s: %s",
                    claim.claim_id,
                    usage_budget_violation,
                )
                break

        return (
            verifications,
            attempts,
            calibration_class,
            citation_violations,
            usage_budget_violation,
        )

    def _evaluate_instructions(
        self,
        answer: str,
        instructions: list[Instruction],
    ) -> list[Violation]:
        """Evaluate adherence to every applicable instruction and flag violations.

        Delegates the actual judgment to the model provider (adherence to an
        instruction is inherently answer-dependent and not purely structural).
        This dimension is UNAVAILABLE in `scores` (no accepted v0.1 formula
        exists yet) -- only the per-instruction violations are formal output.
        """
        if not instructions:
            return []

        violations: list[Violation] = []

        for instruction in instructions:
            score, reason = self.model_provider.evaluate_instruction(answer, instruction)

            if score < INSTRUCTION_ADHERENCE_THRESHOLD:
                violations.append(
                    Violation(
                        code="INSTRUCTION_VIOLATION",
                        severity=_INSTRUCTION_SEVERITY.get(instruction.priority, Severity.MEDIUM),
                        message=reason
                        or (
                            f"Instruction '{instruction.instruction_id}' not adequately followed "
                            f"(adherence={score:.2f})"
                        ),
                        metadata={
                            "instruction_id": instruction.instruction_id,
                            "adherence_score": score,
                        },
                    )
                )

        return violations

    def _evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> list[Violation]:
        """Evaluate scope compliance against the policy's scope constraints.

        This dimension is UNAVAILABLE in `scores` (no accepted v0.1 formula
        exists yet) -- only the violation is formal output.
        """
        _breach_score, breach_domains = self.model_provider.evaluate_scope(answer, policy)

        if not breach_domains:
            return []

        return [
            Violation(
                code="SCOPE_BREACH",
                severity=Severity.HIGH,
                message=f"Answer breaches defined scope: {', '.join(breach_domains)}",
                metadata={"domains": breach_domains},
            )
        ]

    def _evaluate_citations(
        self,
        citations: list[Citation],
        claim_verifications: list[ClaimVerification],
        evidence: list[Evidence],
    ) -> tuple[float | None, list[Violation]]:
        """Evaluate whether supplied citations actually support their claims.

        This is a deterministic structural check (does the cited source exist,
        does it match evidence the claim was actually verified against) rather
        than a model judgment.

        Returns:
            Tuple of (fraction of valid citations, or None if no citations
            were supplied; violations for invalid ones).
        """
        if not citations:
            return None, []

        evidence_ids = {e.evidence_id for e in evidence}
        verification_by_claim = {v.claim_id: v for v in claim_verifications}

        valid = 0
        violations: list[Violation] = []

        for citation in citations:
            claim_ids = [citation.claim_id] if citation.claim_id else []

            if citation.evidence_id not in evidence_ids:
                violations.append(
                    Violation(
                        code="CITATION_MISMATCH",
                        severity=Severity.MEDIUM,
                        claim_ids=claim_ids,
                        evidence_ids=[citation.evidence_id],
                        message=(
                            f"Citation '{citation.citation_id}' references unknown source "
                            f"'{citation.evidence_id}'"
                        ),
                    )
                )
                continue

            verification = (
                verification_by_claim.get(citation.claim_id) if citation.claim_id else None
            )
            if verification is not None and not (
                citation.evidence_id in verification.evidence_ids
                and verification.verdict == ClaimVerdict.SUPPORTED
            ):
                violations.append(
                    Violation(
                        code="CITATION_MISMATCH",
                        severity=Severity.MEDIUM,
                        claim_ids=claim_ids,
                        evidence_ids=[citation.evidence_id],
                        message=(
                            f"Citation '{citation.citation_id}' does not support claim "
                            f"'{citation.claim_id}'"
                        ),
                    )
                )
                continue

            valid += 1

        return round(valid / len(citations), 4), violations

    def _evaluate_tools(
        self,
        tools: list[ToolExecution],
    ) -> tuple[float | None, list[Violation]]:
        """Evaluate agent tool usage correctness from execution status.

        Deterministic by design: correctness here means "did the tool call
        complete successfully," which is a fact recorded on the ToolExecution.

        Returns:
            Tuple of (fraction of successful executions, or None if no tool
            executions were supplied; violations for failures).
        """
        if not tools:
            return None, []

        successful = 0
        violations: list[Violation] = []

        for tool in tools:
            if tool.status == ToolStatus.SUCCESS:
                successful += 1
            else:
                violations.append(
                    Violation(
                        code="TOOL_ERROR",
                        severity=Severity.HIGH
                        if tool.status in (ToolStatus.ERROR, ToolStatus.TIMEOUT)
                        else Severity.MEDIUM,
                        message=tool.error_message
                        or (f"Tool '{tool.tool_name}' finished with status {tool.status.value}"),
                        metadata={
                            "tool_id": tool.tool_execution_id,
                            "tool_name": tool.tool_name,
                            "status": tool.status.value,
                        },
                    )
                )

        return round(successful / len(tools), 4), violations

    def _generate_violations(
        self,
        verifications: list[ClaimVerification],
    ) -> list[Violation]:
        """Generate violations from claim verifications."""
        violations: list[Violation] = []

        for v in verifications:
            if v.verdict == ClaimVerdict.UNSUPPORTED:
                violations.append(
                    Violation(
                        code="UNSUPPORTED_CLAIM",
                        severity=Severity.MEDIUM,
                        claim_ids=[v.claim_id],
                        message="Claim is not supported by supplied evidence",
                    )
                )
            elif v.verdict == ClaimVerdict.CONTRADICTED:
                violations.append(
                    Violation(
                        code="CONTRADICTED_CLAIM",
                        severity=Severity.CRITICAL,
                        claim_ids=[v.claim_id],
                        message="Claim contradicts the evidence",
                    )
                )

        return violations


# Default verifier instance
_default_verifier: Verifier | None = None


def get_default_verifier() -> Verifier:
    """Get or create the default verifier instance."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = Verifier()
    return _default_verifier


def verify(
    answer: str,
    question: str | None = None,
    evidence: list[Evidence] | None = None,
    request_id: str | None = None,
    tenant_id: str = "default",
    project_id: str = "default-project",
    application_id: str = "default-app",
    **kwargs: Any,
) -> VerificationResult:
    """Convenience function for simple verification.

    This is the main entry point for verification. It creates a
    VerificationRequest and processes it through the default verifier.

    Args:
        answer: The AI-generated answer to verify.
        question: Optional user question.
        evidence: Optional list of evidence.
        request_id: Optional request ID (auto-generated if not provided).
        tenant_id: Trusted tenant identifier, passed separately from the
            request payload (default: "default").
        project_id: Project ID within the tenant (default: "default-project").
        application_id: Application ID within the tenant (default: "default-app").
        **kwargs: Additional arguments passed to VerificationRequest.

    Returns:
        VerificationResult with verdict, scores, and claims.

    Example:
        ```python
        from core import verify
        from schemas import Evidence

        result = verify(
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
            evidence=[
                Evidence(
                    evidence_id="doc_001",
                    content="Company X was founded in 2018.",
                )
            ]
        )

        print(result.verdict)  # PASS
        print(result.scores["groundedness"].value)  # 1.0
        ```
    """
    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:12]}"

    request = VerificationRequest(
        request_id=request_id,
        project_id=project_id,
        application_id=application_id,
        question=question,
        answer=answer,
        evidence=evidence or [],
        **kwargs,
    )

    verifier = get_default_verifier()
    return verifier.verify(request, tenant_id=tenant_id)
