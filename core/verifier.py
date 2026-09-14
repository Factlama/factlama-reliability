"""Main verification module."""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from core.claims import ClaimExtractor, EnhancedClaimExtractor
from core.evidence import EvidenceMapper, SimpleEvidenceMapper
from core.policy import PolicyEngine
from core.result import VerificationResultBuilder
from core.scoring import ScoringEngine, derive_calibration_class, determine_verdict
from judges.port import (
    CancellationToken,
    JudgeProvider,
    JudgeRequest,
    apply_citation_support_check,
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

# Per-claim judge dispatch budget. No ADR-012 budget enforcement exists yet;
# this only bounds a single synchronous call's deadline.
DEFAULT_JUDGE_TIMEOUT_SECONDS = 30.0

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

        try:
            policy = request.policy or Policy(id="default")

            claims = self._extract_claims(request.answer)

            claim_verifications, attempts, calibration_class, judge_citation_violations = (
                self._verify_claims(
                    claims,
                    request.evidence,
                )
            )

            instruction_violations = self._evaluate_instructions(
                request.answer, request.instructions
            )
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
                + judge_citation_violations
                + tool_violations
            )

            status, verdict, abstention_reason = self._determine_status(
                claims, claim_verifications, attempts
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

            result = (
                builder.with_status(status)
                .with_abstention_reason(abstention_reason)
                .with_verdict(verdict)
                .with_scores(scores)
                .with_claims(claim_verifications)
                .with_violations(violations + policy_violations)
                .with_policy_action(policy_action)
                .with_provenance(provenance)
                .with_metadata(
                    {
                        "provider": self.model_provider.name,
                        "claim_count": len(claims),
                        "evidence_count": len(request.evidence),
                        "instruction_count": len(request.instructions),
                        "citation_count": len(request.citations),
                        "tool_count": len(request.tool_executions),
                    }
                )
                .build()
            )

            return result

        except Exception:
            logger.exception(
                "Unexpected pipeline failure during verification of request %s", request.request_id
            )
            completed_at = datetime.now(timezone.utc)
            failure_attempt = Attempt(
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
                    qualification_status="UNQUALIFIED",
                ),
                outcome=AttemptOutcome.FAILED,
                error="CONFIGURATION",
                started_at=started_at,
                completed_at=completed_at,
            )
            provenance = Provenance(
                evaluator_id=EVALUATOR_ID,
                evaluator_version=EVALUATOR_VERSION,
                policy_version=None,
                mode=request.mode,
                routing_profile_version=f"{request.mode.value.lower()}-0.1",
                started_at=started_at,
                completed_at=completed_at,
                attempts=[failure_attempt],
            )
            return (
                builder.with_status(ResultStatus.FAILED)
                .with_abstention_reason(AbstentionReason.PROVIDER_FAILURE)
                .with_verdict(OverallVerdict.ABSTAIN)
                .with_scores(self.scoring_engine.calculate_scores([], calibration_class=None))
                .with_provenance(provenance)
                .with_metadata({"failure": "internal pipeline error"})
                .build()
            )

    def _determine_status(
        self,
        claims: list,
        claim_verifications: list[ClaimVerification],
        attempts: list[Attempt],
    ) -> tuple[ResultStatus, OverallVerdict, AbstentionReason | None]:
        """Determine pipeline status, factual verdict, and abstention reason.

        The factual verdict itself always comes from `determine_verdict()`
        (independent of policy); this only decides whether the pipeline
        counts as COMPLETED or ABSTAINED and, if abstained, why.
        """
        if not claims:
            return (
                ResultStatus.ABSTAINED,
                OverallVerdict.ABSTAIN,
                AbstentionReason.NO_CHECKABLE_CLAIMS,
            )

        if attempts and all(a.outcome == AttemptOutcome.FAILED for a in attempts):
            return ResultStatus.ABSTAINED, OverallVerdict.ABSTAIN, AbstentionReason.PROVIDER_FAILURE

        verdict = determine_verdict(claim_verifications)
        if verdict == OverallVerdict.ABSTAIN:
            applicable = [
                v for v in claim_verifications if v.verdict != ClaimVerdict.NOT_APPLICABLE
            ]
            if not applicable:
                return (
                    ResultStatus.ABSTAINED,
                    OverallVerdict.ABSTAIN,
                    AbstentionReason.NO_CHECKABLE_CLAIMS,
                )
            return (
                ResultStatus.ABSTAINED,
                OverallVerdict.ABSTAIN,
                AbstentionReason.INSUFFICIENT_EVIDENCE,
            )

        return ResultStatus.COMPLETED, verdict, None

    def _extract_claims(self, answer: str) -> list:
        """Extract claims from the answer."""
        return self.claim_extractor.extract(answer)

    def _verify_claims(
        self,
        claims: list,
        evidence: list[Evidence],
    ) -> tuple[list[ClaimVerification], list[Attempt], str | None, list[Violation]]:
        """Verify all claims against evidence via one bounded JudgeProvider call each.

        Returns:
            Tuple of (claim verifications, judge attempts, calibration class
            shared by every attempt -- or None when there were no claims to
            dispatch at all -- and violations raised by the judge-boundary
            citation checks below).
        """
        verifications: list[ClaimVerification] = []
        attempts: list[Attempt] = []
        citation_violations: list[Violation] = []
        calibration_class: str | None = None

        for claim in claims:
            cancellation = CancellationToken()
            deadline = time.monotonic() + DEFAULT_JUDGE_TIMEOUT_SECONDS
            judge_request = JudgeRequest(claim=claim, evidence=evidence)

            calibration_class = derive_calibration_class(
                evaluator_id=EVALUATOR_ID,
                evaluator_version=EVALUATOR_VERSION,
                provider_id=self.model_provider.name,
                pinned_model_id=self.model_provider.name,
                configuration_version=judge_request.configuration_version,
                qualification_status=QualificationStatus.UNQUALIFIED.value,
            )

            attempt_started = datetime.now(timezone.utc)
            judge_result = self.model_provider.evaluate(judge_request, deadline, cancellation)
            # Reject a SUPPORTED verdict citing no evidence ID, or an ID absent
            # from this request (a hallucinated citation), before trusting it.
            judge_result = validate_judge_result(judge_result, judge_request)
            # Then run the conservative, non-model overlap/support check on
            # whatever real evidence was cited (CONTRACTS.md).
            judge_result, citation_downgraded = apply_citation_support_check(
                judge_result, judge_request
            )
            attempt_completed = datetime.now(timezone.utc)

            attempt_id = f"attempt_{uuid.uuid4().hex[:12]}"

            if citation_downgraded:
                citation_violations.append(
                    Violation(
                        code="CITATION_MISMATCH",
                        severity=Severity.MEDIUM,
                        claim_ids=[claim.claim_id],
                        evidence_ids=judge_result.evidence_ids,
                        message=(
                            "Judge-cited evidence shares no content with the claim; "
                            "downgraded from SUPPORTED to INSUFFICIENT_EVIDENCE"
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
                        configuration_version=judge_request.configuration_version,
                        qualification_status=QualificationStatus.UNQUALIFIED,
                        calibration_class=calibration_class,
                        outcome=AttemptOutcome.COMPLETED,
                        started_at=attempt_started,
                        completed_at=attempt_completed,
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
                        configuration_version=judge_request.configuration_version,
                        qualification_status=QualificationStatus.UNQUALIFIED,
                        calibration_class=calibration_class,
                        outcome=AttemptOutcome.FAILED,
                        error=judge_result.error.code.value,
                        started_at=attempt_started,
                        completed_at=attempt_completed,
                    )
                )

        return verifications, attempts, calibration_class, citation_violations

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
