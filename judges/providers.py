"""Reference JudgeProvider implementations with no vendor SDK dependency.

`MockModelProvider` and `RuleBasedProvider` are deterministic, in-process
adapters that need nothing beyond the standard library and this repository's
own code (ADR-011's T0 tier). Vendor-coupled adapters (`EmbeddingProvider`,
`NLIProvider`) live in `judges.vendor_adapters` instead, so that importing
this module -- or `judges.port` -- never pulls in sentence-transformers,
transformers or torch.
"""

from judges.port import (
    CancellationToken,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
    bounded_check,
)
from schemas.claims import ClaimVerdict, RationaleCode
from schemas.instruction import Instruction
from schemas.policy import Policy


class MockModelProvider(JudgeProvider):
    """Mock provider for testing that returns predictable results.

    This provider is useful for testing the core verification pipeline
    without requiring an actual model. It uses deterministic logic to
    produce consistent results for the same inputs.
    """

    def __init__(
        self,
        default_verdict: str = "SUPPORTED",
        default_confidence: float = 0.95,
    ) -> None:
        """Initialize the mock provider.

        Args:
            default_verdict: Default claim verdict to return.
            default_confidence: Default confidence score.
        """
        self.default_verdict = default_verdict
        self.default_confidence = default_confidence

    @property
    def name(self) -> str:
        return "mock"

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using simple text matching."""
        error = bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        claim, evidence = request.claim, request.evidence

        if not evidence:
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                rationale_code=RationaleCode.NO_EVIDENCE_SUPPLIED,
                reason="No evidence provided",
            )

        # Check if claim text appears in any evidence
        claim_lower = claim.text.lower()
        supporting_ids: list[str] = []

        for ev in evidence:
            ev_text = (ev.content or "").lower()
            if claim_lower in ev_text or ev_text in claim_lower:
                supporting_ids.append(ev.evidence_id)

        if supporting_ids:
            return JudgeResult(
                verdict=ClaimVerdict.SUPPORTED,
                confidence=self.default_confidence,
                evidence_ids=supporting_ids,
                rationale_code=RationaleCode.DIRECT_SUPPORT,
                reason="Claim found in evidence",
            )
        else:
            return JudgeResult(
                verdict=ClaimVerdict.UNSUPPORTED,
                confidence=0.7,
                rationale_code=RationaleCode.NO_SUPPORT,
                reason="Claim not found in evidence",
            )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, str | None]:
        """Return perfect adherence for mock."""
        return 1.0, "Mock evaluation - assumed compliant"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Return no scope breach for mock."""
        return 0.0, []


class RuleBasedProvider(JudgeProvider):
    """Rule-based verification provider using heuristics and text matching.

    This provider implements verification using deterministic rules without
    any machine learning. It's suitable for:
    - Testing and validation
    - Simple verification scenarios
    - Baseline comparisons
    - Environments where ML models cannot be deployed
    """

    def __init__(
        self,
        support_threshold: float = 0.7,
        contradiction_threshold: float = 0.8,
    ) -> None:
        """Initialize the rule-based provider.

        Args:
            support_threshold: Threshold for considering evidence as supporting.
            contradiction_threshold: Threshold for detecting contradictions.
        """
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    @property
    def name(self) -> str:
        return "rule-based"

    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Verify claim using rule-based text analysis."""
        error = bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)

        from core.evidence import SimpleEvidenceMapper

        mapper = SimpleEvidenceMapper(
            match_threshold=self.support_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )
        mapping = mapper.map_evidence(request.claim, request.evidence)
        return JudgeResult(
            verdict=mapping.verdict,
            evidence_ids=mapping.evidence_ids,
            confidence=mapping.confidence,
            rationale_code=mapping.rationale_code,
            reason=mapping.reason,
        )

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, str | None]:
        """Evaluate instruction adherence using simple rules."""
        instruction_text = instruction.text.lower()

        # Check for format instructions
        if "json" in instruction_text and "return json" in instruction_text:
            # Simple check for JSON-like structure
            if answer.strip().startswith("{") and answer.strip().endswith("}"):
                return 1.0, "Answer appears to be JSON format"
            else:
                return 0.3, "Answer does not appear to be JSON format"

        # Check for grounding instructions
        if "only" in instruction_text and "context" in instruction_text:
            # This would need evidence to verify properly
            return 0.8, "Grounding instruction - cannot fully verify without evidence"

        # Default: assume compliance
        return 0.9, "Rule-based evaluation - assumed compliant"

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance using keyword matching."""
        import re

        if not policy.scope.enabled:
            return 0.0, []

        breach_domains: list[str] = []
        answer_lower = answer.lower()

        # Check for forbidden domains using word boundary matching
        for forbidden in policy.scope.forbidden:
            forbidden_lower = forbidden.lower()
            # Check for the forbidden term as a word or phrase
            if re.search(rf"\b{re.escape(forbidden_lower)}\b", answer_lower):
                breach_domains.append(forbidden)

        # If forbidden domains found, calculate breach score
        if breach_domains:
            breach_score = min(1.0, len(breach_domains) * 0.3)
            return breach_score, breach_domains

        return 0.0, []
