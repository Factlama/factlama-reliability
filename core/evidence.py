"""Evidence mapping module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from schemas.claims import Claim, ClaimVerdict, RationaleCode
from schemas.evidence import Evidence
from schemas.tenancy import TenantContext


@dataclass(frozen=True)
class MappingResult:
    """An `EvidenceMapper`'s judgment for one claim: internal to a JudgeProvider,
    not a wire type -- a provider translates this into a `JudgeResult`."""

    verdict: ClaimVerdict
    evidence_ids: list[str]
    confidence: float
    rationale_code: RationaleCode
    reason: str | None


class EvidenceMapper(ABC):
    """Abstract base class for evidence mapping."""

    @abstractmethod
    def map_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> MappingResult:
        """Map evidence to a claim and produce a verdict.

        Args:
            claim: The claim to verify.
            evidence: Available evidence.

        Returns:
            MappingResult with verdict and supporting evidence IDs.
        """
        ...


class EvidenceRetriever(ABC):
    """Optional retrieval port (evidence-engine.md): "Retrieval is an
    optional EvidenceRetriever port, called only when policy and tenant
    authorization explicitly allow it. No vector database is an MVP
    dependency."

    MVP evidence comes from the request; this exists so a future retrieval
    backend has a real interface to implement rather than a bespoke one
    invented per integration. There is deliberately no concrete
    implementation and no caller wiring it in yet -- there is no policy
    field or tenant-authorization check to gate it on (that gate would be
    new, unreviewed scope, not this interface). This mirrors
    `core.repository.TenantAwareRepository`: a typed contract a future
    implementation must satisfy, not a claim that retrieval itself works
    today.
    """

    @abstractmethod
    def retrieve(
        self, tenant_context: TenantContext, claim: Claim, max_results: int
    ) -> list[Evidence]:
        """Return candidate evidence for a claim, or an empty list.

        `tenant_context` is the authenticated identity CONTRACTS.md's
        `EvidenceRetriever.retrieve(tenant_context, query, limits) ->
        Evidence[]` requires: retrieval is never tenant-implicit, the same
        rule `core.repository.TenantAwareRepository` already enforces for
        storage. An implementation must scope its query -- and any
        authorization check on what it returns -- to this context, never to
        an ambient/global one; this is also the "policy and tenant
        authorization explicitly allow it" gate evidence-engine.md requires
        before retrieval runs at all.

        Implementations must resolve any reference through an authorized
        source before returning it; an unresolved reference is an
        unavailable-evidence condition (evidence-engine.md), never a raised
        exception. Candidate similarity is routing only, never a support
        verdict -- the judge still decides support/contradiction.

        Args:
            tenant_context: Authenticated tenant identity to scope retrieval
                and authorization to.
            claim: The claim to find candidate evidence for.
            max_results: Upper bound on returned candidates.

        Returns:
            Candidate evidence, most relevant first. May be empty.
        """
        ...


class SimpleEvidenceMapper(EvidenceMapper):
    """Simple evidence mapper using text matching.

    This is a baseline implementation. Production implementations should use
    semantic similarity, NLI models, or the FactLama SLM.
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        contradiction_threshold: float = 0.7,
    ) -> None:
        """Initialize the evidence mapper.

        Args:
            match_threshold: Threshold for considering evidence as supporting.
            contradiction_threshold: Threshold for detecting contradictions.
        """
        self.match_threshold = match_threshold
        self.contradiction_threshold = contradiction_threshold

    def map_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> MappingResult:
        """Map evidence to a claim using simple text matching.

        Args:
            claim: The claim to verify.
            evidence: Available evidence.

        Returns:
            MappingResult with verdict and supporting evidence IDs.
        """
        if not evidence:
            return MappingResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                evidence_ids=[],
                confidence=0.5,
                rationale_code=RationaleCode.NO_EVIDENCE_SUPPLIED,
                reason="No evidence available for verification",
            )

        # Find relevant evidence: (evidence_id, support, relevance) per match.
        matches: list[tuple[str, float, float]] = []
        best_support = 0.0

        for ev in evidence:
            support, relevance = self._calculate_support(claim.text, ev.content or "")

            if relevance >= self.match_threshold:
                matches.append((ev.evidence_id, support, relevance))
                best_support = max(best_support, support)

        # Check for numerical contradictions
        contradicting_id = self._check_numerical_contradiction(claim.text, evidence)
        if contradicting_id is not None:
            verdict = ClaimVerdict.CONTRADICTED
            evidence_ids = [contradicting_id]
            rationale_code = RationaleCode.NUMERICAL_CONTRADICTION
            reason = "Numerical values in claim contradict evidence"
        # Determine verdict based on evidence
        elif matches:
            if best_support >= self.contradiction_threshold:
                verdict = ClaimVerdict.SUPPORTED
                evidence_ids = [eid for eid, support, _ in matches if support >= 0]
                rationale_code = RationaleCode.PARAPHRASED_SUPPORT
                reason = "Claim is supported by evidence"
            elif best_support <= -self.contradiction_threshold:
                verdict = ClaimVerdict.CONTRADICTED
                evidence_ids = [eid for eid, support, _ in matches if support < 0]
                rationale_code = RationaleCode.CONTRADICTION_DETECTED
                reason = "Claim contradicts evidence"
            else:
                verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
                evidence_ids = []
                rationale_code = RationaleCode.AMBIGUOUS_EVIDENCE
                reason = "Evidence exists but is insufficient to determine support"
        else:
            verdict = ClaimVerdict.UNSUPPORTED
            evidence_ids = []
            rationale_code = RationaleCode.NO_SUPPORT
            reason = "No relevant evidence found"

        confidence = self._calculate_confidence(matches, verdict)

        return MappingResult(
            verdict=verdict,
            evidence_ids=evidence_ids,
            confidence=confidence,
            rationale_code=rationale_code,
            reason=reason,
        )

    def _calculate_support(self, claim: str, evidence_text: str) -> tuple[float, float]:
        """Calculate support and relevance scores.

        Args:
            claim: The claim text.
            evidence_text: The evidence text.

        Returns:
            Tuple of (support, relevance) scores.
        """
        # Simple word overlap for relevance
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence_text.lower().split())

        overlap = claim_words & evidence_words
        relevance = len(overlap) / len(claim_words) if claim_words else 0.0

        # Check for negation (simple heuristic for contradiction)
        negation_words = {"not", "no", "never", "neither", "nobody", "nothing", "none"}
        has_negation_in_claim = bool(claim_words & negation_words)
        has_negation_in_evidence = bool(evidence_words & negation_words)

        # If both have same negation status and high overlap, likely supported
        # If different negation status and high overlap, likely contradicted
        if relevance >= self.match_threshold:
            # Negative support = contradiction; positive = agreement.
            support = -relevance if has_negation_in_claim != has_negation_in_evidence else relevance
        else:
            support = 0.0

        return support, relevance

    def _calculate_confidence(
        self,
        matches: list[tuple[str, float, float]],
        verdict: ClaimVerdict,
    ) -> float:
        """Calculate confidence in the verdict.

        Args:
            matches: (evidence_id, support, relevance) tuples.
            verdict: The determined verdict.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if verdict == ClaimVerdict.UNSUPPORTED:
            # Low confidence when no evidence found
            return 0.5

        if not matches:
            return 0.5

        # Base confidence on the best evidence relevance
        best_relevance = max(relevance for _, _, relevance in matches)
        best_support = max(abs(support) for _, support, _ in matches)

        # Higher relevance and support = higher confidence
        confidence = (best_relevance + best_support) / 2

        return min(1.0, max(0.0, confidence))

    def _check_numerical_contradiction(self, claim: str, evidence: list[Evidence]) -> str | None:
        """Check for numerical contradictions between claim and evidence.

        Args:
            claim: The claim text.
            evidence: List of evidence.

        Returns:
            The ID of the first evidence found to numerically contradict the
            claim, or None if none do.
        """
        import re

        # Extract numbers from claim
        claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim)
        if not claim_numbers:
            return None

        for ev in evidence:
            ev_text = ev.content or ""
            # Extract numbers from evidence
            evidence_numbers = re.findall(r"\d+(?:\.\d+)?", ev_text)
            if not evidence_numbers:
                continue

            # Check if there's high text similarity but different numbers
            claim_words = set(claim.lower().split())
            ev_words = set(ev_text.lower().split())
            overlap = claim_words & ev_words
            similarity = len(overlap) / len(claim_words) if claim_words else 0.0

            # If texts are similar (> 60% word overlap) but numbers differ, it's a contradiction
            if similarity > 0.6 and claim_numbers != evidence_numbers:
                return ev.evidence_id

        return None
