"""Evidence mapping module."""

from abc import ABC, abstractmethod
from typing import Optional

from schemas.claims import Claim, ClaimVerdict, ClaimVerification, EvidenceReference
from schemas.evidence import Evidence


class EvidenceMapper(ABC):
    """Abstract base class for evidence mapping."""

    @abstractmethod
    def map_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> ClaimVerification:
        """Map evidence to a claim and produce verification.

        Args:
            claim: The claim to verify.
            evidence: Available evidence.

        Returns:
            ClaimVerification with verdict and evidence references.
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
    ) -> ClaimVerification:
        """Map evidence to a claim using simple text matching.

        Args:
            claim: The claim to verify.
            evidence: Available evidence.

        Returns:
            ClaimVerification with verdict and evidence references.
        """
        if not evidence:
            return ClaimVerification(
                claim_id=claim.id,
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                evidence=[],
                reason="No evidence available for verification",
            )

        # Find relevant evidence
        evidence_refs: list[EvidenceReference] = []
        best_support = 0.0
        best_verdict = ClaimVerdict.UNSUPPORTED
        best_reason: Optional[str] = None

        for ev in evidence:
            support, relevance = self._calculate_support(claim.text, ev.extracted_text)

            if relevance >= self.match_threshold:
                evidence_refs.append(
                    EvidenceReference(
                        evidence_id=ev.id,
                        support=support,
                        relevance=relevance,
                    )
                )

                if support > best_support:
                    best_support = support

        # Check for numerical contradictions
        numerical_contradiction = self._check_numerical_contradiction(claim.text, evidence)
        if numerical_contradiction:
            best_verdict = ClaimVerdict.CONTRADICTED
            best_reason = "Numerical values in claim contradict evidence"
        # Determine verdict based on evidence
        elif evidence_refs:
            if best_support >= self.contradiction_threshold:
                best_verdict = ClaimVerdict.SUPPORTED
                best_reason = "Claim is supported by evidence"
            elif best_support <= -self.contradiction_threshold:
                best_verdict = ClaimVerdict.CONTRADICTED
                best_reason = "Claim contradicts evidence"
            else:
                best_verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
                best_reason = "Evidence exists but is insufficient to determine support"
        else:
            best_verdict = ClaimVerdict.UNSUPPORTED
            best_reason = "No relevant evidence found"

        # Calculate confidence based on evidence quality
        confidence = self._calculate_confidence(evidence_refs, best_verdict)

        return ClaimVerification(
            claim_id=claim.id,
            verdict=best_verdict,
            confidence=confidence,
            evidence=evidence_refs,
            reason=best_reason,
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
            if has_negation_in_claim != has_negation_in_evidence:
                support = -relevance  # Contradiction
            else:
                support = relevance  # Support
        else:
            support = 0.0

        return support, relevance

    def _calculate_confidence(
        self,
        evidence_refs: list[EvidenceReference],
        verdict: ClaimVerdict,
    ) -> float:
        """Calculate confidence in the verdict.

        Args:
            evidence_refs: Evidence references.
            verdict: The determined verdict.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if verdict == ClaimVerdict.UNSUPPORTED:
            # Low confidence when no evidence found
            return 0.5

        if not evidence_refs:
            return 0.5

        # Base confidence on the best evidence relevance
        best_relevance = max(ref.relevance for ref in evidence_refs) if evidence_refs else 0.5
        best_support = max(abs(ref.support) for ref in evidence_refs) if evidence_refs else 0.5

        # Higher relevance and support = higher confidence
        confidence = (best_relevance + best_support) / 2

        return min(1.0, max(0.0, confidence))

    def _check_numerical_contradiction(self, claim: str, evidence: list[Evidence]) -> bool:
        """Check for numerical contradictions between claim and evidence.

        Args:
            claim: The claim text.
            evidence: List of evidence.

        Returns:
            True if numerical contradiction detected.
        """
        import re

        # Extract numbers from claim
        claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim)
        if not claim_numbers:
            return False

        for ev in evidence:
            # Extract numbers from evidence
            evidence_numbers = re.findall(r"\d+(?:\.\d+)?", ev.extracted_text)
            if not evidence_numbers:
                continue

            # Check if there's high text similarity but different numbers
            claim_words = set(claim.lower().split())
            ev_words = set(ev.extracted_text.lower().split())
            overlap = claim_words & ev_words
            similarity = len(overlap) / len(claim_words) if claim_words else 0.0

            # If texts are similar (> 60% word overlap) but numbers differ, it's a contradiction
            if similarity > 0.6 and claim_numbers != evidence_numbers:
                return True

        return False
