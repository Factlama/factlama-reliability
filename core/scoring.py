"""Scoring engine for verification metrics.

Implements Reliability's v0.1 scoring specification (scoring.md): groundedness,
hallucination_risk and contradiction_risk are claim-ratio ("count(...) /
count(applicable claims)") measures with no invented partial credit. Every
other dimension is UNAVAILABLE until an evaluator and method version
explicitly supply an accepted formula for it -- there is no weighted
composite "reliability" score in CONTRACTS.md; the aggregate verdict is
computed independently by `determine_verdict()` from claim verdicts alone.
"""

import hashlib
from collections.abc import Callable

from schemas.claims import ClaimVerdict, ClaimVerification
from schemas.verification import (
    AbstentionReason,
    Attempt,
    AttemptOutcome,
    DisputeReason,
    OverallVerdict,
    ResultStatus,
    ScoreStatus,
    ScoreValue,
)

NONE_CALIBRATION_CLASS = "NONE"


def derive_calibration_class(
    evaluator_id: str,
    evaluator_version: str,
    provider_id: str,
    pinned_model_id: str,
    configuration_version: str,
    qualification_status: str,
    derivation_version: str = "cc-0.1",
) -> str:
    """Derive an opaque, stable calibration class per ADR-010.

    Deterministic: identical inputs always produce the same class, and
    changing any single input (including qualification_status) changes it.
    No registry exists yet -- this only computes the identifier, it does not
    look anything up.
    """
    basis = (
        f"{derivation_version}:{evaluator_id}:{evaluator_version}:{provider_id}:"
        f"{pinned_model_id}:{configuration_version}:{qualification_status}"
    )
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


def determine_verdict(claim_verifications: list[ClaimVerification]) -> OverallVerdict:
    """Determine the factual aggregate verdict from claim verdicts alone.

    Implements scoring.md's precedence exactly. This is independent of any
    policy action (CONTRACTS.md: "Factual verdict is computed independently
    of policy action").

    G5/ADR-015: a `DISPUTED` claim takes precedence over every other signal
    (checked before CONTRADICTED/SUPPORTED) -- CONTRACTS.md: "A disputed
    claim prevents an aggregate PASS/FAIL until an explicit, versioned
    reconciliation policy exists; MVP does not perform majority voting." A
    single `Verifier.verify()` call can never itself produce a disputed
    claim (one judge attempt per claim); this precedence only ever fires
    once `worker.reconciliation` has merged two or more attempts' judgments
    for the same claim and found disagreement.
    """
    applicable = [v for v in claim_verifications if v.verdict != ClaimVerdict.NOT_APPLICABLE]

    if not applicable:
        return OverallVerdict.ABSTAIN
    if any(v.verdict == ClaimVerdict.DISPUTED for v in applicable):
        return OverallVerdict.DISPUTED
    if all(v.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE for v in applicable):
        return OverallVerdict.ABSTAIN
    if any(v.verdict == ClaimVerdict.CONTRADICTED for v in applicable):
        return OverallVerdict.FAIL
    if all(v.verdict == ClaimVerdict.SUPPORTED for v in applicable):
        return OverallVerdict.PASS
    return OverallVerdict.PARTIAL


def determine_status(
    claims: list,
    claim_verifications: list[ClaimVerification],
    attempts: list[Attempt],
    usage_budget_exceeded: bool = False,
) -> tuple[ResultStatus, OverallVerdict, AbstentionReason | None, DisputeReason | None]:
    """Determine pipeline status, factual verdict, and abstention/dispute
    reason for one `VerificationResult`.

    Shared by `Verifier.verify()`'s single-attempt path and
    `worker.reconciliation`'s cross-attempt merge -- moved here (from a
    `Verifier` method that never referenced `self`) so both can produce the
    exact same status/verdict/reason precedence without duplicating it.
    """
    if usage_budget_exceeded:
        # ADR-012: exceeding the per-request usage/cost ceiling abstains
        # BUDGET_EXHAUSTED, the same as the pre-dispatch claim/evidence
        # caps -- even though, unlike those, one or more real judge
        # attempts already happened (preserved in provenance).
        return (
            ResultStatus.ABSTAINED,
            OverallVerdict.ABSTAIN,
            AbstentionReason.BUDGET_EXHAUSTED,
            None,
        )

    if not claims:
        return (
            ResultStatus.ABSTAINED,
            OverallVerdict.ABSTAIN,
            AbstentionReason.NO_CHECKABLE_CLAIMS,
            None,
        )

    if attempts and all(a.outcome == AttemptOutcome.FAILED for a in attempts):
        return (
            ResultStatus.ABSTAINED,
            OverallVerdict.ABSTAIN,
            AbstentionReason.PROVIDER_FAILURE,
            None,
        )

    verdict = determine_verdict(claim_verifications)
    if verdict == OverallVerdict.DISPUTED:
        return (
            ResultStatus.DISPUTED,
            OverallVerdict.DISPUTED,
            None,
            DisputeReason.JUDGE_DISAGREEMENT,
        )
    if verdict == OverallVerdict.ABSTAIN:
        applicable = [v for v in claim_verifications if v.verdict != ClaimVerdict.NOT_APPLICABLE]
        if not applicable:
            return (
                ResultStatus.ABSTAINED,
                OverallVerdict.ABSTAIN,
                AbstentionReason.NO_CHECKABLE_CLAIMS,
                None,
            )
        return (
            ResultStatus.ABSTAINED,
            OverallVerdict.ABSTAIN,
            AbstentionReason.INSUFFICIENT_EVIDENCE,
            None,
        )

    return ResultStatus.COMPLETED, verdict, None, None


class ScoringEngine:
    """Engine for calculating verification score dimensions."""

    VERSION = "0.1"

    def __init__(self, scoring_version: str = VERSION) -> None:
        """Initialize the scoring engine.

        Args:
            scoring_version: Version of scoring algorithm to use.
        """
        self.scoring_version = scoring_version

    def calculate_scores(
        self,
        claim_verifications: list[ClaimVerification],
        calibration_class: str | None = None,
        citation_support_score: float | None = None,
        tool_correctness_score: float | None = None,
    ) -> dict[str, ScoreValue]:
        """Calculate all named score dimensions.

        Args:
            claim_verifications: Verified claims with verdicts.
            calibration_class: The calibration class of the judge attempt(s)
                behind these claim verdicts, or None if no judge attempt
                occurred (e.g. zero claims extracted).
            citation_support_score: Pre-calculated fraction of valid
                citations (structural, deterministic), or None if no
                citations were supplied.
            tool_correctness_score: Pre-calculated fraction of successful
                tool executions (structural, deterministic), or None if no
                tool executions were supplied.

        Returns:
            Dict of dimension name to ScoreValue.
        """
        applicable = [v for v in claim_verifications if v.verdict != ClaimVerdict.NOT_APPLICABLE]
        claim_class: str = (
            calibration_class
            if applicable and calibration_class is not None
            else NONE_CALIBRATION_CLASS
        )

        scores: dict[str, ScoreValue] = {
            "groundedness": self._claim_ratio_score(
                applicable,
                lambda v: v.verdict == ClaimVerdict.SUPPORTED,
                "groundedness-0.1",
                claim_class,
            ),
            "hallucination_risk": self._claim_ratio_score(
                applicable,
                lambda v: v.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED),
                "hallucination-risk-0.1",
                claim_class,
            ),
            "contradiction_risk": self._claim_ratio_score(
                applicable,
                lambda v: v.verdict == ClaimVerdict.CONTRADICTED,
                "contradiction-risk-0.1",
                claim_class,
            ),
            "citation_support": self._structural_score(
                citation_support_score, "citation-support-structural-0.1"
            ),
            "tool_correctness": self._structural_score(
                tool_correctness_score, "tool-correctness-structural-0.1"
            ),
            "scope_breach": ScoreValue(status=ScoreStatus.UNAVAILABLE),
            "instruction_adherence": ScoreValue(status=ScoreStatus.UNAVAILABLE),
            "confidence_alignment": ScoreValue(status=ScoreStatus.UNAVAILABLE),
        }
        return scores

    def _claim_ratio_score(
        self,
        applicable: list[ClaimVerification],
        predicate: Callable[[ClaimVerification], bool],
        method_version: str,
        calibration_class: str,
    ) -> ScoreValue:
        """A count(predicate)/count(applicable) ratio score, or NOT_APPLICABLE when empty."""
        if not applicable:
            return ScoreValue(
                status=ScoreStatus.NOT_APPLICABLE,
                method_version=method_version,
                calibration_class=calibration_class,
            )
        value = sum(1 for v in applicable if predicate(v)) / len(applicable)
        return ScoreValue(
            value=round(value, 4),
            status=ScoreStatus.MEASURED,
            method_version=method_version,
            calibration_class=calibration_class,
        )

    def _structural_score(self, value: float | None, method_version: str) -> ScoreValue:
        """A deterministic structural score (no judge involved), or NOT_APPLICABLE when unset."""
        if value is None:
            return ScoreValue(status=ScoreStatus.NOT_APPLICABLE, method_version=method_version)
        return ScoreValue(
            value=round(value, 4),
            status=ScoreStatus.MEASURED,
            method_version=method_version,
            calibration_class=NONE_CALIBRATION_CLASS,
        )
