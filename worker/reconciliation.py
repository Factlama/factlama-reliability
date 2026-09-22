"""Cross-attempt reconciliation (G5, ADR-015): merges per-claim judgments
from one job's bounded retry/fallback attempts into one claim list,
producing `DISPUTED` for any claim where two or more *valid* attempts
disagree.

"Valid" excludes a claim whose dispatch itself failed
(`RationaleCode.PROVIDER_DISPATCH_FAILED`) in a given attempt -- a failed
dispatch contributed no real judgment to agree or disagree with. Never a
silent majority vote (CONTRACTS.md, ADR-015): with 3+ valid judgments
split 2-1, the minority is not discarded -- any disagreement at all
disputes the claim.
"""

from schemas.claims import ClaimVerdict, ClaimVerification, RationaleCode
from schemas.verification import Severity, Violation


def merge_claim_verifications(
    attempts_claim_verifications: list[list[ClaimVerification]],
) -> tuple[list[ClaimVerification], list[Violation]]:
    """`attempts_claim_verifications` is one `Verifier.dispatch_claims()`
    call's `claim_verifications` return value per bounded-retry attempt, in
    attempt order. Returns the merged per-claim list (same claim_ids, same
    order as first seen) and the `JUDGE_DISAGREEMENT` violations for any
    disputed claim.
    """
    if len(attempts_claim_verifications) <= 1:
        return (attempts_claim_verifications[0] if attempts_claim_verifications else []), []

    by_claim_id: dict[str, list[ClaimVerification]] = {}
    order: list[str] = []
    for attempt_cvs in attempts_claim_verifications:
        for cv in attempt_cvs:
            if cv.claim_id not in by_claim_id:
                order.append(cv.claim_id)
                by_claim_id[cv.claim_id] = []
            by_claim_id[cv.claim_id].append(cv)

    merged: list[ClaimVerification] = []
    violations: list[Violation] = []

    for claim_id in order:
        candidates = by_claim_id[claim_id]
        valid = [
            cv for cv in candidates if cv.rationale_code != RationaleCode.PROVIDER_DISPATCH_FAILED
        ]

        if not valid:
            # Every attempt's dispatch failed for this claim -- keep the
            # most recent failure marker; there is nothing to reconcile.
            merged.append(candidates[-1])
            continue

        if len(valid) == 1:
            merged.append(valid[0])
            continue

        distinct_verdicts = {cv.verdict for cv in valid}
        contributing = [judgment for cv in valid for judgment in cv.contributing_judgments]

        if len(distinct_verdicts) == 1:
            agreed = valid[0]
            merged.append(
                agreed.model_copy(
                    update={
                        "evidence_ids": sorted({eid for cv in valid for eid in cv.evidence_ids}),
                        "contributing_judgments": contributing,
                        "confidence": sum(cv.confidence for cv in valid) / len(valid),
                    }
                )
            )
            continue

        first = valid[0]
        merged.append(
            first.model_copy(
                update={
                    "verdict": ClaimVerdict.DISPUTED,
                    "evidence_ids": [],
                    "rationale_code": RationaleCode.JUDGE_DISAGREEMENT,
                    "rationale": (
                        f"{len(valid)} valid attempts disagreed on this claim's verdict "
                        f"({sorted(v.value for v in distinct_verdicts)})"
                    ),
                    "contributing_judgments": contributing,
                    "confidence": 0.0,
                }
            )
        )
        violations.append(
            Violation(
                code="JUDGE_DISAGREEMENT",
                severity=Severity.HIGH,
                claim_ids=[claim_id],
                message=(
                    f"Bounded retry/fallback attempts disagreed on claim {claim_id!r}'s "
                    "verdict; no majority vote is applied (ADR-015)."
                ),
            )
        )

    return merged, violations
