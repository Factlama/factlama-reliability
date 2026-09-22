"""Tests for `worker.reconciliation.merge_claim_verifications` (G5, ADR-015)."""

from schemas.claims import ClaimVerdict, ClaimVerification, ContributingJudgment, RationaleCode
from worker.reconciliation import merge_claim_verifications


def _cv(
    claim_id: str,
    verdict: ClaimVerdict,
    attempt_id: str,
    evidence_ids: list[str] | None = None,
    rationale_code: RationaleCode = RationaleCode.DIRECT_SUPPORT,
) -> ClaimVerification:
    evidence_ids = evidence_ids or []
    return ClaimVerification(
        claim_id=claim_id,
        verdict=verdict,
        evidence_ids=evidence_ids,
        rationale_code=rationale_code,
        contributing_judgments=[
            ContributingJudgment(
                attempt_id=attempt_id,
                verdict=verdict,
                evidence_ids=evidence_ids,
                rationale_code=rationale_code,
            )
        ],
    )


class TestMergeClaimVerifications:
    def test_empty_input_returns_empty(self) -> None:
        merged, violations = merge_claim_verifications([])
        assert merged == []
        assert violations == []

    def test_single_attempt_is_passed_through_unchanged(self) -> None:
        claims = [_cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"])]
        merged, violations = merge_claim_verifications([claims])
        assert merged == claims
        assert violations == []

    def test_agreeing_attempts_keep_the_verdict_and_combine_judgments(self) -> None:
        attempt1 = [_cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"])]
        attempt2 = [_cv("c1", ClaimVerdict.SUPPORTED, "a2", ["e2"])]

        merged, violations = merge_claim_verifications([attempt1, attempt2])

        assert violations == []
        assert len(merged) == 1
        assert merged[0].verdict == ClaimVerdict.SUPPORTED
        assert merged[0].evidence_ids == ["e1", "e2"]
        assert {j.attempt_id for j in merged[0].contributing_judgments} == {"a1", "a2"}

    def test_disagreeing_attempts_produce_disputed_with_judge_disagreement_violation(
        self,
    ) -> None:
        attempt1 = [_cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"])]
        attempt2 = [_cv("c1", ClaimVerdict.CONTRADICTED, "a2")]

        merged, violations = merge_claim_verifications([attempt1, attempt2])

        assert len(merged) == 1
        assert merged[0].verdict == ClaimVerdict.DISPUTED
        assert merged[0].rationale_code == RationaleCode.JUDGE_DISAGREEMENT
        assert merged[0].evidence_ids == []
        assert {j.attempt_id for j in merged[0].contributing_judgments} == {"a1", "a2"}

        assert len(violations) == 1
        assert violations[0].code == "JUDGE_DISAGREEMENT"
        assert violations[0].claim_ids == ["c1"]

    def test_a_failed_dispatch_does_not_count_as_a_valid_judgment(self) -> None:
        """A PROVIDER_DISPATCH_FAILED entry contributed no real judgment --
        it must not trigger a false disagreement against a real one."""
        failed = _cv(
            "c1",
            ClaimVerdict.INSUFFICIENT_EVIDENCE,
            "a1",
            rationale_code=RationaleCode.PROVIDER_DISPATCH_FAILED,
        )
        succeeded = [_cv("c1", ClaimVerdict.SUPPORTED, "a2", ["e1"])]

        merged, violations = merge_claim_verifications([[failed], succeeded])

        assert violations == []
        assert merged[0].verdict == ClaimVerdict.SUPPORTED

    def test_all_attempts_failed_for_a_claim_keeps_the_latest_failure_marker(self) -> None:
        failed1 = _cv(
            "c1",
            ClaimVerdict.INSUFFICIENT_EVIDENCE,
            "a1",
            rationale_code=RationaleCode.PROVIDER_DISPATCH_FAILED,
        )
        failed2 = _cv(
            "c1",
            ClaimVerdict.INSUFFICIENT_EVIDENCE,
            "a2",
            rationale_code=RationaleCode.PROVIDER_DISPATCH_FAILED,
        )

        merged, violations = merge_claim_verifications([[failed1], [failed2]])

        assert violations == []
        assert len(merged) == 1
        assert merged[0].contributing_judgments[0].attempt_id == "a2"

    def test_claims_present_in_only_one_attempt_are_kept(self) -> None:
        attempt1 = [
            _cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"]),
            _cv("c2", ClaimVerdict.CONTRADICTED, "a1"),
        ]
        attempt2 = [_cv("c1", ClaimVerdict.SUPPORTED, "a2", ["e1"])]

        merged, violations = merge_claim_verifications([attempt1, attempt2])

        merged_by_id = {cv.claim_id: cv for cv in merged}
        assert violations == []
        assert set(merged_by_id) == {"c1", "c2"}
        assert merged_by_id["c2"].verdict == ClaimVerdict.CONTRADICTED

    def test_claim_order_matches_first_appearance(self) -> None:
        attempt1 = [
            _cv("c2", ClaimVerdict.SUPPORTED, "a1", ["e1"]),
            _cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"]),
        ]
        merged, _violations = merge_claim_verifications([attempt1])
        assert [cv.claim_id for cv in merged] == ["c2", "c1"]

    def test_three_way_split_disputes_even_without_a_true_majority(self) -> None:
        attempt1 = [_cv("c1", ClaimVerdict.SUPPORTED, "a1", ["e1"])]
        attempt2 = [_cv("c1", ClaimVerdict.CONTRADICTED, "a2")]
        attempt3 = [_cv("c1", ClaimVerdict.UNSUPPORTED, "a3")]

        merged, violations = merge_claim_verifications([attempt1, attempt2, attempt3])

        assert merged[0].verdict == ClaimVerdict.DISPUTED
        assert len(violations) == 1
