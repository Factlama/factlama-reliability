"""Tests for `worker.outbox.build_outbox_event` (CONTRACTS.md's
`ReliabilityEvent` projection)."""

from datetime import datetime, timezone

from schemas.claims import ClaimVerdict, ClaimVerification, RationaleCode
from schemas.verification import (
    AbstentionReason,
    Attempt,
    AttemptOutcome,
    OverallVerdict,
    Provenance,
    QualificationStatus,
    ResultStatus,
    VerificationMode,
    VerificationResult,
)
from worker.outbox import build_outbox_event


def _attempt(
    calibration_class: str,
    qualification_status: QualificationStatus = QualificationStatus.UNQUALIFIED,
    outcome: AttemptOutcome = AttemptOutcome.COMPLETED,
    error: str | None = None,
) -> Attempt:
    now = datetime.now(timezone.utc)
    return Attempt(
        attempt_id=f"attempt_{calibration_class}_{outcome.value}",
        provider_id="test",
        configuration_version="0.1",
        qualification_status=qualification_status,
        calibration_class=calibration_class,
        outcome=outcome,
        error=error,
        started_at=now,
        completed_at=now,
    )


def _result(
    attempts: list[Attempt], status: ResultStatus = ResultStatus.COMPLETED
) -> VerificationResult:
    now = datetime.now(timezone.utc)
    kwargs: dict = {}
    if status == ResultStatus.ABSTAINED:
        kwargs["abstention_reason"] = AbstentionReason.PROVIDER_FAILURE
        verdict = OverallVerdict.ABSTAIN
    else:
        verdict = OverallVerdict.PASS

    claims = (
        [
            ClaimVerification(
                claim_id="c1",
                verdict=ClaimVerdict.SUPPORTED,
                rationale_code=RationaleCode.DIRECT_SUPPORT,
                evidence_ids=["e1"],
            )
        ]
        if status == ResultStatus.COMPLETED
        else []
    )

    return VerificationResult(
        evaluation_id="eval-1",
        request_id="req-1",
        tenant_id="tenant-a",
        project_id="proj-a",
        application_id="app-a",
        status=status,
        verdict=verdict,
        claims=claims,
        provenance=Provenance(
            evaluator_id="test",
            evaluator_version="0.1",
            mode=VerificationMode.STANDARD,
            routing_profile_version="standard-0.1",
            started_at=now,
            completed_at=now,
            attempts=attempts,
        ),
        **kwargs,
    )


class TestBuildOutboxEvent:
    def test_single_completed_attempt_reports_its_own_calibration_class(self) -> None:
        result = _result([_attempt("class-a")])
        event = build_outbox_event(result)
        assert event.calibration_class == "class-a"
        assert event.calibration_classes == []

    def test_no_completed_attempts_reports_none_sentinel(self) -> None:
        result = _result(
            [_attempt("class-a", outcome=AttemptOutcome.FAILED, error="UNAVAILABLE")],
            status=ResultStatus.ABSTAINED,
        )
        event = build_outbox_event(result)
        assert event.calibration_class == "NONE"
        assert event.calibration_classes == []

    def test_two_distinct_completed_classes_report_mixed(self) -> None:
        result = _result([_attempt("class-a"), _attempt("class-b")])
        event = build_outbox_event(result)
        assert event.calibration_class == "MIXED"
        assert event.calibration_classes == ["class-a", "class-b"]

    def test_a_failed_attempts_class_does_not_count_toward_the_set(self) -> None:
        result = _result(
            [
                _attempt("class-a"),
                _attempt("class-b", outcome=AttemptOutcome.FAILED, error="TIMEOUT"),
            ]
        )
        event = build_outbox_event(result)
        assert event.calibration_class == "class-a"

    def test_mixed_qualification_status_across_completed_attempts(self) -> None:
        result = _result(
            [
                _attempt("class-a", qualification_status=QualificationStatus.QUALIFIED),
                _attempt("class-a", qualification_status=QualificationStatus.PROVISIONAL),
            ]
        )
        event = build_outbox_event(result)
        assert event.qualification_status == "MIXED"

    def test_metadata_only_no_raw_content(self) -> None:
        """CONTRACTS.md: the event carries references and metadata, never
        prompt/response/evidence bodies."""
        result = _result([_attempt("class-a")])
        event = build_outbox_event(result)
        assert not hasattr(event, "answer")
        assert not hasattr(event, "text")
        assert event.evaluation_id == "eval-1"
        assert event.tenant_id == "tenant-a"


def _claim(
    claim_id: str,
    verdict: ClaimVerdict,
    rationale_code: RationaleCode,
    *,
    text: str | None = None,
    rationale: str | None = None,
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        verdict=verdict,
        rationale_code=rationale_code,
        # SUPPORTED is the one verdict requiring a cited evidence ID.
        evidence_ids=["e1"] if verdict == ClaimVerdict.SUPPORTED else [],
        text=text,
        rationale=rationale,
    )


def _result_with_claims(claims: list[ClaimVerification]) -> VerificationResult:
    result = _result([_attempt("class-a")])
    return result.model_copy(update={"claims": claims})


class TestClaimSummaryProjection:
    """ADR-022: the non-content claim projection on `ReliabilityEvent`."""

    def test_projects_each_finding_status_and_rationale_code(self) -> None:
        event = build_outbox_event(
            _result_with_claims(
                [
                    _claim("c1", ClaimVerdict.SUPPORTED, RationaleCode.DIRECT_SUPPORT),
                    _claim("c2", ClaimVerdict.UNSUPPORTED, RationaleCode.NO_SUPPORT),
                    _claim("c3", ClaimVerdict.CONTRADICTED, RationaleCode.CONTRADICTION_DETECTED),
                    _claim(
                        "c4",
                        ClaimVerdict.INSUFFICIENT_EVIDENCE,
                        RationaleCode.AMBIGUOUS_EVIDENCE,
                    ),
                ]
            )
        )
        assert event.claim_summary is not None
        assert event.claim_summary.total_claim_count == 4
        assert {cid: f.status for cid, f in event.claim_summary.findings.items()} == {
            "c1": "SUPPORTED",
            "c2": "UNSUPPORTED",
            "c3": "CONTRADICTED",
            "c4": "INSUFFICIENT_EVIDENCE",
        }
        assert event.claim_summary.findings["c3"].rationale_code == "CONTRADICTION_DETECTED"

    def test_not_applicable_claims_are_counted_but_not_projected_as_findings(self) -> None:
        """ADR-022: `ClaimFindingStatus` is narrower than `ClaimVerdict`. A
        NOT_APPLICABLE claim is not a reliability *finding*, but dropping it
        from the count entirely would make the findings fail to sum to the
        claims actually evaluated -- so it stays in `total_claim_count`."""
        event = build_outbox_event(
            _result_with_claims(
                [
                    _claim("c1", ClaimVerdict.SUPPORTED, RationaleCode.DIRECT_SUPPORT),
                    _claim(
                        "c2",
                        ClaimVerdict.NOT_APPLICABLE,
                        RationaleCode.NOT_APPLICABLE_CLAIM_TYPE,
                    ),
                ]
            )
        )
        assert event.claim_summary is not None
        assert event.claim_summary.total_claim_count == 2
        assert list(event.claim_summary.findings) == ["c1"]
        # The derivable no-finding count, the whole reason total is carried.
        assert event.claim_summary.total_claim_count - len(event.claim_summary.findings) == 1

    def test_zero_claims_is_a_present_projection_not_an_absent_one(self) -> None:
        """ADR-022 rejects conflating the two: an evaluation that genuinely
        found no claims reports `total_claim_count == 0` with an empty
        findings map, which is NOT the same as sending no projection."""
        event = build_outbox_event(_result_with_claims([]))
        assert event.claim_summary is not None
        assert event.claim_summary.total_claim_count == 0
        assert event.claim_summary.findings == {}

    def test_claim_text_and_free_form_rationale_are_never_projected(self) -> None:
        """ADR-022's privacy boundary, at the producer. Even when the claim
        carries authorized text and a free-form judge rationale, neither
        reaches the projection -- only the bounded `rationale_code` does."""
        event = build_outbox_event(
            _result_with_claims(
                [
                    _claim(
                        "c1",
                        ClaimVerdict.CONTRADICTED,
                        RationaleCode.CONTRADICTION_DETECTED,
                        text="Acme was founded in 1999 by Jane Doe.",
                        rationale="The evidence says 2018, so this is wrong.",
                    )
                ]
            )
        )
        assert event.claim_summary is not None
        finding = event.claim_summary.findings["c1"]
        assert finding.status == "CONTRADICTED"
        assert finding.rationale_code == "CONTRADICTION_DETECTED"
        # The projected finding has no field that could carry either, and
        # the serialized wire form carries nothing beyond the two keys.
        wire = event.claim_summary.to_wire_json()
        assert wire["findings"] == {
            "c1": {"status": "CONTRADICTED", "rationale_code": "CONTRADICTION_DETECTED"}
        }
        serialized = str(wire)
        assert "Acme" not in serialized
        assert "evidence says" not in serialized
