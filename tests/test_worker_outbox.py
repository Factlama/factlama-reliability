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
