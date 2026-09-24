"""Unit tests for `core.governance.apply_capture_mode` (G5/REL-11,
ADR-008). Pure-function tests -- no storage/worker involved; the end-to-end
"a worker actually persists the governed copy" behavior is covered by
`tests/conformance/test_worker_runner.py`.
"""

from datetime import datetime, timezone

from core.governance import apply_capture_mode
from schemas.claims import ClaimVerdict, ClaimVerification, RationaleCode, TextStatus
from schemas.policy import CaptureMode
from schemas.verification import (
    OverallVerdict,
    Provenance,
    ResultStatus,
    Severity,
    VerificationMode,
    VerificationResult,
    Violation,
)


def _claim(claim_id: str = "c1") -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        verdict=ClaimVerdict.SUPPORTED,
        evidence_ids=["e1"],
        rationale_code=RationaleCode.DIRECT_SUPPORT,
        rationale="Paris is directly stated as the capital in the cited evidence.",
        text_status=TextStatus.AVAILABLE,
        text="Paris is the capital of France.",
    )


def _violation() -> Violation:
    return Violation(
        code="CITATION_MISMATCH",
        severity=Severity.MEDIUM,
        claim_ids=["c1"],
        evidence_ids=["e1"],
        message="The cited evidence only partially supports this claim.",
    )


def _result(claims: list[ClaimVerification], violations: list[Violation]) -> VerificationResult:
    now = datetime.now(timezone.utc)
    return VerificationResult(
        evaluation_id="eval-1",
        request_id="req-1",
        tenant_id="tenant-a",
        project_id="proj-a",
        application_id="app-a",
        status=ResultStatus.COMPLETED,
        verdict=OverallVerdict.PASS,
        claims=claims,
        violations=violations,
        provenance=Provenance(
            evaluator_id="factlama-reliability",
            evaluator_version="0.0.0",
            mode=VerificationMode.STANDARD,
            routing_profile_version="standard-0.1",
            started_at=now,
            completed_at=now,
        ),
    )


def test_full_mode_is_a_pure_passthrough() -> None:
    result = _result([_claim()], [_violation()])
    governed = apply_capture_mode(result, CaptureMode.FULL)
    assert governed is result  # not merely equal -- no copy made at all


def test_metadata_only_strips_text_and_rationale_and_sets_not_stored() -> None:
    result = _result([_claim()], [_violation()])
    governed = apply_capture_mode(result, CaptureMode.METADATA_ONLY)

    claim = governed.claims[0]
    assert claim.text is None
    assert claim.rationale is None
    assert claim.text_status == TextStatus.NOT_STORED
    # Identity/verdict fields are untouched -- only content is governed.
    assert claim.claim_id == "c1"
    assert claim.verdict == ClaimVerdict.SUPPORTED
    assert claim.evidence_ids == ["e1"]

    assert governed.violations[0].message is None
    assert governed.violations[0].code == "CITATION_MISMATCH"


def test_none_mode_behaves_identically_to_metadata_only() -> None:
    """ADR-008 treats NONE and METADATA_ONLY as the same "never write raw
    content" tier for this pass -- the finer distinction is G8/G10's."""
    result = _result([_claim()], [_violation()])
    assert apply_capture_mode(result, CaptureMode.NONE) == apply_capture_mode(
        result, CaptureMode.METADATA_ONLY
    )


def test_redacted_mode_replaces_text_and_marks_redacted_without_persisting_original() -> None:
    result = _result([_claim()], [_violation()])
    governed = apply_capture_mode(result, CaptureMode.REDACTED)

    claim = governed.claims[0]
    assert claim.text_status == TextStatus.REDACTED
    assert claim.text is not None
    assert "Paris" not in claim.text
    assert claim.rationale is not None
    assert "Paris" not in claim.rationale

    assert governed.violations[0].message is not None
    assert "partially supports" not in governed.violations[0].message


def test_claim_with_no_text_is_left_alone_under_every_mode() -> None:
    """A claim whose text_status is already NOT_STORED/REDACTED (e.g. a
    result read back a second time) must not be re-processed."""
    already_governed = _claim().model_copy(
        update={"text": None, "rationale": None, "text_status": TextStatus.NOT_STORED}
    )
    result = _result([already_governed], [])
    for mode in (CaptureMode.NONE, CaptureMode.METADATA_ONLY, CaptureMode.REDACTED):
        governed = apply_capture_mode(result, mode)
        assert governed.claims[0].text_status == TextStatus.NOT_STORED
        assert governed.claims[0].text is None


def test_violation_with_no_message_is_left_alone_under_every_mode() -> None:
    bare_violation = Violation(
        code="HALLUCINATION", severity=Severity.HIGH, claim_ids=["c1"], evidence_ids=[]
    )
    result = _result([], [bare_violation])
    for mode in (CaptureMode.NONE, CaptureMode.METADATA_ONLY, CaptureMode.REDACTED):
        governed = apply_capture_mode(result, mode)
        assert governed.violations[0].message is None


def test_default_capture_policy_mode_is_metadata_only() -> None:
    from schemas.policy import CapturePolicy

    assert CapturePolicy().mode == CaptureMode.METADATA_ONLY
