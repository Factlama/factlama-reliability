"""Builds the metadata-only `ReliabilityEvent` projection
(`storage.models.OutboxEvent`) delivered from Reliability's outbox to
Observability's ingress (G6, not yet built) from a committed
`VerificationResult`. Never carries prompt/response/evidence bodies
(CONTRACTS.md) -- only references, scores, and usage.
"""

import uuid
from datetime import datetime, timezone

from schemas.verification import AttemptOutcome, VerificationResult
from storage.models import OutboxEvent

_NONE_CALIBRATION_CLASS = "NONE"
_MIXED = "MIXED"


def _calibration_fields(result: VerificationResult) -> tuple[str, list[str]]:
    """CONTRACTS.md: "When multiple valid attempts from different classes
    contribute, event calibration_class=MIXED, calibration_classes[] lists
    each class." Only `COMPLETED` attempts count as a real contributing
    judgment -- a `FAILED` dispatch attempt's calibration_class describes
    the provider that failed, not a judgment that contributed to this
    result."""
    classes = sorted(
        {
            a.calibration_class
            for a in result.provenance.attempts
            if a.outcome == AttemptOutcome.COMPLETED
        }
    )
    if not classes:
        return _NONE_CALIBRATION_CLASS, []
    if len(classes) == 1:
        return classes[0], []
    return _MIXED, classes


def _qualification_status(result: VerificationResult) -> str:
    """Same MIXED convention as calibration_class
    (`common.schema.json`'s `QualificationStatusOrMixed`)."""
    statuses = sorted(
        {
            a.qualification_status.value
            for a in result.provenance.attempts
            if a.outcome == AttemptOutcome.COMPLETED
        }
    )
    if not statuses:
        return "UNQUALIFIED"
    if len(statuses) == 1:
        return statuses[0]
    return _MIXED


def build_outbox_event(result: VerificationResult) -> OutboxEvent:
    calibration_class, calibration_classes = _calibration_fields(result)
    return OutboxEvent(
        event_id=f"event_{uuid.uuid4().hex[:16]}",
        tenant_id=result.tenant_id,
        evaluation_id=result.evaluation_id,
        occurred_at=datetime.now(timezone.utc),
        status=result.status.value,
        verdict=result.verdict.value,
        scores=result.scores,
        evaluator_version=result.provenance.evaluator_version,
        calibration_class=calibration_class,
        qualification_status=_qualification_status(result),
        usage_summary=result.usage_summary,
        project_id=result.project_id,
        application_id=result.application_id,
        interaction_id=result.interaction_id,
        trace_id=result.trace_id,
        span_id=result.span_id,
        calibration_classes=calibration_classes,
    )
