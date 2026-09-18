"""Evaluator qualification registry (G4, ADR-010/014).

This is G4's own minimal scope, not G10's full audited/persisted registry
(see `evaluator-registry.md`): a typed state machine and a default-judge
eligibility check, with no storage and no wiring into `Verifier`'s dispatch
path yet -- there is no concept of "a tenant's configured default provider"
anywhere in this codebase today, and building that selection mechanism is
G5/G10 territory (it needs persistence). This module exists so the *rule*
("qualification plus tenant approval before default selection") is real and
tested now, rather than invented later against production data.

Passing this repo's own `pytest` suite never sets `qualification_status` to
anything but `UNQUALIFIED` (see `core.scoring.derive_calibration_class`
callers) -- passing implementation tests establishes that the code behaves
as its own tests expect, never that its judgments agree with a held-out
label set. Only a real agreement report (`evaluator-agreement-harness.md`)
can advance a `QualificationRecord` past `CONFORMANCE_PASSED`.
"""

from dataclasses import dataclass, field, replace
from enum import Enum

from core.compliance import is_provider_compliant


class QualificationState(str, Enum):
    """ADR-014's lifecycle: REGISTERED -> CONFORMANCE_PASSED ->
    AGREEMENT_REPORTED -> TENANT_APPROVED -> {DEPRECATED, REVOKED}."""

    REGISTERED = "REGISTERED"
    CONFORMANCE_PASSED = "CONFORMANCE_PASSED"
    AGREEMENT_REPORTED = "AGREEMENT_REPORTED"
    TENANT_APPROVED = "TENANT_APPROVED"
    DEPRECATED = "DEPRECATED"
    REVOKED = "REVOKED"


_TERMINAL_STATES = frozenset({QualificationState.DEPRECATED, QualificationState.REVOKED})


class IllegalTransitionError(ValueError):
    """Raised when a transition would skip a required lifecycle stage."""


@dataclass(frozen=True)
class QualificationRecord:
    """One `(provider_id, pinned_model_id, configuration_version)` tuple's
    qualification state -- the same tuple `derive_calibration_class()`
    hashes. Immutable: every transition returns a new record.
    """

    provider_id: str
    pinned_model_id: str
    configuration_version: str
    is_pinned: bool = True
    state: QualificationState = QualificationState.REGISTERED
    dataset_version: str | None = None
    adversarial_flips: int | None = None
    approved_tenant_ids: frozenset[str] = field(default_factory=frozenset)
    compliance_tags: frozenset[str] = field(default_factory=frozenset)


def mark_conformance_passed(record: QualificationRecord) -> QualificationRecord:
    """`REGISTERED -> CONFORMANCE_PASSED`: the adapter passed the port-level
    success/ambiguity/timeout/rate-limit/malformed-response fixture suite.
    """
    if record.state != QualificationState.REGISTERED:
        raise IllegalTransitionError(
            f"conformance requires state=REGISTERED, got {record.state.value}"
        )
    return replace(record, state=QualificationState.CONFORMANCE_PASSED)


def report_agreement(
    record: QualificationRecord,
    *,
    dataset_version: str,
    adversarial_flips: int,
) -> QualificationRecord:
    """`CONFORMANCE_PASSED -> AGREEMENT_REPORTED`, only if the report shows
    zero adversarial fixtures flipped to SUPPORTED (ADR-013). A nonzero
    count refuses the transition -- the record stays `CONFORMANCE_PASSED`,
    per evaluator-registry.md, rather than raising: a failed agreement
    report is an expected outcome, not a programming error.
    """
    if record.state != QualificationState.CONFORMANCE_PASSED:
        raise IllegalTransitionError(
            f"agreement reporting requires state=CONFORMANCE_PASSED, got {record.state.value}"
        )
    if adversarial_flips != 0:
        return record
    return replace(
        record,
        state=QualificationState.AGREEMENT_REPORTED,
        dataset_version=dataset_version,
        adversarial_flips=adversarial_flips,
    )


def approve_for_tenant(record: QualificationRecord, tenant_id: str) -> QualificationRecord:
    """`AGREEMENT_REPORTED -> TENANT_APPROVED` (or add another tenant to an
    already-approved record). Approval is per tenant, never global."""
    if record.state not in (
        QualificationState.AGREEMENT_REPORTED,
        QualificationState.TENANT_APPROVED,
    ):
        raise IllegalTransitionError(
            f"tenant approval requires state=AGREEMENT_REPORTED or TENANT_APPROVED, "
            f"got {record.state.value}"
        )
    return replace(
        record,
        state=QualificationState.TENANT_APPROVED,
        approved_tenant_ids=record.approved_tenant_ids | {tenant_id},
    )


def deprecate(record: QualificationRecord) -> QualificationRecord:
    """Voluntary sunset: new dispatch discouraged, existing approvals stand
    until the caller stops using this record. Terminal."""
    if record.state in _TERMINAL_STATES:
        raise IllegalTransitionError(f"cannot deprecate a {record.state.value} record")
    return replace(record, state=QualificationState.DEPRECATED)


def revoke(record: QualificationRecord) -> QualificationRecord:
    """Blocks new dispatch immediately (ADR-014). Terminal; revoking an
    already-revoked record is idempotent, not an error."""
    if record.state == QualificationState.DEPRECATED:
        raise IllegalTransitionError("cannot revoke a DEPRECATED record")
    return replace(record, state=QualificationState.REVOKED)


def is_default_eligible(
    record: QualificationRecord,
    tenant_id: str,
    required_compliance_tags: list[str] | None = None,
) -> bool:
    """Whether `record` may be selected as `tenant_id`'s default judge
    (evaluator-registry.md): TENANT_APPROVED for this exact tenant, pinned
    (not a floating alias), and satisfying every compliance tag the caller
    requires. False for every other state, including PROVISIONAL-equivalent
    states usable for non-default/explicit use (ADR-010: "any judge must
    not become any judge we've certified").
    """
    if record.state != QualificationState.TENANT_APPROVED:
        return False
    if tenant_id not in record.approved_tenant_ids:
        return False
    if not record.is_pinned:
        return False
    return is_provider_compliant(record.compliance_tags, required_compliance_tags or [])
