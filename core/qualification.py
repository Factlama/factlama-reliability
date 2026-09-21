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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from core.compliance import is_provider_compliant

#: ADR-018's per-label qualification bar. EVERY one of these four labels
#: must independently reach at least ADR_018_MIN_LABEL_N scored examples --
#: a label that is missing from a report, or under-sampled, refuses the
#: whole report (see `_undersampled_labels`) rather than being silently
#: skipped. An adapter that was never actually exercised against, say,
#: CONTRADICTED evidence has not been agreement-tested, regardless of how
#: well it did on SUPPORTED. SUPPORTED/CONTRADICTED carry a higher
#: precision/recall bar because they are the labels a downstream policy
#: action (BLOCK, PASS) most directly depends on; UNSUPPORTED/
#: INSUFFICIENT_EVIDENCE get a lower bar because the boundary between "no
#: support" and "insufficient evidence" is inherently fuzzier for a T0
#: adapter.
ADR_018_MIN_LABEL_N = 5
ADR_018_THRESHOLDS: dict[str, tuple[float, float]] = {
    "SUPPORTED": (0.80, 0.80),
    "CONTRADICTED": (0.80, 0.80),
    "UNSUPPORTED": (0.70, 0.70),
    "INSUFFICIENT_EVIDENCE": (0.70, 0.70),
}


def _undersampled_labels(per_label: Mapping[str, Mapping[str, float | int | None]]) -> list[str]:
    """Every ADR-018 label that is missing from `per_label`, or below
    `ADR_018_MIN_LABEL_N` scored examples. Non-empty means the report does
    not cover all four labels well enough to be evidence at all -- a report
    that only tested SUPPORTED, however perfectly, has not tested the other
    three verdicts and must not qualify on that partial evidence. Extra
    keys in `per_label` outside the four ADR-018 labels (e.g. an `ambiguous`
    calibration bucket accidentally passed through) are ignored here, not
    counted toward sufficiency -- only the four scored labels count.
    """
    missing: list[str] = []
    for label in ADR_018_THRESHOLDS:
        metrics = per_label.get(label)
        n = (metrics.get("n") or 0) if metrics is not None else 0
        if n < ADR_018_MIN_LABEL_N:
            missing.append(f"{label} (n={n})")
    return missing


def _invalid_metric(value: float | int | None) -> bool:
    """True if `value` cannot be a real precision/recall: missing,
    non-numeric, NaN/infinite, or outside `[0, 1]`. A threshold comparison
    against NaN is `False` either way in Python (`float("nan") < 0.8` is
    `False`, and so is `>=`), so a corrupted or fabricated metric would
    silently pass `precision < min_precision` instead of failing it -- this
    check exists so a non-finite or out-of-range value refuses the report
    instead of being compared as if it were a real probability.
    """
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0)


def evaluate_agreement_thresholds(
    per_label: Mapping[str, Mapping[str, float | int | None]],
) -> list[str]:
    """ADR-018's per-label precision/recall bar, applied to an agreement
    report's `per_label` section (`evaluator-agreement-harness.md`'s report
    schema). Returns human-readable failure reasons; empty means every one
    of the four labels met its bar. Callers must check `_undersampled_labels`
    first (via `report_agreement()`) -- this function alone does not refuse
    a report for a label that is entirely missing from `per_label`, since a
    missing label has no precision/recall to compare against a threshold.
    """
    failures: list[str] = []
    for label, (min_precision, min_recall) in ADR_018_THRESHOLDS.items():
        metrics = per_label.get(label)
        if metrics is None:
            continue
        n = metrics.get("n") or 0
        if n < ADR_018_MIN_LABEL_N:
            continue
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        if _invalid_metric(precision):
            failures.append(
                f"{label}: precision is not a valid probability ({precision!r}) (n={n})"
            )
        elif precision < min_precision:  # type: ignore[operator]
            failures.append(f"{label}: precision {precision} < {min_precision} (n={n})")
        if _invalid_metric(recall):
            failures.append(f"{label}: recall is not a valid probability ({recall!r}) (n={n})")
        elif recall < min_recall:  # type: ignore[operator]
            failures.append(f"{label}: recall {recall} < {min_recall} (n={n})")
    return failures


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
    threshold_failures: tuple[str, ...] = ()


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
    per_label: Mapping[str, Mapping[str, float | int | None]],
    held_out_hash: str,
    held_out_fixture_count: int,
    leakage_attested: bool,
) -> QualificationRecord:
    """`CONFORMANCE_PASSED -> AGREEMENT_REPORTED`, only if all of (ADR-013,
    ADR-018, evaluator-agreement-harness.md):

    1. Zero adversarial fixtures flipped to SUPPORTED.
    2. All four ADR-018 labels reached `ADR_018_MIN_LABEL_N` scored
       examples -- a report that never exercised, say, CONTRADICTED has not
       agreement-tested the adapter on it, however well the labels it did
       cover scored. A missing or under-sampled label refuses the whole
       report, not just that label (`_undersampled_labels`).
    3. Every label meets its ADR-018 precision/recall bar
       (`evaluate_agreement_thresholds`).
    4. The report actually includes the FactLama-held-out set, not the dev
       set alone: `held_out_hash` is non-empty and `held_out_fixture_count`
       is positive. `evaluator-agreement-harness.md`: the dev set's answers
       are public, so a dev-only report is not independent evidence.
    5. `leakage_attested` is `True` -- the held-out set's human
       leakage-control review (`evaluator-agreement-harness.md`: "a dataset
       reviewer, human, attests to this per release") has actually
       happened. This is a caller-supplied claim, not something this
       function can verify; passing `True` without a real attestation
       having occurred violates the process this parameter exists to
       enforce, not just this function's contract.

    Any failure refuses the transition -- the record stays
    `CONFORMANCE_PASSED`, per evaluator-registry.md, rather than raising: a
    failed agreement report is an expected outcome, not a programming
    error. `record.threshold_failures` carries the specific reasons either
    way (empty on success), so a caller can inspect why a report was
    refused instead of only seeing that it was.
    """
    if record.state != QualificationState.CONFORMANCE_PASSED:
        raise IllegalTransitionError(
            f"agreement reporting requires state=CONFORMANCE_PASSED, got {record.state.value}"
        )

    failures: list[str] = []
    if adversarial_flips != 0:
        failures.append(
            f"adversarial: {adversarial_flips} fixture(s) flipped to SUPPORTED (ADR-013 requires zero)"
        )

    undersampled = _undersampled_labels(per_label)
    if undersampled:
        failures.append(
            f"insufficient sample size (ADR_018_MIN_LABEL_N={ADR_018_MIN_LABEL_N}): "
            f"{', '.join(undersampled)}"
        )
    else:
        failures.extend(evaluate_agreement_thresholds(per_label))

    if not held_out_hash or held_out_fixture_count <= 0:
        failures.append(
            "held-out set not included: held_out_hash/held_out_fixture_count are required "
            "(evaluator-agreement-harness.md: a dev-only report is not sufficient evidence)"
        )
    if not leakage_attested:
        failures.append(
            "leakage-control attestation missing: a human reviewer must attest to the "
            "held-out set per evaluator-agreement-harness.md before it counts as evidence"
        )

    if failures:
        return replace(record, threshold_failures=tuple(failures))

    return replace(
        record,
        state=QualificationState.AGREEMENT_REPORTED,
        dataset_version=dataset_version,
        adversarial_flips=adversarial_flips,
        threshold_failures=(),
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
