"""Content governance (G5/REL-11, ADR-008): applies a `CaptureMode` to a
computed `VerificationResult` immediately before it is persisted.

CONTRACTS.md: "The immediate response may contain claim text even when
persistence is metadata-only; later reads return `text_status=NOT_STORED`
and omit `text`." This module is what makes that distinction real. It never
touches the result an immediate caller already has -- that copy is built by
`core.verifier`/`worker.runner` before this function ever runs, and this
module's return value is a separate copy (every model involved is frozen);
it only governs what `worker.runner._commit` goes on to persist via
`storage.backend.StorageBackend.commit_evaluation_and_outbox`. Free-text
rationale and violation messages are content-governed the same way as claim
text (CONTRACTS.md: "Free-text rationale is optional and content-governed"
/ "`message` is optional free text, content-governed like claim rationale").

Deliberately out of scope for this pass: the *request* side
(`storage.postgres.tables.jobs.request_json`) is not governed here -- a
worker must still read the real answer/evidence out of it to actually
dispatch the job, and ADR-008's `InteractionStore` (a separate, tenant-
scoped historical archive with its own retention) is explicitly "design
only; implementation deferred," not this table. Governing the long-lived
`evaluations` row -- the one artifact CONTRACTS.md's `text_status` field
describes and the one a later `GET /v0.1/verifications/{evaluation_id}`
actually reads back -- is this pass's whole scope.
"""

from schemas.claims import ClaimVerification, TextStatus
from schemas.policy import CaptureMode
from schemas.verification import VerificationResult, Violation

_REDACTED_PLACEHOLDER = "[REDACTED]"


def _redact(text: str) -> str:
    """A deterministic, content-independent placeholder -- not a PII
    scrubber or an NER-based partial redactor. ADR-008's own bar for
    `REDACTED` is "writes only after configured redaction"; replacing the
    original text outright satisfies that honestly, rather than claiming a
    sophistication this pass does not build. A real redaction engine is
    future work, swapped in here without changing this function's shape or
    any caller.
    """
    del text  # content-independent by design; never inspected
    return _REDACTED_PLACEHOLDER


def _governed_claim(claim: ClaimVerification, mode: CaptureMode) -> ClaimVerification:
    if claim.text_status != TextStatus.AVAILABLE:
        return claim  # already governed (or never had text) -- nothing to do
    if mode in (CaptureMode.NONE, CaptureMode.METADATA_ONLY):
        return claim.model_copy(
            update={"text": None, "text_status": TextStatus.NOT_STORED, "rationale": None}
        )
    if mode == CaptureMode.REDACTED:
        return claim.model_copy(
            update={
                "text": _redact(claim.text) if claim.text is not None else None,
                "text_status": TextStatus.REDACTED,
                "rationale": _redact(claim.rationale) if claim.rationale is not None else None,
            }
        )
    return claim  # FULL: pass through unchanged


def _governed_violation(violation: Violation, mode: CaptureMode) -> Violation:
    if violation.message is None:
        return violation
    if mode in (CaptureMode.NONE, CaptureMode.METADATA_ONLY):
        return violation.model_copy(update={"message": None})
    if mode == CaptureMode.REDACTED:
        return violation.model_copy(update={"message": _redact(violation.message)})
    return violation  # FULL: pass through unchanged


def apply_capture_mode(result: VerificationResult, mode: CaptureMode) -> VerificationResult:
    """Return a copy of `result` governed for persistence under `mode`.
    `result` itself is never mutated (every model here is frozen) -- calling
    this has no effect on a caller's own in-memory/immediate-response copy.
    """
    if mode == CaptureMode.FULL:
        return result
    return result.model_copy(
        update={
            "claims": [_governed_claim(claim, mode) for claim in result.claims],
            "violations": [_governed_violation(v, mode) for v in result.violations],
        }
    )
