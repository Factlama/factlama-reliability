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

2026-10-13 G0-G6 code review (F1): the *request* side
(`storage.postgres.tables.jobs.request_json`) is now also governed, at
terminal-state commit -- see `governed_request_payload()` below. A worker
still needs the real answer/evidence while a job is `QUEUED`/`RUNNING` to
actually dispatch it (nothing governs that in-flight read), but once a job
reaches a terminal state nothing in this codebase ever reads
`jobs.request_json` back into a `VerificationRequest` again
(`claim_due_job`'s own claim predicate excludes terminal states) -- so
there is no remaining reason for the raw content to persist indefinitely
after that point. This closes the clearest violation the review found
(raw content surviving forever in a completed metadata-only job), matching
EXECUTION_PLAN.md's G5 exit row ("raw content absent in metadata-only
DB/queues/logs") directly. It does not resolve a stricter "zero write at
any point, including in-flight" reading some might take from ADR-008's
`InteractionStore` language -- that would need a different storage
primitive entirely (e.g. encrypted working storage with a key discarded
after dispatch) and is a genuine architecture decision, not made here;
see `factlama-reliability/docs/implementation.md`'s REL-11 entry for this
named, still-open distinction.
"""

from typing import Any

from schemas.claims import ClaimVerification, TextStatus
from schemas.policy import CaptureMode
from schemas.verification import VerificationRequest, VerificationResult, Violation

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


def governed_request_payload(request: VerificationRequest, mode: CaptureMode) -> dict[str, Any]:
    """A governed JSON projection of `request`, for storing as
    `jobs.request_json` -- at DURABLE_ENCRYPTED submission time (ADR-020;
    the real content lives only in the encrypted sibling table from the
    start) and again, idempotently, at terminal transition (F1 of the
    2026-10-13 G0-G6 code review).

    A2 of the 2026-10-14 ADR-020 review: the first two passes of this
    function took the complete serialized request and blacklisted specific
    known content-bearing fields one at a time -- each pass found a new
    nested field the previous one missed (tool payloads and top-level
    `metadata`, then this pass's `evidence[].metadata` and
    `evidence[].source.metadata`/`.locator.metadata`, none of which any
    blacklist here ever named). A blacklist over an open-ended, nested,
    caller-controlled schema can never be complete this way -- `Evidence`,
    `Source`, `Locator`, `Citation.locator` and `ToolExecution` each carry
    at least one arbitrary dict or `Any` field, and nothing stops a caller
    from putting content in any of them.

    For every mode except FULL (ADR-008's explicit tenant opt-in to raw
    retention, passed through unchanged), this builds a NEW dict from an
    explicit allowlist of known-safe, non-content fields -- identifiers,
    enums and counts -- rather than scrubbing a full serialization. No
    nested caller-controlled structure (`evidence`, `claims`, `citations`,
    `instructions`, `tool_executions`, `metadata`) is ever included here at
    all; a field added to any of those models tomorrow is safe by
    construction, because reaching this projection would require adding it
    to the allowlist by name, not merely forgetting to blacklist it.
    `NONE`/`METADATA_ONLY`/`REDACTED` collapse to the same allowlisted
    projection -- the same precedent `_governed_claim` already set for
    `NONE`/`METADATA_ONLY` on the result side extended one step further,
    since nothing here that `REDACTED` could meaningfully replace with a
    placeholder survives into the allowlist in the first place.

    Operates on `request` directly, not a serialized-then-rebuilt dict --
    once a job is terminal, nothing reconstructs a `VerificationRequest`
    from this column again, so this projection only needs to be a safe JSON
    value to store, never a value that re-validates as a live request.
    """
    if mode == CaptureMode.FULL:
        return request.model_dump(mode="json", exclude_none=True)
    return {
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "project_id": request.project_id,
        "application_id": request.application_id,
        "mode": request.mode.value,
        "interaction_id": request.interaction_id,
        "trace_id": request.trace_id,
        "span_id": request.span_id,
        "model": request.model,
        "prompt_version": request.prompt_version,
        "claim_count": len(request.claims),
        "evidence_count": len(request.evidence),
        "citation_count": len(request.citations),
        "instruction_count": len(request.instructions),
        "tool_execution_count": len(request.tool_executions),
        "capture_mode": mode.value,
    }


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
