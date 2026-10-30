"""Domain types for G5's durable job/outbox storage (ADR-019).

These are internal to Reliability's persistence boundary, not contracts/v0.1
wire types: `docs/API.md` and `docs/async-evaluation.md` describe the job
state machine and status-response shape in prose, but no schema file in
`factlama-architecture/contracts/v0.1` defines a job or outbox row -- ADR-019
itself notes this gap and leaves the internal representation to Reliability.
`OutboxEvent`, by contrast, is persisted storage for a real wire type
(`ReliabilityEvent`) and its fields are kept in lockstep with
`contracts/v0.1/schemas/reliability_event.schema.json`.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from schemas.tenancy import TenantContext
from schemas.verification import ScoreValue, Usage, VerificationRequest


def to_wire_timestamp(value: datetime) -> str:
    """`contracts/v0.1`'s `Timestamp`: "RFC 3339 UTC timestamp (Z suffix
    required)", pattern-enforced as `...Z`.

    `datetime.isoformat()` renders a UTC-aware value as `+00:00`, which is
    the same instant but does NOT match that pattern -- so the outbox's
    `occurred_at` was failing contract validation before this existed
    (found 2026-10-29 by `tests/test_wire_output_schema.py`'s new
    `ReliabilityEvent` producer-output suite). `datetime.fromisoformat`
    accepts either spelling, so readers are unaffected."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


CLAIM_FINDING_STATUSES = frozenset(
    {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
)
"""ADR-022's `ClaimFindingStatus` -- deliberately narrower than
`schemas.claims.ClaimVerdict`, which also has `NOT_APPLICABLE` and
`DISPUTED`. A claim with either of those is not a reliability *finding*
and is omitted from the projection's `findings` while still counted in
`total_claim_count` (see `ClaimSummary` below)."""


class JobState(str, Enum):
    """`docs/async-evaluation.md`: "States are QUEUED -> RUNNING ->
    SUCCEEDED|FAILED|CANCELLED; expired leases return to QUEUED until
    bounded attempts/deadline are exhausted." `SUCCEEDED` covers both a
    completed and an abstained `VerificationResult` -- API.md: "FAILED is
    operational and has a typed error, never a factual verdict."
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_JOB_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})
"""`docs/async-evaluation.md`: "Terminal states cannot be overwritten by
late workers." Every write path that transitions a job's state checks
membership here before allowing the transition."""


@dataclass(frozen=True)
class Job:
    """The handle returned by `submit_or_get_job`, and by
    `get_job_status` -- enough to answer API.md's `POST
    /v0.1/verification-jobs` (202) and `GET
    /v0.1/verification-jobs/{job_id}` (200) responses without exposing
    internal lease/fencing state to a caller."""

    job_id: str
    tenant_id: str
    project_id: str
    application_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    attempt_count: int
    evaluation_id: str | None = None
    """Set once a worker has produced a result; present even when `state`
    is not yet terminal is never valid -- only a terminal `SUCCEEDED` job
    carries one."""


@dataclass(frozen=True)
class ClaimedJob:
    """What `claim_due_job` hands a worker: the stored request plus the
    lease/fencing state every subsequent call on this claim must present
    back to prove current ownership (ADR-019 SS4.3)."""

    job_id: str
    tenant: TenantContext
    request: VerificationRequest
    fencing_token: int
    attempt_count: int
    deadline: datetime
    """Overall request deadline (docs/async-evaluation.md's "total
    deadline" for the retry/backoff budget), not the per-lease expiry --
    `lease_expires_at` is a worker-liveness mechanism the caller renews;
    `deadline` is a fixed budget the caller cannot extend."""
    input_expires_at: datetime
    """A5 of the 2026-10-14 ADR-020 review: the execution input's own
    absolute expiry (always <= `deadline`, per A7's fix to
    `submit_or_get_job`), carried through the claim so dispatch can bound
    itself by the earlier of the two, and so a late commit can be rejected
    if this has passed even though `deadline` technically has not (it
    cannot, given the invariant above, but this makes the real constraint
    explicit rather than re-deriving it from `deadline` at every call
    site)."""
    lease_expires_at: datetime


@dataclass(frozen=True)
class ClaimFinding:
    """ADR-022: one claim's non-content finding. Deliberately carries only
    these two fields -- the wire schema's own `ClaimFinding` sets
    `additionalProperties: false` precisely so claim text, evidence,
    citations or free-form judge rationale cannot cross into the
    observability projection. Do not add a content field here; that is a
    contract change requiring an ADR, not a local edit."""

    status: str
    rationale_code: str | None = None


@dataclass(frozen=True)
class ClaimSummary:
    """ADR-022's claim-summary projection on `ReliabilityEvent`.

    `findings` is keyed by `claim_id` (a map, not a list) so one claim_id
    structurally cannot appear twice. `total_claim_count` counts *every*
    claim in the evaluation, including ones with no projectable finding,
    so a consumer computes the no-finding count as `total_claim_count -
    len(findings)` and never mistakes it for zero findings. The whole
    object being absent (`OutboxEvent.claim_summary is None`) means
    "unavailable", which is a different state again -- never zero."""

    total_claim_count: int
    findings: dict[str, ClaimFinding] = field(default_factory=dict)

    def to_wire_json(self) -> dict[str, object]:
        """The exact `claim_summary` wire shape
        (`contracts/v0.1/schemas/common.schema.json#/$defs/ClaimSummary`).

        Lives on the model rather than in each serializer because both
        `worker.outbox_http_delivery.event_to_reliability_event_json` and
        `storage.postgres.backend._outbox_event_to_dict` need it, and
        those two are deliberately kept independent of each other
        (`worker/` may not import the concrete `storage.postgres`
        adapter). A method on the shared model they both already import is
        the one place that cannot drift. `rationale_code` is omitted, not
        nulled, when absent -- the wire schema's `ClaimFinding` sets
        `additionalProperties: false` and declares only `status` as
        required."""
        findings: dict[str, object] = {}
        for claim_id, finding in self.findings.items():
            payload: dict[str, object] = {"status": finding.status}
            if finding.rationale_code is not None:
                payload["rationale_code"] = finding.rationale_code
            findings[claim_id] = payload
        return {"total_claim_count": self.total_claim_count, "findings": findings}

    @classmethod
    def from_wire_json(cls, payload: dict[str, Any]) -> "ClaimSummary":
        """Inverse of `to_wire_json`, for `_dict_to_outbox_event`'s
        rehydration of a persisted outbox payload. Without this the
        projection would survive unit tests (which build the event
        in-process) but be silently dropped on the real async delivery
        path, which stores the payload and reads it back before
        delivering."""
        return cls(
            total_claim_count=int(payload["total_claim_count"]),
            findings={
                claim_id: ClaimFinding(
                    status=finding["status"], rationale_code=finding.get("rationale_code")
                )
                for claim_id, finding in payload.get("findings", {}).items()
            },
        )


@dataclass(frozen=True)
class OutboxEvent:
    """Persisted form of a `ReliabilityEvent`
    (`contracts/v0.1/schemas/reliability_event.schema.json`). Field names
    and requiredness mirror that schema exactly; `qualification_status` is
    `str`, not `schemas.verification.QualificationStatus`, because the
    wire vocabulary also allows `"MIXED"` here
    (`QualificationStatusOrMixed` in `common.schema.json`), a value that
    enum does not carry since a single judge attempt's own
    `QualificationStatus` is never `MIXED` (PR3 constructs this field once
    genuine multi-attempt dispute reconciliation exists)."""

    event_id: str
    tenant_id: str
    evaluation_id: str
    occurred_at: datetime
    status: str
    verdict: str
    scores: dict[str, ScoreValue]
    evaluator_version: str
    calibration_class: str
    qualification_status: str
    usage_summary: Usage
    project_id: str | None = None
    application_id: str | None = None
    interaction_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    calibration_classes: list[str] = field(default_factory=list)
    # ADR-022: optional projection. `None` means "this producer sent no
    # claim projection" -- which consumers must render as *unavailable*,
    # never as zero findings.
    claim_summary: ClaimSummary | None = None


__all__ = [
    "CLAIM_FINDING_STATUSES",
    "TERMINAL_JOB_STATES",
    "ClaimFinding",
    "ClaimSummary",
    "ClaimedJob",
    "Job",
    "JobState",
    "OutboxEvent",
    "to_wire_timestamp",
]
