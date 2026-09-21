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
from datetime import datetime
from enum import Enum

from schemas.tenancy import TenantContext
from schemas.verification import ScoreValue, Usage, VerificationRequest


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
    lease_expires_at: datetime


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


__all__ = [
    "TERMINAL_JOB_STATES",
    "ClaimedJob",
    "Job",
    "JobState",
    "OutboxEvent",
]
