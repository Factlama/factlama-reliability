"""Verification request and result schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.citation import Citation
from schemas.claims import Claim, ClaimVerification
from schemas.evidence import Evidence
from schemas.instruction import Instruction
from schemas.policy import Policy
from schemas.tools import ToolExecution


class VerificationMode(str, Enum):
    """Verification mode determining depth and cost."""

    FAST = "FAST"
    STANDARD = "STANDARD"
    DEEP = "DEEP"


class OverallVerdict(str, Enum):
    """Overall verification verdict.

    DISPUTED cannot be produced by this pipeline yet (see ResultStatus); it
    is present for the same fixture-compatibility reason.
    """

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    DISPUTED = "DISPUTED"


class ResultStatus(str, Enum):
    """Pipeline execution status for a verification result.

    This pipeline cannot itself produce DISPUTED yet: it dispatches one
    judge attempt per claim, so multi-judge disagreement (G5's bounded
    retries/fallback) cannot occur. It is present so this type can still
    parse every valid contracts/v0.1 fixture, including dispute cases.
    """

    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"
    DISPUTED = "DISPUTED"


class AbstentionReason(str, Enum):
    """Why a result abstained rather than returning a factual verdict.

    BUDGET_EXHAUSTED, NO_COMPLIANT_PROVIDER and REVOKED cannot be produced
    by this pipeline yet -- there is no budget enforcement (G3) or provider
    registry (G10) in this pass. They are present so this type can still
    parse every valid contracts/v0.1 fixture that uses them.
    """

    NO_CHECKABLE_CLAIMS = "NO_CHECKABLE_CLAIMS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_COMPLIANT_PROVIDER = "NO_COMPLIANT_PROVIDER"
    REVOKED = "REVOKED"
    CANCELLED = "CANCELLED"


class DisputeReason(str, Enum):
    """Why a result is DISPUTED. Single value today; additive vocabulary."""

    JUDGE_DISAGREEMENT = "JUDGE_DISAGREEMENT"


class Severity(str, Enum):
    """Violation severity (contracts/v0.1's uppercase wire values)."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ScoreStatus(str, Enum):
    """Whether a score dimension has a usable measured value."""

    MEASURED = "MEASURED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ScoreValue(BaseModel):
    """One named score dimension with its measurement status and provenance.

    `value` is only meaningful when `status == MEASURED`; no score may be
    inferred from a missing/omitted field (CONTRACTS.md).
    """

    model_config = ConfigDict(frozen=True)

    value: float | None = Field(
        None, ge=0.0, le=1.0, description="Score in [0,1], only set when status is MEASURED"
    )
    status: ScoreStatus = Field(..., description="MEASURED | NOT_APPLICABLE | UNAVAILABLE")
    method_version: str | None = Field(
        None, description="Versioned method that produced this score"
    )
    calibration_class: str | None = Field(
        None, description="Opaque calibration class; 'NONE' when no judge attempt backs this score"
    )


class QualificationStatus(str, Enum):
    """Evaluator qualification status for a judge attempt (ADR-010)."""

    QUALIFIED = "QUALIFIED"
    PROVISIONAL = "PROVISIONAL"
    UNQUALIFIED = "UNQUALIFIED"


class AttemptOutcome(str, Enum):
    """Outcome of a single judge attempt."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Cost(BaseModel):
    """Cost of a judge call. Unknown cost is UNAVAILABLE, never zero."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="MEASURED | UNAVAILABLE")
    amount: float | None = Field(None, description="Cost amount, only set when status is MEASURED")
    currency: str | None = Field(
        None, description="ISO currency code, only set when status is MEASURED"
    )
    pricing_version: str | None = Field(None, description="Versioned price source, when known")


class Usage(BaseModel):
    """Token/cost/latency usage for a judge attempt or a result-level summary."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: Cost = Field(default_factory=lambda: Cost(status="UNAVAILABLE"))
    latency_ms: float | None = None


class Attempt(BaseModel):
    """A single judge dispatch, successful or failed. Append-only, ordered."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    provider_id: str
    model_id: str | None = None
    pinned_model_version: str | None = None
    configuration_version: str
    qualification_status: QualificationStatus
    calibration_class: str
    outcome: AttemptOutcome
    error: str | None = Field(None, description="JudgeErrorCode value when outcome is FAILED")
    started_at: datetime
    completed_at: datetime
    usage: Usage = Field(default_factory=Usage)


class Provenance(BaseModel):
    """How a result was produced, without raw interaction content."""

    model_config = ConfigDict(frozen=True)

    evaluator_id: str
    evaluator_version: str
    policy_version: str | None = None
    mode: VerificationMode
    routing_profile_version: str
    started_at: datetime
    completed_at: datetime
    attempts: list[Attempt] = Field(default_factory=list)


class Violation(BaseModel):
    """A violation detected during verification."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="Violation code")
    severity: Severity = Field(..., description="Severity level")
    claim_ids: list[str] = Field(default_factory=list, description="IDs of related claims")
    evidence_ids: list[str] = Field(default_factory=list, description="IDs of related evidence")
    message: str | None = Field(None, description="Human-readable violation message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Standard violation codes
VIOLATION_CODES = {
    "HALLUCINATION": "Claim not supported by evidence",
    "CONTRADICTION": "Claim contradicts evidence",
    "SCOPE_BREACH": "Claim outside defined scope",
    "CITATION_MISMATCH": "Citation does not support claim",
    "INSTRUCTION_VIOLATION": "Instruction not followed",
    "TOOL_ERROR": "Tool usage error",
    "INSUFFICIENT_EVIDENCE": "Not enough evidence to verify",
    "CONFLICTING_EVIDENCE": "Evidence sources conflict with each other",
}


class VerificationRequest(BaseModel):
    """Canonical request for verification.

    `tenant_id` is deliberately NOT a field here: per CONTRACTS.md, tenant
    identity is established from authenticated context at ingress and must
    never be trusted from an unverified client payload. Callers pass a
    trusted `tenant_id` directly to `Verifier.verify()` / `verify()` instead.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="0.1", description="Schema version for compatibility")
    request_id: str = Field(..., description="Unique verification request ID")
    project_id: str = Field(
        ..., description="Client-supplied project ID, authorized within tenant scope"
    )
    application_id: str = Field(
        ..., description="Client-supplied application ID, authorized within tenant scope"
    )
    question: str | None = Field(None, description="User task/question")
    answer: str = Field(..., min_length=1, description="AI-generated answer to verify")
    evidence: list[Evidence] = Field(default_factory=list, description="Available evidence")
    claims: list[Claim] = Field(
        default_factory=list,
        description=(
            "Caller-supplied explicit claims. When non-empty, the Verifier uses "
            "these unchanged instead of extracting claims from `answer` -- "
            "LOW_LEVEL_IMPLEMENTATION.md: 'explicit claims and supplied evidence "
            "first.'"
        ),
    )
    instructions: list[Instruction] = Field(
        default_factory=list, description="Applicable instructions"
    )
    citations: list[Citation] = Field(default_factory=list, description="Citations supplied by AI")
    tool_executions: list[ToolExecution] = Field(
        default_factory=list, description="Agent tool activity"
    )
    policy: Policy | None = Field(None, description="Policy for evaluation")
    policy_id: str | None = Field(
        None, description="Reference to a registered policy, when not inlined"
    )
    mode: VerificationMode = Field(
        default=VerificationMode.STANDARD, description="Verification mode"
    )
    interaction_id: str | None = Field(
        None, description="Links this evaluation to a user-visible AI operation"
    )
    trace_id: str | None = Field(
        None, description="Incoming OpenTelemetry trace ID, echoed if supplied"
    )
    span_id: str | None = Field(
        None, description="Incoming OpenTelemetry span ID, echoed if supplied"
    )
    model: str | None = Field(None, description="Model that produced the answer being verified")
    prompt_version: str | None = Field(None, description="Prompt version that produced the answer")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Application metadata")

    @model_validator(mode="before")
    @classmethod
    def reject_client_supplied_tenant_id(cls, data: Any) -> Any:
        """contracts/v0.1: a `tenant_id` field on this payload is invalid, not merely
        ignored -- silently dropping it would hide a client bug or a spoofing attempt
        instead of rejecting the malformed request outright."""
        if isinstance(data, dict) and "tenant_id" in data:
            raise ValueError(
                "VerificationRequest must not carry tenant_id; tenant identity comes "
                "from authenticated ingress context, never a client-supplied field"
            )
        return data

    @model_validator(mode="after")
    def validate_schema_version(self) -> "VerificationRequest":
        """Validate schema version compatibility."""
        supported_versions = ["0.1"]
        if self.schema_version not in supported_versions:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version}. Supported: {supported_versions}"
            )
        return self

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> "VerificationRequest":
        """LOW_LEVEL_IMPLEMENTATION.md: "If claims is supplied, validate
        unique IDs and use it unchanged." A duplicate ID is rejected here,
        before the Verifier ever sees it."""
        if self.claims:
            ids = [c.claim_id for c in self.claims]
            if len(ids) != len(set(ids)):
                raise ValueError("VerificationRequest.claims must have unique claim_id values")
        return self

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "VerificationRequest":
        """LOW_LEVEL_IMPLEMENTATION.md: "Validate unique evidence IDs and
        exactly one of content/reference." A duplicate evidence_id is
        rejected here -- every ID-indexed lookup in `core`/`judges`
        (citation checks, judge-cited-evidence validation) assumes
        uniqueness and would otherwise let one entry silently shadow
        another instead of failing loudly."""
        if self.evidence:
            ids = [e.evidence_id for e in self.evidence]
            if len(ids) != len(set(ids)):
                raise ValueError("VerificationRequest.evidence must have unique evidence_id values")
        return self


class VerificationResult(BaseModel):
    """Result of verification with scores, verdict, and provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="0.1", description="Schema version")
    evaluation_id: str = Field(..., description="Unique, immutable ID for this evaluation")
    request_id: str = Field(..., description="ID of the verification request")
    tenant_id: str = Field(..., description="Tenant identifier from trusted authenticated context")
    project_id: str = Field(..., description="Echoed from the request")
    application_id: str = Field(..., description="Echoed from the request")
    interaction_id: str | None = Field(None, description="Echoed from the request")
    trace_id: str | None = Field(None, description="OpenTelemetry trace ID for correlation")
    span_id: str | None = Field(None, description="OpenTelemetry span ID for correlation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Result creation time"
    )
    status: ResultStatus = Field(..., description="Pipeline execution status")
    abstention_reason: AbstentionReason | None = Field(
        None, description="Required when status is FAILED or ABSTAINED"
    )
    dispute_reason: DisputeReason | None = Field(
        None, description="Required when status is DISPUTED"
    )
    verdict: OverallVerdict = Field(..., description="Overall verification verdict")
    scores: dict[str, ScoreValue] = Field(
        default_factory=dict, description="Named score dimensions"
    )
    claims: list[ClaimVerification] = Field(
        default_factory=list, description="Individual claim verifications"
    )
    violations: list[Violation] = Field(default_factory=list, description="Detected violations")
    policy_action: str | None = Field(None, description="Recommended action based on policy")
    policy_version: str | None = Field(None, description="Version of the policy applied")
    scoring_version: str = Field(default="0.1", description="Scoring algorithm version")
    provenance: Provenance = Field(..., description="How this result was produced")
    usage_summary: Usage = Field(
        default_factory=Usage, description="Known usage summed across attempts"
    )
    supersedes: str | None = Field(
        None, description="ID of an earlier evaluation this result supersedes"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional non-sensitive metadata"
    )

    @model_validator(mode="after")
    def validate_abstention_and_dispute_reason(self) -> "VerificationResult":
        """FAILED/ABSTAINED require verdict=ABSTAIN + abstention_reason; DISPUTED requires
        verdict=DISPUTED + dispute_reason. This pipeline cannot produce DISPUTED yet
        (see ResultStatus), but the rule is enforced here so it is ready when G5 can."""
        if self.status in (ResultStatus.FAILED, ResultStatus.ABSTAINED):
            if self.verdict != OverallVerdict.ABSTAIN:
                raise ValueError(f"status={self.status} requires verdict=ABSTAIN")
            if self.abstention_reason is None:
                raise ValueError(f"status={self.status} requires an abstention_reason")
        if self.status == ResultStatus.DISPUTED:
            if self.verdict != OverallVerdict.DISPUTED:
                raise ValueError(f"status={self.status} requires verdict=DISPUTED")
            if self.dispute_reason is None:
                raise ValueError(f"status={self.status} requires a dispute_reason")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return self.model_dump_json(indent=2)
