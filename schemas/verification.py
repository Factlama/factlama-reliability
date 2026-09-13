"""Verification request and result schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.claims import Claim, ClaimVerification
from schemas.citation import Citation
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
    """Overall verification verdict."""

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"


class ResultStatus(str, Enum):
    """Pipeline execution status for a verification result.

    DISPUTED is deliberately omitted: this MVP dispatches one judge attempt
    per claim, so multi-judge disagreement (ADR-015, deferred) cannot occur.
    """

    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


class AbstentionReason(str, Enum):
    """Why a result abstained rather than returning a factual verdict.

    BUDGET_EXHAUSTED, NO_COMPLIANT_PROVIDER and REVOKED are omitted: there is
    no budget enforcement or provider registry in this pass.
    """

    NO_CHECKABLE_CLAIMS = "NO_CHECKABLE_CLAIMS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    CANCELLED = "CANCELLED"


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

    value: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score in [0,1], only set when status is MEASURED")
    status: ScoreStatus = Field(..., description="MEASURED | NOT_APPLICABLE | UNAVAILABLE")
    method_version: Optional[str] = Field(None, description="Versioned method that produced this score")
    calibration_class: Optional[str] = Field(
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
    amount: Optional[float] = Field(None, description="Cost amount, only set when status is MEASURED")
    currency: Optional[str] = Field(None, description="ISO currency code, only set when status is MEASURED")


class Usage(BaseModel):
    """Token/cost/latency usage for a judge attempt or a result-level summary."""

    model_config = ConfigDict(frozen=True)

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Cost = Field(default_factory=lambda: Cost(status="UNAVAILABLE"))
    latency_ms: Optional[float] = None


class Attempt(BaseModel):
    """A single judge dispatch, successful or failed. Append-only, ordered."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    provider_id: str
    model_id: Optional[str] = None
    pinned_model_version: Optional[str] = None
    configuration_version: str
    qualification_status: QualificationStatus
    calibration_class: str
    outcome: AttemptOutcome
    error: Optional[str] = Field(None, description="JudgeErrorCode value when outcome is FAILED")
    started_at: datetime
    completed_at: datetime
    usage: Usage = Field(default_factory=Usage)


class Provenance(BaseModel):
    """How a result was produced, without raw interaction content."""

    model_config = ConfigDict(frozen=True)

    evaluator_id: str
    evaluator_version: str
    policy_version: Optional[str] = None
    mode: VerificationMode
    routing_profile_version: str
    started_at: datetime
    completed_at: datetime
    attempts: list[Attempt] = Field(default_factory=list)


class Violation(BaseModel):
    """A violation detected during verification."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="Violation code")
    severity: str = Field(..., description="Severity level: info, low, medium, high, critical")
    claim_id: Optional[str] = Field(None, description="ID of the related claim")
    message: str = Field(..., description="Human-readable violation message")
    evidence_id: Optional[str] = Field(None, description="ID of related evidence")
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
    project_id: str = Field(..., description="Client-supplied project ID, authorized within tenant scope")
    application_id: str = Field(..., description="Client-supplied application ID, authorized within tenant scope")
    question: Optional[str] = Field(None, description="User task/question")
    answer: str = Field(..., min_length=1, description="AI-generated answer to verify")
    evidence: list[Evidence] = Field(default_factory=list, description="Available evidence")
    instructions: list[Instruction] = Field(default_factory=list, description="Applicable instructions")
    citations: list[Citation] = Field(default_factory=list, description="Citations supplied by AI")
    tool_executions: list[ToolExecution] = Field(default_factory=list, description="Agent tool activity")
    policy: Optional[Policy] = Field(None, description="Policy for evaluation")
    policy_id: Optional[str] = Field(None, description="Reference to a registered policy, when not inlined")
    mode: VerificationMode = Field(default=VerificationMode.STANDARD, description="Verification mode")
    interaction_id: Optional[str] = Field(None, description="Links this evaluation to a user-visible AI operation")
    trace_id: Optional[str] = Field(None, description="Incoming OpenTelemetry trace ID, echoed if supplied")
    span_id: Optional[str] = Field(None, description="Incoming OpenTelemetry span ID, echoed if supplied")
    model: Optional[str] = Field(None, description="Model that produced the answer being verified")
    prompt_version: Optional[str] = Field(None, description="Prompt version that produced the answer")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Application metadata")

    @model_validator(mode="after")
    def validate_schema_version(self) -> "VerificationRequest":
        """Validate schema version compatibility."""
        supported_versions = ["0.1"]
        if self.schema_version not in supported_versions:
            raise ValueError(f"Unsupported schema version: {self.schema_version}. Supported: {supported_versions}")
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
    interaction_id: Optional[str] = Field(None, description="Echoed from the request")
    trace_id: Optional[str] = Field(None, description="OpenTelemetry trace ID for correlation")
    span_id: Optional[str] = Field(None, description="OpenTelemetry span ID for correlation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Result creation time")
    status: ResultStatus = Field(..., description="Pipeline execution status")
    abstention_reason: Optional[AbstentionReason] = Field(None, description="Required when status is ABSTAINED")
    verdict: OverallVerdict = Field(..., description="Overall verification verdict")
    scores: dict[str, ScoreValue] = Field(default_factory=dict, description="Named score dimensions")
    claims: list[ClaimVerification] = Field(default_factory=list, description="Individual claim verifications")
    violations: list[Violation] = Field(default_factory=list, description="Detected violations")
    policy_action: Optional[str] = Field(None, description="Recommended action based on policy")
    policy_version: Optional[str] = Field(None, description="Version of the policy applied")
    scoring_version: str = Field(default="0.1", description="Scoring algorithm version")
    provenance: Provenance = Field(..., description="How this result was produced")
    usage_summary: Usage = Field(default_factory=Usage, description="Known usage summed across attempts")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive metadata")

    @model_validator(mode="after")
    def validate_abstention_reason(self) -> "VerificationResult":
        """FAILED and ABSTAINED must carry verdict=ABSTAIN with a reason; others must not."""
        if self.status in (ResultStatus.FAILED, ResultStatus.ABSTAINED):
            if self.verdict != OverallVerdict.ABSTAIN:
                raise ValueError(f"status={self.status} requires verdict=ABSTAIN")
            if self.abstention_reason is None:
                raise ValueError(f"status={self.status} requires an abstention_reason")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return self.model_dump_json(indent=2)
