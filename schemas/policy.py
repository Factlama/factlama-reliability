"""Policy schema models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyAction(str, Enum):
    """Actions that can be taken based on policy evaluation."""

    PASS = "PASS"
    FAIL = "FAIL"
    REGENERATE = "REGENERATE"
    RETRIEVE_AGAIN = "RETRIEVE_AGAIN"
    SWITCH_MODEL = "SWITCH_MODEL"
    ASK_USER = "ASK_USER"
    BLOCK = "BLOCK"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class CaptureMode(str, Enum):
    """ADR-008's persistence-time content-governance control (G5/REL-11's
    baseline scope, `core.governance.apply_capture_mode`): whether a
    persisted `VerificationResult`'s claim text, rationale and violation
    messages are written raw, redacted, or not written at all.

    `NONE` and `METADATA_ONLY` are treated identically by this pass -- both
    "never write raw content" (ADR-008); the distinction between a fully
    disabled telemetry tier and a metadata-only one is G8/G10's
    InteractionStore, not this pass's. `REDACTED` writes only after
    `core.governance`'s own redaction step. `FULL` persists raw content --
    choosing it here *is* ADR-008's required "explicit tenant opt-in," since
    no separate authorization mechanism exists yet.
    """

    NONE = "NONE"
    METADATA_ONLY = "METADATA_ONLY"
    REDACTED = "REDACTED"
    FULL = "FULL"


class CapturePolicy(BaseModel):
    """Content-governance configuration for one policy (ADR-008: "tenant-
    scoped... capture controls"). `METADATA_ONLY` is the default -- ADR-014:
    "FactLama's own recommended metadata-only default."
    """

    mode: CaptureMode = Field(
        default=CaptureMode.METADATA_ONLY,
        description="Persistence-time capture control (ADR-008); does not affect the immediate "
        "response, only what a worker persists (CONTRACTS.md).",
    )


class GroundingPolicy(BaseModel):
    """Policy for groundedness requirements."""

    minimum: float | None = Field(None, ge=0.0, le=1.0, description="Minimum groundedness score")
    required: bool | None = Field(None, description="Whether groundedness is required")


class HallucinationPolicy(BaseModel):
    """Policy for hallucination risk."""

    maximum: float | None = Field(
        None, ge=0.0, le=1.0, description="Maximum allowed hallucination risk"
    )


class ScopePolicy(BaseModel):
    """Policy for scope compliance."""

    enabled: bool = Field(default=True, description="Whether scope checking is enabled")
    allowed: list[str] = Field(default_factory=list, description="Allowed scope domains")
    forbidden: list[str] = Field(default_factory=list, description="Forbidden scope domains")


class CitationPolicy(BaseModel):
    """Policy for citation requirements."""

    required: bool = Field(default=False, description="Whether citations are required")
    minimum_support: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum citation support score"
    )


class InstructionPolicy(BaseModel):
    """Policy for instruction adherence."""

    minimum: float | None = Field(None, ge=0.0, le=1.0, description="Minimum adherence score")
    critical_only: bool = Field(default=False, description="Only evaluate critical instructions")


class ToolPolicy(BaseModel):
    """Policy for tool correctness."""

    minimum: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum tool correctness score"
    )
    fail_on_error: bool = Field(default=True, description="Fail if any tool execution errored")


class ActionPolicy(BaseModel):
    """Policy for actions on failures."""

    on_failure: PolicyAction = Field(
        default=PolicyAction.FAIL, description="Action on verification failure"
    )
    on_partial: PolicyAction = Field(
        default=PolicyAction.PASS, description="Action on partial verification"
    )
    on_abstain: PolicyAction = Field(
        default=PolicyAction.HUMAN_REVIEW, description="Action when abstaining"
    )


class Policy(BaseModel):
    """Complete policy configuration for verification."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Policy identifier")
    tenant_id: str = Field(default="default", description="Tenant identifier for policy scoping")
    version: str = Field(default="0.1", description="Policy schema version")
    grounding: GroundingPolicy = Field(default_factory=GroundingPolicy)
    hallucination: HallucinationPolicy = Field(default_factory=HallucinationPolicy)
    scope: ScopePolicy = Field(default_factory=ScopePolicy)
    citations: CitationPolicy = Field(default_factory=CitationPolicy)
    instructions: InstructionPolicy = Field(default_factory=InstructionPolicy)
    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    actions: ActionPolicy = Field(default_factory=ActionPolicy)
    required_provider_compliance: list[str] = Field(
        default_factory=list,
        description=(
            "Compliance tags the dispatched JudgeProvider must declare (see "
            "JudgeProvider.compliance_tags). Empty means no constraint. A "
            "provider missing a required tag is never dispatched -- the "
            "request abstains with NO_COMPLIANT_PROVIDER instead."
        ),
    )
    capture: CapturePolicy = Field(default_factory=CapturePolicy)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional policy metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary representation."""
        return self.model_dump()
