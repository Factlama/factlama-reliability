"""Claim schema models."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimType(str, Enum):
    """Types of claims that can be extracted from an answer."""

    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    INSTRUCTIONAL = "instructional"
    OPINION = "opinion"
    ACTION = "action"
    OTHER = "other"


class ClaimVerdict(str, Enum):
    """Verdict for an individual claim verification.

    `DISPUTED` cannot be produced by this pipeline yet: it dispatches one
    judge attempt per claim (G5's bounded retries/fallback is what would let
    two attempts on the same claim disagree). It is present so this type can
    still parse every valid contracts/v0.1 fixture, including dispute cases.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DISPUTED = "DISPUTED"


class RationaleCode(str, Enum):
    """Short, stable, versioned rationale tag (CONTRACTS.md v0.1 vocabulary, additive only)."""

    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    PARAPHRASED_SUPPORT = "PARAPHRASED_SUPPORT"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    NUMERICAL_CONTRADICTION = "NUMERICAL_CONTRADICTION"
    NO_SUPPORT = "NO_SUPPORT"
    NO_EVIDENCE_SUPPLIED = "NO_EVIDENCE_SUPPLIED"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    CITATION_UNSUPPORTED = "CITATION_UNSUPPORTED"
    PROVIDER_DISPATCH_FAILED = "PROVIDER_DISPATCH_FAILED"
    NOT_APPLICABLE_CLAIM_TYPE = "NOT_APPLICABLE_CLAIM_TYPE"
    JUDGE_DISAGREEMENT = "JUDGE_DISAGREEMENT"


class TextStatus(str, Enum):
    """Whether a claim's raw text is present on this result (content governance)."""

    AVAILABLE = "AVAILABLE"
    REDACTED = "REDACTED"
    NOT_STORED = "NOT_STORED"


class Claim(BaseModel):
    """A claim is the fundamental unit of verification."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="Unique identifier for this claim")
    text: str = Field(..., description="The claim text")
    type: ClaimType = Field(default=ClaimType.FACTUAL, description="Type of claim")
    importance: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Importance weight (0.0-1.0)"
    )
    start_char: int | None = Field(None, description="Start character position in original answer")
    end_char: int | None = Field(None, description="End character position in original answer")


class ContributingJudgment(BaseModel):
    """One judge attempt's contribution to a claim's aggregate result.

    Today there is exactly one per claim (one-judge-attempt-per-claim); this
    is a list because G5's multi-judge reconciliation will add more without
    changing the shape.
    """

    model_config = ConfigDict(frozen=True)

    attempt_id: str = Field(..., description="ID of the Attempt this judgment came from")
    verdict: ClaimVerdict = Field(..., description="This attempt's verdict")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence IDs this attempt cited"
    )
    rationale_code: RationaleCode = Field(..., description="Why this attempt reached its verdict")


class ClaimVerification(BaseModel):
    """Verification result for an individual claim (contracts/v0.1's ClaimResult shape)."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="ID of the verified claim")
    verdict: ClaimVerdict = Field(..., description="Verification verdict")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting this verdict; non-empty when verdict is SUPPORTED",
    )
    rationale_code: RationaleCode = Field(..., description="Why this verdict was reached")
    rationale: str | None = Field(None, description="Human-readable explanation")
    text_status: TextStatus = Field(
        default=TextStatus.AVAILABLE, description="Whether `text` is present on this result"
    )
    text: str | None = Field(None, description="The claim text, when text_status is AVAILABLE")
    contributing_judgments: list[ContributingJudgment] = Field(
        default_factory=list, description="Judge attempts that produced this verdict"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the verdict (0.0-1.0)"
    )
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def validate_supported_has_evidence(self) -> "ClaimVerification":
        """contracts/v0.1: a SUPPORTED verdict requires at least one evidence ID."""
        if self.verdict == ClaimVerdict.SUPPORTED and not self.evidence_ids:
            raise ValueError("verdict=SUPPORTED requires at least one evidence_id")
        return self
