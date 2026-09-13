"""Claim schema models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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
    """Verdict for an individual claim verification."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Claim(BaseModel):
    """A claim is the fundamental unit of verification."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this claim")
    text: str = Field(..., description="The claim text")
    type: ClaimType = Field(default=ClaimType.FACTUAL, description="Type of claim")
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="Importance weight (0.0-1.0)")
    start_char: Optional[int] = Field(None, description="Start character position in original answer")
    end_char: Optional[int] = Field(None, description="End character position in original answer")


class EvidenceReference(BaseModel):
    """Reference to evidence supporting or contradicting a claim."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="ID of the evidence")
    support: float = Field(default=1.0, ge=-1.0, le=1.0, description="Support score (-1.0 to 1.0)")
    relevance: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score (0.0-1.0)")


class ClaimVerification(BaseModel):
    """Verification result for an individual claim."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="ID of the verified claim")
    verdict: ClaimVerdict = Field(..., description="Verification verdict")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the verdict (0.0-1.0)")
    evidence: list[EvidenceReference] = Field(default_factory=list, description="Evidence references")
    conflicting_evidence: list[EvidenceReference] = Field(
        default_factory=list, description="Evidence that conflicts with or contradicts the claim"
    )
    reason: Optional[str] = Field(None, description="Human-readable explanation")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
