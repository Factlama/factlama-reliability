"""FactLama schemas - Verification contracts and data models."""

from schemas.verification import (
    AbstentionReason,
    Attempt,
    AttemptOutcome,
    Cost,
    OverallVerdict,
    Provenance,
    QualificationStatus,
    ResultStatus,
    ScoreStatus,
    ScoreValue,
    Usage,
    VerificationMode,
    VerificationRequest,
    VerificationResult,
    Violation,
)
from schemas.evidence import Evidence, EvidenceType, Source
from schemas.claims import Claim, ClaimType, ClaimVerification, ClaimVerdict
from schemas.policy import Policy, PolicyAction
from schemas.instruction import Instruction, InstructionType, Priority
from schemas.citation import Citation, Locator
from schemas.tools import ToolExecution, ToolStatus
from schemas.telemetry import TelemetryData
from schemas.errors import (
    FactLamaError,
    InvalidVerificationRequestError,
    InvalidEvidenceError,
    InvalidPolicyError,
    UnsupportedVerificationModeError,
    ModelProviderError,
    VerificationTimeoutError,
    VerificationUnavailableError,
    SchemaVersionError,
)

__all__ = [
    # Verification
    "VerificationMode",
    "VerificationRequest",
    "VerificationResult",
    "Violation",
    "OverallVerdict",
    "ResultStatus",
    "AbstentionReason",
    "ScoreStatus",
    "ScoreValue",
    "QualificationStatus",
    "AttemptOutcome",
    "Attempt",
    "Provenance",
    "Cost",
    "Usage",
    # Evidence
    "Evidence",
    "EvidenceType",
    "Source",
    # Claims
    "Claim",
    "ClaimType",
    "ClaimVerification",
    "ClaimVerdict",
    # Policy
    "Policy",
    "PolicyAction",
    # Instruction
    "Instruction",
    "InstructionType",
    "Priority",
    # Citation
    "Citation",
    "Locator",
    # Tools
    "ToolExecution",
    "ToolStatus",
    # Telemetry
    "TelemetryData",
    # Errors
    "FactLamaError",
    "InvalidVerificationRequestError",
    "InvalidEvidenceError",
    "InvalidPolicyError",
    "UnsupportedVerificationModeError",
    "ModelProviderError",
    "VerificationTimeoutError",
    "VerificationUnavailableError",
    "SchemaVersionError",
]
