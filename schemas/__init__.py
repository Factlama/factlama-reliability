"""FactLama schemas - Verification contracts and data models."""

from schemas.citation import Citation, Locator
from schemas.claims import Claim, ClaimType, ClaimVerdict, ClaimVerification
from schemas.errors import (
    FactLamaError,
    InvalidEvidenceError,
    InvalidPolicyError,
    InvalidVerificationRequestError,
    ModelProviderError,
    SchemaVersionError,
    UnsupportedVerificationModeError,
    VerificationTimeoutError,
    VerificationUnavailableError,
)
from schemas.evidence import Evidence, EvidenceType, Source
from schemas.instruction import Instruction, InstructionType, Priority
from schemas.policy import Policy, PolicyAction
from schemas.telemetry import TelemetryData
from schemas.tools import ToolExecution, ToolStatus
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

__all__ = [
    "AbstentionReason",
    "Attempt",
    "AttemptOutcome",
    # Citation
    "Citation",
    # Claims
    "Claim",
    "ClaimType",
    "ClaimVerdict",
    "ClaimVerification",
    "Cost",
    # Evidence
    "Evidence",
    "EvidenceType",
    # Errors
    "FactLamaError",
    # Instruction
    "Instruction",
    "InstructionType",
    "InvalidEvidenceError",
    "InvalidPolicyError",
    "InvalidVerificationRequestError",
    "Locator",
    "ModelProviderError",
    "OverallVerdict",
    # Policy
    "Policy",
    "PolicyAction",
    "Priority",
    "Provenance",
    "QualificationStatus",
    "ResultStatus",
    "SchemaVersionError",
    "ScoreStatus",
    "ScoreValue",
    "Source",
    # Telemetry
    "TelemetryData",
    # Tools
    "ToolExecution",
    "ToolStatus",
    "UnsupportedVerificationModeError",
    "Usage",
    # Verification
    "VerificationMode",
    "VerificationRequest",
    "VerificationResult",
    "VerificationTimeoutError",
    "VerificationUnavailableError",
    "Violation",
]
