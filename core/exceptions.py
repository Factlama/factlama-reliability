"""Core exception classes."""

from typing import Any, Optional


class CoreError(Exception):
    """Base exception for core module errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "CORE_ERROR"
        self.details = details or {}


class ClaimExtractionError(CoreError):
    """Raised when claim extraction fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="CLAIM_EXTRACTION_ERROR", details=details)


class EvidenceMappingError(CoreError):
    """Raised when evidence mapping fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="EVIDENCE_MAPPING_ERROR", details=details)


class ScoringError(CoreError):
    """Raised when scoring fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="SCORING_ERROR", details=details)


class PolicyEvaluationError(CoreError):
    """Raised when policy evaluation fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="POLICY_EVALUATION_ERROR", details=details)
