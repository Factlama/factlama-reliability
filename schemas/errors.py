"""Error classes for FactLama."""

from typing import Any, Optional


class FactLamaError(Exception):
    """Base exception for all FactLama errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "FACTLAMA_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class InvalidVerificationRequestError(FactLamaError):
    """Raised when a verification request is invalid."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="INVALID_VERIFICATION_REQUEST", details=details)


class InvalidEvidenceError(FactLamaError):
    """Raised when evidence is invalid."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="INVALID_EVIDENCE", details=details)


class InvalidPolicyError(FactLamaError):
    """Raised when a policy is invalid."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="INVALID_POLICY", details=details)


class UnsupportedVerificationModeError(FactLamaError):
    """Raised when an unsupported verification mode is requested."""

    def __init__(self, mode: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Unsupported verification mode: {mode}",
            code="UNSUPPORTED_VERIFICATION_MODE",
            details=details or {"mode": mode},
        )


class ModelProviderError(FactLamaError):
    """Raised when a model provider encounters an error."""

    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[dict[str, Any]] = None) -> None:
        details = details or {}
        if provider:
            details["provider"] = provider
        super().__init__(message, code="MODEL_PROVIDER_ERROR", details=details)


class VerificationTimeoutError(FactLamaError):
    """Raised when verification times out."""

    def __init__(self, timeout_ms: Optional[float] = None, details: Optional[dict[str, Any]] = None) -> None:
        details = details or {}
        if timeout_ms is not None:
            details["timeout_ms"] = timeout_ms
        super().__init__(
            "Verification timed out",
            code="VERIFICATION_TIMEOUT",
            details=details,
        )


class VerificationUnavailableError(FactLamaError):
    """Raised when verification is unavailable."""

    def __init__(self, message: str = "Verification unavailable", details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, code="VERIFICATION_UNAVAILABLE", details=details)


class SchemaVersionError(FactLamaError):
    """Raised when there's a schema version mismatch."""

    def __init__(
        self,
        expected: str,
        actual: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Schema version mismatch: expected {expected}, got {actual}",
            code="SCHEMA_VERSION_ERROR",
            details=details or {"expected": expected, "actual": actual},
        )
