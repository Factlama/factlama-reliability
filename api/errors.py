"""The shared HTTP error contract (CONTRACTS.md): `{schema_version, error:
{code, message, request_id, retryable, details?}}` with a fixed, stable code
vocabulary. `ApiError` is the one exception type every route raises for a
client-visible failure; `app.py` registers a single handler for it so every
route produces this exact shape without repeating the wiring.
"""

from enum import Enum
from typing import Any


class ApiErrorCode(str, Enum):
    """CONTRACTS.md's stable HTTP error codes."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL = "INTERNAL"


_STATUS_BY_CODE: dict[ApiErrorCode, int] = {
    ApiErrorCode.INVALID_ARGUMENT: 400,
    ApiErrorCode.UNSUPPORTED_VERSION: 400,
    ApiErrorCode.UNAUTHENTICATED: 401,
    ApiErrorCode.FORBIDDEN: 403,
    ApiErrorCode.NOT_FOUND: 404,
    ApiErrorCode.CONFLICT: 409,
    ApiErrorCode.PAYLOAD_TOO_LARGE: 413,
    ApiErrorCode.RATE_LIMITED: 429,
    ApiErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    ApiErrorCode.INTERNAL: 500,
}

# API.md: "`retryable` is true only when a retry may succeed."
_RETRYABLE_CODES = frozenset({ApiErrorCode.RATE_LIMITED, ApiErrorCode.DEPENDENCY_UNAVAILABLE})


class ApiError(Exception):
    """A typed, client-visible API failure. Never carries cross-tenant
    existence or secrets in `message` (CONTRACTS.md)."""

    def __init__(
        self,
        code: ApiErrorCode,
        message: str,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details
        super().__init__(message)

    @property
    def status_code(self) -> int:
        return _STATUS_BY_CODE[self.code]

    def to_body(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "request_id": self.request_id,
            "retryable": self.code in _RETRYABLE_CODES,
        }
        if self.details:
            error["details"] = self.details
        return {"schema_version": "0.1", "error": error}
