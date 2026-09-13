"""Tests for error classes."""

import pytest

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


class TestFactLamaError:
    """Tests for base FactLamaError."""

    def test_error_creation(self) -> None:
        """Test creating a base error."""
        error = FactLamaError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == "FACTLAMA_ERROR"

    def test_error_with_code(self) -> None:
        """Test error with custom code."""
        error = FactLamaError("Error", code="CUSTOM_ERROR")
        assert error.code == "CUSTOM_ERROR"

    def test_error_to_dict(self) -> None:
        """Test converting error to dictionary."""
        error = FactLamaError(
            "Error message",
            code="TEST_ERROR",
            details={"key": "value"},
        )
        error_dict = error.to_dict()
        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Error message"
        assert error_dict["details"]["key"] == "value"


class TestInvalidVerificationRequestError:
    """Tests for InvalidVerificationRequestError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = InvalidVerificationRequestError("Missing required field")
        assert error.code == "INVALID_VERIFICATION_REQUEST"
        assert "Missing required field" in error.message


class TestInvalidEvidenceError:
    """Tests for InvalidEvidenceError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = InvalidEvidenceError("Invalid evidence format")
        assert error.code == "INVALID_EVIDENCE"


class TestInvalidPolicyError:
    """Tests for InvalidPolicyError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = InvalidPolicyError("Policy threshold out of range")
        assert error.code == "INVALID_POLICY"


class TestUnsupportedVerificationModeError:
    """Tests for UnsupportedVerificationModeError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = UnsupportedVerificationModeError("ULTRA_DEEP")
        assert error.code == "UNSUPPORTED_VERIFICATION_MODE"
        assert "ULTRA_DEEP" in error.message
        assert error.details["mode"] == "ULTRA_DEEP"


class TestModelProviderError:
    """Tests for ModelProviderError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = ModelProviderError(
            "Model failed to load",
            provider="factlama-slm",
        )
        assert error.code == "MODEL_PROVIDER_ERROR"
        assert error.details["provider"] == "factlama-slm"


class TestVerificationTimeoutError:
    """Tests for VerificationTimeoutError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = VerificationTimeoutError(timeout_ms=5000)
        assert error.code == "VERIFICATION_TIMEOUT"
        assert error.details["timeout_ms"] == 5000


class TestVerificationUnavailableError:
    """Tests for VerificationUnavailableError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = VerificationUnavailableError("Service temporarily unavailable")
        assert error.code == "VERIFICATION_UNAVAILABLE"


class TestSchemaVersionError:
    """Tests for SchemaVersionError."""

    def test_error_creation(self) -> None:
        """Test creating the error."""
        error = SchemaVersionError(expected="0.1", actual="99.0")
        assert error.code == "SCHEMA_VERSION_ERROR"
        assert "0.1" in error.message
        assert "99.0" in error.message
        assert error.details["expected"] == "0.1"
        assert error.details["actual"] == "99.0"
