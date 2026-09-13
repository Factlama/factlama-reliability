"""FactLama Core - Verification engine and domain logic."""

from core.claims import ClaimExtractor
from core.evidence import EvidenceMapper
from core.exceptions import (
    ClaimExtractionError,
    CoreError,
    EvidenceMappingError,
    PolicyEvaluationError,
    ScoringError,
)
from core.policy import PolicyEngine
from core.result import VerificationResultBuilder
from core.scoring import ScoringEngine, derive_calibration_class, determine_verdict
from core.verifier import Verifier, verify
from judges.port import JudgeProvider
from judges.providers import MockModelProvider, RuleBasedProvider

__all__ = [
    "ClaimExtractionError",
    # Components
    "ClaimExtractor",
    # Exceptions
    "CoreError",
    "EvidenceMapper",
    "EvidenceMappingError",
    # Providers
    "JudgeProvider",
    "MockModelProvider",
    "PolicyEngine",
    "PolicyEvaluationError",
    "RuleBasedProvider",
    # Engines
    "ScoringEngine",
    "ScoringError",
    "VerificationResultBuilder",
    "Verifier",
    "derive_calibration_class",
    "determine_verdict",
    # Main API
    "verify",
]
