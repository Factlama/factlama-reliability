"""FactLama Core - Verification engine and domain logic."""

from core.verifier import verify, Verifier
from judges.providers import JudgeProvider, MockModelProvider, RuleBasedProvider
from core.scoring import ScoringEngine, determine_verdict, derive_calibration_class
from core.policy import PolicyEngine
from core.claims import ClaimExtractor
from core.evidence import EvidenceMapper
from core.result import VerificationResultBuilder
from core.exceptions import (
    CoreError,
    ClaimExtractionError,
    EvidenceMappingError,
    ScoringError,
    PolicyEvaluationError,
)

__all__ = [
    # Main API
    "verify",
    "Verifier",
    # Providers
    "JudgeProvider",
    "MockModelProvider",
    "RuleBasedProvider",
    # Engines
    "ScoringEngine",
    "determine_verdict",
    "derive_calibration_class",
    "PolicyEngine",
    # Components
    "ClaimExtractor",
    "EvidenceMapper",
    "VerificationResultBuilder",
    # Exceptions
    "CoreError",
    "ClaimExtractionError",
    "EvidenceMappingError",
    "ScoringError",
    "PolicyEvaluationError",
]
