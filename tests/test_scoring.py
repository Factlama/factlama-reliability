"""Tests for scoring engine."""

import pytest

from core.scoring import ScoringEngine, derive_calibration_class, determine_verdict
from schemas.claims import ClaimVerification, ClaimVerdict
from schemas.verification import OverallVerdict, ScoreStatus


class TestScoringEngine:
    """Tests for ScoringEngine."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = ScoringEngine()

    def test_engine_initialization(self) -> None:
        """Test engine initialization."""
        assert self.engine.scoring_version == "0.1"

    def test_calculate_scores_empty_claims(self) -> None:
        """With no claims, the claim-ratio scores are NOT_APPLICABLE, not an invented value."""
        scores = self.engine.calculate_scores([])
        assert scores["groundedness"].status == ScoreStatus.NOT_APPLICABLE
        assert scores["groundedness"].value is None
        assert scores["hallucination_risk"].status == ScoreStatus.NOT_APPLICABLE
        assert scores["contradiction_risk"].status == ScoreStatus.NOT_APPLICABLE

    def test_calculate_scores_supported_claims(self) -> None:
        """Test scoring with supported claims."""
        verifications = [
            ClaimVerification(claim_id="claim_001", verdict=ClaimVerdict.SUPPORTED, confidence=0.95),
            ClaimVerification(claim_id="claim_002", verdict=ClaimVerdict.SUPPORTED, confidence=0.90),
        ]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-1")
        assert scores["groundedness"].value == 1.0
        assert scores["groundedness"].status == ScoreStatus.MEASURED
        assert scores["hallucination_risk"].value == 0.0
        assert scores["contradiction_risk"].value == 0.0

    def test_calculate_scores_unsupported_claims(self) -> None:
        """Test scoring with unsupported claims."""
        verifications = [
            ClaimVerification(claim_id="claim_001", verdict=ClaimVerdict.SUPPORTED, confidence=0.95),
            ClaimVerification(claim_id="claim_002", verdict=ClaimVerdict.UNSUPPORTED, confidence=0.90),
        ]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-1")
        assert scores["groundedness"].value == 0.5
        assert scores["hallucination_risk"].value == 0.5
        assert scores["contradiction_risk"].value == 0.0

    def test_calculate_scores_insufficient_evidence_gets_no_partial_credit(self) -> None:
        """scoring.md: INSUFFICIENT_EVIDENCE contributes zero to groundedness, not partial credit."""
        verifications = [
            ClaimVerification(claim_id="claim_001", verdict=ClaimVerdict.SUPPORTED, confidence=0.95),
            ClaimVerification(claim_id="claim_002", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE, confidence=0.5),
        ]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-1")
        assert scores["groundedness"].value == 0.5

    def test_calculate_scores_contradicted_claims(self) -> None:
        """Test scoring with contradicted claims."""
        verifications = [
            ClaimVerification(claim_id="claim_001", verdict=ClaimVerdict.CONTRADICTED, confidence=0.95),
        ]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-1")
        assert scores["groundedness"].value == 0.0
        assert scores["hallucination_risk"].value == 1.0
        assert scores["contradiction_risk"].value == 1.0

    def test_calculate_scores_not_applicable_claims_are_excluded(self) -> None:
        """NOT_APPLICABLE claims are excluded from the applicable denominator."""
        verifications = [
            ClaimVerification(claim_id="claim_001", verdict=ClaimVerdict.SUPPORTED, confidence=0.95),
            ClaimVerification(claim_id="claim_002", verdict=ClaimVerdict.NOT_APPLICABLE, confidence=1.0),
        ]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-1")
        assert scores["groundedness"].value == 1.0

    def test_calibration_class_is_none_sentinel_with_no_applicable_claims(self) -> None:
        """With zero applicable claims (no judge attempt happened), calibration_class is 'NONE'."""
        scores = self.engine.calculate_scores([])
        assert scores["groundedness"].calibration_class == "NONE"

    def test_calibration_class_propagates_to_claim_ratio_scores(self) -> None:
        verifications = [ClaimVerification(claim_id="c1", verdict=ClaimVerdict.SUPPORTED)]
        scores = self.engine.calculate_scores(verifications, calibration_class="class-xyz")
        assert scores["groundedness"].calibration_class == "class-xyz"
        assert scores["hallucination_risk"].calibration_class == "class-xyz"
        assert scores["contradiction_risk"].calibration_class == "class-xyz"

    def test_citation_support_not_applicable_when_no_citations(self) -> None:
        """No citations supplied -> NOT_APPLICABLE, not an optimistic default of 1.0."""
        scores = self.engine.calculate_scores([], citation_support_score=None)
        assert scores["citation_support"].status == ScoreStatus.NOT_APPLICABLE
        assert scores["citation_support"].value is None

    def test_citation_support_measured_when_supplied(self) -> None:
        scores = self.engine.calculate_scores([], citation_support_score=0.6)
        assert scores["citation_support"].status == ScoreStatus.MEASURED
        assert scores["citation_support"].value == 0.6
        assert scores["citation_support"].calibration_class == "NONE"

    def test_tool_correctness_not_applicable_when_no_tools(self) -> None:
        scores = self.engine.calculate_scores([], tool_correctness_score=None)
        assert scores["tool_correctness"].status == ScoreStatus.NOT_APPLICABLE

    def test_tool_correctness_measured_when_supplied(self) -> None:
        scores = self.engine.calculate_scores([], tool_correctness_score=0.5)
        assert scores["tool_correctness"].value == 0.5

    def test_scope_instruction_confidence_are_always_unavailable(self) -> None:
        """No accepted v0.1 formula exists for these -- never fabricate one."""
        scores = self.engine.calculate_scores([])
        assert scores["scope_breach"].status == ScoreStatus.UNAVAILABLE
        assert scores["scope_breach"].value is None
        assert scores["instruction_adherence"].status == ScoreStatus.UNAVAILABLE
        assert scores["confidence_alignment"].status == ScoreStatus.UNAVAILABLE


class TestDeriveCalibrationClass:
    """Tests for ADR-010's calibration class derivation."""

    def _basis(self, **overrides):
        base = dict(
            evaluator_id="reliability-verifier",
            evaluator_version="0.1",
            provider_id="rule-based",
            pinned_model_id="rule-based",
            configuration_version="0.1",
            qualification_status="UNQUALIFIED",
        )
        base.update(overrides)
        return base

    def test_deterministic_for_same_inputs(self) -> None:
        a = derive_calibration_class(**self._basis())
        b = derive_calibration_class(**self._basis())
        assert a == b

    def test_changes_when_provider_changes(self) -> None:
        a = derive_calibration_class(**self._basis())
        b = derive_calibration_class(**self._basis(provider_id="mock"))
        assert a != b

    def test_changes_when_qualification_status_changes(self) -> None:
        a = derive_calibration_class(**self._basis())
        b = derive_calibration_class(**self._basis(qualification_status="QUALIFIED"))
        assert a != b

    def test_changes_when_configuration_version_changes(self) -> None:
        a = derive_calibration_class(**self._basis())
        b = derive_calibration_class(**self._basis(configuration_version="0.2"))
        assert a != b


class TestDetermineVerdict:
    """Tests for scoring.md's verdict precedence."""

    def test_no_applicable_claims_abstains(self) -> None:
        assert determine_verdict([]) == OverallVerdict.ABSTAIN

    def test_all_not_applicable_abstains(self) -> None:
        claims = [ClaimVerification(claim_id="c1", verdict=ClaimVerdict.NOT_APPLICABLE)]
        assert determine_verdict(claims) == OverallVerdict.ABSTAIN

    def test_all_insufficient_evidence_abstains(self) -> None:
        claims = [
            ClaimVerification(claim_id="c1", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE),
            ClaimVerification(claim_id="c2", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE),
        ]
        assert determine_verdict(claims) == OverallVerdict.ABSTAIN

    def test_any_contradicted_fails(self) -> None:
        claims = [
            ClaimVerification(claim_id="c1", verdict=ClaimVerdict.SUPPORTED),
            ClaimVerification(claim_id="c2", verdict=ClaimVerdict.CONTRADICTED),
        ]
        assert determine_verdict(claims) == OverallVerdict.FAIL

    def test_all_supported_passes(self) -> None:
        claims = [
            ClaimVerification(claim_id="c1", verdict=ClaimVerdict.SUPPORTED),
            ClaimVerification(claim_id="c2", verdict=ClaimVerdict.SUPPORTED),
        ]
        assert determine_verdict(claims) == OverallVerdict.PASS

    def test_mixture_with_unsupported_is_partial(self) -> None:
        claims = [
            ClaimVerification(claim_id="c1", verdict=ClaimVerdict.SUPPORTED),
            ClaimVerification(claim_id="c2", verdict=ClaimVerdict.UNSUPPORTED),
        ]
        assert determine_verdict(claims) == OverallVerdict.PARTIAL

    def test_mixture_with_insufficient_evidence_is_partial(self) -> None:
        claims = [
            ClaimVerification(claim_id="c1", verdict=ClaimVerdict.SUPPORTED),
            ClaimVerification(claim_id="c2", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE),
        ]
        assert determine_verdict(claims) == OverallVerdict.PARTIAL
