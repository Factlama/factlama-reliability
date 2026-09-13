"""Tests for judge providers."""

import time

import pytest

from judges.providers import (
    CancellationToken,
    JudgeErrorCode,
    JudgeRequest,
    MockModelProvider,
    NLIProvider,
    RuleBasedProvider,
)
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence
from schemas.instruction import Instruction, InstructionType
from schemas.policy import Policy, ScopePolicy

FAR_DEADLINE = time.monotonic() + 3600


class TestMockModelProvider:
    """Tests for MockModelProvider."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = MockModelProvider()

    def test_provider_name(self) -> None:
        """Test provider name."""
        assert self.provider.name == "mock"

    def test_evaluate_supported(self) -> None:
        """Test claim verification with supporting evidence."""
        claim = Claim(id="claim_001", text="Company X was founded in 2018.")
        evidence = [Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")]
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken())
        assert result.error is None
        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.confidence > 0.9
        assert len(result.evidence) == 1

    def test_evaluate_unsupported(self) -> None:
        """Test claim verification without supporting evidence."""
        claim = Claim(id="claim_001", text="Company X was founded in 2018.")
        evidence = [Evidence(id="doc_001", extracted_text="Company X is a technology company.")]
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken())
        assert result.verdict == ClaimVerdict.UNSUPPORTED

    def test_evaluate_no_evidence(self) -> None:
        """Test claim verification with no evidence."""
        claim = Claim(id="claim_001", text="Company X was founded in 2018.")
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, CancellationToken())
        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.5

    def test_evaluate_returns_cancelled_error_when_already_cancelled(self) -> None:
        """A cancelled token must short-circuit before any evaluation logic runs."""
        claim = Claim(id="claim_001", text="Company X was founded in 2018.")
        cancellation = CancellationToken()
        cancellation.cancel()
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, cancellation)
        assert result.verdict is None
        assert result.error is not None
        assert result.error.code == JudgeErrorCode.CANCELLED

    def test_evaluate_returns_timeout_error_when_deadline_passed(self) -> None:
        """A deadline already in the past must short-circuit as TIMEOUT."""
        claim = Claim(id="claim_001", text="Company X was founded in 2018.")
        past_deadline = time.monotonic() - 1.0
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=[]), past_deadline, CancellationToken())
        assert result.error is not None
        assert result.error.code == JudgeErrorCode.TIMEOUT

    def test_evaluate_instruction(self) -> None:
        """Test instruction evaluation."""
        instruction = Instruction(id="inst_001", text="Return JSON only.", type=InstructionType.FORMAT)
        score, reason = self.provider.evaluate_instruction("Some answer", instruction)
        assert score == 1.0
        assert "Mock" in reason

    def test_evaluate_scope(self) -> None:
        """Test scope evaluation."""
        policy = Policy(id="test")
        score, domains = self.provider.evaluate_scope("Some answer", policy)
        assert score == 0.0
        assert len(domains) == 0


class TestRuleBasedProvider:
    """Tests for RuleBasedProvider."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = RuleBasedProvider()

    def test_provider_name(self) -> None:
        """Test provider name."""
        assert self.provider.name == "rule-based"

    def test_evaluate_with_matching_evidence(self) -> None:
        """Test verification when evidence matches claim."""
        claim = Claim(id="claim_001", text="The product weighs 2.4 kg.")
        evidence = [Evidence(id="doc_001", extracted_text="The product weighs 2.4 kg.")]
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken())
        assert result.verdict == ClaimVerdict.SUPPORTED

    def test_evaluate_with_contradiction(self) -> None:
        """Test verification when evidence contradicts claim."""
        claim = Claim(id="claim_001", text="The product weighs 2.4 kg.")
        evidence = [Evidence(id="doc_001", extracted_text="The product weighs 3.4 kg.")]
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken())
        assert result.verdict in (
            ClaimVerdict.SUPPORTED,
            ClaimVerdict.UNSUPPORTED,
            ClaimVerdict.CONTRADICTED,
            ClaimVerdict.INSUFFICIENT_EVIDENCE,
        )

    def test_evaluate_no_relevant_evidence(self) -> None:
        """Test verification when evidence is irrelevant."""
        claim = Claim(id="claim_001", text="The product costs $500.")
        evidence = [Evidence(id="doc_001", extracted_text="The product is blue.")]
        result = self.provider.evaluate(JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken())
        assert result.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.INSUFFICIENT_EVIDENCE)

    def test_evaluate_instruction_json_format(self) -> None:
        """Test instruction evaluation for JSON format."""
        instruction = Instruction(id="inst_001", text="Return JSON only.", type=InstructionType.FORMAT)

        score, _ = self.provider.evaluate_instruction('{"key": "value"}', instruction)
        assert score >= 0.9

        score, _ = self.provider.evaluate_instruction("This is plain text", instruction)
        assert score < 0.5

    def test_evaluate_scope_forbidden_domain(self) -> None:
        """Test scope evaluation with forbidden domain."""
        policy = Policy(id="test", scope=ScopePolicy(enabled=True, forbidden=["invest"]))
        score, domains = self.provider.evaluate_scope("You should invest in Company X stock.", policy)
        assert "invest" in domains
        assert score > 0.0

    def test_evaluate_scope_allowed_domain(self) -> None:
        """Test scope evaluation with allowed content."""
        policy = Policy(id="test", scope=ScopePolicy(enabled=True, forbidden=["invest"]))
        score, domains = self.provider.evaluate_scope("Company X was founded in 2018.", policy)
        assert len(domains) == 0
        assert score == 0.0


class TestNLIProviderLabelResolution:
    """Tests for NLIProvider's model-agnostic label-index resolution.

    This is the fix for a real bug: the code used to assume every NLI
    checkpoint orders its 3 output classes as
    [entailment, neutral, contradiction], which is untrue for some widely
    used checkpoints (e.g. `cross-encoder/nli-deberta-v3-xsmall` orders them
    [contradiction, entailment, neutral]). These tests don't need
    torch/transformers installed since they exercise pure dict logic, not
    model loading.
    """

    def setup_method(self) -> None:
        self.provider = NLIProvider()

    def test_resolves_standard_ordering(self) -> None:
        """e.g. typeform/distilbert-base-uncased-mnli, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli."""
        indices = self.provider._resolve_label_indices(
            {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        )
        assert indices == {"entailment": 0, "neutral": 1, "contradiction": 2}

    def test_resolves_nonstandard_ordering(self) -> None:
        """e.g. cross-encoder/nli-deberta-v3-xsmall."""
        indices = self.provider._resolve_label_indices(
            {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        assert indices == {"entailment": 1, "neutral": 2, "contradiction": 0}

    def test_resolves_string_keys(self) -> None:
        """HF configs sometimes round-trip id2label keys as strings."""
        indices = self.provider._resolve_label_indices(
            {"0": "entailment", "1": "neutral", "2": "contradiction"}
        )
        assert indices == {"entailment": 0, "neutral": 1, "contradiction": 2}

    def test_rejects_non_nli_label_set(self) -> None:
        """A base (non-NLI-finetuned) model's generic labels must be rejected,
        not silently misinterpreted as entailment/neutral/contradiction."""
        with pytest.raises(RuntimeError, match="does not expose recognizable"):
            self.provider._resolve_label_indices({0: "LABEL_0", 1: "LABEL_1"})
