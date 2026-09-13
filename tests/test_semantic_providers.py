"""Tests for EmbeddingProvider and NLIProvider against real downloaded models.

Unlike the rest of the test suite, these exercise actual model inference
(sentence-transformers cosine similarity, transformers NLI classification) --
not mocks -- because that is what the "semantic evidence matching" phase-1
item actually requires verifying. They're skipped automatically when the
optional `embeddings`/`nli` extras aren't installed (`pip install
factlama-reliability[all]`), matching how these providers are meant to be opt-in.

Model weights are cached under ~/.cache/huggingface after the first run.
"""

import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from judges.port import CancellationToken, JudgeRequest
from judges.vendor_adapters import EmbeddingProvider, NLIProvider
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence
from schemas.instruction import Instruction
from schemas.policy import Policy, ScopePolicy

FAR_DEADLINE = time.monotonic() + 3600


@pytest.fixture(scope="module")
def embedding_provider() -> EmbeddingProvider:
    """Shared instance so the sentence-transformer model loads once per run."""
    return EmbeddingProvider()


@pytest.fixture(scope="module")
def nli_provider() -> NLIProvider:
    """Shared instance so the NLI model loads once per run."""
    return NLIProvider()


class TestEmbeddingProviderReal:
    """EmbeddingProvider against the real all-MiniLM-L6-v2 model."""

    def test_name(self, embedding_provider: EmbeddingProvider) -> None:
        assert embedding_provider.name == "embedding:sentence-transformers/all-MiniLM-L6-v2"

    def test_paraphrased_claim_is_supported(self, embedding_provider: EmbeddingProvider) -> None:
        """Different wording, same meaning, should still match via embeddings
        (this is the whole point of semantic over exact-text matching)."""
        claim = Claim(id="c1", text="Company X was founded in 2018.")
        evidence = [
            Evidence(id="doc_001", extracted_text="Company X was established in the year 2018.")
        ]
        result = embedding_provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.confidence > 0.7

    def test_unrelated_claim_is_unsupported(self, embedding_provider: EmbeddingProvider) -> None:
        claim = Claim(id="c2", text="The stock market crashed yesterday.")
        evidence = [
            Evidence(id="doc_001", extracted_text="Company X was established in the year 2018.")
        ]
        result = embedding_provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.UNSUPPORTED

    def test_no_evidence_is_insufficient(self, embedding_provider: EmbeddingProvider) -> None:
        claim = Claim(id="c3", text="Company X was founded in 2018.")
        result = embedding_provider.evaluate(
            JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE

    def test_evaluate_instruction_returns_bounded_score(
        self, embedding_provider: EmbeddingProvider
    ) -> None:
        instruction = Instruction(id="i1", text="Discuss the company's financial history.")
        score, reason = embedding_provider.evaluate_instruction(
            "Company X was established in 2018 and has grown steadily.", instruction
        )
        assert 0.0 <= score <= 1.0
        assert reason

    def test_evaluate_scope_returns_bounded_score(
        self, embedding_provider: EmbeddingProvider
    ) -> None:
        policy = Policy(id="p1", scope=ScopePolicy(enabled=True, forbidden=["cryptocurrency"]))
        breach, domains = embedding_provider.evaluate_scope(
            "You should invest heavily in Bitcoin and other cryptocurrency right now.", policy
        )
        assert 0.0 <= breach <= 1.0
        assert isinstance(domains, list)

    def test_evaluate_scope_disabled_short_circuits(
        self, embedding_provider: EmbeddingProvider
    ) -> None:
        """Scope evaluation should skip model inference entirely when disabled."""
        policy = Policy(id="p1", scope=ScopePolicy(enabled=False, forbidden=["cryptocurrency"]))
        breach, domains = embedding_provider.evaluate_scope("Bitcoin is a cryptocurrency.", policy)
        assert breach == 0.0
        assert domains == []


class TestNLIProviderReal:
    """NLIProvider against the real typeform/distilbert-base-uncased-mnli model."""

    def test_name(self, nli_provider: NLIProvider) -> None:
        assert nli_provider.name == "nli:typeform/distilbert-base-uncased-mnli"

    def test_label_indices_resolved_from_model_config(self, nli_provider: NLIProvider) -> None:
        """The fix under test: indices must come from the model's own config,
        not an assumed fixed position (see TestNLIProviderLabelResolution in
        test_providers.py for the pure-logic version of this check)."""
        nli_provider._get_model()
        assert set(nli_provider._label_indices) == {"entailment", "neutral", "contradiction"}

    def test_entailing_claim_is_supported(self, nli_provider: NLIProvider) -> None:
        claim = Claim(id="c1", text="Company X was founded in 2018.")
        evidence = [
            Evidence(id="doc_001", extracted_text="Company X was founded in 2018 in California.")
        ]
        result = nli_provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.confidence > 0.9

    def test_contradicting_claim_is_contradicted(self, nli_provider: NLIProvider) -> None:
        claim = Claim(id="c2", text="Company X was founded in 2005.")
        evidence = [
            Evidence(id="doc_001", extracted_text="Company X was founded in 2018 in California.")
        ]
        result = nli_provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.CONTRADICTED
        assert result.confidence > 0.9

    def test_unrelated_claim_is_neutral(self, nli_provider: NLIProvider) -> None:
        claim = Claim(id="c3", text="Bananas are a good source of potassium.")
        evidence = [
            Evidence(id="doc_001", extracted_text="Company X was founded in 2018 in California.")
        ]
        result = nli_provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE

    def test_no_evidence_is_insufficient(self, nli_provider: NLIProvider) -> None:
        claim = Claim(id="c4", text="Company X was founded in 2018.")
        result = nli_provider.evaluate(
            JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.5
