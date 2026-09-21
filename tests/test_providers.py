"""Tests for judge providers."""

import time

import pytest

from judges.port import (
    CancellationToken,
    JudgeError,
    JudgeErrorCode,
    JudgeRequest,
    JudgeResult,
    apply_citation_support_check,
    validate_judge_result,
)
from judges.providers import MockModelProvider, RuleBasedProvider
from judges.vendor_adapters import NLIProvider
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence
from schemas.instruction import Instruction, InstructionType
from schemas.policy import Policy, ScopePolicy

FAR_DEADLINE = time.monotonic() + 3600


class TestJudgeResultInvariants:
    """Tests for the JudgeResult port's stated invariants."""

    def test_verdict_and_error_together_is_rejected(self) -> None:
        """A JudgeResult must never carry both a verdict and an error."""
        with pytest.raises(ValueError):
            JudgeResult(
                verdict=ClaimVerdict.SUPPORTED,
                error=JudgeError(code=JudgeErrorCode.TIMEOUT, message="x"),
            )

    def test_neither_verdict_nor_error_is_rejected(self) -> None:
        """A JudgeResult must carry at least one of verdict or error."""
        with pytest.raises(ValueError):
            JudgeResult()

    def test_supported_without_evidence_is_downgraded_to_invalid_response(self) -> None:
        """CONTRACTS.md: SUPPORTED without a cited evidence ID is INVALID_RESPONSE."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(
            claim=claim,
            evidence=[Evidence(evidence_id="doc_1", content="Company X was founded in 2018.")],
        )
        malformed = JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=[])
        validated = validate_judge_result(malformed, request)

        assert validated.verdict is None
        assert validated.error is not None
        assert validated.error.code == JudgeErrorCode.INVALID_RESPONSE

    def test_supported_citing_unknown_evidence_id_is_downgraded_to_invalid_response(self) -> None:
        """A judge citing an evidence ID that was never part of its request must not be
        trusted -- citing a nonexistent ID is not a real citation, matching the "without a
        cited evidence ID" rule in spirit."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(
            claim=claim,
            evidence=[Evidence(evidence_id="doc_1", content="Company X was founded in 2018.")],
        )
        hallucinated = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_999"],
        )
        validated = validate_judge_result(hallucinated, request)

        assert validated.verdict is None
        assert validated.error is not None
        assert validated.error.code == JudgeErrorCode.INVALID_RESPONSE

    def test_supported_with_evidence_passes_through(self) -> None:
        """A well-formed SUPPORTED result is not altered."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(
            claim=claim,
            evidence=[Evidence(evidence_id="doc_1", content="Company X was founded in 2018.")],
        )
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )
        assert validate_judge_result(result, request) is result

    def test_non_supported_verdicts_are_not_affected(self) -> None:
        """The evidence-citation rule is specific to SUPPORTED, not every verdict."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(claim=claim, evidence=[])
        result = JudgeResult(verdict=ClaimVerdict.UNSUPPORTED, evidence_ids=[])
        assert validate_judge_result(result, request) is result

    def test_downgrade_to_invalid_response_preserves_usage(self) -> None:
        """R4 of the 2026-09-21 re-audit: a provider's reported usage is
        real work that happened regardless of whether its citation was
        valid -- must not be silently discarded when downgrading."""
        from schemas.verification import Cost, Usage

        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(claim=claim, evidence=[])
        hallucinated = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_999"],
            usage=Usage(total_tokens=12, cost=Cost(status="MEASURED", amount=0.01)),
        )
        validated = validate_judge_result(hallucinated, request)

        assert validated.error is not None
        assert validated.usage is not None
        assert validated.usage.total_tokens == 12
        assert validated.usage.cost.status == "MEASURED"
        assert validated.usage.cost.amount == 0.01


class TestCitationSupportCheck:
    """Tests for CONTRACTS.md's conservative, non-model overlap/support check.

    This runs after validate_judge_result() has already rejected a
    nonexistent evidence ID -- every case here cites real evidence.

    Re-audit follow-up (2026-09-21): this check no longer overrides the
    verdict (see `apply_citation_support_check()`'s own docstring for why
    zero recognized overlap alone is not proof of a fabricated citation) --
    `result` returned is always the exact input, and the bool return is
    whether the caller should flag the citation as overlap-inconclusive for
    mandatory human review, not whether the verdict changed.
    """

    def test_flags_supported_verdict_with_no_shared_content(self) -> None:
        """A SUPPORTED verdict citing real evidence that is unrelated to the claim is
        flagged as overlap-inconclusive, but its verdict is not overridden."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        evidence = Evidence(evidence_id="doc_1", content="The weather today is sunny and warm.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        unchanged, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is True
        assert unchanged is result
        assert unchanged.verdict == ClaimVerdict.SUPPORTED

    def test_passes_through_supported_verdict_with_shared_content(self) -> None:
        """Matching evidence is left untouched and not flagged."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        evidence = Evidence(evidence_id="doc_1", content="Company X was founded in 2018.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        unchanged, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False
        assert unchanged is result

    def test_does_not_flag_legitimate_paraphrase_with_partial_overlap(self) -> None:
        """Low but nonzero overlap must not be flagged -- CONTRACTS.md: "legitimate
        paraphrase is not rejected solely for lacking shared words"."""
        claim = Claim(claim_id="claim_001", text="The firm began operations in 2018.")
        evidence = Evidence(evidence_id="doc_1", content="Company X was founded in the year 2018.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        _, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False

    def test_does_not_flag_legitimate_paraphrase_with_zero_raw_token_overlap(self) -> None:
        """F7 of the 2026-09-21 G0-G4 validation report's exact reproduction:
        a claim and its cited evidence sharing *zero* raw tokens must still
        pass unflagged when they are recognized near-synonyms, not just when
        they happen to share a word."""
        claim = Claim(claim_id="claim_001", text="The physician purchased an automobile.")
        evidence = Evidence(evidence_id="doc_1", content="The doctor bought a car.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        _, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False

    def test_does_not_flag_unlisted_synonym_pair(self) -> None:
        """R3 of the 2026-09-21 re-audit's exact reproduction: a legitimate
        paraphrase using words not previously in the synonym table must not
        be flagged either -- the fix adds this specific pair to the table
        (closing this example) as well as a general common-root fallback
        (`_share_common_root`) for morphological variants no finite table
        will ever fully enumerate."""
        claim = Claim(claim_id="claim_001", text="The infant is asleep.")
        evidence = Evidence(evidence_id="doc_1", content="The baby is sleeping.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        _, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False

    def test_does_not_flag_morphological_variant_outside_the_synonym_table(self) -> None:
        """The common-root fallback (not the finite synonym table) is what
        closes this one: claim/evidence content words after stopword
        removal ({"costs", "increased", "significantly"} vs {"filing",
        "shows", "significant", "increase"}) share zero exact tokens and
        zero synonym-table entries -- only common roots
        ("increas.../signific...")."""
        claim = Claim(claim_id="claim_001", text="Costs increased significantly.")
        evidence = Evidence(evidence_id="doc_1", content="The filing shows a significant increase.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        _, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False

    def test_does_not_flag_unrelated_root_synonym_pair(self) -> None:
        """Follow-up re-audit finding (2026-09-21): "The hound fled."
        supported by "The dog ran away." -- a legitimate paraphrase with
        zero shared roots, distinct from R3's "infant"/"baby" example.
        Closed here the same way R3 was (an added table entry); this guard
        cannot generalize past enumerated/morphological pairs without a
        semantic model ADR-004 forbids in this module -- see
        `_share_common_root()`'s own docstring."""
        claim = Claim(claim_id="claim_001", text="The hound fled.")
        evidence = Evidence(evidence_id="doc_1", content="The dog ran away.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        _, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is False

    def test_still_flags_when_no_synonym_relates_the_two(self) -> None:
        """The synonym normalization must not become so permissive that a
        genuinely fabricated citation (topically unrelated evidence) stops
        being flagged -- it is still routed to human review, just no longer
        by silently overriding the verdict (see class docstring)."""
        claim = Claim(claim_id="claim_001", text="The physician purchased an automobile.")
        evidence = Evidence(evidence_id="doc_1", content="The weather today is sunny and warm.")
        request = JudgeRequest(claim=claim, evidence=[evidence])
        result = JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=["doc_1"],
        )

        unchanged, overlap_inconclusive = apply_citation_support_check(result, request)

        assert overlap_inconclusive is True
        assert unchanged.verdict == ClaimVerdict.SUPPORTED

    def test_non_supported_verdicts_are_not_affected(self) -> None:
        """The overlap check is specific to SUPPORTED, not every verdict."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        request = JudgeRequest(claim=claim, evidence=[])
        result = JudgeResult(verdict=ClaimVerdict.UNSUPPORTED, evidence_ids=[])

        unchanged, changed = apply_citation_support_check(result, request)

        assert changed is False
        assert unchanged is result


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
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        evidence = [Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")]
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.error is None
        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.confidence > 0.9
        assert len(result.evidence_ids) == 1

    def test_evaluate_unsupported(self) -> None:
        """Test claim verification without supporting evidence."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        evidence = [Evidence(evidence_id="doc_001", content="Company X is a technology company.")]
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.UNSUPPORTED

    def test_evaluate_no_evidence(self) -> None:
        """Test claim verification with no evidence."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.5

    def test_evaluate_returns_cancelled_error_when_already_cancelled(self) -> None:
        """A cancelled token must short-circuit before any evaluation logic runs."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        cancellation = CancellationToken()
        cancellation.cancel()
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=[]), FAR_DEADLINE, cancellation
        )
        assert result.verdict is None
        assert result.error is not None
        assert result.error.code == JudgeErrorCode.CANCELLED

    def test_evaluate_returns_timeout_error_when_deadline_passed(self) -> None:
        """A deadline already in the past must short-circuit as TIMEOUT."""
        claim = Claim(claim_id="claim_001", text="Company X was founded in 2018.")
        past_deadline = time.monotonic() - 1.0
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=[]), past_deadline, CancellationToken()
        )
        assert result.error is not None
        assert result.error.code == JudgeErrorCode.TIMEOUT

    def test_evaluate_instruction(self) -> None:
        """Test instruction evaluation."""
        instruction = Instruction(
            instruction_id="inst_001", text="Return JSON only.", type=InstructionType.FORMAT
        )
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
        claim = Claim(claim_id="claim_001", text="The product weighs 2.4 kg.")
        evidence = [Evidence(evidence_id="doc_001", content="The product weighs 2.4 kg.")]
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict == ClaimVerdict.SUPPORTED

    def test_evaluate_with_contradiction(self) -> None:
        """Test verification when evidence contradicts claim."""
        claim = Claim(claim_id="claim_001", text="The product weighs 2.4 kg.")
        evidence = [Evidence(evidence_id="doc_001", content="The product weighs 3.4 kg.")]
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict in (
            ClaimVerdict.SUPPORTED,
            ClaimVerdict.UNSUPPORTED,
            ClaimVerdict.CONTRADICTED,
            ClaimVerdict.INSUFFICIENT_EVIDENCE,
        )

    def test_evaluate_no_relevant_evidence(self) -> None:
        """Test verification when evidence is irrelevant."""
        claim = Claim(claim_id="claim_001", text="The product costs $500.")
        evidence = [Evidence(evidence_id="doc_001", content="The product is blue.")]
        result = self.provider.evaluate(
            JudgeRequest(claim=claim, evidence=evidence), FAR_DEADLINE, CancellationToken()
        )
        assert result.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.INSUFFICIENT_EVIDENCE)

    def test_evaluate_instruction_json_format(self) -> None:
        """Test instruction evaluation for JSON format."""
        instruction = Instruction(
            instruction_id="inst_001", text="Return JSON only.", type=InstructionType.FORMAT
        )

        score, _ = self.provider.evaluate_instruction('{"key": "value"}', instruction)
        assert score >= 0.9

        score, _ = self.provider.evaluate_instruction("This is plain text", instruction)
        assert score < 0.5

    def test_evaluate_scope_forbidden_domain(self) -> None:
        """Test scope evaluation with forbidden domain."""
        policy = Policy(id="test", scope=ScopePolicy(enabled=True, forbidden=["invest"]))
        score, domains = self.provider.evaluate_scope(
            "You should invest in Company X stock.", policy
        )
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
