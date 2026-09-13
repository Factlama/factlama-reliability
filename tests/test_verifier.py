"""Tests for the main Verifier."""

from core import verify
from core.verifier import Verifier
from judges.port import JudgeProvider, JudgeRequest, JudgeResult
from judges.providers import MockModelProvider, RuleBasedProvider
from schemas.citation import Citation
from schemas.claims import ClaimVerdict
from schemas.evidence import Evidence
from schemas.instruction import Instruction, InstructionType, Priority
from schemas.policy import (
    ActionPolicy,
    CitationPolicy,
    GroundingPolicy,
    HallucinationPolicy,
    Policy,
    PolicyAction,
    ScopePolicy,
    ToolPolicy,
)
from schemas.tools import ToolExecution, ToolStatus
from schemas.verification import OverallVerdict, ResultStatus, ScoreStatus, VerificationRequest


def _request(**overrides) -> VerificationRequest:
    """A minimal valid VerificationRequest, project/application defaulted."""
    kwargs = {
        "request_id": "req_001",
        "project_id": "proj_test",
        "application_id": "app_test",
        "question": "Question?",
        "answer": "Answer.",
    }
    kwargs.update(overrides)
    return VerificationRequest(**kwargs)


class TestVerifier:
    """Tests for Verifier class."""

    def test_verifier_initialization(self) -> None:
        """Test verifier can be initialized."""
        verifier = Verifier()
        assert verifier is not None
        assert verifier.model_provider is not None

    def test_verifier_with_custom_provider(self) -> None:
        """Test verifier with custom provider."""
        provider = MockModelProvider()
        verifier = Verifier(model_provider=provider)
        assert verifier.model_provider == provider

    def test_verify_simple_supported(self) -> None:
        """Test verification with clearly supported claim."""
        verifier = Verifier()
        request = _request(
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
            evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
        )
        result = verifier.verify(request, tenant_id="tenant_test")
        assert result.request_id == "req_001"
        assert result.tenant_id == "tenant_test"
        assert result.trace_id is not None
        assert result.verdict in (OverallVerdict.PASS, OverallVerdict.PARTIAL)
        assert result.scores["groundedness"].value > 0.5

    def test_verify_simple_unsupported(self) -> None:
        """Test verification with unsupported claim."""
        verifier = Verifier()
        request = _request(
            question="When was Company X founded?",
            answer="Company X was founded in 2018 by John Smith.",
            evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
        )
        result = verifier.verify(request)
        # Should detect partial support (founded in 2018) and unsupported (by John Smith)
        assert result.verdict in (OverallVerdict.PARTIAL, OverallVerdict.FAIL)

    def test_verify_no_evidence(self) -> None:
        """Without evidence, the judge cannot confirm a claim -- it abstains, honestly.

        This replaces the old "no context = perfect groundedness by default"
        optimistic behavior: a claim-ratio score never invents support.
        """
        verifier = Verifier()
        request = _request(
            question="When was Company X founded?", answer="Company X was founded in 2018."
        )
        result = verifier.verify(request)
        assert result.verdict == OverallVerdict.ABSTAIN
        assert result.status == ResultStatus.ABSTAINED
        assert result.scores["groundedness"].status == ScoreStatus.MEASURED
        assert result.scores["groundedness"].value == 0.0

    def test_verify_with_policy(self) -> None:
        """Test verification with policy constraints."""
        verifier = Verifier()
        policy = Policy(
            id="strict",
            grounding=GroundingPolicy(minimum=0.95),
            hallucination=HallucinationPolicy(maximum=0.05),
            actions=ActionPolicy(on_failure=PolicyAction.REGENERATE),
        )
        request = _request(
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
            evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
            policy=policy,
        )
        result = verifier.verify(request)
        assert result.policy_action is not None
        assert result.policy_version == "0.1"

    def test_verify_result_structure(self) -> None:
        """Test that result has all expected fields."""
        verifier = Verifier()
        result = verifier.verify(_request())

        assert result.request_id == "req_001"
        assert result.tenant_id == "default"
        assert result.evaluation_id.startswith("eval_")
        assert result.trace_id is not None
        assert result.schema_version == "0.1"
        assert result.verdict is not None
        assert result.status is not None
        assert result.scores is not None
        assert result.scoring_version == "0.1"
        assert result.provenance is not None
        assert isinstance(result.claims, list)
        assert isinstance(result.violations, list)


class TestVerifyFunction:
    """Tests for the convenience verify() function."""

    def test_verify_function_basic(self) -> None:
        """Test basic verify function usage."""
        result = verify(
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
            evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
        )
        assert result is not None
        assert result.request_id is not None
        assert result.verdict is not None

    def test_verify_function_auto_request_id(self) -> None:
        """Test that request ID is auto-generated."""
        result = verify(question="Question?", answer="Answer.")
        assert result.request_id.startswith("req_")

    def test_verify_function_with_custom_id(self) -> None:
        """Test with custom request ID."""
        result = verify(question="Question?", answer="Answer.", request_id="custom_001")
        assert result.request_id == "custom_001"

    def test_verify_function_with_tenant_id(self) -> None:
        """Test verify function with tenant_id."""
        result = verify(question="Question?", answer="Answer.", tenant_id="tenant_acme")
        assert result.tenant_id == "tenant_acme"


class TestGoldenCases:
    """Golden test cases from the specification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.verifier = Verifier(model_provider=RuleBasedProvider())

    def test_golden_supported(self) -> None:
        """Context: "Product X weighs 2.4 kg." Answer: same. Expected: PASS."""
        result = self.verifier.verify(
            _request(
                request_id="golden_001",
                question="What does Product X weigh?",
                answer="Product X weighs 2.4 kg.",
                evidence=[Evidence(id="doc_001", extracted_text="Product X weighs 2.4 kg.")],
            )
        )
        assert result.verdict == OverallVerdict.PASS

    def test_golden_partial(self) -> None:
        """Context: "Product X weighs 2.4 kg." Answer adds an unsupported clause. Expected: PARTIAL."""
        result = self.verifier.verify(
            _request(
                request_id="golden_002",
                question="What does Product X weigh?",
                answer="Product X weighs 2.4 kg and costs $500.",
                evidence=[Evidence(id="doc_001", extracted_text="Product X weighs 2.4 kg.")],
            )
        )
        assert result.verdict == OverallVerdict.PARTIAL

    def test_golden_contradiction(self) -> None:
        """Context: "Product X weighs 2.4 kg." Answer: "3.4 kg." Expected: FAIL."""
        result = self.verifier.verify(
            _request(
                request_id="golden_003",
                question="What does Product X weigh?",
                answer="Product X weighs 3.4 kg.",
                evidence=[Evidence(id="doc_001", extracted_text="Product X weighs 2.4 kg.")],
            )
        )
        assert result.verdict == OverallVerdict.FAIL
        assert result.scores["contradiction_risk"].value == 1.0

    def test_golden_insufficient_evidence(self) -> None:
        """No relevant evidence for the answer's claim -> not SUPPORTED, not ABSTAIN (still a
        checkable claim), so PARTIAL under scoring.md's precedence."""
        result = self.verifier.verify(
            _request(
                request_id="golden_004",
                question="What is the company's future revenue?",
                answer="Revenue will be $10 billion.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X makes widgets.")],
            )
        )
        assert result.verdict == OverallVerdict.PARTIAL

    def test_golden_multiple_claims_mixed(self) -> None:
        """Answer: two claims, one supported one not. Expected: PARTIAL."""
        result = self.verifier.verify(
            _request(
                request_id="golden_005",
                question="Tell me about Company X.",
                answer="Company X was founded in 2018. It has 5000 employees.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
            )
        )
        assert result.verdict == OverallVerdict.PARTIAL
        assert len([c for c in result.claims if c.verdict == ClaimVerdict.UNSUPPORTED]) >= 1


class TestErrorHandling:
    """Tests for error handling in verification."""

    def test_verification_error_returns_failed_abstain(self) -> None:
        """An unexpected pipeline failure never leaks internals into the public result."""
        from core.claims import ClaimExtractor

        class ErrorExtractor(ClaimExtractor):
            def extract(self, answer: str):
                raise RuntimeError("Simulated error with a secret detail")

        verifier = Verifier(claim_extractor=ErrorExtractor())
        result = verifier.verify(_request())

        assert result.status == ResultStatus.FAILED
        assert result.verdict == OverallVerdict.ABSTAIN
        assert result.abstention_reason is not None
        assert result.metadata.get("failure") == "internal pipeline error"
        assert "error" not in result.metadata
        assert "Simulated error" not in str(result.metadata)
        assert len(result.provenance.attempts) == 1
        assert result.provenance.attempts[0].outcome.value == "FAILED"


class _MalformedProvider(JudgeProvider):
    """A judge provider that violates the evidence-citation rule, for testing."""

    @property
    def name(self) -> str:
        return "malformed-test-provider"

    def evaluate(self, request: JudgeRequest, deadline: float, cancellation) -> JudgeResult:
        return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence=[])


class TestResponseValidation:
    """Tests that a provider cannot smuggle an unsupported SUPPORTED verdict past the port."""

    def test_supported_without_evidence_downgrades_to_insufficient_evidence(self) -> None:
        """CONTRACTS.md: SUPPORTED without cited evidence is INVALID_RESPONSE, never trusted as-is."""
        verifier = Verifier(model_provider=_MalformedProvider())
        result = verifier.verify(_request(answer="Company X was founded in 2018."))

        assert all(c.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE for c in result.claims)
        assert len(result.provenance.attempts) >= 1
        assert all(a.error == "INVALID_RESPONSE" for a in result.provenance.attempts)
        # Never UNSUPPORTED/FAIL from an infrastructure/response-validation failure.
        assert result.verdict != OverallVerdict.FAIL


class TestTenantIsolation:
    """Tests for multi-tenant isolation."""

    def test_tenant_id_propagated_to_result(self) -> None:
        """Test that a trusted tenant_id argument appears in the result."""
        verifier = Verifier()
        result = verifier.verify(
            _request(question="What is the price?", answer="The price is $100."),
            tenant_id="tenant_acme",
        )
        assert result.tenant_id == "tenant_acme"

    def test_default_tenant_id(self) -> None:
        """Test that default tenant_id is used when not specified."""
        verifier = Verifier()
        result = verifier.verify(_request())
        assert result.tenant_id == "default"

    def test_different_tenants_isolated(self) -> None:
        """Test that verification results are tenant-scoped."""
        verifier = Verifier()

        result_a = verifier.verify(
            _request(
                request_id="req_a",
                question="What is the revenue?",
                answer="Revenue is $1M.",
                evidence=[Evidence(id="doc_a", extracted_text="Revenue is $1M.")],
            ),
            tenant_id="tenant_a",
        )
        result_b = verifier.verify(
            _request(
                request_id="req_b",
                question="What is the revenue?",
                answer="Revenue is $2M.",
                evidence=[Evidence(id="doc_b", extracted_text="Revenue is $2M.")],
            ),
            tenant_id="tenant_b",
        )

        assert result_a.tenant_id == "tenant_a"
        assert result_b.tenant_id == "tenant_b"
        assert result_a.request_id != result_b.request_id


class TestScopeInstructionCitationToolWiring:
    """Tests that scope/instruction/citation/tool signals reach violations (and, where a v0.1
    formula exists, scores)."""

    def setup_method(self) -> None:
        self.verifier = Verifier(model_provider=RuleBasedProvider())

    def test_scope_breach_detected_and_flagged(self) -> None:
        """Forbidden-domain content raises a violation. No v0.1 score formula exists for this
        dimension yet, so `scores["scope_breach"]` stays UNAVAILABLE by design."""
        policy = Policy(id="test", scope=ScopePolicy(enabled=True, forbidden=["invest"]))
        result = self.verifier.verify(
            _request(
                request_id="req_scope_001",
                question="What should I do with my savings?",
                answer="You should invest in Company X stock.",
                policy=policy,
            )
        )
        assert result.scores["scope_breach"].status == ScoreStatus.UNAVAILABLE
        assert any(v.code == "SCOPE_BREACH" for v in result.violations)

    def test_no_scope_breach_when_answer_is_clean(self) -> None:
        """Content outside forbidden domains should not be flagged."""
        policy = Policy(id="test", scope=ScopePolicy(enabled=True, forbidden=["invest"]))
        result = self.verifier.verify(
            _request(
                request_id="req_scope_002",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
                policy=policy,
            )
        )
        assert not any(v.code == "SCOPE_BREACH" for v in result.violations)

    def test_instruction_violation_flags(self) -> None:
        """An unmet instruction raises a violation; instruction_adherence stays UNAVAILABLE."""
        instruction = Instruction(
            id="inst_json",
            text="Return JSON only.",
            type=InstructionType.FORMAT,
            priority=Priority.HIGH,
        )
        result = self.verifier.verify(
            _request(
                request_id="req_instr_001",
                question="Give me the data.",
                answer="Here is the data in plain text, not JSON.",
                instructions=[instruction],
            )
        )
        assert result.scores["instruction_adherence"].status == ScoreStatus.UNAVAILABLE
        violation = next(v for v in result.violations if v.code == "INSTRUCTION_VIOLATION")
        assert violation.severity == "high"
        assert violation.metadata["instruction_id"] == "inst_json"

    def test_instruction_followed_raises_no_violation(self) -> None:
        """A satisfied instruction should not be flagged."""
        instruction = Instruction(
            id="inst_json", text="Return JSON only.", type=InstructionType.FORMAT
        )
        result = self.verifier.verify(
            _request(
                request_id="req_instr_002",
                question="Give me the data.",
                answer='{"key": "value"}',
                instructions=[instruction],
            )
        )
        assert not any(v.code == "INSTRUCTION_VIOLATION" for v in result.violations)

    def test_citation_to_unknown_source_is_flagged(self) -> None:
        """A citation pointing at a source_id absent from evidence is invalid."""
        result = self.verifier.verify(
            _request(
                request_id="req_cite_001",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
                citations=[Citation(id="cite_1", source_id="doc_999")],
            )
        )
        assert result.scores["citation_support"].value < 1.0
        violation = next(v for v in result.violations if v.code == "CITATION_MISMATCH")
        assert violation.evidence_id == "doc_999"

    def test_citation_to_real_source_that_does_not_back_the_claim_is_flagged(self) -> None:
        """A citation to a real source that isn't the claim's supporting evidence is invalid."""
        result = self.verifier.verify(
            _request(
                request_id="req_cite_003",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[
                    Evidence(id="doc_001", extracted_text="Company X was founded in 2018."),
                    Evidence(id="doc_002", extracted_text="Company X makes widgets."),
                ],
                citations=[Citation(id="cite_1", claim_id="claim_001", source_id="doc_002")],
            )
        )
        assert result.scores["citation_support"].value < 1.0
        violation = next(v for v in result.violations if v.code == "CITATION_MISMATCH")
        assert violation.claim_id == "claim_001"
        assert "does not support" in violation.message

    def test_citation_to_known_source_is_valid(self) -> None:
        """A citation pointing at real evidence should score as fully supported."""
        result = self.verifier.verify(
            _request(
                request_id="req_cite_002",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
                citations=[Citation(id="cite_1", source_id="doc_001")],
            )
        )
        assert result.scores["citation_support"].value == 1.0
        assert not any(v.code == "CITATION_MISMATCH" for v in result.violations)

    def test_tool_error_flags_and_forces_failure_action(self) -> None:
        """A failed tool execution lowers tool_correctness, raises a violation, and (per the
        ToolPolicy.fail_on_error default of True) forces the on_failure policy action -- without
        overriding the independently-computed factual verdict."""
        result = self.verifier.verify(
            _request(
                request_id="req_tool_001",
                question="What is the weather?",
                answer="The weather is sunny.",
                tool_executions=[
                    ToolExecution(
                        id="tool_1",
                        tool_name="weather_api",
                        status=ToolStatus.ERROR,
                        error_message="API timeout",
                    )
                ],
            )
        )
        assert result.scores["tool_correctness"].value == 0.0
        violation = next(v for v in result.violations if v.code == "TOOL_ERROR")
        assert violation.message == "API timeout"
        assert result.policy_action == PolicyAction.FAIL.value

    def test_tool_error_with_fail_on_error_disabled_does_not_force_failure_action(self) -> None:
        """Explicitly disabling fail_on_error should let the action follow the verdict normally."""
        policy = Policy(id="lenient-tools", tools=ToolPolicy(fail_on_error=False))
        result = self.verifier.verify(
            _request(
                request_id="req_tool_003",
                question="What is the weather?",
                answer="The weather is sunny.",
                tool_executions=[
                    ToolExecution(id="tool_1", tool_name="weather_api", status=ToolStatus.ERROR)
                ],
                policy=policy,
            )
        )
        assert result.policy_action != PolicyAction.FAIL.value

    def test_citations_required_but_none_supplied(self) -> None:
        """A policy requiring citations should flag a request with none."""
        policy = Policy(id="cite-required", citations=CitationPolicy(required=True))
        result = self.verifier.verify(
            _request(
                request_id="req_cite_004",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")],
                policy=policy,
            )
        )
        assert any(v.code == "CITATION_MISMATCH" for v in result.violations)

    def test_successful_tools_keep_perfect_correctness(self) -> None:
        """Successful tool executions should not be penalized."""
        result = self.verifier.verify(
            _request(
                request_id="req_tool_002",
                question="What is the weather?",
                answer="The weather is sunny.",
                tool_executions=[
                    ToolExecution(id="tool_1", tool_name="weather_api", status=ToolStatus.SUCCESS)
                ],
            )
        )
        assert result.scores["tool_correctness"].value == 1.0
        assert not any(v.code == "TOOL_ERROR" for v in result.violations)


class TestConflictingEvidence:
    """Tests for conflicting evidence detection."""

    def test_conflicting_evidence_detected(self) -> None:
        """Test that conflicting evidence is identified."""
        verifier = Verifier(model_provider=RuleBasedProvider())

        request = _request(
            request_id="req_conflict_001",
            question="What is the population?",
            answer="The population is 5 million.",
            evidence=[
                Evidence(id="doc_001", extracted_text="The population is 5 million."),
                Evidence(id="doc_002", extracted_text="The population is 7 million."),
            ],
        )
        result = verifier.verify(request)

        assert result.verdict in (OverallVerdict.PASS, OverallVerdict.PARTIAL, OverallVerdict.FAIL)
        for claim in result.claims:
            assert hasattr(claim, "conflicting_evidence")
            assert isinstance(claim.conflicting_evidence, list)

    def test_claim_verification_with_explicit_conflict(self) -> None:
        """Test explicit conflict in claim verification."""
        from schemas.claims import ClaimVerdict, ClaimVerification, EvidenceReference

        verification = ClaimVerification(
            claim_id="claim_001",
            verdict=ClaimVerdict.CONTRADICTED,
            confidence=0.8,
            evidence=[EvidenceReference(evidence_id="doc_001", support=0.9, relevance=0.95)],
            conflicting_evidence=[
                EvidenceReference(evidence_id="doc_002", support=-0.8, relevance=0.9)
            ],
            reason="Evidence sources conflict",
        )

        assert verification.verdict == ClaimVerdict.CONTRADICTED
        assert len(verification.evidence) == 1
        assert len(verification.conflicting_evidence) == 1
        assert verification.conflicting_evidence[0].evidence_id == "doc_002"
