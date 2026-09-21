"""Tests for the main Verifier."""

import pytest
from pydantic import ValidationError

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
from schemas.verification import (
    OverallVerdict,
    ResultStatus,
    ScoreStatus,
    Severity,
    VerificationRequest,
)


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
            evidence=[Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")],
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
            evidence=[Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")],
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
            evidence=[Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")],
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
            evidence=[Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")],
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
                evidence=[Evidence(evidence_id="doc_001", content="Product X weighs 2.4 kg.")],
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
                evidence=[Evidence(evidence_id="doc_001", content="Product X weighs 2.4 kg.")],
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
                evidence=[Evidence(evidence_id="doc_001", content="Product X weighs 2.4 kg.")],
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
                evidence=[Evidence(evidence_id="doc_001", content="Company X makes widgets.")],
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
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")
                ],
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
        return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=[])


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


class _UnrelatedCitationProvider(JudgeProvider):
    """Always SUPPORTED, citing whatever real evidence it was given, regardless of
    whether it actually relates to the claim -- for exercising
    `apply_citation_support_check()`'s zero-overlap path end to end."""

    @property
    def name(self) -> str:
        return "unrelated-citation-test-provider"

    def evaluate(self, request: JudgeRequest, deadline: float, cancellation) -> JudgeResult:
        return JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=[e.evidence_id for e in request.evidence],
        )


class TestCitationOverlapInconclusive:
    """End-to-end coverage for the re-audit follow-up (2026-09-21) behavior
    of `apply_citation_support_check()`'s zero-overlap case: the factual
    verdict is never overridden by this deterministic guard alone (see its
    own docstring for why), and a genuine ambiguity is routed to mandatory
    human review (`core.policy`, the same ADR-013 pattern used for
    `EVIDENCE_INJECTION_SUSPECTED`) instead of silently guessed either way.
    """

    def test_zero_overlap_citation_keeps_verdict_but_forces_human_review(self) -> None:
        """A citation with no lexical relation to its claim no longer downgrades
        the verdict to INSUFFICIENT_EVIDENCE -- it stays SUPPORTED, flagged with
        CITATION_OVERLAP_INCONCLUSIVE, and the policy action is forced to
        HUMAN_REVIEW so a person resolves whether it was fabricated."""
        result = Verifier(model_provider=_UnrelatedCitationProvider()).verify(
            _request(
                answer="Company X was founded in 2018.",
                evidence=[
                    Evidence(evidence_id="doc_1", content="The weather today is sunny and warm.")
                ],
            )
        )

        assert all(c.verdict == ClaimVerdict.SUPPORTED for c in result.claims)
        assert result.verdict == OverallVerdict.PASS
        violation = next(v for v in result.violations if v.code == "CITATION_OVERLAP_INCONCLUSIVE")
        assert violation.evidence_ids == ["doc_1"]
        assert result.policy_action == PolicyAction.HUMAN_REVIEW.value

    def test_legitimate_zero_overlap_paraphrase_is_not_flagged_or_reviewed(self) -> None:
        """CONTRACTS.md: "legitimate paraphrase is not rejected solely for lacking
        shared words" -- a recognized paraphrase with zero raw token overlap stays
        SUPPORTED, unflagged, with the normal (not forced) policy action."""
        result = Verifier(model_provider=_UnrelatedCitationProvider()).verify(
            _request(
                answer="The physician purchased an automobile.",
                evidence=[Evidence(evidence_id="doc_1", content="The doctor bought a car.")],
            )
        )

        assert all(c.verdict == ClaimVerdict.SUPPORTED for c in result.claims)
        assert not any(v.code == "CITATION_OVERLAP_INCONCLUSIVE" for v in result.violations)
        assert result.policy_action == PolicyAction.PASS.value


class TestProviderIdentityInAttempts:
    """F2 of the 2026-09-21 G0-G4 validation report: `Attempt.model_id`/
    `pinned_model_version`/`configuration_version`/`calibration_class` must
    reflect the real provider instance's identity -- including its own
    tunable configuration, which `.name` alone does not capture -- not a
    value shared by every differently-configured instance of it. Real
    `EmbeddingProvider` instances at support thresholds 0.70 and 0.95
    previously produced identical `configuration_version`/`calibration_class`
    because only `.name` (constant across both) fed calibration derivation."""

    class _ConfigurableProvider(JudgeProvider):
        """Same `.name` regardless of `support_threshold` -- mirrors
        `EmbeddingProvider`'s own naming convention, where `.name` is
        `f"embedding:{model_name}"` and does not vary with tunable
        thresholds."""

        name = "configurable:model-x"

        def __init__(self, support_threshold: float) -> None:
            self.support_threshold = support_threshold

        def evaluate(self, request: JudgeRequest, deadline: float, cancellation) -> JudgeResult:
            return JudgeResult(
                verdict=ClaimVerdict.SUPPORTED, evidence_ids=[request.evidence[0].evidence_id]
            )

        @property
        def compliance_tags(self) -> frozenset[str]:
            return frozenset({"IN_PROCESS", "NO_EXTERNAL_EGRESS"})

    def test_differently_configured_providers_get_different_identity(self) -> None:
        evidence = [Evidence(evidence_id="e1", content="Paris is the capital of France.")]
        result_a = Verifier(
            model_provider=self._ConfigurableProvider(support_threshold=0.70)
        ).verify(_request(answer="Paris is the capital of France.", evidence=evidence))
        result_b = Verifier(
            model_provider=self._ConfigurableProvider(support_threshold=0.95)
        ).verify(_request(answer="Paris is the capital of France.", evidence=evidence))

        attempt_a = result_a.provenance.attempts[0]
        attempt_b = result_b.provenance.attempts[0]
        assert attempt_a.configuration_version != attempt_b.configuration_version
        assert attempt_a.calibration_class != attempt_b.calibration_class

    def test_model_id_and_pinned_version_populated_when_provider_exposes_them(self) -> None:
        class _PinnedProvider(JudgeProvider):
            name = "embedding:some/model"
            resolved_revision = "abc123deadbeef"

            def evaluate(self, request: JudgeRequest, deadline: float, cancellation) -> JudgeResult:
                return JudgeResult(
                    verdict=ClaimVerdict.SUPPORTED, evidence_ids=[request.evidence[0].evidence_id]
                )

            @property
            def compliance_tags(self) -> frozenset[str]:
                return frozenset({"IN_PROCESS", "NO_EXTERNAL_EGRESS"})

        result = Verifier(model_provider=_PinnedProvider()).verify(
            _request(
                answer="Paris is the capital of France.",
                evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
            )
        )

        attempt = result.provenance.attempts[0]
        assert attempt.model_id == "some/model"
        assert attempt.pinned_model_version == "abc123deadbeef"

    def test_provider_with_no_pinned_model_gets_none_not_guessed(self) -> None:
        """Mock/RuleBased-shaped providers (no colon in `.name`) have no
        underlying pinned model -- `model_id`/`pinned_model_version` must
        stay honestly `None`, never fabricated."""
        result = Verifier(model_provider=RuleBasedProvider()).verify(
            _request(
                answer="Paris is the capital of France.",
                evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
            )
        )

        attempt = result.provenance.attempts[0]
        assert attempt.model_id is None
        assert attempt.pinned_model_version is None


class TestPolicyReferenceRejection:
    """F4 of the 2026-09-21 G0-G4 validation report: no tenant-scoped policy
    registry/lookup exists yet, so a request naming a `policy_id` was
    previously silently served under the default policy instead of being
    rejected -- a request for `policy_id="nonexistent-strict-policy"`
    completed with action PASS under a policy the caller never got."""

    def test_policy_id_is_rejected_at_request_construction(self) -> None:
        with pytest.raises(ValidationError, match="policy_id cannot be resolved"):
            _request(policy_id="nonexistent-strict-policy")


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
                evidence=[Evidence(evidence_id="doc_a", content="Revenue is $1M.")],
            ),
            tenant_id="tenant_a",
        )
        result_b = verifier.verify(
            _request(
                request_id="req_b",
                question="What is the revenue?",
                answer="Revenue is $2M.",
                evidence=[Evidence(evidence_id="doc_b", content="Revenue is $2M.")],
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
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")
                ],
                policy=policy,
            )
        )
        assert not any(v.code == "SCOPE_BREACH" for v in result.violations)

    def test_instruction_violation_flags(self) -> None:
        """An unmet instruction raises a violation; instruction_adherence stays UNAVAILABLE."""
        instruction = Instruction(
            instruction_id="inst_json",
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
        assert violation.severity == Severity.HIGH
        assert violation.metadata["instruction_id"] == "inst_json"

    def test_instruction_followed_raises_no_violation(self) -> None:
        """A satisfied instruction should not be flagged."""
        instruction = Instruction(
            instruction_id="inst_json", text="Return JSON only.", type=InstructionType.FORMAT
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
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")
                ],
                citations=[Citation(citation_id="cite_1", evidence_id="doc_999")],
            )
        )
        assert result.scores["citation_support"].value < 1.0
        violation = next(v for v in result.violations if v.code == "CITATION_MISMATCH")
        assert violation.evidence_ids == ["doc_999"]

    def test_citation_to_real_source_that_does_not_back_the_claim_is_flagged(self) -> None:
        """A citation to a real source that isn't the claim's supporting evidence is invalid."""
        result = self.verifier.verify(
            _request(
                request_id="req_cite_003",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018."),
                    Evidence(evidence_id="doc_002", content="Company X makes widgets."),
                ],
                citations=[
                    Citation(citation_id="cite_1", claim_id="claim_001", evidence_id="doc_002")
                ],
            )
        )
        assert result.scores["citation_support"].value < 1.0
        violation = next(v for v in result.violations if v.code == "CITATION_MISMATCH")
        assert violation.claim_ids == ["claim_001"]
        assert "does not support" in violation.message

    def test_citation_to_known_source_is_valid(self) -> None:
        """A citation pointing at real evidence should score as fully supported."""
        result = self.verifier.verify(
            _request(
                request_id="req_cite_002",
                question="When was Company X founded?",
                answer="Company X was founded in 2018.",
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")
                ],
                citations=[Citation(citation_id="cite_1", evidence_id="doc_001")],
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
                        tool_execution_id="tool_1",
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
                    ToolExecution(
                        tool_execution_id="tool_1", tool_name="weather_api", status=ToolStatus.ERROR
                    )
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
                evidence=[
                    Evidence(evidence_id="doc_001", content="Company X was founded in 2018.")
                ],
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
                    ToolExecution(
                        tool_execution_id="tool_1",
                        tool_name="weather_api",
                        status=ToolStatus.SUCCESS,
                    )
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
                Evidence(evidence_id="doc_001", content="The population is 5 million."),
                Evidence(evidence_id="doc_002", content="The population is 7 million."),
            ],
        )
        result = verifier.verify(request)

        assert result.verdict in (OverallVerdict.PASS, OverallVerdict.PARTIAL, OverallVerdict.FAIL)
        for claim in result.claims:
            assert isinstance(claim.evidence_ids, list)

    def test_claim_verification_with_explicit_conflict(self) -> None:
        """A CONTRADICTED claim records the contradicting evidence's ID in evidence_ids --
        contracts/v0.1 has no separate conflicting-evidence list; evidence_ids alone plus
        the verdict is the claim's full story."""
        from schemas.claims import ClaimVerdict, ClaimVerification, RationaleCode

        verification = ClaimVerification(
            claim_id="claim_001",
            verdict=ClaimVerdict.CONTRADICTED,
            confidence=0.8,
            evidence_ids=["doc_002"],
            rationale_code=RationaleCode.CONTRADICTION_DETECTED,
            rationale="Evidence sources conflict",
        )

        assert verification.verdict == ClaimVerdict.CONTRADICTED
        assert verification.evidence_ids == ["doc_002"]
