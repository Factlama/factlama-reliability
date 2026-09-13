"""Tests for core schema models."""

import pytest
from pydantic import ValidationError

from schemas import (
    Attempt,
    AttemptOutcome,
    Citation,
    Claim,
    ClaimType,
    ClaimVerdict,
    ClaimVerification,
    Evidence,
    EvidenceType,
    Instruction,
    InstructionType,
    Locator,
    OverallVerdict,
    Policy,
    Priority,
    Provenance,
    QualificationStatus,
    ResultStatus,
    ScoreStatus,
    ScoreValue,
    Source,
    TelemetryData,
    ToolExecution,
    ToolStatus,
    VerificationRequest,
    VerificationResult,
    Violation,
)
from schemas.verification import AbstentionReason, VerificationMode


def _provenance() -> Provenance:
    return Provenance(
        evaluator_id="reliability-verifier",
        evaluator_version="0.1",
        mode=VerificationMode.STANDARD,
        routing_profile_version="standard-0.1",
        started_at="2026-09-13T00:00:00Z",
        completed_at="2026-09-13T00:00:00Z",
    )


class TestEvidence:
    """Tests for Evidence model."""

    def test_evidence_creation(self) -> None:
        """Test creating a basic evidence object."""
        evidence = Evidence(
            id="doc_001",
            type=EvidenceType.DOCUMENT,
            extracted_text="Company X was founded in 2018.",
        )
        assert evidence.id == "doc_001"
        assert evidence.type == EvidenceType.DOCUMENT
        assert evidence.extracted_text == "Company X was founded in 2018."

    def test_evidence_with_source(self) -> None:
        """Test evidence with source information."""
        source = Source(uri="https://example.com/doc", title="Company History", author="John Doe")
        evidence = Evidence(id="doc_001", extracted_text="Some text", source=source)
        assert evidence.source is not None
        assert evidence.source.uri == "https://example.com/doc"
        assert evidence.source.title == "Company History"

    def test_evidence_default_type(self) -> None:
        """Test that default evidence type is DOCUMENT."""
        evidence = Evidence(id="doc_001", extracted_text="Some text")
        assert evidence.type == EvidenceType.DOCUMENT

    def test_evidence_hash(self) -> None:
        """Test that evidence is hashable by id."""
        ev1 = Evidence(id="doc_001", extracted_text="Text 1")
        ev2 = Evidence(id="doc_001", extracted_text="Text 2")
        ev3 = Evidence(id="doc_002", extracted_text="Text 1")

        assert hash(ev1) == hash(ev2)
        assert hash(ev1) != hash(ev3)


class TestClaim:
    """Tests for Claim model."""

    def test_claim_creation(self) -> None:
        """Test creating a claim."""
        claim = Claim(
            id="claim_001",
            text="Company X was founded in 2018.",
            type=ClaimType.FACTUAL,
            importance=0.9,
        )
        assert claim.id == "claim_001"
        assert claim.type == ClaimType.FACTUAL
        assert claim.importance == 0.9

    def test_claim_default_importance(self) -> None:
        """Test default claim importance."""
        claim = Claim(id="claim_001", text="Some claim")
        assert claim.importance == 1.0

    def test_claim_importance_validation(self) -> None:
        """Test that importance must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Claim(id="claim_001", text="Claim", importance=1.5)
        with pytest.raises(ValidationError):
            Claim(id="claim_001", text="Claim", importance=-0.1)


class TestClaimVerification:
    """Tests for ClaimVerification model."""

    def test_claim_verification_creation(self) -> None:
        """Test creating a claim verification."""
        verification = ClaimVerification(
            claim_id="claim_001",
            verdict=ClaimVerdict.SUPPORTED,
            confidence=0.95,
            reason="Claim is supported by evidence",
        )
        assert verification.claim_id == "claim_001"
        assert verification.verdict == ClaimVerdict.SUPPORTED
        assert verification.confidence == 0.95

    def test_claim_verification_with_evidence(self) -> None:
        """Test claim verification with evidence references."""
        from schemas.claims import EvidenceReference

        verification = ClaimVerification(
            claim_id="claim_001",
            verdict=ClaimVerdict.SUPPORTED,
            evidence=[EvidenceReference(evidence_id="doc_001", support=0.9, relevance=0.95)],
        )
        assert len(verification.evidence) == 1
        assert verification.evidence[0].evidence_id == "doc_001"


class TestVerificationRequest:
    """Tests for VerificationRequest model."""

    def test_request_creation_minimal(self) -> None:
        """Test creating a minimal verification request."""
        request = VerificationRequest(
            request_id="req_001",
            project_id="proj_001",
            application_id="app_001",
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
        )
        assert request.request_id == "req_001"
        assert request.question == "When was Company X founded?"
        assert request.answer == "Company X was founded in 2018."
        assert request.evidence == []
        assert request.schema_version == "0.1"

    def test_request_question_is_optional(self) -> None:
        """CONTRACTS.md: question is optional."""
        request = VerificationRequest(
            request_id="req_001",
            project_id="proj_001",
            application_id="app_001",
            answer="Answer.",
        )
        assert request.question is None

    def test_request_has_no_tenant_id_field(self) -> None:
        """Tenant identity must never be a client-supplied request field."""
        assert "tenant_id" not in VerificationRequest.model_fields

    def test_request_with_evidence(self) -> None:
        """Test request with evidence."""
        evidence = Evidence(id="doc_001", extracted_text="Company X was founded in 2018.")
        request = VerificationRequest(
            request_id="req_001",
            project_id="proj_001",
            application_id="app_001",
            question="When was Company X founded?",
            answer="Company X was founded in 2018.",
            evidence=[evidence],
        )
        assert len(request.evidence) == 1
        assert request.evidence[0].id == "doc_001"

    def test_request_schema_version_validation(self) -> None:
        """Test that unsupported schema versions are rejected."""
        with pytest.raises(ValidationError, match="Unsupported schema version"):
            VerificationRequest(
                request_id="req_001",
                project_id="proj_001",
                application_id="app_001",
                answer="Answer.",
                schema_version="99.0",
            )


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def _base_kwargs(self, **overrides) -> dict:
        kwargs = {
            "evaluation_id": "eval_001",
            "request_id": "req_001",
            "tenant_id": "tenant_test",
            "project_id": "proj_001",
            "application_id": "app_001",
            "status": ResultStatus.COMPLETED,
            "verdict": OverallVerdict.PASS,
            "scores": {"groundedness": ScoreValue(value=0.95, status=ScoreStatus.MEASURED)},
            "provenance": _provenance(),
        }
        kwargs.update(overrides)
        return kwargs

    def test_result_creation(self) -> None:
        """Test creating a verification result."""
        result = VerificationResult(**self._base_kwargs())
        assert result.request_id == "req_001"
        assert result.verdict == OverallVerdict.PASS
        assert result.scores["groundedness"].value == 0.95

    def test_result_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = VerificationResult(**self._base_kwargs())
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["request_id"] == "req_001"

    def test_result_to_json(self) -> None:
        """Test converting result to JSON."""
        result = VerificationResult(**self._base_kwargs())
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "req_001" in json_str

    def test_abstained_status_requires_abstain_verdict(self) -> None:
        """FAILED/ABSTAINED must carry verdict=ABSTAIN (CONTRACTS.md)."""
        with pytest.raises(ValidationError, match="requires verdict=ABSTAIN"):
            VerificationResult(
                **self._base_kwargs(status=ResultStatus.ABSTAINED, verdict=OverallVerdict.PASS)
            )

    def test_abstained_status_requires_abstention_reason(self) -> None:
        with pytest.raises(ValidationError, match="requires an abstention_reason"):
            VerificationResult(
                **self._base_kwargs(status=ResultStatus.ABSTAINED, verdict=OverallVerdict.ABSTAIN)
            )

    def test_abstained_status_with_reason_is_valid(self) -> None:
        result = VerificationResult(
            **self._base_kwargs(
                status=ResultStatus.ABSTAINED,
                verdict=OverallVerdict.ABSTAIN,
                abstention_reason=AbstentionReason.NO_CHECKABLE_CLAIMS,
            )
        )
        assert result.abstention_reason == AbstentionReason.NO_CHECKABLE_CLAIMS


class TestScoreValue:
    """Tests for the ScoreValue map replacing the old flat-float VerificationScores."""

    def test_measured_score(self) -> None:
        score = ScoreValue(
            value=0.75, status=ScoreStatus.MEASURED, method_version="groundedness-0.1"
        )
        assert score.value == 0.75
        assert score.status == ScoreStatus.MEASURED

    def test_not_applicable_score_has_no_value(self) -> None:
        score = ScoreValue(status=ScoreStatus.NOT_APPLICABLE)
        assert score.value is None

    def test_value_bounds_validation(self) -> None:
        with pytest.raises(ValidationError):
            ScoreValue(value=1.5, status=ScoreStatus.MEASURED)
        with pytest.raises(ValidationError):
            ScoreValue(value=-0.1, status=ScoreStatus.MEASURED)


class TestAttemptAndProvenance:
    """Tests for provenance/attempt/usage models."""

    def test_attempt_creation(self) -> None:
        attempt = Attempt(
            attempt_id="attempt_1",
            provider_id="rule-based",
            configuration_version="0.1",
            qualification_status=QualificationStatus.UNQUALIFIED,
            calibration_class="abc123",
            outcome=AttemptOutcome.COMPLETED,
            started_at="2026-09-13T00:00:00Z",
            completed_at="2026-09-13T00:00:00Z",
        )
        assert attempt.outcome == AttemptOutcome.COMPLETED

    def test_provenance_attempts_default_empty(self) -> None:
        assert _provenance().attempts == []


class TestPolicy:
    """Tests for Policy model."""

    def test_policy_creation(self) -> None:
        """Test creating a policy."""
        policy = Policy(id="customer-support", version="0.1")
        assert policy.id == "customer-support"
        assert policy.version == "0.1"

    def test_policy_with_thresholds(self) -> None:
        """Test policy with threshold settings."""
        from schemas.policy import GroundingPolicy, HallucinationPolicy

        policy = Policy(
            id="strict",
            grounding=GroundingPolicy(minimum=0.90),
            hallucination=HallucinationPolicy(maximum=0.05),
        )
        assert policy.grounding.minimum == 0.90
        assert policy.hallucination.maximum == 0.05


class TestInstruction:
    """Tests for Instruction model."""

    def test_instruction_creation(self) -> None:
        """Test creating an instruction."""
        instruction = Instruction(
            id="inst_001",
            text="Use only the supplied context.",
            type=InstructionType.GROUNDING,
            priority=Priority.HIGH,
        )
        assert instruction.id == "inst_001"
        assert instruction.type == InstructionType.GROUNDING
        assert instruction.priority == Priority.HIGH


class TestCitation:
    """Tests for Citation model."""

    def test_citation_creation(self) -> None:
        """Test creating a citation."""
        locator = Locator(page=5, section="Introduction")
        citation = Citation(id="cite_001", source_id="doc_001", locator=locator)
        assert citation.id == "cite_001"
        assert citation.source_id == "doc_001"
        assert citation.locator is not None
        assert citation.locator.page == 5


class TestToolExecution:
    """Tests for ToolExecution model."""

    def test_tool_execution_success(self) -> None:
        """Test successful tool execution."""
        tool = ToolExecution(
            id="tool_001",
            tool_name="get_balance",
            arguments={"customer_id": "123"},
            result={"balance": 100},
            status=ToolStatus.SUCCESS,
        )
        assert tool.status == ToolStatus.SUCCESS
        assert tool.result == {"balance": 100}

    def test_tool_execution_error(self) -> None:
        """Test failed tool execution."""
        tool = ToolExecution(
            id="tool_001",
            tool_name="get_balance",
            arguments={"customer_id": "invalid"},
            status=ToolStatus.ERROR,
            error_message="Customer not found",
        )
        assert tool.status == ToolStatus.ERROR
        assert tool.error_message == "Customer not found"


class TestViolation:
    """Tests for Violation model."""

    def test_violation_creation(self) -> None:
        """Test creating a violation."""
        violation = Violation(
            code="UNSUPPORTED_CLAIM",
            severity="medium",
            claim_id="claim_001",
            message="Claim is not supported by evidence",
        )
        assert violation.code == "UNSUPPORTED_CLAIM"
        assert violation.severity == "medium"
        assert violation.claim_id == "claim_001"


class TestTelemetryData:
    """Tests for TelemetryData model."""

    def test_telemetry_creation(self) -> None:
        """Test creating telemetry data."""
        telemetry = TelemetryData(
            trace_id="trace_001", verdict="PASS", reliability=0.95, groundedness=0.98
        )
        assert telemetry.trace_id == "trace_001"
        assert telemetry.verdict == "PASS"

    def test_telemetry_to_open_telemetry(self) -> None:
        """Test converting to OpenTelemetry attributes."""
        telemetry = TelemetryData(
            trace_id="trace_001",
            service_name="my-service",
            verdict="PASS",
            reliability=0.95,
            groundedness=0.98,
        )
        attrs = telemetry.to_open_telemetry_attributes()
        assert attrs["factlama.verdict"] == "PASS"
        assert attrs["service.name"] == "my-service"
        assert attrs["factlama.reliability"] == 0.95
