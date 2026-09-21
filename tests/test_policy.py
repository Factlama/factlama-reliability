"""Tests for policy engine."""

from core.policy import PolicyEngine
from schemas.policy import (
    ActionPolicy,
    CitationPolicy,
    GroundingPolicy,
    HallucinationPolicy,
    Policy,
    PolicyAction,
    ToolPolicy,
)
from schemas.verification import OverallVerdict, ScoreStatus, ScoreValue, Severity, Violation


def _score(value: float | None, status: ScoreStatus = ScoreStatus.MEASURED) -> ScoreValue:
    return ScoreValue(
        value=value, status=status, method_version="test-0.1", calibration_class="NONE"
    )


def _na() -> ScoreValue:
    return ScoreValue(status=ScoreStatus.NOT_APPLICABLE)


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = PolicyEngine()

    def test_engine_initialization(self) -> None:
        """Test engine initialization."""
        assert self.engine is not None

    def test_evaluate_perfect_scores(self) -> None:
        """Test evaluation with perfect scores and a PASS verdict."""
        scores = {"groundedness": _score(1.0), "hallucination_risk": _score(0.0)}
        action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS,
            scores=scores,
            violations=[],
        )
        assert action == PolicyAction.PASS
        assert len(violations) == 0

    def test_evaluate_low_groundedness(self) -> None:
        """Test evaluation with low groundedness flags a threshold violation."""
        policy = Policy(id="test", grounding=GroundingPolicy(minimum=0.90))
        scores = {"groundedness": _score(0.50)}
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PARTIAL,
            scores=scores,
            violations=[],
            policy=policy,
        )
        assert any(v.code == "GROUNDING_BELOW_THRESHOLD" for v in violations)

    def test_threshold_skipped_when_score_not_measured(self) -> None:
        """A threshold check must not fire against a NOT_APPLICABLE/UNAVAILABLE score."""
        policy = Policy(id="test", grounding=GroundingPolicy(minimum=0.90))
        scores = {"groundedness": _na()}
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.ABSTAIN,
            scores=scores,
            violations=[],
            policy=policy,
        )
        assert not any(v.code == "GROUNDING_BELOW_THRESHOLD" for v in violations)

    def test_evaluate_high_hallucination_risk(self) -> None:
        """Test evaluation with high hallucination risk."""
        policy = Policy(id="test", hallucination=HallucinationPolicy(maximum=0.10))
        scores = {"hallucination_risk": _score(0.50)}
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.FAIL,
            scores=scores,
            violations=[],
            policy=policy,
        )
        assert any(v.code == "HALLUCINATION_ABOVE_THRESHOLD" for v in violations)

    def test_evaluate_does_not_recompute_verdict_from_violations(self) -> None:
        """The passed-in factual verdict is authoritative; policy no longer upgrades it."""
        existing_violation = Violation(
            code="UNSUPPORTED_CLAIM", severity=Severity.MEDIUM, message="Unsupported claim found"
        )
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.PARTIAL,
            scores={},
            violations=[existing_violation],
        )
        assert action == PolicyAction.PASS  # PARTIAL maps to default on_partial=PASS

    def test_policy_action_on_failure(self) -> None:
        """Test policy action configuration on failure."""
        policy = Policy(id="test", actions=ActionPolicy(on_failure=PolicyAction.REGENERATE))
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.FAIL,
            scores={},
            violations=[],
            policy=policy,
        )
        assert action == PolicyAction.REGENERATE

    def test_policy_action_on_partial(self) -> None:
        """Test policy action configuration on partial."""
        policy = Policy(id="test", actions=ActionPolicy(on_partial=PolicyAction.HUMAN_REVIEW))
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.PARTIAL,
            scores={},
            violations=[],
            policy=policy,
        )
        assert action == PolicyAction.HUMAN_REVIEW

    def test_citations_required_but_missing(self) -> None:
        """Test that a policy requiring citations flags their absence."""
        policy = Policy(id="test", citations=CitationPolicy(required=True))
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS,
            scores={},
            violations=[],
            policy=policy,
            has_citations=False,
        )
        assert any(v.code == "CITATION_MISMATCH" for v in violations)

    def test_citations_required_and_present_is_fine(self) -> None:
        """Test that supplying citations satisfies a required-citations policy."""
        policy = Policy(id="test", citations=CitationPolicy(required=True))
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS,
            scores={},
            violations=[],
            policy=policy,
            has_citations=True,
        )
        assert not any(v.code == "CITATION_MISMATCH" for v in violations)

    def test_citation_support_below_minimum(self) -> None:
        """Test that low citation support against a minimum_support policy is flagged."""
        policy = Policy(id="test", citations=CitationPolicy(minimum_support=0.9))
        scores = {"citation_support": _score(0.5)}
        _action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PARTIAL,
            scores=scores,
            violations=[],
            policy=policy,
        )
        assert any(v.code == "CITATION_MISMATCH" for v in violations)

    def test_citation_overlap_inconclusive_forces_human_review(self) -> None:
        """Re-audit follow-up (2026-09-21): CITATION_OVERLAP_INCONCLUSIVE
        (judges.port.apply_citation_support_check() no longer overriding the
        verdict for zero recognized overlap) forces HUMAN_REVIEW the same
        way EVIDENCE_INJECTION_SUSPECTED already does, even for an otherwise
        PASSing verdict -- distinct from CITATION_MISMATCH above, which does
        not force review on its own."""
        violation = Violation(
            code="CITATION_OVERLAP_INCONCLUSIVE",
            severity=Severity.MEDIUM,
            claim_ids=["claim_001"],
            evidence_ids=["doc_1"],
        )
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.PASS,
            scores={},
            violations=[violation],
        )
        assert action == PolicyAction.HUMAN_REVIEW

    def test_tool_error_forces_failure_action_by_default(self) -> None:
        """Any tool error forces the on_failure action when fail_on_error is on (the default),
        without touching the passed-in factual verdict."""
        tool_error_violation = Violation(
            code="TOOL_ERROR",
            severity=Severity.HIGH,
            message="Tool 'search' finished with status error",
        )
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.ABSTAIN,
            scores={},
            violations=[tool_error_violation],
        )
        assert action == Policy(id="default").actions.on_failure

    def test_tool_error_does_not_force_failure_when_disabled(self) -> None:
        """fail_on_error=False lets a single tool error stay non-fatal to the action."""
        policy = Policy(id="test", tools=ToolPolicy(fail_on_error=False))
        tool_error_violation = Violation(
            code="TOOL_ERROR",
            severity=Severity.HIGH,
            message="Tool 'search' finished with status error",
        )
        action, _ = self.engine.evaluate(
            verdict=OverallVerdict.PARTIAL,
            scores={},
            violations=[tool_error_violation],
            policy=policy,
        )
        assert action == policy.actions.on_partial

    def test_default_policy(self) -> None:
        """Test evaluation with default policy."""
        action, _violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS,
            scores={},
            violations=[],
            policy=None,
        )
        assert action == PolicyAction.PASS
