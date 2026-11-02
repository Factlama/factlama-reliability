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


class TestPolicyEngineDecisionProvenance:
    """ADR-023: `evaluate_with_decision()`'s `matched_rule_id`/`inputs_used`
    must track `evaluate()`'s own action precedence exactly -- one test per
    forced-override path, plus the default verdict-to-action mappings and
    `inputs_used`'s token vocabulary."""

    def setup_method(self) -> None:
        self.engine = PolicyEngine()

    def test_evaluate_returns_the_same_tuple_evaluate_with_decision_wraps(self) -> None:
        """`evaluate()` must stay a thin wrapper: same action/violations as
        the richer method, for every existing call site's sake."""
        scores = {"groundedness": _score(1.0)}
        action, violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS, scores=scores, violations=[]
        )
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores=scores, violations=[]
        )
        assert (action, violations) == (draft.action, draft.additional_violations)

    def test_default_pass_rule_id(self) -> None:
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[]
        )
        assert draft.matched_rule_id == "DEFAULT_PASS"
        assert draft.action == PolicyAction.PASS

    def test_actions_on_failure_rule_id(self) -> None:
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.FAIL, scores={}, violations=[]
        )
        assert draft.matched_rule_id == "ACTIONS_ON_FAILURE"
        assert draft.action == PolicyAction.FAIL

    def test_actions_on_partial_rule_id(self) -> None:
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PARTIAL, scores={}, violations=[]
        )
        assert draft.matched_rule_id == "ACTIONS_ON_PARTIAL"
        assert draft.action == PolicyAction.PASS

    def test_actions_on_abstain_rule_id(self) -> None:
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.ABSTAIN, scores={}, violations=[]
        )
        assert draft.matched_rule_id == "ACTIONS_ON_ABSTAIN"
        assert draft.action == PolicyAction.HUMAN_REVIEW

    def test_tool_error_forces_fail_rule_id(self) -> None:
        tool_error = Violation(code="TOOL_ERROR", severity=Severity.HIGH)
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.ABSTAIN, scores={}, violations=[tool_error]
        )
        assert draft.matched_rule_id == "TOOL_ERROR_FORCES_FAIL"
        assert draft.action == Policy(id="default").actions.on_failure

    def test_evidence_injection_suspected_rule_id(self) -> None:
        violation = Violation(code="EVIDENCE_INJECTION_SUSPECTED", severity=Severity.HIGH)
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[violation]
        )
        assert draft.matched_rule_id == "EVIDENCE_INJECTION_SUSPECTED_FORCES_HUMAN_REVIEW"
        assert draft.action == PolicyAction.HUMAN_REVIEW

    def test_citation_overlap_inconclusive_rule_id(self) -> None:
        violation = Violation(code="CITATION_OVERLAP_INCONCLUSIVE", severity=Severity.MEDIUM)
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[violation]
        )
        assert draft.matched_rule_id == "CITATION_OVERLAP_INCONCLUSIVE_FORCES_HUMAN_REVIEW"
        assert draft.action == PolicyAction.HUMAN_REVIEW

    def test_inputs_used_includes_verdict_and_inspected_scores_only(self) -> None:
        """`contradiction_risk` is not one of the five dimensions this
        engine's threshold checks look up -- it must not appear as a
        SCORE: token even though it's present in `scores`."""
        scores = {
            "groundedness": _score(1.0),
            "hallucination_risk": _na(),
            "contradiction_risk": _score(0.0),
        }
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores=scores, violations=[], has_citations=False
        )
        assert draft.inputs_used == ["SCORE:groundedness", "SCORE:hallucination_risk", "VERDICT"]

    def test_inputs_used_citations_present_token_is_conditional(self) -> None:
        draft_with = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], has_citations=True
        )
        draft_without = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], has_citations=False
        )
        assert "CITATIONS_PRESENT" in draft_with.inputs_used
        assert "CITATIONS_PRESENT" not in draft_without.inputs_used

    def test_inputs_used_includes_mode_and_application_tokens_when_supplied(self) -> None:
        from schemas.verification import VerificationMode

        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS,
            scores={},
            violations=[],
            has_citations=False,
            mode=VerificationMode.DEEP,
            application_id="app-1",
        )
        assert "MODE:DEEP" in draft.inputs_used
        assert "APPLICATION_PRESENT" in draft.inputs_used

    def test_inputs_used_omits_mode_and_application_tokens_when_absent(self) -> None:
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], has_citations=False
        )
        assert not any(t.startswith("MODE:") for t in draft.inputs_used)
        assert "APPLICATION_PRESENT" not in draft.inputs_used

    def test_inputs_used_violation_tokens_are_sorted_and_deduplicated(self) -> None:
        violations = [
            Violation(code="TOOL_ERROR", severity=Severity.HIGH),
            Violation(code="CITATION_MISMATCH", severity=Severity.HIGH),
            Violation(code="TOOL_ERROR", severity=Severity.HIGH, claim_ids=["c2"]),
        ]
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.ABSTAIN, scores={}, violations=violations
        )
        assert "VIOLATION:CITATION_MISMATCH" in draft.inputs_used
        assert draft.inputs_used.count("VIOLATION:TOOL_ERROR") == 1
        assert draft.inputs_used == sorted(draft.inputs_used)


class TestRiskTierEscalation:
    """EPIC-08 risk/cost routing: `policy.risk_tier` in
    `policy.actions.escalate_risk_tiers` forces at least HUMAN_REVIEW, but
    only ever escalates FROM a PASS -- it must never downgrade a stricter
    action a tenant's own config, or another forced-override rule, already
    selected."""

    def setup_method(self) -> None:
        self.engine = PolicyEngine()

    def test_matching_risk_tier_escalates_pass_to_human_review(self) -> None:
        policy = Policy(
            id="test", actions=ActionPolicy(escalate_risk_tiers=["HIGH"]), risk_tier="HIGH"
        )
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], policy=policy
        )
        assert draft.action == PolicyAction.HUMAN_REVIEW
        assert draft.matched_rule_id == "RISK_TIER_ESCALATES_TO_HUMAN_REVIEW"

    def test_non_matching_risk_tier_does_not_escalate(self) -> None:
        policy = Policy(
            id="test", actions=ActionPolicy(escalate_risk_tiers=["CRITICAL"]), risk_tier="LOW"
        )
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], policy=policy
        )
        assert draft.action == PolicyAction.PASS
        assert draft.matched_rule_id == "DEFAULT_PASS"

    def test_no_risk_tier_does_not_escalate(self) -> None:
        policy = Policy(id="test", actions=ActionPolicy(escalate_risk_tiers=["HIGH"]))
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[], policy=policy
        )
        assert draft.action == PolicyAction.PASS

    def test_escalation_never_downgrades_an_already_stricter_action(self) -> None:
        """A FAIL verdict already selects on_failure -- a matching risk tier
        must not downgrade that back toward HUMAN_REVIEW/PASS."""
        policy = Policy(
            id="test",
            actions=ActionPolicy(on_failure=PolicyAction.BLOCK, escalate_risk_tiers=["HIGH"]),
            risk_tier="HIGH",
        )
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.FAIL, scores={}, violations=[], policy=policy
        )
        assert draft.action == PolicyAction.BLOCK
        assert draft.matched_rule_id == "ACTIONS_ON_FAILURE"

    def test_evidence_injection_override_still_wins_over_risk_tier_escalation(self) -> None:
        """Both rules pick HUMAN_REVIEW here, but the more specific
        judge-boundary signal should still be named as the matched rule."""
        policy = Policy(
            id="test", actions=ActionPolicy(escalate_risk_tiers=["HIGH"]), risk_tier="HIGH"
        )
        violation = Violation(code="EVIDENCE_INJECTION_SUSPECTED", severity=Severity.HIGH)
        draft = self.engine.evaluate_with_decision(
            verdict=OverallVerdict.PASS, scores={}, violations=[violation], policy=policy
        )
        assert draft.action == PolicyAction.HUMAN_REVIEW
        assert draft.matched_rule_id == "EVIDENCE_INJECTION_SUSPECTED_FORCES_HUMAN_REVIEW"

    def test_evaluate_wrapper_also_applies_risk_tier_escalation(self) -> None:
        """The thin `evaluate()` wrapper must not lose this behavior."""
        policy = Policy(
            id="test", actions=ActionPolicy(escalate_risk_tiers=["HIGH"]), risk_tier="HIGH"
        )
        action, _violations = self.engine.evaluate(
            verdict=OverallVerdict.PASS, scores={}, violations=[], policy=policy
        )
        assert action == PolicyAction.HUMAN_REVIEW
