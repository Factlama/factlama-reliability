"""Policy evaluation engine.

Determines only the policy action, never the factual verdict -- that is
`core.scoring.determine_verdict()`'s job (CONTRACTS.md: "Factual verdict is
computed independently of policy action").
"""

from typing import NamedTuple

from schemas.policy import Policy, PolicyAction
from schemas.verification import (
    OverallVerdict,
    ScoreStatus,
    ScoreValue,
    Severity,
    VerificationMode,
    Violation,
)

# ADR-023: the five score dimensions this engine's threshold checks actually
# look up via `scores.get(name)`. `inputs_used`'s `SCORE:<dimension>` tokens
# are drawn from this fixed list, not from every key in the caller's
# `scores` dict -- an unrelated dimension a caller happens to include is
# never "inspected" by this engine.
_SCORE_DIMENSIONS_INSPECTED = (
    "groundedness",
    "hallucination_risk",
    "instruction_adherence",
    "tool_correctness",
    "citation_support",
)


class PolicyDecisionDraft(NamedTuple):
    """`evaluate()`'s richer internal result -- action and violations exactly
    as before, plus ADR-023's decision-provenance fields. Kept as a separate
    method/type (`evaluate_with_decision`) rather than widening `evaluate()`
    itself, so every existing `tests/test_policy.py` call site keeps its
    `(action, violations)` tuple unpacking unmodified.
    """

    action: PolicyAction
    additional_violations: list[Violation]
    matched_rule_id: str
    inputs_used: list[str]


class PolicyEngine:
    """Engine for evaluating verification scores against policies."""

    def __init__(self) -> None:
        """Initialize the policy engine."""

    def evaluate(
        self,
        verdict: OverallVerdict,
        scores: dict[str, ScoreValue],
        violations: list[Violation],
        policy: Policy | None = None,
        has_citations: bool = True,
        mode: VerificationMode | None = None,
        application_id: str | None = None,
    ) -> tuple[PolicyAction, list[Violation]]:
        """Evaluate scores against policy thresholds and determine the policy action.

        Args:
            verdict: The already-determined factual verdict.
            scores: Named score dimensions; a threshold check is skipped
                (not fired) whenever its dimension's status isn't MEASURED --
                a threshold cannot be compared against a value that doesn't
                exist.
            violations: Violations already detected by the Verifier.
            policy: Policy for evaluation (uses defaults if None).
            has_citations: Whether any citations were supplied for this
                request, used to check `policy.citations.required`.
            mode: EPIC-08 risk/cost routing input -- recorded in
                `inputs_used` when supplied; no rule branches on it yet.
            application_id: EPIC-08 risk/cost routing input -- recorded as
                `APPLICATION_PRESENT` in `inputs_used` when supplied, never
                as the raw ID itself. No rule branches on it yet.

        Returns:
            Tuple of (policy_action, additional_violations).
        """
        draft = self.evaluate_with_decision(
            verdict, scores, violations, policy, has_citations, mode, application_id
        )
        return draft.action, draft.additional_violations

    def evaluate_with_decision(
        self,
        verdict: OverallVerdict,
        scores: dict[str, ScoreValue],
        violations: list[Violation],
        policy: Policy | None = None,
        has_citations: bool = True,
        mode: VerificationMode | None = None,
        application_id: str | None = None,
    ) -> PolicyDecisionDraft:
        """Same evaluation as `evaluate()`, plus ADR-023's `matched_rule_id`
        and `inputs_used` -- computed alongside the action, from the same
        conditions, so the two can never drift apart from each other.

        EPIC-08 risk/cost routing: `mode`/`application_id` are accepted and
        recorded in `inputs_used` -- policy-engine.md names both as decision
        inputs, but no rule branches on either yet, named here rather than
        silently absent. `policy.risk_tier` (read from `policy`, not a
        separate parameter, since it already travels on the Policy object)
        is the one of the three that drives real behavior: see the
        escalate_risk_tiers check below.
        """
        policy = policy or Policy(id="default")
        additional_violations: list[Violation] = []
        inputs_used: list[str] = ["VERDICT"]
        for dimension in _SCORE_DIMENSIONS_INSPECTED:
            if scores.get(dimension) is not None:
                inputs_used.append(f"SCORE:{dimension}")
        if has_citations:
            inputs_used.append("CITATIONS_PRESENT")
        if mode is not None:
            inputs_used.append(f"MODE:{mode.value}")
        if application_id:
            inputs_used.append("APPLICATION_PRESENT")
        if policy.risk_tier is not None:
            inputs_used.append(f"RISK_TIER:{policy.risk_tier}")

        groundedness = scores.get("groundedness")
        if (
            policy.grounding.minimum is not None
            and groundedness is not None
            and groundedness.status == ScoreStatus.MEASURED
            and (groundedness_value := groundedness.value) is not None
            and groundedness_value < policy.grounding.minimum
        ):
            additional_violations.append(
                Violation(
                    code="GROUNDING_BELOW_THRESHOLD",
                    severity=Severity.HIGH,
                    message=f"Groundedness {groundedness_value:.2f} below minimum {policy.grounding.minimum:.2f}",
                )
            )

        hallucination_risk = scores.get("hallucination_risk")
        if (
            policy.hallucination.maximum is not None
            and hallucination_risk is not None
            and hallucination_risk.status == ScoreStatus.MEASURED
            and (hallucination_risk_value := hallucination_risk.value) is not None
            and hallucination_risk_value > policy.hallucination.maximum
        ):
            additional_violations.append(
                Violation(
                    code="HALLUCINATION_ABOVE_THRESHOLD",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Hallucination risk {hallucination_risk_value:.2f} above maximum "
                        f"{policy.hallucination.maximum:.2f}"
                    ),
                )
            )

        instruction_adherence = scores.get("instruction_adherence")
        if (
            policy.instructions.minimum is not None
            and instruction_adherence is not None
            and instruction_adherence.status == ScoreStatus.MEASURED
            and (instruction_adherence_value := instruction_adherence.value) is not None
            and instruction_adherence_value < policy.instructions.minimum
        ):
            additional_violations.append(
                Violation(
                    code="INSTRUCTION_VIOLATION",
                    severity=Severity.HIGH,
                    message=(
                        f"Instruction adherence {instruction_adherence_value:.2f} below minimum "
                        f"{policy.instructions.minimum:.2f}"
                    ),
                )
            )

        tool_correctness = scores.get("tool_correctness")
        if (
            policy.tools.minimum is not None
            and tool_correctness is not None
            and tool_correctness.status == ScoreStatus.MEASURED
            and (tool_correctness_value := tool_correctness.value) is not None
            and tool_correctness_value < policy.tools.minimum
        ):
            additional_violations.append(
                Violation(
                    code="TOOL_ERROR",
                    severity=Severity.HIGH,
                    message=f"Tool correctness {tool_correctness_value:.2f} below minimum {policy.tools.minimum:.2f}",
                )
            )

        # Check citation requirements
        if policy.citations.required and not has_citations:
            additional_violations.append(
                Violation(
                    code="CITATION_MISMATCH",
                    severity=Severity.HIGH,
                    message="Citations are required by policy but none were supplied",
                )
            )

        citation_support = scores.get("citation_support")
        if (
            policy.citations.minimum_support is not None
            and citation_support is not None
            and citation_support.status == ScoreStatus.MEASURED
            and (citation_support_value := citation_support.value) is not None
            and citation_support_value < policy.citations.minimum_support
        ):
            additional_violations.append(
                Violation(
                    code="CITATION_MISMATCH",
                    severity=Severity.HIGH,
                    message=(
                        f"Citation support {citation_support_value:.2f} below minimum "
                        f"{policy.citations.minimum_support:.2f}"
                    ),
                )
            )

        # A tool execution that actually errored is critical whenever the
        # policy says tool failures should fail the request outright (the
        # ToolPolicy default), regardless of the tool_correctness threshold.
        # This can only escalate the *policy action*, not the already-fixed
        # factual verdict.
        all_violations = violations + additional_violations
        for code in sorted({v.code for v in all_violations}):
            inputs_used.append(f"VIOLATION:{code}")
        force_fail = policy.tools.fail_on_error and any(
            v.code == "TOOL_ERROR" for v in all_violations
        )

        effective_verdict = OverallVerdict.FAIL if force_fail else verdict
        action = self._determine_action(effective_verdict, policy)
        matched_rule_id = "TOOL_ERROR_FORCES_FAIL" if force_fail else self._default_rule_id(verdict)

        # EPIC-08 risk/cost routing: a tenant-named risk tier (policy.
        # risk_tier) can force at least HUMAN_REVIEW even when the verdict-
        # to-action mapping above selected PASS. Only ever escalates FROM
        # PASS -- never downgrades a stricter action a tenant's own
        # on_failure/on_partial/on_abstain config, or the TOOL_ERROR
        # override above, already selected.
        if (
            action == PolicyAction.PASS
            and policy.risk_tier is not None
            and policy.risk_tier in policy.actions.escalate_risk_tiers
        ):
            action = PolicyAction.HUMAN_REVIEW
            matched_rule_id = "RISK_TIER_ESCALATES_TO_HUMAN_REVIEW"

        # ADR-013: "a pattern-based scan over evidence content sets
        # EVIDENCE_INJECTION_SUSPECTED and routes to HUMAN_REVIEW via
        # policy." This is a dedicated routing rule, unlike TOOL_ERROR
        # above: it never escalates the factual verdict (the verdict above
        # was already reached independently of the injection attempt), it
        # always routes the *action* to HUMAN_REVIEW so a human sees the
        # attempt even when the verdict would otherwise default to PASS
        # (e.g. injected text riding along with genuine support).
        if any(v.code == "EVIDENCE_INJECTION_SUSPECTED" for v in all_violations):
            action = PolicyAction.HUMAN_REVIEW
            matched_rule_id = "EVIDENCE_INJECTION_SUSPECTED_FORCES_HUMAN_REVIEW"

        # Re-audit follow-up (2026-09-21): the same pattern as
        # EVIDENCE_INJECTION_SUSPECTED above, for the same reason. A cited
        # evidence text sharing no recognized lexical/synonym/morphological
        # overlap with its claim is genuinely ambiguous -- as consistent
        # with a paraphrase this deterministic guard's vocabulary doesn't
        # yet cover as with a fabricated citation
        # (`judges.port.apply_citation_support_check()`'s own docstring) --
        # so it no longer silently overrides the factual verdict either
        # direction. It still must not be silently ignored: force human
        # review so a person, not a guess, resolves the ambiguity.
        if any(v.code == "CITATION_OVERLAP_INCONCLUSIVE" for v in all_violations):
            action = PolicyAction.HUMAN_REVIEW
            matched_rule_id = "CITATION_OVERLAP_INCONCLUSIVE_FORCES_HUMAN_REVIEW"

        return PolicyDecisionDraft(
            action=action,
            additional_violations=additional_violations,
            matched_rule_id=matched_rule_id,
            inputs_used=sorted(set(inputs_used)),
        )

    def _determine_action(
        self,
        verdict: OverallVerdict,
        policy: Policy,
    ) -> PolicyAction:
        """Determine the policy action based on verdict.

        Args:
            verdict: The overall verdict (or an escalated verdict for the
                purpose of choosing an action; the returned VerificationResult
                still records the true factual verdict separately).
            policy: The policy configuration.

        Returns:
            The policy action to take.
        """
        if verdict == OverallVerdict.PASS:
            return PolicyAction.PASS
        elif verdict == OverallVerdict.FAIL:
            return policy.actions.on_failure
        elif verdict == OverallVerdict.PARTIAL:
            return policy.actions.on_partial
        else:  # ABSTAIN
            return policy.actions.on_abstain

    def _default_rule_id(self, verdict: OverallVerdict) -> str:
        """ADR-023: names which verdict-to-action mapping rule
        `_determine_action` would apply, before any forced override. Kept
        separate from `_determine_action` itself since that method also
        runs (unlabeled) against the tool-error-escalated verdict."""
        if verdict == OverallVerdict.PASS:
            return "DEFAULT_PASS"
        elif verdict == OverallVerdict.FAIL:
            return "ACTIONS_ON_FAILURE"
        elif verdict == OverallVerdict.PARTIAL:
            return "ACTIONS_ON_PARTIAL"
        else:  # ABSTAIN
            return "ACTIONS_ON_ABSTAIN"
