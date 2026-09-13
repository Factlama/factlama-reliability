"""Policy evaluation engine.

Determines only the policy action, never the factual verdict -- that is
`core.scoring.determine_verdict()`'s job (CONTRACTS.md: "Factual verdict is
computed independently of policy action").
"""

from typing import Optional

from schemas.policy import Policy, PolicyAction
from schemas.verification import OverallVerdict, ScoreStatus, ScoreValue, Violation


class PolicyEngine:
    """Engine for evaluating verification scores against policies."""

    def __init__(self) -> None:
        """Initialize the policy engine."""
        pass

    def evaluate(
        self,
        verdict: OverallVerdict,
        scores: dict[str, ScoreValue],
        violations: list[Violation],
        policy: Optional[Policy] = None,
        has_citations: bool = True,
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

        Returns:
            Tuple of (policy_action, additional_violations).
        """
        policy = policy or Policy(id="default")
        additional_violations: list[Violation] = []

        groundedness = scores.get("groundedness")
        if (
            policy.grounding.minimum is not None
            and groundedness is not None
            and groundedness.status == ScoreStatus.MEASURED
            and groundedness.value < policy.grounding.minimum
        ):
            additional_violations.append(
                Violation(
                    code="GROUNDING_BELOW_THRESHOLD",
                    severity="high",
                    message=f"Groundedness {groundedness.value:.2f} below minimum {policy.grounding.minimum:.2f}",
                )
            )

        hallucination_risk = scores.get("hallucination_risk")
        if (
            policy.hallucination.maximum is not None
            and hallucination_risk is not None
            and hallucination_risk.status == ScoreStatus.MEASURED
            and hallucination_risk.value > policy.hallucination.maximum
        ):
            additional_violations.append(
                Violation(
                    code="HALLUCINATION_ABOVE_THRESHOLD",
                    severity="critical",
                    message=(
                        f"Hallucination risk {hallucination_risk.value:.2f} above maximum "
                        f"{policy.hallucination.maximum:.2f}"
                    ),
                )
            )

        instruction_adherence = scores.get("instruction_adherence")
        if (
            policy.instructions.minimum is not None
            and instruction_adherence is not None
            and instruction_adherence.status == ScoreStatus.MEASURED
            and instruction_adherence.value < policy.instructions.minimum
        ):
            additional_violations.append(
                Violation(
                    code="INSTRUCTION_VIOLATION",
                    severity="high",
                    message=(
                        f"Instruction adherence {instruction_adherence.value:.2f} below minimum "
                        f"{policy.instructions.minimum:.2f}"
                    ),
                )
            )

        tool_correctness = scores.get("tool_correctness")
        if (
            policy.tools.minimum is not None
            and tool_correctness is not None
            and tool_correctness.status == ScoreStatus.MEASURED
            and tool_correctness.value < policy.tools.minimum
        ):
            additional_violations.append(
                Violation(
                    code="TOOL_ERROR",
                    severity="high",
                    message=f"Tool correctness {tool_correctness.value:.2f} below minimum {policy.tools.minimum:.2f}",
                )
            )

        # Check citation requirements
        if policy.citations.required and not has_citations:
            additional_violations.append(
                Violation(
                    code="CITATION_MISMATCH",
                    severity="high",
                    message="Citations are required by policy but none were supplied",
                )
            )

        citation_support = scores.get("citation_support")
        if (
            policy.citations.minimum_support is not None
            and citation_support is not None
            and citation_support.status == ScoreStatus.MEASURED
            and citation_support.value < policy.citations.minimum_support
        ):
            additional_violations.append(
                Violation(
                    code="CITATION_MISMATCH",
                    severity="high",
                    message=(
                        f"Citation support {citation_support.value:.2f} below minimum "
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
        force_fail = policy.tools.fail_on_error and any(v.code == "TOOL_ERROR" for v in all_violations)

        effective_verdict = OverallVerdict.FAIL if force_fail else verdict
        action = self._determine_action(effective_verdict, policy)

        return action, additional_violations

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
