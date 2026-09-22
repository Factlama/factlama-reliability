"""Result builder for verification."""

import uuid
from typing import Any

from schemas.claims import ClaimVerification
from schemas.policy import PolicyAction
from schemas.verification import (
    AbstentionReason,
    DisputeReason,
    OverallVerdict,
    Provenance,
    ResultStatus,
    ScoreValue,
    Usage,
    VerificationResult,
    Violation,
)


class VerificationResultBuilder:
    """Builder for creating verification results."""

    def __init__(
        self,
        request_id: str,
        tenant_id: str,
        project_id: str,
        application_id: str,
        interaction_id: str | None = None,
    ) -> None:
        """Initialize the result builder.

        Args:
            request_id: ID of the verification request.
            tenant_id: Trusted tenant identifier (never from the request payload).
            project_id: Echoed from the request.
            application_id: Echoed from the request.
            interaction_id: Echoed from the request, if supplied.
        """
        self.evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.application_id = application_id
        self.interaction_id = interaction_id
        self.trace_id: str | None = None
        self.span_id: str | None = None
        self.status: ResultStatus = ResultStatus.COMPLETED
        self.abstention_reason: AbstentionReason | None = None
        self.dispute_reason: DisputeReason | None = None
        self.verdict: OverallVerdict = OverallVerdict.PASS
        self.scores: dict[str, ScoreValue] = {}
        self.claims: list[ClaimVerification] = []
        self.violations: list[Violation] = []
        self.policy_action: str | None = None
        self.policy_version: str | None = None
        self.provenance: Provenance | None = None
        self.usage_summary: Usage = Usage()
        self.metadata: dict[str, Any] = {}

    def with_status(self, status: ResultStatus) -> "VerificationResultBuilder":
        self.status = status
        return self

    def with_abstention_reason(
        self, reason: AbstentionReason | None
    ) -> "VerificationResultBuilder":
        self.abstention_reason = reason
        return self

    def with_dispute_reason(self, reason: DisputeReason | None) -> "VerificationResultBuilder":
        self.dispute_reason = reason
        return self

    def with_verdict(self, verdict: OverallVerdict) -> "VerificationResultBuilder":
        self.verdict = verdict
        return self

    def with_scores(self, scores: dict[str, ScoreValue]) -> "VerificationResultBuilder":
        self.scores = scores
        return self

    def with_claims(self, claims: list[ClaimVerification]) -> "VerificationResultBuilder":
        self.claims = claims
        return self

    def with_violations(self, violations: list[Violation]) -> "VerificationResultBuilder":
        self.violations = violations
        return self

    def with_policy_action(self, action: PolicyAction) -> "VerificationResultBuilder":
        self.policy_action = action.value
        return self

    def with_metadata(self, metadata: dict[str, Any]) -> "VerificationResultBuilder":
        self.metadata.update(metadata)
        return self

    def with_trace_id(self, trace_id: str | None) -> "VerificationResultBuilder":
        self.trace_id = trace_id
        return self

    def with_span_id(self, span_id: str | None) -> "VerificationResultBuilder":
        self.span_id = span_id
        return self

    def with_policy_version(self, policy_version: str | None) -> "VerificationResultBuilder":
        self.policy_version = policy_version
        return self

    def with_provenance(self, provenance: Provenance) -> "VerificationResultBuilder":
        self.provenance = provenance
        return self

    def with_usage_summary(self, usage_summary: Usage) -> "VerificationResultBuilder":
        self.usage_summary = usage_summary
        return self

    def build(self) -> VerificationResult:
        """Build the verification result.

        Raises:
            ValueError: If `provenance` was never set -- every result must
                record how it was produced.
        """
        if self.provenance is None:
            raise ValueError("VerificationResultBuilder.build() requires provenance to be set")

        return VerificationResult(
            evaluation_id=self.evaluation_id,
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            application_id=self.application_id,
            interaction_id=self.interaction_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            status=self.status,
            abstention_reason=self.abstention_reason,
            dispute_reason=self.dispute_reason,
            verdict=self.verdict,
            scores=self.scores,
            claims=self.claims,
            violations=self.violations,
            policy_action=self.policy_action,
            policy_version=self.policy_version,
            provenance=self.provenance,
            usage_summary=self.usage_summary,
            metadata=self.metadata,
        )
