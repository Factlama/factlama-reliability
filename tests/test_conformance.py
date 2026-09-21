"""Tests for `core.conformance`'s T0 suite (F8 of the 2026-09-21 G0-G4
validation report).

Two kinds of coverage:
1. Every real T0 provider (Mock, RuleBased always; Embedding/NLI when their
   extras are installed) actually clears the suite -- proving it isn't
   vacuous against the adapters it exists to gate.
2. Deliberately non-conformant stub providers each trip exactly the check
   they're built to violate -- proving the suite detects real violations,
   not just that it passes everything.
"""

import time

from core.conformance import run_t0_conformance_suite
from judges.port import (
    CancellationToken,
    JudgeError,
    JudgeErrorCode,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
)
from judges.providers import MockModelProvider, RuleBasedProvider
from schemas.claims import ClaimVerdict, RationaleCode


class TestRealProvidersPassConformance:
    def test_mock_provider(self) -> None:
        assert run_t0_conformance_suite(MockModelProvider()) == []

    def test_rule_based_provider(self) -> None:
        assert run_t0_conformance_suite(RuleBasedProvider()) == []

    def test_embedding_provider(self) -> None:
        import pytest

        pytest.importorskip("sentence_transformers")
        from judges.vendor_adapters import EmbeddingProvider

        assert run_t0_conformance_suite(EmbeddingProvider()) == []

    def test_nli_provider(self) -> None:
        import pytest

        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("sentence_transformers")
        from judges.vendor_adapters import NLIProvider

        assert run_t0_conformance_suite(NLIProvider()) == []


class _IgnoresDeadlineAndCancellationProvider(JudgeProvider):
    """Never checks `deadline`/`cancellation` -- always returns SUPPORTED."""

    @property
    def name(self) -> str:
        return "stub:ignores-deadline"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        return JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=[e.evidence_id for e in request.evidence],
        )


class _AlwaysErrorsProvider(JudgeProvider):
    """Returns an error even for a well-formed, in-time request."""

    @property
    def name(self) -> str:
        return "stub:always-errors"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        return JudgeResult(error=JudgeError(code=JudgeErrorCode.UNAVAILABLE, message="stub"))


class _CrashesOnAmbiguousProvider(JudgeProvider):
    """Raises instead of returning a typed result."""

    @property
    def name(self) -> str:
        return "stub:crashes"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)
        raise RuntimeError("boom")


class _FabricatesSupportFromNothingProvider(JudgeProvider):
    """Returns SUPPORTED even for reference-only (unreadable) evidence."""

    @property
    def name(self) -> str:
        return "stub:fabricates"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)
        return JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=[e.evidence_id for e in request.evidence],
        )


class _SupportedWithNoCitationProvider(JudgeProvider):
    """R2 of the 2026-09-21 re-audit's exact reproduction: honors timeout/
    cancellation, returns INSUFFICIENT_EVIDENCE for unreadable references,
    but returns SUPPORTED with `evidence_ids=[]` for perfectly readable
    evidence -- a citation-invariant violation the old suite (which only
    checked `result.error is None`) let straight through with zero
    failures."""

    @property
    def name(self) -> str:
        return "stub:supported-no-citation"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        error = _bounded_check(deadline, cancellation)
        if error is not None:
            return JudgeResult(error=error)
        if not any((e.content or "") for e in request.evidence):
            return JudgeResult(
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                rationale_code=RationaleCode.NO_EVIDENCE_SUPPLIED,
            )
        return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=[])


def _bounded_check(deadline: float, cancellation: CancellationToken) -> JudgeError | None:
    if cancellation.is_cancelled:
        return JudgeError(code=JudgeErrorCode.CANCELLED, message="cancelled")
    if time.monotonic() > deadline:
        return JudgeError(code=JudgeErrorCode.TIMEOUT, message="deadline exceeded")
    return None


class TestConformanceSuiteDetectsViolations:
    def test_ignoring_deadline_and_cancellation_fails_timeout_and_cancellation_checks(
        self,
    ) -> None:
        failures = run_t0_conformance_suite(_IgnoresDeadlineAndCancellationProvider())
        assert any(f.startswith("timeout:") for f in failures)
        assert any(f.startswith("cancellation:") for f in failures)

    def test_always_erroring_fails_the_success_check(self) -> None:
        failures = run_t0_conformance_suite(_AlwaysErrorsProvider())
        assert any(f.startswith("success:") for f in failures)

    def test_crashing_fails_ambiguity_and_malformed_response_checks(self) -> None:
        failures = run_t0_conformance_suite(_CrashesOnAmbiguousProvider())
        assert any(f.startswith("ambiguity:") for f in failures)
        assert any(f.startswith("malformed-response:") for f in failures)

    def test_fabricating_support_fails_the_malformed_response_check(self) -> None:
        failures = run_t0_conformance_suite(_FabricatesSupportFromNothingProvider())
        assert any(f.startswith("malformed-response:") for f in failures)

    def test_supported_with_no_citation_fails_the_success_check(self) -> None:
        """R2 of the 2026-09-21 re-audit's exact reproduction: this provider
        honors timeout/cancellation and handles unreadable evidence
        correctly -- only the success case (readable evidence, SUPPORTED
        with no citation) is non-conformant. The pre-fix suite reported
        zero failures for this provider entirely."""
        failures = run_t0_conformance_suite(_SupportedWithNoCitationProvider())
        assert any(f.startswith("success:") for f in failures)
        assert any("citation" in f for f in failures)
