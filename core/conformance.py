"""T0 conformance suite (G4's subset of REL-13's acceptance bar).

`evaluator-agreement-harness.md`: "Conformance (ADR-014's first lifecycle
transition, REGISTERED -> CONFORMANCE_PASSED): the existing success/
ambiguity/timeout/rate-limit/malformed-response fixture suite (REL-13's
acceptance bar) against the JudgeProvider port itself -- does the adapter
return well-formed JudgeResults and typed errors under the documented
failure modes." REL-13's full ecosystem-tier acceptance (two adapters,
provider-enforced quota, a real vendor HTTP call) is NOT_STARTED and stays
out of scope here (G10) -- this module is the T0 subset F8 of the
2026-09-21 G0-G4 validation report assigned to G4: every check a purely
in-process (no vendor HTTP call) adapter can actually be held to.

Every current adapter (Mock, RuleBased, Embedding, NLI -- ADR-011's T0
tier) runs in-process and has no vendor rate limiter to trip, so RATE_LIMIT
is not simulated here; it is REL-13's concern once a real HTTP vendor call
exists. Cooperative cancellation (`CancellationToken`) is the bounded
port's own mechanism in its place, so it is checked instead.
"""

import time
from collections.abc import Callable

from judges.port import (
    CancellationToken,
    JudgeErrorCode,
    JudgeProvider,
    JudgeRequest,
    JudgeResult,
    validate_judge_result,
)
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence

_FAR_DEADLINE_SECONDS = 30.0

_SUCCESS_CLAIM = Claim(
    claim_id="conformance-success", text="Water boils at 100 degrees Celsius at sea level."
)
_SUCCESS_EVIDENCE = [
    Evidence(
        evidence_id="conformance-success-e1",
        content="At sea level, water boils at 100 degrees Celsius.",
    )
]

#: Evidence present but genuinely inconclusive -- the "ambiguous" family
#: (evaluator-agreement-harness.md), used here only to prove a provider
#: doesn't crash or return a malformed result on a genuinely hard case, not
#: to score its verdict (that is `scripts/run_agreement_harness.py`'s job).
_AMBIGUOUS_CLAIM = Claim(claim_id="conformance-ambiguous", text="The policy was broadly popular.")
_AMBIGUOUS_EVIDENCE = [
    Evidence(
        evidence_id="conformance-ambiguous-e1",
        content="Polling showed mixed reactions, with support varying widely across regions.",
    )
]

#: Reference-only evidence (no inline content) is the T0 analog of a
#: malformed vendor response: the adapter has no real text body to read and
#: must degrade to a typed result, never crash and never fabricate support
#: from evidence it cannot actually read.
_MALFORMED_CLAIM = Claim(
    claim_id="conformance-malformed", text="The report was published last year."
)
_MALFORMED_EVIDENCE = [
    Evidence(evidence_id="conformance-malformed-e1", reference={"source": "redacted"})
]


def _citation_invariant_failure(result: JudgeResult, request: JudgeRequest) -> str | None:
    """Every SUPPORTED verdict must cite at least one evidence ID that is
    actually part of `request` -- the same invariant
    `judges.port.validate_judge_result()` enforces at the live dispatch
    boundary. Re-checked here (R2 of the 2026-09-21 re-audit) because a
    provider's *own raw output* must satisfy this, not merely whatever
    `core.verifier` happens to repair downstream -- a provider that returns
    `SUPPORTED` with `evidence_ids=[]` (or citing an ID never given to it)
    is not port-conformant even though a caller who remembers to run it
    through `validate_judge_result()` would never see the bad citation."""
    validated = validate_judge_result(result, request)
    if validated.error is not None and result.error is None:
        return f"returned a result that fails port-level citation validation: {validated.error}"
    return None


def _check_success(provider: JudgeProvider) -> str | None:
    request = JudgeRequest(claim=_SUCCESS_CLAIM, evidence=_SUCCESS_EVIDENCE)
    result = provider.evaluate(
        request, time.monotonic() + _FAR_DEADLINE_SECONDS, CancellationToken()
    )
    if result.error is not None:
        return f"well-formed request returned an error instead of a verdict: {result.error}"
    return _citation_invariant_failure(result, request)


def _check_ambiguity(provider: JudgeProvider) -> str | None:
    request = JudgeRequest(claim=_AMBIGUOUS_CLAIM, evidence=_AMBIGUOUS_EVIDENCE)
    result = provider.evaluate(
        request, time.monotonic() + _FAR_DEADLINE_SECONDS, CancellationToken()
    )
    if result.verdict is None and result.error is None:
        return "JudgeResult set neither verdict nor error"
    return _citation_invariant_failure(result, request)


def _check_timeout(provider: JudgeProvider) -> str | None:
    already_past_deadline = time.monotonic() - 1.0
    result = provider.evaluate(
        JudgeRequest(claim=_SUCCESS_CLAIM, evidence=_SUCCESS_EVIDENCE),
        already_past_deadline,
        CancellationToken(),
    )
    if result.error is None or result.error.code != JudgeErrorCode.TIMEOUT:
        return f"an already-past deadline did not produce a TIMEOUT JudgeError (got {result!r})"
    return None


def _check_cancellation(provider: JudgeProvider) -> str | None:
    cancellation = CancellationToken()
    cancellation.cancel()
    result = provider.evaluate(
        JudgeRequest(claim=_SUCCESS_CLAIM, evidence=_SUCCESS_EVIDENCE),
        time.monotonic() + _FAR_DEADLINE_SECONDS,
        cancellation,
    )
    if result.error is None or result.error.code != JudgeErrorCode.CANCELLED:
        return f"a pre-cancelled token did not produce a CANCELLED JudgeError (got {result!r})"
    return None


def _check_malformed_response_robustness(provider: JudgeProvider) -> str | None:
    request = JudgeRequest(claim=_MALFORMED_CLAIM, evidence=_MALFORMED_EVIDENCE)
    result = provider.evaluate(
        request, time.monotonic() + _FAR_DEADLINE_SECONDS, CancellationToken()
    )
    if result.verdict is None and result.error is None:
        return "JudgeResult set neither verdict nor error"
    if result.verdict == ClaimVerdict.SUPPORTED:
        return "fabricated a SUPPORTED verdict from evidence with no readable content"
    return _citation_invariant_failure(result, request)


#: (label, check) pairs, ordered so a failure list reads in the same
#: success/ambiguity/timeout/rate-limit/malformed-response order
#: evaluator-agreement-harness.md names them (rate-limit is T0-out-of-scope;
#: see module docstring). The label prefixes every failure string
#: `run_t0_conformance_suite` returns, whether the check function itself
#: reported a failure or raised -- a crash is exactly the kind of
#: not-well-formed behavior this suite exists to catch, not a suite bug.
_CHECKS: tuple[tuple[str, Callable[[JudgeProvider], str | None]], ...] = (
    ("success", _check_success),
    ("ambiguity", _check_ambiguity),
    ("timeout", _check_timeout),
    ("cancellation", _check_cancellation),
    ("malformed-response", _check_malformed_response_robustness),
)


def run_t0_conformance_suite(provider: JudgeProvider) -> list[str]:
    """Run every T0 conformance check against `provider`.

    Returns human-readable failure reasons; empty means every check passed
    and `core.qualification.mark_conformance_passed()` may be called for
    this provider. This is a pass/fail gate on the port contract itself
    (well-formed results, typed errors under documented failure modes), not
    a quality measurement -- an adapter can pass every check here and still
    fail `evaluator-agreement-harness.md`'s agreement thresholds.
    """
    failures: list[str] = []
    for label, check in _CHECKS:
        try:
            failure = check(provider)
        except Exception as exc:
            failures.append(
                f"{label}: evaluate() raised {exc!r} instead of returning a typed result"
            )
            continue
        if failure is not None:
            failures.append(f"{label}: {failure}")
    return failures
