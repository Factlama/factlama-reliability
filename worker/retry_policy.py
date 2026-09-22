"""Bounded-retry classification (docs/async-evaluation.md): "Retry
RATE_LIMIT and UNAVAILABLE with exponential backoff plus jitter under a
total deadline; do not retry invalid schema, authorization, configuration
or malformed provider responses."

`Verifier.dispatch_claims()` never raises for a provider failure -- every
claim's own dispatch failure already becomes a typed `Attempt` (outcome
FAILED, a `JudgeErrorCode` value in `.error`). Retryability is therefore
decided from that batch of per-claim attempts directly, not from a caught
exception or a single error code.
"""

import random

from schemas.verification import Attempt, AttemptOutcome

_RETRYABLE_JUDGE_ERRORS = frozenset({"RATE_LIMIT", "UNAVAILABLE", "TIMEOUT"})
"""CONFIGURATION, INVALID_RESPONSE, CANCELLED, BUDGET_EXHAUSTED,
NO_COMPLIANT_PROVIDER and REVOKED are all excluded deliberately -- none of
them describe a transient condition another attempt could plausibly fix
(docs/async-evaluation.md's "do not retry invalid schema, authorization,
configuration or malformed provider responses")."""


def is_retryable(attempts: list[Attempt]) -> bool:
    """Whether at least one claim's dispatch in this attempt failed with a
    transient error worth retrying the whole claim set for.

    A mix of successes and transient failures within the same attempt is
    still retryable -- the retry re-dispatches every claim (not just the
    failed ones), so a claim that already succeeded gets a second,
    independent judgment too; `worker.reconciliation` is what decides
    whether that second judgment agrees.
    """
    return any(
        attempt.outcome == AttemptOutcome.FAILED and attempt.error in _RETRYABLE_JUDGE_ERRORS
        for attempt in attempts
    )


def compute_backoff_seconds(
    attempt_number: int,
    base_seconds: float,
    max_seconds: float,
    jitter_fraction: float,
    *,
    _random: random.Random | None = None,
) -> float:
    """Exponential backoff capped at `max_seconds`, plus up to
    `jitter_fraction` of the capped value as random jitter. `attempt_number`
    is 1-based (the attempt that just finished); this is the delay before
    the next one. `_random` is a seam for deterministic tests, never set in
    production."""
    rng = _random or random
    exponential = min(base_seconds * (2 ** (attempt_number - 1)), max_seconds)
    jitter = exponential * jitter_fraction * rng.random()
    return exponential + jitter
