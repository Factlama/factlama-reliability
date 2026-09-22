"""Tests for `worker.retry_policy`."""

import random
from datetime import datetime, timezone

from schemas.verification import Attempt, AttemptOutcome, QualificationStatus
from worker.retry_policy import compute_backoff_seconds, is_retryable


def _attempt(outcome: AttemptOutcome, error: str | None = None) -> Attempt:
    now = datetime.now(timezone.utc)
    return Attempt(
        attempt_id="attempt_1",
        provider_id="test",
        configuration_version="0.1",
        qualification_status=QualificationStatus.UNQUALIFIED,
        calibration_class="NONE",
        outcome=outcome,
        error=error,
        started_at=now,
        completed_at=now,
    )


class TestIsRetryable:
    def test_no_attempts_is_not_retryable(self) -> None:
        assert is_retryable([]) is False

    def test_all_completed_is_not_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.COMPLETED)]) is False

    def test_rate_limit_failure_is_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.FAILED, error="RATE_LIMIT")]) is True

    def test_unavailable_failure_is_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.FAILED, error="UNAVAILABLE")]) is True

    def test_timeout_failure_is_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.FAILED, error="TIMEOUT")]) is True

    def test_configuration_failure_is_not_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.FAILED, error="CONFIGURATION")]) is False

    def test_invalid_response_failure_is_not_retryable(self) -> None:
        assert is_retryable([_attempt(AttemptOutcome.FAILED, error="INVALID_RESPONSE")]) is False

    def test_one_retryable_failure_among_successes_is_retryable(self) -> None:
        attempts = [
            _attempt(AttemptOutcome.COMPLETED),
            _attempt(AttemptOutcome.FAILED, error="RATE_LIMIT"),
        ]
        assert is_retryable(attempts) is True


class TestComputeBackoffSeconds:
    def test_grows_exponentially_before_the_cap(self) -> None:
        rng = random.Random(0)
        first = compute_backoff_seconds(
            1, base_seconds=1.0, max_seconds=100.0, jitter_fraction=0.0, _random=rng
        )
        second = compute_backoff_seconds(
            2, base_seconds=1.0, max_seconds=100.0, jitter_fraction=0.0, _random=rng
        )
        third = compute_backoff_seconds(
            3, base_seconds=1.0, max_seconds=100.0, jitter_fraction=0.0, _random=rng
        )
        assert first == 1.0
        assert second == 2.0
        assert third == 4.0

    def test_capped_at_max_seconds(self) -> None:
        rng = random.Random(0)
        delay = compute_backoff_seconds(
            10, base_seconds=1.0, max_seconds=5.0, jitter_fraction=0.0, _random=rng
        )
        assert delay == 5.0

    def test_jitter_only_adds_never_subtracts(self) -> None:
        rng = random.Random(0)
        delay = compute_backoff_seconds(
            1, base_seconds=2.0, max_seconds=100.0, jitter_fraction=0.5, _random=rng
        )
        assert 2.0 <= delay <= 3.0
