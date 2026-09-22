"""The worker's job-processing loop (G5, `docs/async-evaluation.md`):
claim a due job, run bounded dispatch/retry, reconcile cross-attempt
disagreement, then commit a terminal outcome.

States (`docs/async-evaluation.md`): `QUEUED -> RUNNING ->
SUCCEEDED|FAILED|CANCELLED`. `SUCCEEDED` covers a factual
`COMPLETED`/`ABSTAINED`/`DISPUTED` result alike -- `FAILED` is always
operational (a typed error), never a factual verdict (API.md).
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone

from core.budgets import check_request_budget
from core.compliance import is_provider_compliant
from core.verifier import Verifier
from judges.port import JudgeErrorCode
from schemas.claims import ClaimVerification
from schemas.policy import Policy
from schemas.verification import AbstentionReason, Attempt, VerificationResult, Violation
from storage.backend import StorageBackend
from storage.errors import LeaseExpired
from storage.models import ClaimedJob
from worker.outbox import build_outbox_event
from worker.reconciliation import merge_claim_verifications
from worker.retry_policy import compute_backoff_seconds, is_retryable
from worker.settings import WorkerSettings

logger = logging.getLogger(__name__)


class _CancelledMidDispatch(Exception):
    """Internal signal only: cancellation was observed between attempts."""


async def process_claimed_job(
    claimed: ClaimedJob,
    backend: StorageBackend,
    verifier: Verifier,
    settings: WorkerSettings,
) -> None:
    """Run one claimed job to a terminal outcome and commit it.

    Never raises for an ordinary business or operational outcome -- every
    path ends in `commit_evaluation_and_outbox` or `fail_job_terminal`. A
    `storage.errors.LeaseExpired` anywhere (this claim was reclaimed by
    another worker, or expired mid-processing) is logged and swallowed:
    there is nothing more this call can usefully do once that happens, and
    the reclaiming worker owns the job now.
    """
    request = claimed.request
    fencing_token = claimed.fencing_token
    started_at = datetime.now(timezone.utc)
    trace_id = request.trace_id or f"trace_{claimed.job_id}"
    span_id = request.span_id or f"span_{claimed.job_id}"

    try:
        if await backend.is_cancellation_requested(claimed.job_id):
            await backend.fail_job_terminal(
                claimed.job_id, fencing_token, "CANCELLED", "cancelled before dispatch"
            )
            return

        policy = request.policy or Policy(id="default")
        claims = (
            list(request.claims)
            if request.claims
            else verifier.claim_extractor.extract(request.answer)
        )

        budget_violation = check_request_budget(len(claims), len(request.evidence))
        if budget_violation is not None:
            logger.warning("Job %s abstained before dispatch: %s", claimed.job_id, budget_violation)
            result = verifier.abstained_result(
                claimed.tenant.tenant_id,
                started_at,
                request,
                reason=AbstentionReason.BUDGET_EXHAUSTED,
                error_label=JudgeErrorCode.BUDGET_EXHAUSTED.value,
                claim_count=len(claims),
                trace_id=trace_id,
                span_id=span_id,
            )
            await _commit(backend, claimed.job_id, fencing_token, result)
            return

        if not is_provider_compliant(
            verifier.model_provider.compliance_tags, policy.required_provider_compliance
        ):
            logger.warning(
                "Job %s abstained before dispatch: provider '%s' does not satisfy "
                "policy's required_provider_compliance=%s",
                claimed.job_id,
                verifier.model_provider.name,
                policy.required_provider_compliance,
            )
            result = verifier.abstained_result(
                claimed.tenant.tenant_id,
                started_at,
                request,
                reason=AbstentionReason.NO_COMPLIANT_PROVIDER,
                error_label=JudgeErrorCode.NO_COMPLIANT_PROVIDER.value,
                claim_count=len(claims),
                trace_id=trace_id,
                span_id=span_id,
            )
            await _commit(backend, claimed.job_id, fencing_token, result)
            return

        (
            attempt_claim_verifications,
            all_attempts,
            calibration_class,
            judge_boundary_violations,
            usage_budget_violation,
        ) = await _dispatch_with_bounded_retries(claimed, claims, backend, verifier, settings)

        if not attempt_claim_verifications:
            # The overall deadline was already past before any attempt
            # could even be dispatched -- this is an operational timeout,
            # not "zero checkable claims" (determine_status would
            # otherwise mislabel it that way from an empty claim_verifications
            # list alone).
            logger.warning(
                "Job %s: overall deadline exceeded before any attempt could be dispatched",
                claimed.job_id,
            )
            await backend.fail_job_terminal(
                claimed.job_id,
                fencing_token,
                JudgeErrorCode.TIMEOUT.value,
                "overall job deadline exceeded before any attempt could be dispatched",
            )
            return

        merged_claims, disagreement_violations = merge_claim_verifications(
            attempt_claim_verifications
        )

        result = verifier.finalize(
            request=request,
            tenant_id=claimed.tenant.tenant_id,
            started_at=started_at,
            trace_id=trace_id,
            span_id=span_id,
            claims=claims,
            claim_verifications=merged_claims,
            attempts=all_attempts,
            calibration_class=calibration_class,
            judge_boundary_violations=judge_boundary_violations + disagreement_violations,
            usage_budget_violation=usage_budget_violation,
        )
        await _commit(backend, claimed.job_id, fencing_token, result)

    except _CancelledMidDispatch:
        logger.info("Job %s cancelled mid-dispatch", claimed.job_id)
        await _fail_terminal_best_effort(
            backend, claimed.job_id, fencing_token, "CANCELLED", "cancelled mid-dispatch"
        )
    except LeaseExpired:
        logger.warning("Job %s: lease lost during processing", claimed.job_id)
    except Exception:
        logger.exception("Job %s: unexpected worker failure", claimed.job_id)
        await _fail_terminal_best_effort(
            backend, claimed.job_id, fencing_token, "INTERNAL", "unexpected worker failure"
        )


async def _dispatch_with_bounded_retries(
    claimed: ClaimedJob,
    claims: list,
    backend: StorageBackend,
    verifier: Verifier,
    settings: WorkerSettings,
) -> tuple[list[list[ClaimVerification]], list[Attempt], str | None, list[Violation], str | None]:
    """Bounded retry loop (`docs/async-evaluation.md`): up to
    `settings.max_attempts` full re-dispatches of every claim, stopping
    early once an attempt has no retryable failures. Returns each attempt's
    `claim_verifications` (for `worker.reconciliation` to merge), every
    attempt's `Attempt` records concatenated in order, the most recent
    calibration class and judge-boundary violations, and the most recent
    usage-budget violation (each attempt tracks its own request-scoped
    budget independently, per ADR-012's per-request scope)."""
    attempt_claim_verifications: list[list[ClaimVerification]] = []
    all_attempts: list[Attempt] = []
    calibration_class: str | None = None
    judge_boundary_violations: list[Violation] = []
    usage_budget_violation: str | None = None
    fencing_token = claimed.fencing_token

    for attempt_number in range(1, settings.max_attempts + 1):
        if await backend.is_cancellation_requested(claimed.job_id):
            raise _CancelledMidDispatch

        if datetime.now(timezone.utc) >= claimed.deadline:
            logger.warning(
                "Job %s: overall deadline reached before attempt %d",
                claimed.job_id,
                attempt_number,
            )
            break

        (
            claim_verifications,
            attempts,
            this_calibration_class,
            this_judge_boundary_violations,
            this_usage_budget_violation,
        ) = verifier.dispatch_claims(claims, claimed.request.evidence)

        attempt_claim_verifications.append(claim_verifications)
        all_attempts.extend(attempts)
        if this_calibration_class is not None:
            calibration_class = this_calibration_class
        judge_boundary_violations = this_judge_boundary_violations
        usage_budget_violation = this_usage_budget_violation

        if not is_retryable(attempts) or attempt_number == settings.max_attempts:
            break

        fencing_token, lease_held = await _renew_lease(
            backend, claimed.job_id, fencing_token, settings
        )
        if not lease_held:
            break

        delay = compute_backoff_seconds(
            attempt_number,
            settings.backoff_base_seconds,
            settings.backoff_max_seconds,
            settings.backoff_jitter_fraction,
        )
        logger.info(
            "Job %s: attempt %d had retryable failures, backing off %.2fs",
            claimed.job_id,
            attempt_number,
            delay,
        )
        await asyncio.sleep(delay)

    return (
        attempt_claim_verifications,
        all_attempts,
        calibration_class,
        judge_boundary_violations,
        usage_budget_violation,
    )


async def _renew_lease(
    backend: StorageBackend, job_id: str, fencing_token: int, settings: WorkerSettings
) -> tuple[int, bool]:
    """Renew the lease before a backoff wait so it does not expire out from
    under a still-in-progress job. Returns `(fencing_token, held)` --
    `renew_lease` never changes the token itself (only `claim_due_job`
    does), so the token is echoed back unchanged; `held` is `False` once
    another worker has already reclaimed the job."""
    try:
        await backend.renew_lease(
            job_id, fencing_token, timedelta(seconds=settings.lease_duration_seconds)
        )
        return fencing_token, True
    except LeaseExpired:
        logger.warning("Job %s: lease lost while renewing before a retry", job_id)
        return fencing_token, False


async def _commit(
    backend: StorageBackend, job_id: str, fencing_token: int, result: VerificationResult
) -> None:
    event = build_outbox_event(result)
    try:
        await backend.commit_evaluation_and_outbox(job_id, fencing_token, result, event)
    except LeaseExpired:
        logger.warning(
            "Job %s: lease lost before commit; result discarded, another worker owns it now",
            job_id,
        )


async def _fail_terminal_best_effort(
    backend: StorageBackend, job_id: str, fencing_token: int, error_code: str, error_message: str
) -> None:
    try:
        await backend.fail_job_terminal(job_id, fencing_token, error_code, error_message)
    except LeaseExpired:
        logger.warning("Job %s: lease already lost; cannot record terminal failure", job_id)


async def run_poll_loop(
    backend: StorageBackend,
    verifier: Verifier,
    settings: WorkerSettings,
    stop_event: asyncio.Event,
) -> None:
    """The worker's main loop: claim a due job, process it to a terminal
    outcome, repeat; sleep `poll_interval_seconds` whenever nothing is due.
    Runs until `stop_event` is set."""
    lease_duration = timedelta(seconds=settings.lease_duration_seconds)
    while not stop_event.is_set():
        claimed = await backend.claim_due_job(settings.worker_id, lease_duration)
        if claimed is None:
            await _sleep_or_stop(settings.poll_interval_seconds, stop_event)
            continue
        await process_claimed_job(claimed, backend, verifier, settings)


async def _sleep_or_stop(seconds: float, stop_event: asyncio.Event) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
