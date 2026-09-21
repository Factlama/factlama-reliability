"""`StorageBackend`: the one atomic persistence port G5 defines (ADR-019
SS2). Jobs, idempotency, evaluations and outbox form a single logical
backend rather than four independent repositories, because several
operations below must commit changes across those views atomically --
splitting them into separate ports would let a caller (correctly) compose
them non-atomically and reintroduce the exact half-committed states this
protocol exists to rule out.

`PostgresStorageBackend` (`storage/postgres/backend.py`) is the only
implementation today. Nothing outside `storage/` may import it directly
(`pyproject.toml`'s import-linter contract enforces this); everything else
-- the future async API (PR2) and worker (PR3) -- depends on this Protocol
so a post-MVP operator-supplied backend (ADR-019) can substitute for
Postgres without touching API/worker code.

No method here takes or returns a SQLAlchemy `Connection`/`Engine`, a raw
SQL string, or a driver-specific exception (ADR-019 SS2: "Public signatures
must not expose ORM sessions, database transactions, SQL expressions, BSON
documents, driver exceptions, or vendor-specific identifiers."). Failures
are one of `storage.errors`'s typed exceptions.
"""

from datetime import datetime, timedelta
from typing import Protocol

from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest, VerificationResult
from storage.models import ClaimedJob, Job, OutboxEvent


class StorageBackend(Protocol):
    """One tenant-scoped, atomic backend for Reliability's async job,
    idempotency, evaluation and outbox state."""

    async def submit_or_get_job(
        self,
        tenant: TenantContext,
        route: str,
        idempotency_key: str,
        canonical_request_hash: str,
        request: VerificationRequest,
        deadline: datetime,
    ) -> Job:
        """Atomically create the job and its scoped idempotency mapping, or
        return the existing job for a matching `(tenant, route,
        project_id, idempotency_key)` with the same
        `canonical_request_hash`. Never commits a mapping without its job,
        or an orphan job on a lost race between two concurrent identical
        submissions (ADR-019 SS4.2).

        Raises `storage.errors.JobConflict` when the same key is reused
        with a different `canonical_request_hash` (API.md's 409
        `CONFLICT`)."""
        ...

    async def claim_due_job(self, worker_id: str, lease_duration: timedelta) -> ClaimedJob | None:
        """Claim one job that is `QUEUED` (and due, per any
        `release_for_retry` backoff), or `RUNNING` with an expired lease,
        with a fresh fencing token and lease; increments `attempt_count`.
        Returns `None` when no job is due. Concurrent claimers never both
        hold the same lease (ADR-019 SS4.3)."""
        ...

    async def renew_lease(self, job_id: str, fencing_token: int, extension: timedelta) -> datetime:
        """Extend the current lease for a still-owned claim. Returns the
        new `lease_expires_at`. Raises `storage.errors.LeaseExpired` if
        `fencing_token` no longer matches the job's current lease (a newer
        worker has already reclaimed it)."""
        ...

    async def commit_evaluation_and_outbox(
        self,
        job_id: str,
        fencing_token: int,
        result: VerificationResult,
        event: OutboxEvent,
    ) -> None:
        """Verify the fencing token still holds, persist the immutable
        evaluation and its stable outbox event, and transition the job to
        `SUCCEEDED` -- all as one atomic operation (ADR-019 SS4.4). A
        cancelled or reclaimed job's stale commit raises
        `storage.errors.LeaseExpired` instead of silently succeeding. A
        retried commit with byte-identical `result`/`event` for an
        already-committed `evaluation_id` succeeds without raising
        (idempotent); one with different content raises
        `storage.errors.EvaluationConflict`."""
        ...

    async def fail_job_terminal(
        self,
        job_id: str,
        fencing_token: int,
        error_code: str,
        error_message: str,
    ) -> None:
        """Transition a job to terminal `FAILED` after bounded retries are
        exhausted (`docs/async-evaluation.md`: "keep a sanitized error and
        attempt history inspectable to authorized operators"). `FAILED` is
        always operational, never a factual verdict (API.md) -- a factual
        abstention is a `SUCCEEDED` job whose result's status is
        `ABSTAINED`, committed through `commit_evaluation_and_outbox`.
        Raises `storage.errors.LeaseExpired` on a stale fencing token."""
        ...

    async def release_for_retry(
        self, job_id: str, fencing_token: int, not_before: datetime
    ) -> None:
        """Release a claimed job back to `QUEUED`, eligible for
        `claim_due_job` again only once `not_before` has passed (the
        caller's own backoff+jitter delay -- `docs/async-evaluation.md`).
        `attempt_count` already increased when this claim began
        (`claim_due_job`); it is not incremented a second time here.
        Raises `storage.errors.LeaseExpired` on a stale fencing token."""
        ...

    async def request_cancel(self, tenant: TenantContext, job_id: str) -> None:
        """Best-effort cancellation request. Does not itself force a state
        transition -- `docs/async-evaluation.md`: "A cancellation request
        is best effort and must be checked before each external call and
        commit," by the worker holding the current lease. Raises
        `storage.errors.JobNotFound` for a missing or cross-tenant job id
        (CONTRACTS.md: same 404 either way)."""
        ...

    async def is_cancellation_requested(self, job_id: str) -> bool:
        """Whether `request_cancel` has been called for this job. Polled by
        a worker before each external call and before
        `commit_evaluation_and_outbox`; observing `True` does not itself
        change job state -- the worker's own subsequent
        `fail_job_terminal`/cancellation-commit path does."""
        ...

    async def get_job_status(self, tenant: TenantContext, job_id: str) -> Job | None:
        """Tenant-scoped job status for `GET
        /v0.1/verification-jobs/{job_id}`. Returns `None` for a missing or
        cross-tenant job id."""
        ...

    async def get_result(
        self, tenant: TenantContext, evaluation_id: str
    ) -> VerificationResult | None:
        """Tenant-scoped result for `GET /v0.1/verifications/{evaluation_id}`.
        Returns `None` for a missing or cross-tenant evaluation id. Content
        governance (capture-mode-driven redaction, PR4) is applied before
        the result reaches this return value, not by the caller."""
        ...

    async def claim_outbox_batch(
        self, worker_id: str, lease_duration: timedelta, limit: int
    ) -> list[OutboxEvent]:
        """Claim up to `limit` outbox events not yet acknowledged and not
        currently leased by another claimer, for at-least-once delivery to
        Observability's ingress (G6, not yet built -- this method has no
        real consumer yet, only conformance/integration tests)."""
        ...

    async def ack_outbox(self, event_id: str, worker_id: str) -> None:
        """Acknowledge successful delivery of a claimed outbox event,
        removing it from future `claim_outbox_batch` results. A stale
        acknowledgment (event reclaimed by another worker after this
        worker's lease expired) is a no-op, not an error -- delivery is
        at-least-once by design (CONTRACTS.md)."""
        ...
