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
from typing import Any, Protocol

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
        `CONFLICT`). Raises `storage.errors.ExecutionStorageNotAuthorized`
        (ADR-020) when the request's effective `execution_storage.mode` is
        `NO_PERSISTENCE` -- before any insert, not after a partial write."""
        ...

    async def claim_due_job(self, worker_id: str, lease_duration: timedelta) -> ClaimedJob | None:
        """Claim one job that is `QUEUED` (and due, per any
        `release_for_retry` backoff), or `RUNNING` with an expired lease,
        with a fresh fencing token and lease; increments `attempt_count`.
        Returns `None` when no job is due. Concurrent claimers never both
        hold the same lease (ADR-019 SS4.3).

        ADR-020: every job reaching this point is `DURABLE_ENCRYPTED` by
        construction (`NO_PERSISTENCE` never creates one); `ClaimedJob.request`
        is the real, decrypted request, checked against its absolute expiry
        before decryption is even attempted; `ClaimedJob.input_expires_at`
        carries that same absolute expiry forward so a caller can bound
        dispatch by it too (A5 of the 2026-10-14 ADR-020 review). An
        already-expired, missing, corrupt, or previously-terminated
        (tombstoned -- A4 of the same review) input never produces a
        claim: it is terminalized directly, in the same transaction that
        would otherwise have claimed it, and this method returns `None`
        for that call -- never silently returns an empty-input request
        that could fabricate a passing verdict from nothing, and never
        raises in a way that would roll the claim back and leave the job
        claimable (and reselected) again (A1 of the same review)."""
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
        *,
        governed_request_json: dict[str, Any] | None = None,
    ) -> None:
        """Verify the fencing token still holds, persist the immutable
        evaluation and its stable outbox event, and transition the job to
        `SUCCEEDED` -- all as one atomic operation (ADR-019 SS4.4). A
        cancelled or reclaimed job's stale commit raises
        `storage.errors.LeaseExpired` instead of silently succeeding. A
        byte-identical retry for the *same* already-committed
        `evaluation_id` succeeds without raising (idempotent); a
        conflicting retry under the same `evaluation_id` raises
        `storage.errors.EvaluationConflict` (content differs), and so does
        a second, independently-computed completion for a job already
        `SUCCEEDED` under a *different* `evaluation_id` (duplicate delivery
        of the same claimed work item; ADR-019 SS4.4 permits only a retry
        of the same logical completion, not a second one).

        `governed_request_json`, when supplied, atomically overwrites this
        job's stored `request_json` in the same transaction (F1 of the
        2026-10-13 G0-G6 code review: once a job is terminal, nothing reads
        its raw request content back again, so it no longer needs to
        persist). ADR-020: this terminal transition also atomically deletes
        the job's `job_execution_inputs` row and records a permanent
        tombstone (`execution_input_tombstones`, A4 of the 2026-10-14
        ADR-020 review) in the same transaction -- "atomically revoke
        active payload access ... and schedule deletion."

        A5 of the 2026-10-14 ADR-020 review: for a first-time completion
        (job currently `RUNNING`, not an idempotent retry of an
        already-`SUCCEEDED` job), this re-validates the input's own
        absolute expiry before committing -- raises
        `storage.errors.ExecutionInputExpired` if it has expired or was
        destroyed since the claim that produced `result`, refusing to
        commit a factual verdict over input that is no longer verified.
        """
        ...

    async def fail_job_terminal(
        self,
        job_id: str,
        fencing_token: int,
        error_code: str,
        error_message: str,
        *,
        governed_request_json: dict[str, Any] | None = None,
        terminal_provenance_json: dict[str, Any] | None = None,
    ) -> None:
        """Transition a job to terminal `FAILED` after bounded retries are
        exhausted (`docs/async-evaluation.md`: "keep a sanitized error and
        attempt history inspectable to authorized operators"). `FAILED` is
        always operational, never a factual verdict (API.md) -- a factual
        abstention is a `SUCCEEDED` job whose result's status is
        `ABSTAINED`, committed through `commit_evaluation_and_outbox`.
        Raises `storage.errors.LeaseExpired` on a stale fencing token.

        `governed_request_json`: see `commit_evaluation_and_outbox`'s own
        parameter of the same name. ADR-020: this terminal transition also
        atomically deletes the job's `job_execution_inputs` row and records
        a permanent tombstone (`execution_input_tombstones`, A4 of the
        2026-10-14 ADR-020 review), same as `commit_evaluation_and_outbox`.

        `terminal_provenance_json`: P2 of the 2026-10-14 ADR-020 closure
        review -- when a dispatch already produced real attempt/usage
        history before this specific rejection (e.g. the input expired
        between claim and commit), that safe, never content-bearing
        history is persisted here rather than discarded outright. Not
        populated by every caller -- only the ones that have real history
        to preserve.
        """
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
        governance (`core.governance.apply_capture_mode`, ADR-008) already
        ran before this row was written -- `worker.runner._commit` governs
        the result at commit time, so whatever this method reads back is
        already the governed copy; there is nothing left for this method or
        its caller to filter."""
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

    async def purge_expired_execution_inputs(self) -> int:
        """ADR-020 section 4: delete every `job_execution_inputs` row past
        its absolute `expires_at`, regardless of the owning job's state --
        the independent sweep (`worker.expiry_cleanup`) that bounds
        retention even when no worker ever claims the job at all. Unlike
        `claim_outbox_batch`/`ack_outbox`, this needs no per-item
        claim/lease protocol: a plain delete-where-expired is safe to run
        concurrently from multiple workers (redundant, not racy). Returns
        the number of rows purged. `PostgresStorageBackend`'s own
        implementation also logs the oldest expiry age among purged rows
        on every sweep that finds work -- the cleanup-lag monitoring signal
        ADR-020 section 4 asks for."""
        ...
