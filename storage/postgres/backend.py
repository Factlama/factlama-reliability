"""`PostgresStorageBackend`: the concrete `storage.backend.StorageBackend`
implementation for G5 (ADR-019). Every public method opens exactly one
`AsyncEngine.begin()` transaction (or a plain `connect()` for read-only
methods) -- ADR-019 SS2's atomicity boundary is enforced here, not by
composing smaller repository calls at a caller.

Concurrency primitives used throughout:
- `SELECT ... FOR UPDATE SKIP LOCKED` for `claim_due_job`, so multiple
  worker processes claiming concurrently never block on or double-claim
  the same row.
- A monotonic per-job `fencing_token`, bumped only by `claim_due_job`.
  Every other state-changing method requires the caller's token to still
  match the stored one, so a worker that held a lease before it expired
  and was reclaimed cannot silently win a race against the new holder
  (ADR-019 SS4.3).
- Content-hash comparison on `evaluations`/`outbox_events` so a retried
  `commit_evaluation_and_outbox` with byte-identical content is treated as
  success, not as a conflict (ADR-019 SS4.4).
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, delete, exists, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from core.execution_storage import (
    EncryptedPayload,
    ExecutionInputDecryptionFailed,
    ExecutionStorageConfigurationError,
    ExecutionStorageKeyring,
    decrypt_request_payload,
    encrypt_request_payload,
)
from core.governance import governed_request_payload
from schemas.policy import ExecutionStorageMode, Policy, resolve_effective_execution_storage_mode
from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest, VerificationResult
from storage.backend import StorageBackend
from storage.errors import (
    EvaluationConflict,
    ExecutionInputExpired,
    ExecutionStorageNotAuthorized,
    JobConflict,
    JobNotFound,
    LeaseExpired,
)
from storage.models import ClaimedJob, ClaimSummary, Job, JobState, OutboxEvent, to_wire_timestamp
from storage.postgres.tables import (
    evaluations,
    execution_input_tombstones,
    idempotency_keys,
    job_execution_inputs,
    jobs,
    outbox_events,
)

logger = logging.getLogger(__name__)


def _hash_json(value: dict[str, Any]) -> str:
    """Stable content hash for idempotent-commit comparison. `sort_keys`
    makes the hash independent of dict insertion order, which
    `model_dump`/`json.loads` do not otherwise guarantee across calls."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _outbox_event_to_dict(event: OutboxEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": event.event_id,
        "tenant_id": event.tenant_id,
        "evaluation_id": event.evaluation_id,
        "occurred_at": to_wire_timestamp(event.occurred_at),
        "status": event.status,
        "verdict": event.verdict,
        # See `event_to_reliability_event_json`'s own note: this dict is
        # kept field-for-field identical to the delivered wire body, so
        # the same `exclude_none`/Z-suffix contract rules apply here.
        "scores": {
            name: score.model_dump(mode="json", exclude_none=True)
            for name, score in event.scores.items()
        },
        "evaluator_version": event.evaluator_version,
        "calibration_class": event.calibration_class,
        "qualification_status": event.qualification_status,
        "usage_summary": event.usage_summary.model_dump(mode="json", exclude_none=True),
        "project_id": event.project_id,
        "application_id": event.application_id,
        "interaction_id": event.interaction_id,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "calibration_classes": list(event.calibration_classes),
        # ADR-022; see `event_to_reliability_event_json`'s own note on why
        # an absent projection is an omitted key, not a null.
        "claim_summary": (
            event.claim_summary.to_wire_json() if event.claim_summary is not None else None
        ),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _row_to_job(row: Any) -> Job:
    return Job(
        job_id=row.job_id,
        tenant_id=row.tenant_id,
        project_id=row.project_id,
        application_id=row.application_id,
        state=JobState(row.state),
        created_at=row.created_at,
        updated_at=row.updated_at,
        attempt_count=row.attempt_count,
        evaluation_id=row.evaluation_id,
    )


_JOB_STATUS_COLUMNS = (
    jobs.c.job_id,
    jobs.c.tenant_id,
    jobs.c.project_id,
    jobs.c.application_id,
    jobs.c.state,
    jobs.c.created_at,
    jobs.c.updated_at,
    jobs.c.attempt_count,
    jobs.c.evaluation_id,
)


class PostgresStorageBackend(StorageBackend):
    """See `storage.backend.StorageBackend` for the contract this
    implements. Construct one instance per process, sharing one
    `AsyncEngine` from `storage.postgres.engine.create_storage_engine`
    (ADR-019 SS3)."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        execution_storage_keyring: ExecutionStorageKeyring | None = None,
        default_execution_input_ttl_seconds: int = 3600,
    ) -> None:
        self._engine = engine
        # ADR-020: `None` is valid construction (mirrors the keyring's own
        # fail-closed design) -- a DURABLE_ENCRYPTED submission against an
        # instance with no keyring configured fails loudly at that point,
        # never silently falling back to plaintext storage.
        self._execution_storage_keyring = execution_storage_keyring
        self._default_execution_input_ttl = timedelta(seconds=default_execution_input_ttl_seconds)

    async def submit_or_get_job(
        self,
        tenant: TenantContext,
        route: str,
        idempotency_key: str,
        canonical_request_hash: str,
        request: VerificationRequest,
        deadline: datetime,
    ) -> Job:
        # ADR-020: the real backstop against persisting raw inputs under
        # NO_PERSISTENCE, independent of whatever `api/app.py` already
        # checked -- raised before any insert, not after a partial write,
        # so a caller retrying the same idempotency key against a
        # NO_PERSISTENCE policy gets a clean rejection every time, never a
        # half-committed mapping.
        #
        # A3 of the 2026-10-14 ADR-020 review: this must resolve against
        # `tenant`'s own trusted grant, not the request's selection alone --
        # a request can narrow permissions but never grant itself storage
        # rights (ADR-020 section 1). `tenant` comes from authenticated
        # ingress context (never a client-payload field), so
        # `tenant.execution_storage_grant` is the real ceiling here.
        requested_mode = (
            request.policy.execution_storage.mode
            if request.policy
            else ExecutionStorageMode.NO_PERSISTENCE
        )
        execution_storage_mode = resolve_effective_execution_storage_mode(
            requested=requested_mode, tenant_grant=tenant.execution_storage_grant
        )
        if execution_storage_mode == ExecutionStorageMode.NO_PERSISTENCE:
            raise ExecutionStorageNotAuthorized(
                "execution_storage.mode=NO_PERSISTENCE does not authorize durable "
                "persistence of this request's input; submit synchronously instead, "
                "or have the tenant explicitly granted DURABLE_ENCRYPTED"
            )

        # ADR-020: the only other mode is DURABLE_ENCRYPTED -- encrypt the
        # complete request now (before opening the transaction; this is
        # pure CPU work, no need to hold a connection for it) and insert the
        # resulting ciphertext atomically with the job itself below.
        # Fails loudly (ExecutionStorageConfigurationError) if no keyring or
        # no active key is configured -- never silently falls back to
        # plaintext storage.
        if self._execution_storage_keyring is None:
            raise ExecutionStorageConfigurationError(
                "DURABLE_ENCRYPTED execution storage requires a configured "
                "execution-storage keyring; none was provided to this backend"
            )
        job_id = f"job_{uuid4().hex}"
        key_id, key = self._execution_storage_keyring.active_key()
        encrypted_payload = encrypt_request_payload(
            request, tenant_id=tenant.tenant_id, job_id=job_id, key_id=key_id, key=key
        )
        capture_mode = (request.policy or Policy(id="default")).capture.mode
        governed_submission_json = governed_request_payload(request, capture_mode)

        now = datetime.now(timezone.utc)
        # 2026-10-14 review A7: the configured TTL alone could outlive this
        # job's own execution deadline (default 1h retention vs. a 5-minute
        # deadline), leaving the input recoverable long after the job could
        # ever legitimately still be running. ADR-020 section 4: the input
        # expiry must be no later than "the permitted execution lifetime."
        execution_input_expires_at = min(now + self._default_execution_input_ttl, deadline)
        async with self._engine.begin() as conn:
            # Claim the idempotency mapping first, inside this same
            # transaction: if we lose the race, we never touch `jobs` at
            # all, so no orphan job can ever be committed.
            claim_stmt = (
                pg_insert(idempotency_keys)
                .values(
                    tenant_id=tenant.tenant_id,
                    route=route,
                    project_id=request.project_id,
                    idempotency_key=idempotency_key,
                    canonical_request_hash=canonical_request_hash,
                    job_id=job_id,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=["tenant_id", "route", "project_id", "idempotency_key"]
                )
                .returning(idempotency_keys.c.job_id)
            )
            won = (await conn.execute(claim_stmt)).first()
            if won is None:
                existing = (
                    await conn.execute(
                        select(
                            idempotency_keys.c.canonical_request_hash,
                            idempotency_keys.c.job_id,
                        ).where(
                            idempotency_keys.c.tenant_id == tenant.tenant_id,
                            idempotency_keys.c.route == route,
                            idempotency_keys.c.project_id == request.project_id,
                            idempotency_keys.c.idempotency_key == idempotency_key,
                        )
                    )
                ).one()
                if existing.canonical_request_hash != canonical_request_hash:
                    raise JobConflict(
                        f"idempotency key {idempotency_key!r} was already used with a "
                        "different request body"
                    )
                row = (
                    await conn.execute(
                        select(*_JOB_STATUS_COLUMNS).where(jobs.c.job_id == existing.job_id)
                    )
                ).one()
                return _row_to_job(row)

            # ADR-020: `jobs.request_json` stores the already-governed
            # (redacted) projection from the moment of insertion, not the
            # raw request -- the real content lives only in
            # `job_execution_inputs.ciphertext`, inserted atomically below
            # in this same transaction (ADR-020 section 3: "Commit job,
            # idempotency mapping and encrypted payload reference together
            # in the existing PostgreSQL transaction").
            await conn.execute(
                jobs.insert().values(
                    job_id=job_id,
                    tenant_id=tenant.tenant_id,
                    project_id=request.project_id,
                    application_id=request.application_id,
                    route=route,
                    state=JobState.QUEUED.value,
                    request_json=governed_submission_json,
                    deadline=deadline,
                    created_at=now,
                    updated_at=now,
                    attempt_count=0,
                    fencing_token=0,
                    cancel_requested=False,
                )
            )
            await conn.execute(
                job_execution_inputs.insert().values(
                    tenant_id=tenant.tenant_id,
                    job_id=job_id,
                    key_id=encrypted_payload.key_id,
                    payload_version=encrypted_payload.payload_version,
                    nonce=encrypted_payload.nonce,
                    ciphertext=encrypted_payload.ciphertext,
                    created_at=now,
                    expires_at=execution_input_expires_at,
                )
            )
            return Job(
                job_id=job_id,
                tenant_id=tenant.tenant_id,
                project_id=request.project_id,
                application_id=request.application_id,
                state=JobState.QUEUED,
                created_at=now,
                updated_at=now,
                attempt_count=0,
                evaluation_id=None,
            )

    async def claim_due_job(self, worker_id: str, lease_duration: timedelta) -> ClaimedJob | None:
        now = datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            due = or_(
                and_(
                    jobs.c.state == JobState.QUEUED.value,
                    or_(jobs.c.available_at.is_(None), jobs.c.available_at <= now),
                ),
                and_(
                    jobs.c.state == JobState.RUNNING.value,
                    jobs.c.lease_expires_at < now,
                ),
            )
            candidate = (
                await conn.execute(
                    select(
                        jobs.c.job_id,
                        jobs.c.tenant_id,
                        jobs.c.project_id,
                        jobs.c.application_id,
                        jobs.c.deadline,
                    )
                    .where(due)
                    .order_by(jobs.c.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).first()
            if candidate is None:
                return None

            # A4 of the 2026-10-14 ADR-020 review: a job that already
            # reached a terminal state once must never be claimable again,
            # no matter what the (possibly restored, out-of-date)
            # `jobs`/`job_execution_inputs` rows themselves claim -- a
            # tombstone here is never deleted, so restoring only those two
            # tables (e.g. from a pre-termination backup) cannot revive
            # access. See `execution_input_tombstones`'s own docstring for
            # this fix's real scope and its named limit.
            tombstoned = (
                await conn.execute(
                    select(execution_input_tombstones.c.job_id).where(
                        execution_input_tombstones.c.tenant_id == candidate.tenant_id,
                        execution_input_tombstones.c.job_id == candidate.job_id,
                    )
                )
            ).first()
            if tombstoned is not None:
                await self._terminalize_unusable_claim(
                    conn,
                    candidate.tenant_id,
                    candidate.job_id,
                    now,
                    error_code="EXECUTION_INPUT_REVOKED",
                    error_message=(
                        f"job {candidate.job_id} was already terminated once; its "
                        "appearance as claimable again indicates a restored or "
                        "otherwise out-of-date jobs/job_execution_inputs snapshot"
                    ),
                )
                return None

            # A1 of the 2026-10-14 ADR-020 review: validate the encrypted
            # input BEFORE transitioning to RUNNING, not after -- the old
            # order raised from inside this same transaction on a missing/
            # expired/corrupt input, which rolled back the RUNNING
            # transition along with it and left the job QUEUED, still due,
            # and immediately reselected by this same oldest-first query
            # forever. A permanently-unusable input is instead terminalized
            # directly, in this same transaction, and this call returns
            # `None` -- "no work claimed this round" -- so the poll loop
            # moves on to whatever else is due. A genuine keyring
            # misconfiguration is NOT the same kind of problem (it affects
            # every job equally and is operator-fixable) -- that still
            # raises and rolls back, leaving the row claimable once this
            # worker's configuration is corrected, loud enough to stop a
            # misconfigured worker rather than silently failing every job
            # it touches one at a time.
            input_row = (
                await conn.execute(
                    select(
                        job_execution_inputs.c.key_id,
                        job_execution_inputs.c.payload_version,
                        job_execution_inputs.c.nonce,
                        job_execution_inputs.c.ciphertext,
                        job_execution_inputs.c.expires_at,
                    ).where(
                        job_execution_inputs.c.tenant_id == candidate.tenant_id,
                        job_execution_inputs.c.job_id == candidate.job_id,
                    )
                )
            ).first()

            if input_row is None:
                await self._terminalize_unusable_claim(
                    conn,
                    candidate.tenant_id,
                    candidate.job_id,
                    now,
                    error_code="EXECUTION_INPUT_CORRUPTED",
                    error_message=f"job {candidate.job_id} has no execution_input row to decrypt",
                )
                return None
            if now > input_row.expires_at:
                await self._terminalize_unusable_claim(
                    conn,
                    candidate.tenant_id,
                    candidate.job_id,
                    now,
                    error_code="EXECUTION_INPUT_EXPIRED",
                    error_message=(
                        f"job {candidate.job_id}'s execution input expired at "
                        f"{input_row.expires_at.isoformat()}"
                    ),
                )
                return None
            if self._execution_storage_keyring is None:
                raise ExecutionStorageConfigurationError(
                    "DURABLE_ENCRYPTED execution storage requires a configured "
                    "execution-storage keyring; none was provided to this backend"
                )
            try:
                key = self._execution_storage_keyring.resolve(input_row.key_id)
            except ExecutionStorageConfigurationError:
                # P1 of the 2026-10-14 ADR-020 closure review: a backend
                # with *some* keyring configured, but no entry for this
                # specific row's key_id, is not the same kind of problem as
                # no keyring at all. This is the enforcement mechanism
                # behind the mandatory restore procedure ADR-020 section 5
                # requires: retiring (removing from the keyring) every
                # key_id that might have encrypted a now-stale row is a
                # real, tested way to make a restored database's rows --
                # revived-terminal ones and legitimately-unfinished ones
                # alike -- permanently undecryptable, not merely documented
                # as a runbook warning. Terminalize this one row, do not
                # block the rest of the queue the way a blanket
                # misconfiguration correctly does above.
                await self._terminalize_unusable_claim(
                    conn,
                    candidate.tenant_id,
                    candidate.job_id,
                    now,
                    error_code="EXECUTION_INPUT_KEY_RETIRED",
                    error_message=(
                        f"job {candidate.job_id}'s execution input uses key_id="
                        f"{input_row.key_id!r}, which is not present in this worker's "
                        "configured keyring (retired, or a restored row from before a "
                        "key rotation)"
                    ),
                )
                return None
            payload = EncryptedPayload(
                key_id=input_row.key_id,
                payload_version=input_row.payload_version,
                nonce=input_row.nonce,
                ciphertext=input_row.ciphertext,
            )
            try:
                request = decrypt_request_payload(
                    payload, tenant_id=candidate.tenant_id, job_id=candidate.job_id, key=key
                )
            except ExecutionInputDecryptionFailed as exc:
                await self._terminalize_unusable_claim(
                    conn,
                    candidate.tenant_id,
                    candidate.job_id,
                    now,
                    error_code="EXECUTION_INPUT_CORRUPTED",
                    error_message=str(exc),
                )
                return None

            lease_expires_at = now + lease_duration
            claimed = (
                await conn.execute(
                    update(jobs)
                    .where(jobs.c.job_id == candidate.job_id)
                    .values(
                        state=JobState.RUNNING.value,
                        lease_owner=worker_id,
                        lease_expires_at=lease_expires_at,
                        fencing_token=jobs.c.fencing_token + 1,
                        attempt_count=jobs.c.attempt_count + 1,
                        updated_at=now,
                    )
                    .returning(jobs.c.fencing_token, jobs.c.attempt_count)
                )
            ).one()

            return ClaimedJob(
                job_id=candidate.job_id,
                tenant=TenantContext(
                    tenant_id=candidate.tenant_id,
                    project_id=candidate.project_id,
                    application_id=candidate.application_id,
                ),
                request=request,
                fencing_token=claimed.fencing_token,
                attempt_count=claimed.attempt_count,
                deadline=candidate.deadline,
                input_expires_at=input_row.expires_at,
                lease_expires_at=lease_expires_at,
            )

    async def _terminalize_unusable_claim(
        self,
        conn: Any,
        tenant_id: str,
        job_id: str,
        now: datetime,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        """Shared by `claim_due_job`'s four "this input can never be used"
        branches (A1/A4 of the 2026-10-14 ADR-020 review): transition
        straight from QUEUED/RUNNING-with-expired-lease to `FAILED` --
        skipping a RUNNING cycle entirely, since there is nothing left to
        run -- clear the lease, delete any `job_execution_inputs` row
        (already gone for the "missing"/"tombstoned" cases; harmless), and
        record a permanent tombstone (idempotent -- already present for the
        "tombstoned" case) so this job can never again look claimable, no
        matter what a later restored snapshot of `jobs`/
        `job_execution_inputs` claims (A4)."""
        await conn.execute(
            update(jobs)
            .where(jobs.c.job_id == job_id)
            .values(
                state=JobState.FAILED.value,
                error_code=error_code,
                error_message=error_message,
                lease_owner=None,
                lease_expires_at=None,
                updated_at=now,
            )
        )
        await conn.execute(
            delete(job_execution_inputs).where(job_execution_inputs.c.job_id == job_id)
        )
        await conn.execute(
            pg_insert(execution_input_tombstones)
            .values(tenant_id=tenant_id, job_id=job_id, terminated_at=now)
            .on_conflict_do_nothing(index_elements=["tenant_id", "job_id"])
        )

    async def renew_lease(self, job_id: str, fencing_token: int, extension: timedelta) -> datetime:
        now = datetime.now(timezone.utc)
        new_expiry = now + extension
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(jobs)
                .where(
                    jobs.c.job_id == job_id,
                    jobs.c.fencing_token == fencing_token,
                    jobs.c.state == JobState.RUNNING.value,
                )
                .values(lease_expires_at=new_expiry, updated_at=now)
            )
            if result.rowcount == 0:
                raise LeaseExpired(
                    f"job {job_id} lease no longer held (fencing_token={fencing_token})"
                )
        return new_expiry

    async def commit_evaluation_and_outbox(
        self,
        job_id: str,
        fencing_token: int,
        result: VerificationResult,
        event: OutboxEvent,
        *,
        governed_request_json: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        result_json = json.loads(result.model_dump_json(exclude_none=True))
        result_hash = _hash_json(result_json)
        event_payload = _outbox_event_to_dict(event)
        event_hash = _hash_json(event_payload)

        async with self._engine.begin() as conn:
            job_row = (
                await conn.execute(
                    select(
                        jobs.c.tenant_id,
                        jobs.c.project_id,
                        jobs.c.application_id,
                        jobs.c.state,
                        jobs.c.fencing_token,
                        jobs.c.evaluation_id,
                    )
                    .where(jobs.c.job_id == job_id)
                    .with_for_update()
                )
            ).first()
            if job_row is None:
                raise JobNotFound(f"job {job_id} not found")
            if job_row.fencing_token != fencing_token:
                raise LeaseExpired(
                    f"job {job_id} lease no longer held (fencing_token={fencing_token})"
                )
            if job_row.state == JobState.SUCCEEDED.value:
                # ADR-019 SS4.4 permits only a byte-identical retry of the
                # *same* logical completion to succeed without raising --
                # not a second, independently-computed completion under a
                # different evaluation_id. Without this check, a duplicate
                # delivery of the same claimed work item (e.g. a retried
                # worker call after a transient commit failure) would
                # silently create a second evaluation row and move this
                # job's `evaluation_id` pointer out from under the first,
                # violating `storage/models.py`'s own "terminal states
                # cannot be overwritten by late workers" rule.
                if job_row.evaluation_id != result.evaluation_id:
                    raise EvaluationConflict(
                        f"job {job_id} already succeeded with evaluation "
                        f"{job_row.evaluation_id!r}; refusing to commit a different "
                        f"evaluation {result.evaluation_id!r} as a second completion"
                    )
            elif job_row.state != JobState.RUNNING.value:
                raise LeaseExpired(
                    f"job {job_id} is already terminal as {job_row.state}; cannot commit a success"
                )

            if job_row.state == JobState.RUNNING.value:
                # A5 of the 2026-10-14 ADR-020 review: re-validate the
                # input's own absolute expiry at commit time too, not only
                # at claim -- a claim that was valid when issued can still
                # have its input expire (and be purged by the independent
                # sweep) while bounded retries/backoff consume real
                # wall-clock time, and nothing before this re-checked that
                # before trusting a factual PASS/FAIL enough to commit it.
                # Skipped for the idempotent-retry branch above
                # (job_row.state == SUCCEEDED): that job's input row was
                # already, correctly, deleted by its own first successful
                # commit, and re-requiring it here would wrongly reject a
                # legitimate retry of an already-succeeded completion.
                input_row = (
                    await conn.execute(
                        select(job_execution_inputs.c.expires_at).where(
                            job_execution_inputs.c.tenant_id == job_row.tenant_id,
                            job_execution_inputs.c.job_id == job_id,
                        )
                    )
                ).first()
                if input_row is None or now > input_row.expires_at:
                    raise ExecutionInputExpired(
                        f"job {job_id}'s execution input expired or was destroyed "
                        "before this completion could be committed; refusing to "
                        "commit a factual result over input that is no longer "
                        "verified"
                    )

            existing_eval = (
                await conn.execute(
                    select(evaluations.c.content_hash).where(
                        evaluations.c.tenant_id == job_row.tenant_id,
                        evaluations.c.evaluation_id == result.evaluation_id,
                    )
                )
            ).first()
            if existing_eval is not None:
                if existing_eval.content_hash != result_hash:
                    raise EvaluationConflict(
                        f"evaluation {result.evaluation_id} already exists with different content"
                    )
            else:
                await conn.execute(
                    evaluations.insert().values(
                        tenant_id=job_row.tenant_id,
                        evaluation_id=result.evaluation_id,
                        job_id=job_id,
                        project_id=job_row.project_id,
                        application_id=job_row.application_id,
                        result_json=result_json,
                        content_hash=result_hash,
                        created_at=now,
                    )
                )

            existing_event = (
                await conn.execute(
                    select(outbox_events.c.content_hash).where(
                        outbox_events.c.tenant_id == job_row.tenant_id,
                        outbox_events.c.event_id == event.event_id,
                    )
                )
            ).first()
            if existing_event is not None:
                if existing_event.content_hash != event_hash:
                    raise EvaluationConflict(
                        f"outbox event {event.event_id} already exists with different content"
                    )
            else:
                await conn.execute(
                    outbox_events.insert().values(
                        tenant_id=job_row.tenant_id,
                        event_id=event.event_id,
                        evaluation_id=event.evaluation_id,
                        payload_json=event_payload,
                        content_hash=event_hash,
                        created_at=now,
                        acked=False,
                    )
                )

            job_update_values: dict[str, Any] = {
                "state": JobState.SUCCEEDED.value,
                "evaluation_id": result.evaluation_id,
                "updated_at": now,
            }
            if governed_request_json is not None:
                # F1 of the 2026-10-13 G0-G6 code review: once terminal,
                # nothing reads this column back into a VerificationRequest
                # again (`claim_due_job`'s claim predicate excludes
                # terminal states) -- overwrite it in the same transaction
                # that marks the job SUCCEEDED, not as a separate later
                # write that could leave a window with raw content visible
                # on an already-terminal row.
                job_update_values["request_json"] = governed_request_json
            await conn.execute(
                update(jobs).where(jobs.c.job_id == job_id).values(**job_update_values)
            )
            # ADR-020 section 4: "At terminal transition, atomically revoke
            # active payload access within the backend ... schedule
            # deletion" -- a real DELETE, in the same transaction as the
            # state transition, not a separate later write that could leave
            # a window where a terminal job's ciphertext is still readable.
            # A byte-identical retry of this same completion (the branch
            # above) deletes a row that may already be gone -- harmless, a
            # plain DELETE matching zero rows is not an error.
            await conn.execute(
                delete(job_execution_inputs).where(job_execution_inputs.c.job_id == job_id)
            )
            # A4 of the 2026-10-14 ADR-020 review: a permanent record that
            # this job reached a terminal state -- see
            # `execution_input_tombstones`'s own docstring. Idempotent: a
            # byte-identical retry of this same completion (the branch
            # above) inserts a tombstone that is already there.
            await conn.execute(
                pg_insert(execution_input_tombstones)
                .values(tenant_id=job_row.tenant_id, job_id=job_id, terminated_at=now)
                .on_conflict_do_nothing(index_elements=["tenant_id", "job_id"])
            )

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
        now = datetime.now(timezone.utc)
        fail_values: dict[str, Any] = {
            "state": JobState.FAILED.value,
            "error_code": error_code,
            "error_message": error_message,
            "updated_at": now,
        }
        if governed_request_json is not None:
            fail_values["request_json"] = governed_request_json
        if terminal_provenance_json is not None:
            # P2 of the 2026-10-14 ADR-020 closure review: a dispatch that
            # already happened before this rejection produced real, safe
            # (never content-bearing) attempt/usage history -- preserve it
            # on the operational failure outcome rather than losing it
            # outright, the same atomic write as the state transition.
            fail_values["terminal_provenance_json"] = terminal_provenance_json
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(jobs)
                .where(
                    jobs.c.job_id == job_id,
                    jobs.c.fencing_token == fencing_token,
                    jobs.c.state == JobState.RUNNING.value,
                )
                .values(**fail_values)
                .returning(jobs.c.tenant_id)
            )
            row = result.first()
            if row is None:
                raise LeaseExpired(
                    f"job {job_id} lease no longer held (fencing_token={fencing_token})"
                )
            # ADR-020 section 4: FAILED is terminal too -- revoke access to
            # the encrypted input in the same transaction as the state
            # transition, same reasoning as commit_evaluation_and_outbox.
            await conn.execute(
                delete(job_execution_inputs).where(job_execution_inputs.c.job_id == job_id)
            )
            # A4 of the 2026-10-14 ADR-020 review: same permanent tombstone
            # as commit_evaluation_and_outbox -- FAILED is terminal too.
            await conn.execute(
                pg_insert(execution_input_tombstones)
                .values(tenant_id=row.tenant_id, job_id=job_id, terminated_at=now)
                .on_conflict_do_nothing(index_elements=["tenant_id", "job_id"])
            )

    async def release_for_retry(
        self, job_id: str, fencing_token: int, not_before: datetime
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(jobs)
                .where(
                    jobs.c.job_id == job_id,
                    jobs.c.fencing_token == fencing_token,
                    jobs.c.state == JobState.RUNNING.value,
                )
                .values(
                    state=JobState.QUEUED.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    available_at=not_before,
                    updated_at=now,
                )
            )
            if result.rowcount == 0:
                raise LeaseExpired(
                    f"job {job_id} lease no longer held (fencing_token={fencing_token})"
                )

    async def request_cancel(self, tenant: TenantContext, job_id: str) -> None:
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    select(jobs.c.tenant_id, jobs.c.project_id, jobs.c.application_id).where(
                        jobs.c.job_id == job_id
                    )
                )
            ).first()
            if (
                row is None
                or row.tenant_id != tenant.tenant_id
                or not tenant.authorizes(row.project_id, row.application_id)
            ):
                raise JobNotFound(f"job {job_id} not found")
            await conn.execute(
                update(jobs).where(jobs.c.job_id == job_id).values(cancel_requested=True)
            )

    async def is_cancellation_requested(self, job_id: str) -> bool:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(select(jobs.c.cancel_requested).where(jobs.c.job_id == job_id))
            ).first()
            return bool(row.cancel_requested) if row is not None else False

    async def get_job_status(self, tenant: TenantContext, job_id: str) -> Job | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(select(*_JOB_STATUS_COLUMNS).where(jobs.c.job_id == job_id))
            ).first()
            if row is None or row.tenant_id != tenant.tenant_id:
                return None
            if not tenant.authorizes(row.project_id, row.application_id):
                return None
            return _row_to_job(row)

    async def get_result(
        self, tenant: TenantContext, evaluation_id: str
    ) -> VerificationResult | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(
                        evaluations.c.tenant_id,
                        evaluations.c.project_id,
                        evaluations.c.application_id,
                        evaluations.c.result_json,
                    ).where(evaluations.c.evaluation_id == evaluation_id)
                )
            ).first()
            if row is None or row.tenant_id != tenant.tenant_id:
                return None
            if not tenant.authorizes(row.project_id, row.application_id):
                return None
            return VerificationResult.model_validate(row.result_json)

    async def claim_outbox_batch(
        self, worker_id: str, lease_duration: timedelta, limit: int
    ) -> list[OutboxEvent]:
        now = datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            claimable = and_(
                outbox_events.c.acked.is_(False),
                or_(
                    outbox_events.c.lease_expires_at.is_(None),
                    outbox_events.c.lease_expires_at < now,
                ),
            )
            candidates = (
                await conn.execute(
                    select(outbox_events.c.tenant_id, outbox_events.c.event_id)
                    .where(claimable)
                    .order_by(outbox_events.c.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            if not candidates:
                return []

            lease_expires_at = now + lease_duration
            events: list[OutboxEvent] = []
            for candidate in candidates:
                row = (
                    await conn.execute(
                        update(outbox_events)
                        .where(
                            outbox_events.c.tenant_id == candidate.tenant_id,
                            outbox_events.c.event_id == candidate.event_id,
                        )
                        .values(lease_owner=worker_id, lease_expires_at=lease_expires_at)
                        .returning(outbox_events.c.payload_json)
                    )
                ).one()
                events.append(_dict_to_outbox_event(row.payload_json))
            return events

    async def ack_outbox(self, event_id: str, worker_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(outbox_events)
                .where(
                    outbox_events.c.event_id == event_id,
                    outbox_events.c.lease_owner == worker_id,
                )
                .values(acked=True, lease_owner=None, lease_expires_at=None)
            )

    async def purge_expired_execution_inputs(self) -> int:
        """ADR-020 section 4's cleanup-lag monitoring: besides deleting
        every expired row, logs the age of the *oldest* expiry this sweep
        observed (`now - expires_at`, not `now - created_at`) -- the real
        signal for "how far behind is cleanup," since a row's own absolute
        expiry, not its age, is what bounds how long it may stay readable.

        P2 of the 2026-10-14 ADR-020 closure review: deleting the input
        alone left its owning job stuck `QUEUED` (or `RUNNING` with an
        already-dead lease) forever whenever no dispatch worker ever claims
        it again -- "independent of dispatch workers" must mean this sweep
        reaches a terminal outcome on its own, not merely that it deletes a
        row and waits for a claim that may never come. For every expired
        row whose owning job is currently unclaimed by a *live* lease
        (`QUEUED`, or `RUNNING` with `lease_expires_at` already past -- the
        same "due" definition `claim_due_job` itself uses), this
        terminalizes the job to `FAILED` and records its permanent
        tombstone (A4) in the same transaction as the delete. A job
        `RUNNING` under a still-valid lease is left alone: forcing it
        terminal out from under a live worker would itself be a race, and
        `commit_evaluation_and_outbox`'s own expiry recheck (A5) is the
        real backstop that already prevents that worker from committing a
        trusted success over an input this sweep is about to delete --
        fencing enforced there, not by racing this sweep against it.

        P2 of the 2026-10-14 ADR-020 follow-up review: that left-alone
        live-lease job still has its (now expired) ciphertext deleted below
        -- ADR-020 requires prompt destruction of expired input regardless
        of lease state, and retaining it solely so a *future* sweep's
        `expired` query can still find the row is not an acceptable
        alternative. So if that worker then dies without renewing or
        completing, there is nothing left in `job_execution_inputs` for a
        later sweep to key off of. This sweep therefore also separately
        reconciles `RUNNING` jobs whose lease has already expired and which
        have no `job_execution_inputs` row at all -- the same "missing
        input" case `claim_due_job` already terminalizes when a worker
        happens to claim one (A1/A4), done here too so reaching that
        terminal outcome never depends on a worker claiming it again.
        """
        now = datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            expired = (
                await conn.execute(
                    select(
                        job_execution_inputs.c.tenant_id,
                        job_execution_inputs.c.job_id,
                        job_execution_inputs.c.expires_at,
                    ).where(job_execution_inputs.c.expires_at < now)
                )
            ).all()

            for row in expired:
                job_row = (
                    await conn.execute(
                        select(jobs.c.state, jobs.c.lease_expires_at)
                        .where(jobs.c.job_id == row.job_id)
                        .with_for_update()
                    )
                ).first()
                if job_row is None:
                    continue
                unclaimed_by_a_live_lease = job_row.state == JobState.QUEUED.value or (
                    job_row.state == JobState.RUNNING.value
                    and (job_row.lease_expires_at is None or job_row.lease_expires_at < now)
                )
                if not unclaimed_by_a_live_lease:
                    continue
                await conn.execute(
                    update(jobs)
                    .where(jobs.c.job_id == row.job_id)
                    .values(
                        state=JobState.FAILED.value,
                        error_code="EXECUTION_INPUT_EXPIRED",
                        error_message=(
                            f"job {row.job_id}'s execution input expired and was purged "
                            "by the independent cleanup sweep while unclaimed"
                        ),
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                await conn.execute(
                    pg_insert(execution_input_tombstones)
                    .values(tenant_id=row.tenant_id, job_id=row.job_id, terminated_at=now)
                    .on_conflict_do_nothing(index_elements=["tenant_id", "job_id"])
                )

            purged_count = 0
            if expired:
                result = await conn.execute(
                    delete(job_execution_inputs).where(job_execution_inputs.c.expires_at < now)
                )
                purged_count = result.rowcount or 0

            orphaned_leases = (
                await conn.execute(
                    select(jobs.c.tenant_id, jobs.c.job_id)
                    .where(
                        jobs.c.state == JobState.RUNNING.value,
                        jobs.c.lease_expires_at < now,
                        ~exists(
                            select(job_execution_inputs.c.job_id).where(
                                job_execution_inputs.c.job_id == jobs.c.job_id
                            )
                        ),
                    )
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for row in orphaned_leases:
                await self._terminalize_unusable_claim(
                    conn,
                    row.tenant_id,
                    row.job_id,
                    now,
                    error_code="EXECUTION_INPUT_EXPIRED",
                    error_message=(
                        f"job {row.job_id}'s execution input was already purged by an "
                        "earlier sweep while its lease was still live, and that lease "
                        "has since expired with no worker ever completing it"
                    ),
                )

        if expired:
            oldest_expiry = min(row.expires_at for row in expired)
            lag_seconds = (now - oldest_expiry).total_seconds()
            logger.info(
                "execution-input expiry sweep purged %d row(s); oldest cleanup lag %.1fs",
                len(expired),
                lag_seconds,
            )
        return purged_count


def _dict_to_outbox_event(payload: dict[str, Any]) -> OutboxEvent:
    from schemas.verification import ScoreValue, Usage

    return OutboxEvent(
        event_id=payload["event_id"],
        tenant_id=payload["tenant_id"],
        evaluation_id=payload["evaluation_id"],
        occurred_at=datetime.fromisoformat(payload["occurred_at"]),
        status=payload["status"],
        verdict=payload["verdict"],
        scores={
            name: ScoreValue.model_validate(score)
            for name, score in payload.get("scores", {}).items()
        },
        evaluator_version=payload["evaluator_version"],
        calibration_class=payload["calibration_class"],
        qualification_status=payload["qualification_status"],
        usage_summary=Usage.model_validate(payload["usage_summary"]),
        project_id=payload.get("project_id"),
        application_id=payload.get("application_id"),
        interaction_id=payload.get("interaction_id"),
        trace_id=payload.get("trace_id"),
        span_id=payload.get("span_id"),
        calibration_classes=list(payload.get("calibration_classes", [])),
        # ADR-022: absent key -> None -> "unavailable" downstream, which is
        # exactly how a pre-ADR-022 payload already sitting in the outbox
        # must be read. Without this line the projection would be dropped
        # on the real delivery path, since that path stores the payload and
        # reads it back before delivering.
        claim_summary=(
            ClaimSummary.from_wire_json(payload["claim_summary"])
            if payload.get("claim_summary") is not None
            else None
        ),
    )
