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
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest, VerificationResult
from storage.backend import StorageBackend
from storage.errors import (
    EvaluationConflict,
    JobConflict,
    JobNotFound,
    LeaseExpired,
)
from storage.models import ClaimedJob, Job, JobState, OutboxEvent
from storage.postgres.tables import evaluations, idempotency_keys, jobs, outbox_events


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
        "occurred_at": event.occurred_at.isoformat(),
        "status": event.status,
        "verdict": event.verdict,
        "scores": {name: score.model_dump(mode="json") for name, score in event.scores.items()},
        "evaluator_version": event.evaluator_version,
        "calibration_class": event.calibration_class,
        "qualification_status": event.qualification_status,
        "usage_summary": event.usage_summary.model_dump(mode="json"),
        "project_id": event.project_id,
        "application_id": event.application_id,
        "interaction_id": event.interaction_id,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "calibration_classes": list(event.calibration_classes),
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

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def submit_or_get_job(
        self,
        tenant: TenantContext,
        route: str,
        idempotency_key: str,
        canonical_request_hash: str,
        request: VerificationRequest,
        deadline: datetime,
    ) -> Job:
        now = datetime.now(timezone.utc)
        job_id = f"job_{uuid4().hex}"
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

            request_json = json.loads(request.model_dump_json(exclude_none=True))
            await conn.execute(
                jobs.insert().values(
                    job_id=job_id,
                    tenant_id=tenant.tenant_id,
                    project_id=request.project_id,
                    application_id=request.application_id,
                    route=route,
                    state=JobState.QUEUED.value,
                    request_json=request_json,
                    deadline=deadline,
                    created_at=now,
                    updated_at=now,
                    attempt_count=0,
                    fencing_token=0,
                    cancel_requested=False,
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
                    select(jobs.c.job_id)
                    .where(due)
                    .order_by(jobs.c.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).first()
            if candidate is None:
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
                    .returning(
                        jobs.c.tenant_id,
                        jobs.c.project_id,
                        jobs.c.application_id,
                        jobs.c.request_json,
                        jobs.c.deadline,
                        jobs.c.fencing_token,
                        jobs.c.attempt_count,
                    )
                )
            ).one()
            return ClaimedJob(
                job_id=candidate.job_id,
                tenant=TenantContext(
                    tenant_id=claimed.tenant_id,
                    project_id=claimed.project_id,
                    application_id=claimed.application_id,
                ),
                request=VerificationRequest.model_validate(claimed.request_json),
                fencing_token=claimed.fencing_token,
                attempt_count=claimed.attempt_count,
                deadline=claimed.deadline,
                lease_expires_at=lease_expires_at,
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
            if job_row.state not in (JobState.RUNNING.value, JobState.SUCCEEDED.value):
                raise LeaseExpired(
                    f"job {job_id} is already terminal as {job_row.state}; cannot commit a success"
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

            await conn.execute(
                update(jobs)
                .where(jobs.c.job_id == job_id)
                .values(
                    state=JobState.SUCCEEDED.value,
                    evaluation_id=result.evaluation_id,
                    updated_at=now,
                )
            )

    async def fail_job_terminal(
        self, job_id: str, fencing_token: int, error_code: str, error_message: str
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
                    state=JobState.FAILED.value,
                    error_code=error_code,
                    error_message=error_message,
                    updated_at=now,
                )
            )
            if result.rowcount == 0:
                raise LeaseExpired(
                    f"job {job_id} lease no longer held (fencing_token={fencing_token})"
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
    )
