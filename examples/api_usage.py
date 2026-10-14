"""Runnable example of the REL-12 API (docs/API.md), demonstrating both the
synchronous (`POST /v0.1/verifications`) and async-job (`POST
/v0.1/verification-jobs` and friends) surfaces through the real HTTP layer
-- authentication, request validation and the JSON error/response envelope
included, not just `core.verify()` called directly (see
`examples/basic_usage.py` for that library-level path).

Run with:
    pip install -e ".[dev]"
    python examples/api_usage.py

This drives the FastAPI app in-process via `TestClient` (the same mechanism
`tests/test_api.py` uses), so no separate server process or database is
required; the same requests work unchanged against a real `uvicorn`
process backed by a real PostgreSQL database.

The async section below is about the job-submission *contract*
(idempotency, status, cancellation), the same scope the sync section above
has for the HTTP contract rather than judge accuracy -- it uses a small
in-memory stand-in for the real `storage.postgres.PostgresStorageBackend`,
the same pattern `tests/test_api.py::FakeStorageBackend` uses, so this
script needs no Docker/database to run. A worker actually driving a queued
job to a real `SUCCEEDED` result is a separate, already-tested concern
(`tests/conformance/test_worker_runner.py`, against real PostgreSQL) -- not
re-demonstrated here; this stand-in seeds one result directly to show the
read path (`GET /v0.1/verifications/{evaluation_id}`) a real worker would
otherwise have produced.
"""

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from api.app import app, get_storage_backend, get_storage_readiness, get_verifier
from api.auth import TenantCredentialStore
from api.settings import Settings, get_settings
from core.verifier import Verifier
from judges.providers import RuleBasedProvider
from schemas.policy import ExecutionStorageMode
from schemas.tenancy import TenantContext
from schemas.verification import (
    AbstentionReason,
    OverallVerdict,
    Provenance,
    ResultStatus,
    VerificationMode,
    VerificationRequest,
    VerificationResult,
)
from storage.errors import JobConflict, JobNotFound
from storage.models import ClaimedJob, Job, JobState, OutboxEvent

API_KEY = "example-api-key"
# A3 of the 2026-10-14 ADR-020 review: the async demo below requests
# DURABLE_ENCRYPTED -- that alone no longer suffices, the tenant's own
# trusted grant must also permit it.
TENANT_CONTEXT = TenantContext(
    tenant_id="example-tenant", execution_storage_grant=ExecutionStorageMode.DURABLE_ENCRYPTED
)


def _build_settings() -> Settings:
    settings = Settings.__new__(Settings)
    settings.tenant_store = TenantCredentialStore({API_KEY: TENANT_CONTEXT})
    settings.max_body_bytes = 1_000_000
    settings.job_deadline_seconds = 300
    return settings


class _ExampleStorageBackend:
    """A minimal in-memory `StorageBackend` double, scoped to exactly what
    this script's async demo exercises (submit/idempotency, status,
    cancel, seeded result read) -- mirrors
    `tests/test_api.py::FakeStorageBackend`'s same scope note: real
    atomicity/concurrency/fencing guarantees are proven against a live
    PostgreSQL by `tests/conformance/`, not by this script.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._idempotency: dict[tuple[str, str, str, str], tuple[str, str]] = {}
        self._results: dict[tuple[str, str], VerificationResult] = {}

    def seed_result(self, tenant_id: str, evaluation_id: str, result: VerificationResult) -> None:
        self._results[(tenant_id, evaluation_id)] = result

    async def submit_or_get_job(
        self,
        tenant: TenantContext,
        route: str,
        idempotency_key: str,
        canonical_request_hash: str,
        request: VerificationRequest,
        deadline: datetime,
    ) -> Job:
        key = (tenant.tenant_id, route, request.project_id, idempotency_key)
        existing = self._idempotency.get(key)
        if existing is not None:
            existing_hash, job_id = existing
            if existing_hash != canonical_request_hash:
                raise JobConflict("idempotency key reused with a different request body")
            return self._jobs[job_id]

        now = datetime.now(timezone.utc)
        job_id = f"job_{len(self._jobs) + 1}"
        job = Job(
            job_id=job_id,
            tenant_id=tenant.tenant_id,
            project_id=request.project_id,
            application_id=request.application_id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            attempt_count=0,
        )
        self._jobs[job_id] = job
        self._idempotency[key] = (canonical_request_hash, job_id)
        return job

    async def get_job_status(self, tenant: TenantContext, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None or job.tenant_id != tenant.tenant_id:
            return None
        return job

    async def get_result(
        self, tenant: TenantContext, evaluation_id: str
    ) -> VerificationResult | None:
        return self._results.get((tenant.tenant_id, evaluation_id))

    async def request_cancel(self, tenant: TenantContext, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.tenant_id != tenant.tenant_id:
            raise JobNotFound(f"job {job_id} not found")

    # Worker-only methods -- this script never drives a job to completion
    # itself (see the module docstring); raising loudly beats a silent
    # no-op if that ever changes without updating this comment.
    async def claim_due_job(self, worker_id: str, lease_duration: timedelta) -> ClaimedJob | None:
        raise NotImplementedError

    async def renew_lease(self, job_id: str, fencing_token: int, extension: timedelta) -> datetime:
        raise NotImplementedError

    async def commit_evaluation_and_outbox(
        self,
        job_id: str,
        fencing_token: int,
        result: VerificationResult,
        event: OutboxEvent,
        *,
        governed_request_json: dict | None = None,
    ) -> None:
        raise NotImplementedError

    async def fail_job_terminal(
        self,
        job_id: str,
        fencing_token: int,
        error_code: str,
        error_message: str,
        *,
        governed_request_json: dict | None = None,
    ) -> None:
        raise NotImplementedError

    async def release_for_retry(
        self, job_id: str, fencing_token: int, not_before: datetime
    ) -> None:
        raise NotImplementedError

    async def is_cancellation_requested(self, job_id: str) -> bool:
        raise NotImplementedError

    async def claim_outbox_batch(
        self, worker_id: str, lease_duration: timedelta, limit: int
    ) -> list[OutboxEvent]:
        raise NotImplementedError

    async def ack_outbox(self, event_id: str, worker_id: str) -> None:
        raise NotImplementedError


def _seeded_completed_result(evaluation_id: str) -> VerificationResult:
    """Stands in for what a real worker would have committed -- see the
    module docstring on why this script seeds rather than runs one."""
    now = datetime.now(timezone.utc)
    return VerificationResult(
        evaluation_id=evaluation_id,
        request_id="req_example_async",
        tenant_id=TENANT_CONTEXT.tenant_id,
        project_id="proj-a",
        application_id="app-a",
        status=ResultStatus.ABSTAINED,
        abstention_reason=AbstentionReason.NO_CHECKABLE_CLAIMS,
        verdict=OverallVerdict.ABSTAIN,
        provenance=Provenance(
            evaluator_id="example",
            evaluator_version="0.0.0",
            mode=VerificationMode.STANDARD,
            routing_profile_version="0.0.0",
            started_at=now,
            completed_at=now,
        ),
    )


def _print_result(title: str, response) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"HTTP {response.status_code}")
    body = response.json()
    print(f"Verdict: {body.get('verdict')}")
    print(f"Status: {body.get('status')} (abstention_reason: {body.get('abstention_reason')})")
    for claim in body.get("claims", []):
        print(f"  - {claim['claim_id']}: {claim['verdict']} ({claim['rationale_code']})")
    print()


def main() -> None:
    # A deterministic, in-process RuleBasedProvider (ADR-011's T0 tier), not
    # a real vendor call -- this example is about the HTTP contract, not
    # judge accuracy.
    app.dependency_overrides[get_settings] = _build_settings
    app.dependency_overrides[get_verifier] = lambda: Verifier(model_provider=RuleBasedProvider())
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {API_KEY}"}

    supported_request = {
        "schema_version": "0.1",
        "request_id": "req_example_supported",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Paris is the capital of France.",
        "evidence": [
            {"evidence_id": "e1", "content": "Paris is the capital of France."},
        ],
    }
    _print_result(
        "Example 1: Supported",
        client.post("/v0.1/verifications", headers=headers, json=supported_request),
    )

    contradicted_request = {
        "schema_version": "0.1",
        "request_id": "req_example_contradicted",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Product X weighs 3.4 kg.",
        "evidence": [
            {"evidence_id": "e1", "content": "Product X weighs 2.4 kg."},
        ],
    }
    _print_result(
        "Example 2: Contradicted",
        client.post("/v0.1/verifications", headers=headers, json=contradicted_request),
    )

    abstained_request = {
        "schema_version": "0.1",
        "request_id": "req_example_abstained",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Paris is the capital of France.",
        # No evidence supplied: the judge cannot confirm anything, so the
        # request abstains instead of guessing -- no optimistic default.
    }
    _print_result(
        "Example 3: Abstained (no evidence supplied)",
        client.post("/v0.1/verifications", headers=headers, json=abstained_request),
    )

    print("=" * 60)
    print("Example 4: Unauthenticated request (401, contract error envelope)")
    print("=" * 60)
    response = client.post("/v0.1/verifications", json=supported_request)
    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # --- Async job surface (API.md's POST /v0.1/verification-jobs and
    # friends) -- see the module docstring for why this section uses an
    # in-memory storage stand-in instead of a real database/worker.
    storage_backend = _ExampleStorageBackend()
    app.dependency_overrides[get_storage_backend] = lambda: storage_backend
    app.dependency_overrides[get_storage_readiness] = lambda: True
    async_headers = {**headers, "Idempotency-Key": "idem-example-1"}

    # ADR-020: durable async submission requires an explicit execution-
    # storage authorization -- the sync examples above default to
    # NO_PERSISTENCE (no policy at all) and never touch storage, but a
    # caller choosing the durable async surface must opt in.
    async_policy = {
        "id": "example-async-durable",
        "execution_storage": {"mode": "DURABLE_ENCRYPTED"},
    }
    supported_request_async = {**supported_request, "policy": async_policy}
    contradicted_request_async = {**contradicted_request, "policy": async_policy}

    print("=" * 60)
    print("Example 5: Submit an async job (202, QUEUED)")
    print("=" * 60)
    submitted = client.post(
        "/v0.1/verification-jobs", headers=async_headers, json=supported_request_async
    )
    job_id = submitted.json()["job_id"]
    print(f"HTTP {submitted.status_code}")
    print(json.dumps(submitted.json(), indent=2))
    print()

    print("=" * 60)
    print("Example 6: Resubmit with the same Idempotency-Key and body -> same job")
    print("=" * 60)
    resubmitted = client.post(
        "/v0.1/verification-jobs", headers=async_headers, json=supported_request_async
    )
    print(
        f"HTTP {resubmitted.status_code}, job_id={resubmitted.json()['job_id']} "
        f"(matches original: {resubmitted.json()['job_id'] == job_id})"
    )
    print()

    print("=" * 60)
    print("Example 7: Resubmit with the same key but a different body -> 409 CONFLICT")
    print("=" * 60)
    conflicting = client.post(
        "/v0.1/verification-jobs", headers=async_headers, json=contradicted_request_async
    )
    print(f"HTTP {conflicting.status_code}")
    print(json.dumps(conflicting.json(), indent=2))
    print()

    print("=" * 60)
    print(f"Example 8: Check job status -> GET /v0.1/verification-jobs/{job_id}")
    print("=" * 60)
    status_response = client.get(f"/v0.1/verification-jobs/{job_id}", headers=headers)
    print(f"HTTP {status_response.status_code}")
    print(json.dumps(status_response.json(), indent=2))
    print()

    print("=" * 60)
    print(f"Example 9: Request cancellation -> POST /v0.1/verification-jobs/{job_id}/cancel")
    print("=" * 60)
    cancel_response = client.post(f"/v0.1/verification-jobs/{job_id}/cancel", headers=headers)
    print(f"HTTP {cancel_response.status_code}")
    print(
        "Cancellation is best-effort (docs/async-evaluation.md): it does not "
        "itself force a state transition, so state is still "
        f"{cancel_response.json()['state']!r} until a worker observes it."
    )
    print()

    print("=" * 60)
    print("Example 10: Read a completed result -> GET /v0.1/verifications/{evaluation_id}")
    print("=" * 60)
    evaluation_id = "eval_example_async"
    storage_backend.seed_result(
        TENANT_CONTEXT.tenant_id, evaluation_id, _seeded_completed_result(evaluation_id)
    )
    result_response = client.get(f"/v0.1/verifications/{evaluation_id}", headers=headers)
    print(f"HTTP {result_response.status_code}")
    print(f"Verdict: {result_response.json()['verdict']}")
    print(f"Status: {result_response.json()['status']}")
    print()

    app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
