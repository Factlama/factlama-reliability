"""REL-12: the synchronous API is `TenantContext`'s first real caller, so
this is also where REL-02's remaining items get a live boundary to test
against -- a genuine cross-tenant/forged-tenant-field test against a running
API, not typed-context logic against a Python function argument.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import (
    _enforce_content_length_limit,
    _read_body_within_limit,
    _require_json_content_type,
    app,
    get_storage_backend,
    get_storage_readiness,
    get_verifier,
)
from api.auth import TenantCredentialStore
from api.errors import ApiError, ApiErrorCode
from api.settings import DEFAULT_MAX_BODY_BYTES, Settings, get_settings
from core.verifier import Verifier
from judges.port import CancellationToken, JudgeProvider, JudgeRequest, JudgeResult
from judges.providers import MockModelProvider
from schemas.claims import ClaimVerdict
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

TENANT_ONE_KEY = "key-tenant-one"
TENANT_ONE_SCOPED_KEY = "key-tenant-one-scoped-to-proj-a"
TENANT_TWO_KEY = "key-tenant-two"


class FakeStorageBackend:
    """An in-memory `StorageBackend` double for API-layer tests: routing,
    header/status-code handling, and tenant scoping at the HTTP boundary.
    Real atomicity/concurrency/fencing guarantees are proven against a live
    PostgreSQL by `tests/conformance/`, not re-tested here -- this fake only
    implements enough correct behavior (idempotency-key matching, tenant
    scoping) to exercise that boundary honestly.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._idempotency: dict[tuple[str, str, str, str], tuple[str, str]] = {}
        self._results: dict[tuple[str, str], VerificationResult] = {}
        self.cancelled_job_ids: set[str] = set()

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
        if not tenant.authorizes(job.project_id, job.application_id):
            return None
        return job

    async def get_result(
        self, tenant: TenantContext, evaluation_id: str
    ) -> VerificationResult | None:
        result = self._results.get((tenant.tenant_id, evaluation_id))
        if result is None:
            return None
        if not tenant.authorizes(result.project_id, result.application_id):
            return None
        return result

    async def request_cancel(self, tenant: TenantContext, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.tenant_id != tenant.tenant_id:
            raise JobNotFound(f"job {job_id} not found")
        if not tenant.authorizes(job.project_id, job.application_id):
            raise JobNotFound(f"job {job_id} not found")
        self.cancelled_job_ids.add(job_id)

    # The routes under test never reach these -- they belong to the future
    # worker (PR3). Raising loudly means a test that accidentally depends on
    # worker-only behavior fails clearly instead of silently no-op'ing.
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
    ) -> None:
        raise NotImplementedError

    async def fail_job_terminal(
        self, job_id: str, fencing_token: int, error_code: str, error_message: str
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


def _build_settings(max_body_bytes: int = 1_000_000) -> Settings:
    settings = Settings.__new__(Settings)
    settings.tenant_store = TenantCredentialStore(
        {
            TENANT_ONE_KEY: TenantContext(tenant_id="tenant-one"),
            TENANT_ONE_SCOPED_KEY: TenantContext(tenant_id="tenant-one", project_id="proj-a"),
            TENANT_TWO_KEY: TenantContext(tenant_id="tenant-two"),
        }
    )
    settings.max_body_bytes = max_body_bytes
    settings.job_deadline_seconds = 300
    return settings


@pytest.fixture
def storage_backend() -> FakeStorageBackend:
    return FakeStorageBackend()


@pytest.fixture
def client(storage_backend: FakeStorageBackend) -> TestClient:
    """A TestClient wired to a fixed, in-memory tenant credential store, a
    deterministic MockModelProvider, and an in-memory storage backend -- no
    environment variables, no real judge dispatch, no real database. Every
    dependency `api/app.py` defines is overridden here (the same pattern as
    `get_verifier`/`get_settings`), so this fixture never depends on
    `app.state` or on FastAPI's lifespan actually having run."""
    app.dependency_overrides[get_settings] = lambda: _build_settings()
    app.dependency_overrides[get_verifier] = lambda: Verifier(model_provider=MockModelProvider())
    app.dependency_overrides[get_storage_backend] = lambda: storage_backend
    app.dependency_overrides[get_storage_readiness] = lambda: True
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _valid_body(project_id: str = "proj-a", application_id: str = "app-a") -> dict:
    return {
        "schema_version": "0.1",
        "request_id": "req_api_test",
        "project_id": project_id,
        "application_id": application_id,
        "answer": "Paris is the capital of France.",
        "evidence": [{"evidence_id": "e1", "content": "Paris is the capital of France."}],
    }


def _seeded_result(
    evaluation_id: str,
    tenant_id: str = "tenant-one",
    project_id: str = "proj-a",
    application_id: str = "app-a",
) -> VerificationResult:
    now = datetime.now(timezone.utc)
    return VerificationResult(
        evaluation_id=evaluation_id,
        request_id="req-seeded",
        tenant_id=tenant_id,
        project_id=project_id,
        application_id=application_id,
        status=ResultStatus.ABSTAINED,
        abstention_reason=AbstentionReason.NO_CHECKABLE_CLAIMS,
        verdict=OverallVerdict.ABSTAIN,
        provenance=Provenance(
            evaluator_id="test",
            evaluator_version="0.0.0",
            mode=VerificationMode.STANDARD,
            routing_profile_version="0.0.0",
            started_at=now,
            completed_at=now,
        ),
    )


class TestTenantCredentialStore:
    def test_from_json_none_authenticates_nothing(self) -> None:
        store = TenantCredentialStore.from_json(None)
        assert store.authenticate("any-key") is None

    def test_from_json_parses_scoped_credentials(self) -> None:
        raw = json.dumps(
            {
                "k1": {"tenant_id": "t1"},
                "k2": {"tenant_id": "t2", "project_id": "p2", "application_id": "a2"},
            }
        )
        store = TenantCredentialStore.from_json(raw)

        assert store.authenticate("k1") == TenantContext(tenant_id="t1")
        assert store.authenticate("k2") == TenantContext(
            tenant_id="t2", project_id="p2", application_id="a2"
        )
        assert store.authenticate("no-such-key") is None

    def test_authenticate_none_key_is_none(self) -> None:
        store = TenantCredentialStore({"k1": TenantContext(tenant_id="t1")})
        assert store.authenticate(None) is None
        assert store.authenticate("") is None


class TestSettings:
    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FACTLAMA_TENANT_CREDENTIALS", raising=False)
        monkeypatch.delenv("FACTLAMA_MAX_BODY_BYTES", raising=False)

        settings = Settings()

        assert settings.tenant_store.authenticate("any-key") is None
        assert settings.max_body_bytes == DEFAULT_MAX_BODY_BYTES

    def test_reads_credentials_and_body_limit_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "FACTLAMA_TENANT_CREDENTIALS", json.dumps({"env-key": {"tenant_id": "env-tenant"}})
        )
        monkeypatch.setenv("FACTLAMA_MAX_BODY_BYTES", "42")

        settings = Settings()

        assert settings.tenant_store.authenticate("env-key") == TenantContext(
            tenant_id="env-tenant"
        )
        assert settings.max_body_bytes == 42

    def test_get_settings_returns_a_settings_instance(self) -> None:
        get_settings.cache_clear()
        try:
            assert isinstance(get_settings(), Settings)
        finally:
            get_settings.cache_clear()


class TestHealth:
    def test_live(self, client: TestClient) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_not_ready_when_storage_is_unreachable(self, client: TestClient) -> None:
        app.dependency_overrides[get_storage_readiness] = lambda: False
        try:
            response = client.get("/health/ready")
        finally:
            app.dependency_overrides[get_storage_readiness] = lambda: True
        assert response.status_code == 503


class TestAuthentication:
    def test_missing_authorization_header_is_401(self, client: TestClient) -> None:
        response = client.post("/v0.1/verifications", json=_valid_body())
        assert response.status_code == 401
        body = response.json()
        assert body["schema_version"] == "0.1"
        assert body["error"]["code"] == "UNAUTHENTICATED"
        assert body["error"]["retryable"] is False

    def test_malformed_authorization_header_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications",
            json=_valid_body(),
            headers={"Authorization": "not-bearer-scheme"},
        )
        assert response.status_code == 401

    def test_unknown_api_key_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications", json=_valid_body(), headers=_auth("no-such-key")
        )
        assert response.status_code == 401

    def test_valid_key_succeeds(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications", json=_valid_body(), headers=_auth(TENANT_ONE_KEY)
        )
        assert response.status_code == 200
        result = response.json()
        assert result["tenant_id"] == "tenant-one"
        assert result["verdict"] == "PASS"


class TestCrossTenantIsolation:
    """The live version of the test `TenantContext.authorizes()`'s own
    docstring says needs a real boundary -- this is that boundary."""

    def test_scoped_credential_rejects_a_different_project(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications",
            json=_valid_body(project_id="proj-b"),
            headers=_auth(TENANT_ONE_SCOPED_KEY),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_scoped_credential_allows_its_own_project(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications",
            json=_valid_body(project_id="proj-a"),
            headers=_auth(TENANT_ONE_SCOPED_KEY),
        )
        assert response.status_code == 200

    def test_tenant_twos_key_cannot_reach_tenant_ones_result_tenant_id(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v0.1/verifications", json=_valid_body(), headers=_auth(TENANT_TWO_KEY)
        )
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "tenant-two"


class TestRequestValidation:
    def test_client_supplied_tenant_id_is_rejected(self, client: TestClient) -> None:
        """A forged tenant field must fail safely against the real API
        boundary, not merely be ignored (VerificationRequest's own
        validator; exercised here through the live HTTP path)."""
        body = _valid_body()
        body["tenant_id"] = "attacker-supplied-tenant"
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"

    def test_missing_required_field_is_400(self, client: TestClient) -> None:
        body = _valid_body()
        del body["answer"]
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 400

    def test_unsupported_schema_version_is_400(self, client: TestClient) -> None:
        body = _valid_body()
        body["schema_version"] = "9.9"
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "UNSUPPORTED_VERSION"

    def test_unknown_minor_schema_version_is_accepted(self, client: TestClient) -> None:
        """contracts/v0.1: an unknown minor within the known major (e.g.
        "0.2") is additive and must not be rejected as unsupported."""
        body = _valid_body()
        body["schema_version"] = "0.2"
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 200

    def test_oversized_payload_is_413(self, client: TestClient) -> None:
        body = _valid_body()
        body["answer"] = "x" * 2_000_000
        response = client.post(
            "/v0.1/verifications",
            content=json.dumps(body),
            headers={**_auth(TENANT_ONE_KEY), "Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_streamed_oversized_payload_without_content_length_is_413(
        self, client: TestClient
    ) -> None:
        """F3 of the 2026-09-21 G0-G4 validation report: `_enforce_content_
        length_limit`'s fast-path check cannot catch a request sent without a
        `Content-Length` header (e.g. chunked transfer encoding) -- this
        exercises `_read_body_within_limit`'s streaming enforcement, the
        only remaining guard for exactly this case (`TestClient`'s own ASGI
        transport pre-buffers the request client-side regardless of server
        behavior, so this cannot observe the memory-bound benefit directly,
        only that the cap is still correctly enforced without a reliable
        Content-Length to rely on)."""

        def _chunks():
            chunk = b"x" * 100_000
            for _ in range(20):  # 2,000,000 bytes total, over the 1MB test limit
                yield chunk

        response = client.post(
            "/v0.1/verifications",
            content=_chunks(),
            headers={**_auth(TENANT_ONE_KEY), "Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_unresolvable_policy_id_is_404(self, client: TestClient) -> None:
        """F4 of the 2026-09-21 G0-G4 validation report: no tenant-scoped
        policy registry exists yet, so any `policy_id` is unresolvable and
        must be rejected, not silently served under the default policy."""
        body = _valid_body()
        body["policy_id"] = "nonexistent-strict-policy"
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_wrong_content_type_is_400_not_silently_accepted(self, client: TestClient) -> None:
        """API.md requires JSON content type; a text/plain body carrying
        valid JSON must not sneak through as 200."""
        body = _valid_body()
        response = client.post(
            "/v0.1/verifications",
            content=json.dumps(body),
            headers={**_auth(TENANT_ONE_KEY), "Content-Type": "text/plain"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"

    def test_missing_content_type_is_400(self, client: TestClient) -> None:
        body = _valid_body()
        response = client.post(
            "/v0.1/verifications",
            content=json.dumps(body),
            headers=_auth(TENANT_ONE_KEY),
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"

    def test_content_type_with_charset_is_accepted(self, client: TestClient) -> None:
        body = _valid_body()
        response = client.post(
            "/v0.1/verifications",
            content=json.dumps(body),
            headers={**_auth(TENANT_ONE_KEY), "Content-Type": "application/json; charset=utf-8"},
        )
        assert response.status_code == 200


class TestContentLengthGuard:
    """`_enforce_content_length_limit` unit-level: a malformed header must
    raise a typed `ApiError`, never let a bare `int()` crash into an
    uncontracted 500. Tested directly since httpx recomputes a real
    Content-Length from the body, making a genuinely malformed header hard
    to force through a live HTTP round-trip."""

    class _FakeRequest:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    def test_malformed_content_length_raises_typed_400(self) -> None:
        request = self._FakeRequest({"content-length": "abc"})
        with pytest.raises(ApiError) as exc_info:
            _enforce_content_length_limit(request, max_body_bytes=1000)  # type: ignore[arg-type]
        assert exc_info.value.code == ApiErrorCode.INVALID_ARGUMENT
        assert exc_info.value.status_code == 400

    def test_oversized_declared_length_raises_413(self) -> None:
        request = self._FakeRequest({"content-length": "5000"})
        with pytest.raises(ApiError) as exc_info:
            _enforce_content_length_limit(request, max_body_bytes=1000)  # type: ignore[arg-type]
        assert exc_info.value.code == ApiErrorCode.PAYLOAD_TOO_LARGE

    def test_missing_header_is_a_noop(self) -> None:
        request = self._FakeRequest({})
        _enforce_content_length_limit(request, max_body_bytes=1000)  # type: ignore[arg-type]

    def test_within_limit_is_a_noop(self) -> None:
        request = self._FakeRequest({"content-length": "100"})
        _enforce_content_length_limit(request, max_body_bytes=1000)  # type: ignore[arg-type]


class TestReadBodyWithinLimit:
    """F3 of the 2026-09-21 G0-G4 validation report: `_read_body_within_
    limit` unit-level -- `request.body()` would buffer an entire oversized
    payload into memory before any check ran when Content-Length is absent
    or understated. `TestClient`'s own ASGI transport pre-buffers a request
    client-side regardless of server behavior (see `TestRequestValidation.
    test_streamed_oversized_payload_without_content_length_is_413`'s
    docstring), so a real regression test for "aborts without draining the
    whole stream" needs a stream double the route never actually gets from
    `TestClient`."""

    class _FakeStreamingRequest:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks
            self.chunks_consumed = 0

        async def stream(self):
            for chunk in self._chunks:
                self.chunks_consumed += 1
                yield chunk

    async def test_aborts_without_consuming_the_entire_stream(self) -> None:
        chunks = [b"x" * 100_000] * 20  # 2,000,000 bytes total
        request = self._FakeStreamingRequest(chunks)

        with pytest.raises(ApiError) as exc_info:
            await _read_body_within_limit(request, max_body_bytes=1_000_000)  # type: ignore[arg-type]

        assert exc_info.value.code == ApiErrorCode.PAYLOAD_TOO_LARGE
        # The critical regression check: the stream was not drained to
        # completion -- the cap was enforced WHILE streaming, not after
        # buffering everything first.
        assert request.chunks_consumed < len(chunks)

    async def test_returns_the_full_body_when_within_limit(self) -> None:
        chunks = [b"x" * 100, b"y" * 100]
        request = self._FakeStreamingRequest(chunks)

        body = await _read_body_within_limit(request, max_body_bytes=1_000_000)  # type: ignore[arg-type]

        assert body == b"x" * 100 + b"y" * 100
        assert request.chunks_consumed == len(chunks)


class TestJsonContentTypeGuard:
    """`_require_json_content_type` unit-level."""

    class _FakeRequest:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    def test_missing_content_type_raises(self) -> None:
        request = self._FakeRequest({})
        with pytest.raises(ApiError) as exc_info:
            _require_json_content_type(request)  # type: ignore[arg-type]
        assert exc_info.value.code == ApiErrorCode.INVALID_ARGUMENT

    def test_non_json_content_type_raises(self) -> None:
        request = self._FakeRequest({"content-type": "text/plain"})
        with pytest.raises(ApiError):
            _require_json_content_type(request)  # type: ignore[arg-type]

    def test_json_content_type_is_a_noop(self) -> None:
        request = self._FakeRequest({"content-type": "application/json"})
        _require_json_content_type(request)  # type: ignore[arg-type]

    def test_json_content_type_with_charset_is_a_noop(self) -> None:
        request = self._FakeRequest({"content-type": "application/json; charset=utf-8"})
        _require_json_content_type(request)  # type: ignore[arg-type]


class _SlowBlockingProvider(JudgeProvider):
    """Simulates a model-heavy adapter with a genuinely blocking call
    (`time.sleep`, not `asyncio.sleep`) -- exactly the kind of CPU-bound
    inference call that would freeze the event loop if the route dispatched
    it directly instead of through a thread pool."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "slow-blocking-provider"

    def evaluate(
        self, request: JudgeRequest, deadline: float, cancellation: CancellationToken
    ) -> JudgeResult:
        time.sleep(self._delay_seconds)
        return JudgeResult(
            verdict=ClaimVerdict.SUPPORTED, evidence_ids=[request.evidence[0].evidence_id]
        )


class TestAsyncDispatchDoesNotBlockEventLoop:
    """A CPU-bound verification must not stall a concurrent, unrelated
    request on the same worker -- the regression this test guards against
    is `verifier.verify()` being awaited directly in the route coroutine."""

    async def test_health_check_stays_responsive_during_a_slow_verification(self) -> None:
        app.dependency_overrides[get_settings] = lambda: _build_settings()
        app.dependency_overrides[get_verifier] = lambda: Verifier(
            model_provider=_SlowBlockingProvider(delay_seconds=0.3)
        )
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                # Measured from BEFORE the slow request is even scheduled --
                # not from after some fixed pre-sleep. If the route blocks
                # the event loop directly (no `run_in_threadpool`), that
                # block also delays this coroutine's own `asyncio.sleep(0.05)`
                # from firing (the whole thread is frozen), which would hide
                # the regression from a "start timing after the sleep"
                # measurement -- by the time such a measurement started, the
                # blocking call would already be done.
                started = time.monotonic()
                verify_task = asyncio.create_task(
                    ac.post(
                        "/v0.1/verifications", json=_valid_body(), headers=_auth(TENANT_ONE_KEY)
                    )
                )
                await asyncio.sleep(0.05)  # let the slow request actually start
                health_response = await ac.get("/health/live")
                elapsed_to_health_response = time.monotonic() - started

                assert health_response.status_code == 200
                assert elapsed_to_health_response < 0.2, (
                    f"health check wasn't answered until {elapsed_to_health_response:.3f}s "
                    "after a slow verification started -- the event loop was blocked"
                )

                verify_response = await verify_task
                assert verify_response.status_code == 200
        finally:
            app.dependency_overrides.clear()


class TestVerificationResult:
    def test_result_shape_matches_contract_fields(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications", json=_valid_body(), headers=_auth(TENANT_ONE_KEY)
        )
        result = response.json()
        for field in (
            "schema_version",
            "evaluation_id",
            "request_id",
            "tenant_id",
            "project_id",
            "application_id",
            "status",
            "verdict",
            "claims",
            "scores",
            "violations",
            "provenance",
        ):
            assert field in result

    def test_unsupported_claim_still_returns_a_completed_result(self, client: TestClient) -> None:
        body = _valid_body()
        body["answer"] = "Paris is the capital of France."
        body["evidence"] = [{"evidence_id": "e1", "content": "Unrelated evidence text."}]
        response = client.post("/v0.1/verifications", json=body, headers=_auth(TENANT_ONE_KEY))
        assert response.status_code == 200
        result = response.json()
        assert result["claims"][0]["verdict"] == "UNSUPPORTED"


def _idempotency_headers(api_key: str, idempotency_key: str = "idem-1") -> dict[str, str]:
    return {**_auth(api_key), "Idempotency-Key": idempotency_key}


class TestCreateVerificationJob:
    def test_submission_returns_202_with_a_queued_job(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers=_idempotency_headers(TENANT_ONE_KEY),
        )
        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "QUEUED"
        assert "job_id" in body
        assert "evaluation_id" not in body

    def test_missing_idempotency_key_is_400(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verification-jobs", json=_valid_body(), headers=_auth(TENANT_ONE_KEY)
        )
        assert response.status_code == 400

    def test_same_key_and_body_returns_the_same_job(self, client: TestClient) -> None:
        headers = _idempotency_headers(TENANT_ONE_KEY, "idem-same")
        first = client.post("/v0.1/verification-jobs", json=_valid_body(), headers=headers)
        second = client.post("/v0.1/verification-jobs", json=_valid_body(), headers=headers)

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]

    def test_same_key_different_body_is_409(self, client: TestClient) -> None:
        headers = _idempotency_headers(TENANT_ONE_KEY, "idem-conflict")
        client.post("/v0.1/verification-jobs", json=_valid_body(), headers=headers)

        conflicting_body = _valid_body()
        conflicting_body["answer"] = "A completely different answer."
        response = client.post("/v0.1/verification-jobs", json=conflicting_body, headers=headers)

        assert response.status_code == 409

    def test_cross_tenant_body_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(project_id="proj-b"),
            headers=_idempotency_headers(TENANT_ONE_SCOPED_KEY),
        )
        assert response.status_code == 403

    def test_missing_authorization_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers={"Idempotency-Key": "idem-1"},
        )
        assert response.status_code == 401


class TestGetVerificationJob:
    def test_returns_the_submitted_jobs_status(self, client: TestClient) -> None:
        submitted = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers=_idempotency_headers(TENANT_ONE_KEY),
        )
        job_id = submitted.json()["job_id"]

        response = client.get(f"/v0.1/verification-jobs/{job_id}", headers=_auth(TENANT_ONE_KEY))

        assert response.status_code == 200
        assert response.json()["job_id"] == job_id

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        response = client.get(
            "/v0.1/verification-jobs/job_does_not_exist", headers=_auth(TENANT_ONE_KEY)
        )
        assert response.status_code == 404

    def test_other_tenants_job_is_404_not_403(self, client: TestClient) -> None:
        submitted = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers=_idempotency_headers(TENANT_ONE_KEY),
        )
        job_id = submitted.json()["job_id"]

        response = client.get(f"/v0.1/verification-jobs/{job_id}", headers=_auth(TENANT_TWO_KEY))

        # CONTRACTS.md: "missing or inaccessible IDs return the same 404" --
        # a cross-tenant job must not be distinguishable from a nonexistent one.
        assert response.status_code == 404


class TestGetVerificationResult:
    def test_returns_a_seeded_result(
        self, client: TestClient, storage_backend: FakeStorageBackend
    ) -> None:
        storage_backend.seed_result("tenant-one", "eval-seeded", _seeded_result("eval-seeded"))

        response = client.get("/v0.1/verifications/eval-seeded", headers=_auth(TENANT_ONE_KEY))

        assert response.status_code == 200
        assert response.json()["evaluation_id"] == "eval-seeded"

    def test_unknown_evaluation_is_404(self, client: TestClient) -> None:
        response = client.get(
            "/v0.1/verifications/eval_does_not_exist", headers=_auth(TENANT_ONE_KEY)
        )
        assert response.status_code == 404

    def test_other_tenants_result_is_404(
        self, client: TestClient, storage_backend: FakeStorageBackend
    ) -> None:
        storage_backend.seed_result("tenant-one", "eval-seeded", _seeded_result("eval-seeded"))

        response = client.get("/v0.1/verifications/eval-seeded", headers=_auth(TENANT_TWO_KEY))

        assert response.status_code == 404


class TestCancelVerificationJob:
    def test_cancel_marks_the_job_as_cancel_requested(
        self, client: TestClient, storage_backend: FakeStorageBackend
    ) -> None:
        submitted = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers=_idempotency_headers(TENANT_ONE_KEY),
        )
        job_id = submitted.json()["job_id"]

        response = client.post(
            f"/v0.1/verification-jobs/{job_id}/cancel", headers=_auth(TENANT_ONE_KEY)
        )

        assert response.status_code == 200
        assert job_id in storage_backend.cancelled_job_ids

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verification-jobs/job_does_not_exist/cancel", headers=_auth(TENANT_ONE_KEY)
        )
        assert response.status_code == 404

    def test_other_tenants_job_is_404(self, client: TestClient) -> None:
        submitted = client.post(
            "/v0.1/verification-jobs",
            json=_valid_body(),
            headers=_idempotency_headers(TENANT_ONE_KEY),
        )
        job_id = submitted.json()["job_id"]

        response = client.post(
            f"/v0.1/verification-jobs/{job_id}/cancel", headers=_auth(TENANT_TWO_KEY)
        )

        assert response.status_code == 404
