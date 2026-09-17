"""REL-12: the synchronous API is `TenantContext`'s first real caller, so
this is also where REL-02's remaining items get a live boundary to test
against -- a genuine cross-tenant/forged-tenant-field test against a running
API, not typed-context logic against a Python function argument.
"""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import (
    _enforce_content_length_limit,
    _require_json_content_type,
    app,
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

TENANT_ONE_KEY = "key-tenant-one"
TENANT_ONE_SCOPED_KEY = "key-tenant-one-scoped-to-proj-a"
TENANT_TWO_KEY = "key-tenant-two"


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
    return settings


@pytest.fixture
def client() -> TestClient:
    """A TestClient wired to a fixed, in-memory tenant credential store and
    a deterministic MockModelProvider -- no environment variables, no real
    judge dispatch."""
    app.dependency_overrides[get_settings] = lambda: _build_settings()
    app.dependency_overrides[get_verifier] = lambda: Verifier(model_provider=MockModelProvider())
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
