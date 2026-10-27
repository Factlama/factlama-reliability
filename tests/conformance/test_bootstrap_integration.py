"""Every test in `tests/test_api.py` overrides `get_storage_backend`/
`get_storage_readiness` with a fake -- correctly, for fast HTTP-layer
tests, but that means `api/bootstrap.py`'s actual `lifespan` (engine
construction, readiness check, `PostgresStorageBackend` wiring, shutdown
disposal) is otherwise never executed by anything. This test runs the real
app, with its real lifespan, against the same live Postgres container the
rest of this package already proves persistence correctness against
(ADR-019 SS3's startup/readiness contract, exercised end to end rather
than just type-checked)."""

import base64
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from api.app import app
from api.settings import get_settings
from core.execution_storage import ExecutionStorageConfigurationError
from tests.conformance.conftest import TEST_EXECUTION_STORAGE_KEY, TEST_EXECUTION_STORAGE_KEY_ID


@pytest.fixture
def live_app_client(
    engine: AsyncEngine,  # ensures the schema exists before the app connects
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    raw = postgres_container.get_connection_url()
    _, _, rest = raw.partition("://")
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", f"postgresql+asyncpg://{rest}")
    # A3 of the 2026-10-14 ADR-020 review: the request body below requests
    # DURABLE_ENCRYPTED -- that alone no longer suffices, this tenant must
    # be granted it too.
    monkeypatch.setenv(
        "FACTLAMA_TENANT_CREDENTIALS",
        '{"live-key": {"tenant_id": "tenant-live", "execution_storage_grant": "DURABLE_ENCRYPTED"}}',
    )
    # ADR-020: the real composition root needs a configured keyring too --
    # same test key material as the `backend` fixture, so a job encrypted
    # through this real lifespan is decryptable the same way.
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        get_settings.cache_clear()


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer live-key"}


def test_health_ready_reflects_a_real_reachable_database(live_app_client: TestClient) -> None:
    response = live_app_client.get("/health/ready")
    assert response.status_code == 200


def test_submit_then_read_a_job_through_the_real_composition_root(
    live_app_client: TestClient,
) -> None:
    body = {
        "schema_version": "0.1",
        "request_id": "req-live",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Paris is the capital of France.",
        "evidence": [{"evidence_id": "e1", "content": "Paris is the capital of France."}],
        # ADR-020: durable async submission requires explicit authorization;
        # this test is about the real composition root/idempotency, not
        # execution-storage policy itself.
        "policy": {"id": "test-durable", "execution_storage": {"mode": "DURABLE_ENCRYPTED"}},
    }
    headers = {**_auth(), "Idempotency-Key": "idem-live-1"}

    submitted = live_app_client.post("/v0.1/verification-jobs", json=body, headers=headers)
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]

    # Resubmitting with the same key/body must return the same job --
    # proves this hits the real PostgresStorageBackend's idempotency path,
    # not an in-memory fake.
    resubmitted = live_app_client.post("/v0.1/verification-jobs", json=body, headers=headers)
    assert resubmitted.json()["job_id"] == job_id

    status = live_app_client.get(f"/v0.1/verification-jobs/{job_id}", headers=_auth())
    assert status.status_code == 200
    assert status.json()["state"] == "QUEUED"


def _set_database_and_active_key_env(
    postgres_container: PostgresContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = postgres_container.get_connection_url()
    _, _, rest = raw.partition("://")
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", f"postgresql+asyncpg://{rest}")
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)


def test_api_refuses_to_start_when_a_declared_retired_key_is_still_configured(
    engine: AsyncEngine,  # ensures the schema exists, matching live_app_client's own fixture
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-020 section 5's mandatory restore procedure, enforced
    mechanically: declaring a key_id retired (as an operator restoring
    from backup must) while that same key_id is still present in
    FACTLAMA_EXECUTION_STORAGE_KEYS must refuse to start the real
    composition root entirely -- never mind serve traffic -- rather than
    trusting that the runbook step was actually followed. Deliberately
    does not reuse `live_app_client`: that fixture's own `with
    TestClient(app)` is expected to succeed, so a startup failure here
    needs its own `pytest.raises` around entering the context."""
    _set_database_and_active_key_env(postgres_container, monkeypatch)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    # The declared-retired key_id is the exact one still configured above --
    # the operator restoring from backup forgot (or never performed) the
    # actual key removal this procedure exists to make mandatory.
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", TEST_EXECUTION_STORAGE_KEY_ID)
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(ExecutionStorageConfigurationError, match=TEST_EXECUTION_STORAGE_KEY_ID),
            TestClient(app),
        ):
            pass
    finally:
        get_settings.cache_clear()


def test_api_refuses_to_start_when_restore_mode_has_an_empty_manifest(
    engine: AsyncEngine,
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declaring `RESTORE_MODE=RESTORE` without naming any key to retire
    must not be a no-op way to "confirm" the restore procedure without
    doing anything -- the manifest must actually be non-empty."""
    _set_database_and_active_key_env(postgres_container, monkeypatch)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.delenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", raising=False)
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(ExecutionStorageConfigurationError, match="RESTORE_MODE=RESTORE"),
            TestClient(app),
        ):
            pass
    finally:
        get_settings.cache_clear()


def test_api_refuses_to_start_with_an_invalid_restore_mode_value(
    engine: AsyncEngine,
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE must fail loudly
    at real startup, not silently fall back to NORMAL."""
    _set_database_and_active_key_env(postgres_container, monkeypatch)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTOER")
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(ExecutionStorageConfigurationError, match="RESTOER"),
            TestClient(app),
        ):
            pass
    finally:
        get_settings.cache_clear()


def test_api_starts_successfully_once_declared_retired_keys_are_actually_removed(
    engine: AsyncEngine,
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive path this whole procedure exists to allow: an operator
    declares the restore, names the key that could have encrypted a now-
    stale row, and has actually rotated it out of the active keyring --
    startup must succeed."""
    _set_database_and_active_key_env(postgres_container, monkeypatch)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    # Retiring a key_id that was never active in the first place is exactly
    # what "actually removed it" looks like from this gate's perspective --
    # the active keyring above only ever configured TEST_EXECUTION_STORAGE_KEY_ID.
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", "some-retired-key-from-the-backup"
    )
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200
    finally:
        get_settings.cache_clear()
