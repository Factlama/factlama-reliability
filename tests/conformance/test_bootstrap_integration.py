"""Every test in `tests/test_api.py` overrides `get_storage_backend`/
`get_storage_readiness` with a fake -- correctly, for fast HTTP-layer
tests, but that means `api/bootstrap.py`'s actual `lifespan` (engine
construction, readiness check, `PostgresStorageBackend` wiring, shutdown
disposal) is otherwise never executed by anything. This test runs the real
app, with its real lifespan, against the same live Postgres container the
rest of this package already proves persistence correctness against
(ADR-019 SS3's startup/readiness contract, exercised end to end rather
than just type-checked)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from api.app import app
from api.settings import get_settings


@pytest.fixture
def live_app_client(
    engine: AsyncEngine,  # ensures the schema exists before the app connects
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    raw = postgres_container.get_connection_url()
    _, _, rest = raw.partition("://")
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", f"postgresql+asyncpg://{rest}")
    monkeypatch.setenv("FACTLAMA_TENANT_CREDENTIALS", '{"live-key": {"tenant_id": "tenant-live"}}')
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
