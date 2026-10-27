"""G7's smallest vertical slice, proven end to end: one real SDK call
(`factlama-observability`'s `sdk.client.FactLamaClient.submit_verification`,
run as a real subprocess via its own sample app, `examples/
verify_and_correlate.py`) submits a durable verification job to this
repo's own live API and exports one `VERIFICATION` span to a live
Observability collector, sharing one trace_id/span_id. This repo's real
worker pipeline then processes the job and delivers the resulting
`ReliabilityEvent` to that same live Observability instance, which
correlates it against the already-stored span by `(tenant_id, trace_id,
span_id)` (ADR-021) -- all already-built, already-closed-gate machinery
(G5/G6), exercised here for the first time together with a real SDK call
in front of it.

Mirrors `test_outbox_cross_repo_delivery.py`'s own live-subprocess
pattern (duplicated, not imported, from that file -- see the helpers
below) rather than refactoring a file tied to a closed gate. Adds two new
live subprocesses of its own: this repo's own `api.app` (trivial -- same
venv as this test, no sibling-path lookup) and Observability's
`query.api.app` (a second instance of the exact pattern
`observability_service` below already proves for `collector.ingress.app`).

Needs the same local three-repo workspace checkout as
`test_outbox_cross_repo_delivery.py` -- `../factlama-observability` with
its own `.venv` provisioned via `pip install -e ".[dev]"` there (the `sdk`
extra's own runtime deps, `httpx`/`opentelemetry-proto`, are already
included in that repo's `dev` extra). Skips cleanly otherwise, refusing to
skip in CI, same rule as that file.
"""

import base64
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from core.verifier import Verifier
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.registry import PostgresRevocationRegistry
from tests.conformance.conftest import (
    TEST_EXECUTION_STORAGE_KEY,
    TEST_EXECUTION_STORAGE_KEY_ID,
)
from tests.conformance.test_worker_runner import _fast_settings, _ScriptedProvider, _supported
from worker.outbox_http_delivery import build_http_deliver
from worker.runner import process_claimed_job
from worker.settings import WorkerSettings

WORKLOAD_TOKEN = "g7-e2e-cross-repo-shared-secret"
API_KEY = "g7-e2e-shared-key"
TENANT_ID = "tenant-g7"
PROJECT_ID = "proj-g7"
APPLICATION_ID = "app-g7"


# --- Duplicated from test_outbox_cross_repo_delivery.py (deliberately not
# extracted into a shared module -- that file is tied to a closed gate,
# G6; see this file's own module docstring) ---------------------------


def _observability_repo_dir() -> Path | None:
    env = os.environ.get("FACTLAMA_OBSERVABILITY_REPO_DIR")
    path = Path(env) if env else Path(__file__).resolve().parents[3] / "factlama-observability"
    venv_python = path / ".venv" / "bin" / "python"
    return path if path.is_dir() and venv_python.is_file() else None


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("1", "true") or bool(
        os.environ.get("GITHUB_ACTIONS")
    )


OBSERVABILITY_DIR = _observability_repo_dir()

if OBSERVABILITY_DIR is None and _running_in_ci():
    raise RuntimeError(
        "FACTLAMA_OBSERVABILITY_REPO_DIR is unset, or its .venv is not provisioned, in "
        "a CI run. This G7 SDK correlation test is a required acceptance test for the "
        "SDK -> collector -> correlation path, not an optional local-dev check -- "
        "refusing to silently skip it."
    )

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        OBSERVABILITY_DIR is None,
        reason=(
            "factlama-observability checkout with a provisioned .venv not found next to "
            "this repo -- set FACTLAMA_OBSERVABILITY_REPO_DIR, or run inside the "
            "three-repo local workspace with ../factlama-observability's venv installed"
        ),
    ),
]


def _observability_python() -> str:
    assert OBSERVABILITY_DIR is not None
    return str(OBSERVABILITY_DIR / ".venv" / "bin" / "python")


def _asyncpg_url(container: PostgresContainer) -> str:
    raw = container.get_connection_url()
    _, _, rest = raw.partition("://")
    return f"postgresql+asyncpg://{rest}"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health/ready", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.3)
    raise RuntimeError(
        f"service at {base_url} did not become ready within {timeout}s"
    ) from last_error


# --- New fixtures ------------------------------------------------------


@pytest.fixture(scope="module")
def observability_postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="module")
def observability_database_url(observability_postgres_container: PostgresContainer) -> str:
    return _asyncpg_url(observability_postgres_container)


def _observability_tenant_credentials() -> str:
    return json.dumps(
        {API_KEY: {"tenant_id": TENANT_ID, "project_id": None, "application_id": None}}
    )


@pytest.fixture(scope="module")
def observability_service(observability_database_url: str) -> Iterator[str]:
    """Observability's real OTLP ingress (`collector.ingress.app`), the
    SDK's own export target and the internal `ReliabilityEvent` delivery
    target -- same shape as `test_outbox_cross_repo_delivery.py`'s own
    fixture of the same name."""
    env = {
        **os.environ,
        "FACTLAMA_DATABASE_URL": observability_database_url,
        "FACTLAMA_RELIABILITY_WORKLOAD_TOKEN": WORKLOAD_TOKEN,
        "FACTLAMA_TENANT_CREDENTIALS": _observability_tenant_credentials(),
    }
    python = _observability_python()
    subprocess.run(
        [python, "-m", "alembic", "-c", "storage/migrations/alembic.ini", "upgrade", "head"],
        cwd=OBSERVABILITY_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    port = _free_tcp_port()
    proc = subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            "collector.ingress.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=OBSERVABILITY_DIR,
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def observability_query_service(
    observability_database_url: str, observability_service: str
) -> Iterator[str]:
    """Observability's `query.api.app` -- a second live instance of the
    exact same pattern `observability_service` above proves, on its own
    port, against the same already-migrated database (depending on
    `observability_service` only to guarantee migrations have already
    run; this fixture starts its own separate process)."""
    env = {
        **os.environ,
        "FACTLAMA_DATABASE_URL": observability_database_url,
        "FACTLAMA_TENANT_CREDENTIALS": _observability_tenant_credentials(),
    }
    python = _observability_python()
    port = _free_tcp_port()
    proc = subprocess.Popen(
        [python, "-m", "uvicorn", "query.api.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=OBSERVABILITY_DIR,
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _reliability_tenant_credentials() -> str:
    return json.dumps(
        {
            API_KEY: {
                "tenant_id": TENANT_ID,
                "execution_storage_grant": "DURABLE_ENCRYPTED",
            }
        }
    )


@pytest.fixture
def reliability_service(
    postgres_container: PostgresContainer, engine: AsyncEngine
) -> Iterator[str]:
    """This repo's own real `api.app`, live, so the SDK's one real HTTP
    call has something real to submit to. Trivial compared to
    `observability_service` above: same venv as this test process (`sys.
    executable`), no sibling-path lookup, no migration step (`engine`
    above already created the schema fresh for this test function via
    `metadata.create_all`, the same way every other conformance test's
    `backend` fixture does)."""
    env = {
        **os.environ,
        "FACTLAMA_DATABASE_URL": _asyncpg_url(postgres_container),
        "FACTLAMA_TENANT_CREDENTIALS": _reliability_tenant_credentials(),
        "FACTLAMA_EXECUTION_STORAGE_KEYS": json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
        "FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID": TEST_EXECUTION_STORAGE_KEY_ID,
    }
    port = _free_tcp_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _worker_settings() -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.worker_id = "g7-e2e-worker"
    settings.observability_workload_token = WORKLOAD_TOKEN
    settings.observability_ingress_timeout_seconds = 10.0
    settings.outbox_batch_size = 20
    settings.outbox_lease_seconds = 60.0
    settings.outbox_poll_interval_seconds = 0.05
    return settings


def _run_sdk_driver(*, reliability_api_url: str, collector_url: str) -> dict[str, object]:
    """Invokes `examples/verify_and_correlate.py` -- the real sample app,
    not a test-only duplicate -- as a subprocess in Observability's own
    venv, since `sdk`'s runtime deps are not installed in this repo's
    venv (nor should they be; see `sdk/client.py`'s own module docstring
    on why there is no cross-repo Python import here)."""
    assert OBSERVABILITY_DIR is not None
    env = {
        **os.environ,
        "FACTLAMA_SDK_COLLECTOR_URL": collector_url,
        "FACTLAMA_SDK_RELIABILITY_API_URL": reliability_api_url,
        "FACTLAMA_SDK_API_KEY": API_KEY,
        "FACTLAMA_SDK_PROJECT_ID": PROJECT_ID,
        "FACTLAMA_SDK_APPLICATION_ID": APPLICATION_ID,
    }
    payload = json.dumps(
        {
            "request": {
                "schema_version": "0.1",
                "request_id": "g7-e2e-req-1",
                "project_id": PROJECT_ID,
                "application_id": APPLICATION_ID,
                "answer": "placeholder answer -- explicit claims are used unchanged",
                "claims": [{"claim_id": "c1", "text": "Paris is the capital of France."}],
                "evidence": [{"evidence_id": "e1", "content": "Paris is the capital of France."}],
                "policy": {
                    "id": "g7-e2e",
                    "execution_storage": {"mode": "DURABLE_ENCRYPTED"},
                },
            },
            "idempotency_key": "g7-e2e-idem-1",
        }
    )
    result = subprocess.run(
        [_observability_python(), "-m", "examples.verify_and_correlate"],
        cwd=OBSERVABILITY_DIR,
        env=env,
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"SDK driver failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


async def test_sdk_submission_correlates_with_its_exported_span(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
    reliability_service: str,
    observability_service: str,
    observability_query_service: str,
) -> None:
    # 1. One real SDK call: submits to the live Reliability API, exports
    # one VERIFICATION span to the live Observability collector.
    driver_result = _run_sdk_driver(
        reliability_api_url=reliability_service, collector_url=observability_service
    )
    assert "error" not in driver_result, driver_result
    trace_id = driver_result["trace_id"]
    span_id = driver_result["span_id"]
    job_id = driver_result["job_id"]
    assert driver_result["state"] == "QUEUED"

    # 2. In-process: claim and process the SAME job the live API inserted,
    # through the real Verifier/process_claimed_job pipeline.
    claimed = await backend.claim_due_job("g7-e2e-worker", timedelta(minutes=5))
    assert claimed is not None
    assert claimed.job_id == job_id
    provider = _ScriptedProvider([{"c1": _supported("e1")}])
    verifier = Verifier(model_provider=provider)
    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    # 3. In-process: real HTTP outbox delivery to the live collector --
    # the already-closed-gate (G6) path, reused unchanged.
    batch = await backend.claim_outbox_batch("g7-e2e-delivery", timedelta(minutes=5), limit=10)
    assert len(batch) == 1
    evaluation_id = batch[0].evaluation_id
    async with httpx.AsyncClient(base_url=observability_service, timeout=10.0) as client:
        delivered = await build_http_deliver(client, _worker_settings())(batch[0])
    assert delivered is True
    await backend.ack_outbox(batch[0].event_id, "g7-e2e-delivery")

    # 4. Query the live Observability query API for the correlated trace.
    async with httpx.AsyncClient(base_url=observability_query_service, timeout=10.0) as client:
        response = await client.get(
            f"/v0.1/ai/traces/{trace_id}", headers={"Authorization": f"Bearer {API_KEY}"}
        )
    assert response.status_code == 200
    trace = response.json()

    assert len(trace["spans"]) == 1
    assert trace["spans"][0]["span_id"] == span_id
    assert trace["spans"][0]["operation_type"] == "VERIFICATION"

    assert trace["evaluation_state"] == "AVAILABLE"
    assert len(trace["evaluation_links"]) == 1
    link = trace["evaluation_links"][0]
    assert link["evaluation_id"] == evaluation_id
    assert link["state"] == "AVAILABLE"

    serialized = json.dumps(trace)
    assert "Paris is the capital of France" not in serialized
