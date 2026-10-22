"""Real cross-repo integration test for the Reliability -> Observability
outbox handoff (`EXECUTION_PLAN.md`'s handoff table: "Every handoff needs
... one cross-repo integration test against compatible revisions").

Starts Observability's actual `collector.ingress.app` as a live subprocess,
backed by its own real, throwaway PostgreSQL via `testcontainers` -- not an
in-process ASGI transport call and not a mock. Both of those would only
prove this repo's own serialization is self-consistent; the point of this
test is to catch what neither repo's own suite can: a real HTTP round-trip
against another real codebase's actual request parsing, auth and storage,
asserted against that database directly, not just the HTTP response.

Needs a local three-repo workspace checkout of `factlama-observability`
next to this repo (`../factlama-observability`), with its own `.venv`
already provisioned (`pip install -e ".[dev]"` there) -- the same
local-workspace convention `tests/test_contract_fixtures.py` already uses
for `factlama-architecture`. Skips cleanly otherwise, including in CI:
wiring this into both repos' CI (checking out the sibling and provisioning
its venv) is the natural next pass, not done here.
"""

import asyncio
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)
from worker.outbox_delivery import run_outbox_delivery_loop
from worker.outbox_http_delivery import build_http_deliver
from worker.settings import WorkerSettings

WORKLOAD_TOKEN = "cross-repo-test-shared-secret"


def _observability_repo_dir() -> Path | None:
    env = os.environ.get("FACTLAMA_OBSERVABILITY_REPO_DIR")
    path = Path(env) if env else Path(__file__).resolve().parents[3] / "factlama-observability"
    venv_python = path / ".venv" / "bin" / "python"
    return path if path.is_dir() and venv_python.is_file() else None


OBSERVABILITY_DIR = _observability_repo_dir()

pytestmark = pytest.mark.skipif(
    OBSERVABILITY_DIR is None,
    reason=(
        "factlama-observability checkout with a provisioned .venv not found next to "
        "this repo -- set FACTLAMA_OBSERVABILITY_REPO_DIR, or run inside the "
        "three-repo local workspace with ../factlama-observability's venv installed"
    ),
)


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
        f"observability ingress at {base_url} did not become ready within {timeout}s"
    ) from last_error


@pytest.fixture(scope="module")
def observability_postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="module")
def observability_database_url(observability_postgres_container: PostgresContainer) -> str:
    return _asyncpg_url(observability_postgres_container)


@pytest.fixture(scope="module")
def observability_service(observability_database_url: str) -> Iterator[str]:
    """Runs Observability's real ingress as a live process against its own
    real PostgreSQL. Yields the base URL once `/health/ready` passes."""
    env = {
        **os.environ,
        "FACTLAMA_DATABASE_URL": observability_database_url,
        "FACTLAMA_RELIABILITY_WORKLOAD_TOKEN": WORKLOAD_TOKEN,
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


def _worker_settings(outbox_lease_seconds: float = 60.0) -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.worker_id = "cross-repo-test-worker"
    settings.observability_workload_token = WORKLOAD_TOKEN
    settings.observability_ingress_timeout_seconds = 10.0
    settings.outbox_batch_size = 20
    settings.outbox_lease_seconds = outbox_lease_seconds
    settings.outbox_poll_interval_seconds = 0.05
    return settings


async def _commit_one(
    backend: PostgresStorageBackend, key: str, evaluation_id: str, event_id: str
) -> None:
    tenant = make_tenant()
    await backend.submit_or_get_job(
        tenant,
        "verification-jobs",
        key,
        f"hash-{key}",
        make_request(request_id=key),
        default_deadline(),
    )
    claimed = await backend.claim_due_job(f"worker-{key}", timedelta(minutes=5))
    assert claimed is not None
    await backend.commit_evaluation_and_outbox(
        claimed.job_id,
        claimed.fencing_token,
        make_result(evaluation_id),
        make_outbox_event(event_id, evaluation_id),
    )


async def _fetch_event_row(database_url: str, tenant_id: str, event_id: str) -> Any:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT evaluation_id, status, verdict, calibration_class, "
                    "qualification_status FROM reliability_events "
                    "WHERE tenant_id = :tenant_id AND event_id = :event_id"
                ),
                {"tenant_id": tenant_id, "event_id": event_id},
            )
            return result.first()
    finally:
        await engine.dispose()


async def _count_event_rows(database_url: str, tenant_id: str, event_id: str) -> int:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT count(*) FROM reliability_events "
                    "WHERE tenant_id = :tenant_id AND event_id = :event_id"
                ),
                {"tenant_id": tenant_id, "event_id": event_id},
            )
            return int(result.scalar_one())
    finally:
        await engine.dispose()


async def test_http_delivery_persists_through_to_observability_postgres(
    backend: PostgresStorageBackend,
    observability_service: str,
    observability_database_url: str,
) -> None:
    await _commit_one(backend, "cross-key-1", "cross-eval-1", "cross-event-1")
    batch = await backend.claim_outbox_batch("delivery-worker", timedelta(minutes=5), limit=10)
    assert [e.event_id for e in batch] == ["cross-event-1"]

    async with httpx.AsyncClient(base_url=observability_service, timeout=10.0) as client:
        deliver = build_http_deliver(client, _worker_settings())
        delivered = await deliver(batch[0])
    assert delivered is True
    await backend.ack_outbox("cross-event-1", "delivery-worker")

    row = await _fetch_event_row(observability_database_url, "tenant-a", "cross-event-1")
    assert row is not None
    assert row.evaluation_id == "cross-eval-1"
    assert row.status == "ABSTAINED"
    assert row.verdict == "ABSTAIN"
    assert row.calibration_class == "NONE"
    assert row.qualification_status == "UNQUALIFIED"


async def test_duplicate_delivery_is_not_double_counted(
    backend: PostgresStorageBackend,
    observability_service: str,
    observability_database_url: str,
) -> None:
    await _commit_one(backend, "cross-key-2", "cross-eval-2", "cross-event-2")
    batch = await backend.claim_outbox_batch("dup-worker", timedelta(minutes=5), limit=10)
    event = batch[0]

    async with httpx.AsyncClient(base_url=observability_service, timeout=10.0) as client:
        deliver = build_http_deliver(client, _worker_settings())
        # CONTRACTS.md: "duplicate does not double-count" -- redeliver the
        # same event twice, as at-least-once delivery genuinely does (e.g.
        # a delivered-but-unacked event reclaimed by another worker).
        first = await deliver(event)
        second = await deliver(event)
    assert (first, second) == (True, True)

    count = await _count_event_rows(observability_database_url, "tenant-a", "cross-event-2")
    assert count == 1


async def test_unreachable_ingress_leaves_the_event_claimable_for_retry(
    backend: PostgresStorageBackend,
) -> None:
    """Independent of `observability_service`: proves the production
    wiring (`worker.outbox_http_delivery` driving
    `worker.outbox_delivery.run_outbox_delivery_loop`) actually leaves an
    event undelivered against a real dead endpoint, not just a mocked one
    (`tests/test_worker_outbox_http_delivery.py` already proves the
    mocked case)."""
    await _commit_one(backend, "cross-key-3", "cross-eval-3", "cross-event-3")

    dead_port = _free_tcp_port()  # nothing listens here
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{dead_port}", timeout=1.0) as client:
        # A negative lease duration means the delivery loop's own failed
        # claim is already "expired" the instant it's taken (same trick
        # `tests/conformance/test_worker_outbox_delivery.py` uses) -- so
        # the assertion below doesn't have to wait out a real lease window.
        settings = _worker_settings(outbox_lease_seconds=-1.0)
        deliver = build_http_deliver(client, settings)
        stop_event = asyncio.Event()

        async def _stop_soon() -> None:
            await asyncio.sleep(0.5)
            stop_event.set()

        await asyncio.gather(
            run_outbox_delivery_loop(backend, settings, stop_event, deliver=deliver),
            _stop_soon(),
        )

    remaining = await backend.claim_outbox_batch("checker", timedelta(minutes=1), limit=10)
    assert [e.event_id for e in remaining] == ["cross-event-3"]
