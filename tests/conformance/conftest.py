"""Shared fixtures for G5's storage conformance suite. ADR-019 SS4: "The
suite must run against an actual instance of the chosen database... Mock-only
tests do not establish persistence correctness." `testcontainers` launches a
real, throwaway PostgreSQL via the local Docker daemon for every test
function that requests the `backend` fixture below -- a fresh schema per
test, not a shared one, so tests can run concurrency scenarios without
cross-test interference.

This package is deliberately separate from `tests/` proper
(`tests/conformance/`, not `core/storage_conformance.py`): ADR-019 SS4
requires the reusable conformance suite to live in a test/conformance
package, not a pytest-dependent production module.
"""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from schemas.tenancy import TenantContext
from schemas.verification import (
    AbstentionReason,
    OverallVerdict,
    Provenance,
    ResultStatus,
    Usage,
    VerificationMode,
    VerificationRequest,
    VerificationResult,
)
from storage.models import OutboxEvent
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.engine import create_storage_engine
from storage.postgres.tables import metadata


@pytest.fixture(scope="session")
def postgres_container() -> AsyncIterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


def _asyncpg_url(container: PostgresContainer) -> str:
    # testcontainers' own default driver is irrelevant here -- this repo
    # only ever talks to Postgres through asyncpg (storage/postgres/engine.py).
    raw = container.get_connection_url()
    _, _, rest = raw.partition("://")
    return f"postgresql+asyncpg://{rest}"


@pytest_asyncio.fixture
async def engine(postgres_container: PostgresContainer) -> AsyncIterator[AsyncEngine]:
    eng = create_storage_engine(_asyncpg_url(postgres_container))
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def backend(engine: AsyncEngine) -> PostgresStorageBackend:
    return PostgresStorageBackend(engine)


def make_tenant(
    tenant_id: str = "tenant-a",
    project_id: str = "proj-a",
    application_id: str = "app-a",
) -> TenantContext:
    return TenantContext(tenant_id=tenant_id, project_id=project_id, application_id=application_id)


def make_request(
    request_id: str = "req-1",
    project_id: str = "proj-a",
    application_id: str = "app-a",
    answer: str = "The sky is blue.",
) -> VerificationRequest:
    return VerificationRequest(
        request_id=request_id,
        project_id=project_id,
        application_id=application_id,
        answer=answer,
    )


def default_deadline() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=5)


def make_result(
    evaluation_id: str,
    tenant_id: str = "tenant-a",
    project_id: str = "proj-a",
    application_id: str = "app-a",
    request_id: str = "req-1",
) -> VerificationResult:
    """A minimal, validly-shaped result (ABSTAINED/NO_CHECKABLE_CLAIMS --
    no claims/evidence required) for exercising the storage boundary, not
    the verification pipeline itself."""
    now = datetime.now(timezone.utc)
    return VerificationResult(
        evaluation_id=evaluation_id,
        request_id=request_id,
        tenant_id=tenant_id,
        project_id=project_id,
        application_id=application_id,
        status=ResultStatus.ABSTAINED,
        abstention_reason=AbstentionReason.NO_CHECKABLE_CLAIMS,
        verdict=OverallVerdict.ABSTAIN,
        provenance=Provenance(
            evaluator_id="conformance-suite",
            evaluator_version="0.0.0",
            mode=VerificationMode.STANDARD,
            routing_profile_version="0.0.0",
            started_at=now,
            completed_at=now,
        ),
    )


def make_outbox_event(
    event_id: str,
    evaluation_id: str,
    tenant_id: str = "tenant-a",
) -> OutboxEvent:
    return OutboxEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        evaluation_id=evaluation_id,
        occurred_at=datetime.now(timezone.utc),
        status=ResultStatus.ABSTAINED.value,
        verdict=OverallVerdict.ABSTAIN.value,
        scores={},
        evaluator_version="0.0.0",
        calibration_class="NONE",
        qualification_status="UNQUALIFIED",
        usage_summary=Usage(),
    )
