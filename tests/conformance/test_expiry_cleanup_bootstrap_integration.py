"""`worker.expiry_cleanup_bootstrap.run()` is the standalone execution-input
expiry-cleanup service's own composition root (A6 of the 2026-10-14 ADR-020
review) -- a third process that touches storage independently of the API
and the dispatch worker, and so is gated by ADR-020 section 5's mandatory
restore procedure exactly the same way (`worker/bootstrap.py`,
`api/bootstrap.py`). This proves that gate is actually wired into this
process too, the same way `tests/conformance/test_bootstrap_integration.py`
and `test_worker_bootstrap_integration.py` prove it for the other two.
"""

import asyncio
import base64
import json

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from core.execution_storage import ExecutionStorageConfigurationError
from tests.conformance.conftest import TEST_EXECUTION_STORAGE_KEY, TEST_EXECUTION_STORAGE_KEY_ID
from worker.expiry_cleanup_bootstrap import run


def _asyncpg_url(container: PostgresContainer) -> str:
    raw = container.get_connection_url()
    _, _, rest = raw.partition("://")
    return f"postgresql+asyncpg://{rest}"


async def test_run_refuses_to_start_when_a_declared_retired_key_is_still_configured(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", TEST_EXECUTION_STORAGE_KEY_ID)

    with pytest.raises(ExecutionStorageConfigurationError, match=TEST_EXECUTION_STORAGE_KEY_ID):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_refuses_to_start_when_restore_mode_has_an_empty_manifest(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.delenv("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", raising=False)

    with pytest.raises(ExecutionStorageConfigurationError, match="RESTORE_MODE=RESTORE"):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_refuses_to_start_with_an_invalid_restore_mode_value(
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTOER")

    with pytest.raises(ExecutionStorageConfigurationError, match="RESTOER"):
        await asyncio.wait_for(run(), timeout=0.5)


async def test_run_starts_successfully_once_declared_retired_keys_are_actually_removed(
    engine: AsyncEngine,  # ensures the schema exists before this process connects
    postgres_container: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive path: an operator declares the restore, names the key
    that could have encrypted a now-stale row, and has actually rotated it
    out of the active keyring -- startup must succeed and the sweep must
    keep running."""
    monkeypatch.setenv("FACTLAMA_DATABASE_URL", _asyncpg_url(postgres_container))
    monkeypatch.setenv("FACTLAMA_WORKER_EXECUTION_INPUT_SWEEP_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_KEYS",
        json.dumps(
            {TEST_EXECUTION_STORAGE_KEY_ID: base64.b64encode(TEST_EXECUTION_STORAGE_KEY).decode()}
        ),
    )
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID", TEST_EXECUTION_STORAGE_KEY_ID)
    monkeypatch.setenv("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE", "RESTORE")
    monkeypatch.setenv(
        "FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS", "some-retired-key-from-the-backup"
    )

    with pytest.raises(TimeoutError):
        # A clean, still-running sweep never returns on its own; timing out
        # (rather than raising) is the proof startup succeeded.
        await asyncio.wait_for(run(), timeout=0.5)
