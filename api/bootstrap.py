"""Composition root (ADR-019 SS3): the one module in `api/` allowed to
import `storage.postgres` directly. Constructs one `PostgresStorageBackend`
per process during FastAPI startup, validates it can actually reach the
database before the app accepts traffic, and disposes it on shutdown.
Everything else in `api/` depends only on `storage.backend.StorageBackend`
-- `pyproject.toml`'s import-linter contract carries an explicit
`ignore_imports` exception naming this module and no other.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.settings import get_settings
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.engine import (
    StorageConfigurationError,
    check_ready,
    close_engine,
    create_storage_engine,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.database_url is None:
        raise StorageConfigurationError(
            "FACTLAMA_DATABASE_URL must be set to start the API (ADR-019: PostgreSQL "
            "is a required dependency of the whole service, not only the async routes)"
        )
    engine = create_storage_engine(settings.database_url)
    # ADR-019 SS3: "failed initialization must fail startup/readiness
    # explicitly; never silently fall back to another database." An
    # unreachable database at startup raises here and the process never
    # starts serving traffic, rather than surfacing as the first request's
    # 500.
    await check_ready(engine)
    app.state.storage_engine = engine
    app.state.storage_backend = PostgresStorageBackend(engine)
    logger.info("storage backend ready")
    try:
        yield
    finally:
        await close_engine(engine)


async def check_storage_ready(app: FastAPI) -> bool:
    """Live readiness probe for `/health/ready` -- a fresh check against the
    database, not merely "did startup succeed once" (ADR-019 SS3's ongoing
    readiness, not just initial validation)."""
    engine = app.state.storage_engine
    try:
        await check_ready(engine)
    except Exception:
        logger.exception("storage readiness check failed")
        return False
    return True
