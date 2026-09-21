"""Alembic environment, wired for the async `postgresql+asyncpg://` engine
this repo standardizes on (`storage/postgres/engine.py`) rather than adding
a second, sync-only driver dependency solely for migrations. Follows
SQLAlchemy's documented async-Alembic recipe: run migrations inside
`AsyncConnection.run_sync`, since Alembic's own migration machinery is
synchronous.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from storage.postgres.tables import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def _database_url() -> str:
    url = os.environ.get("FACTLAMA_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "FACTLAMA_DATABASE_URL must be set (postgresql+asyncpg://...) to run migrations"
        )
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live database connection (`alembic
    upgrade head --sql`)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
