"""ADR-020 section 5: key-management acceptance evidence against real
PostgreSQL -- outage (no silent plaintext fallback), rotation (old
ciphertext stays decryptable, new writes use the new key), and a
backup/restore simulation proving a restored row cannot re-enable access
to an already-expired input (the row's own absolute `expires_at`, not its
mere presence/absence, is what protects it).
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine

from core.execution_storage import ExecutionStorageConfigurationError, ExecutionStorageKeyring
from core.verifier import Verifier
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.registry import PostgresRevocationRegistry
from storage.postgres.tables import job_execution_inputs, metadata
from tests.conformance.conftest import default_deadline, make_request, make_tenant
from tests.conformance.test_worker_runner import _fast_settings
from worker.runner import process_claimed_job

KEY_V1 = b"\x11" * 32
KEY_V2 = b"\x22" * 32


class TestOutageHasNoSilentPlaintextFallback:
    async def test_no_keyring_configured_fails_loudly_not_plaintext(
        self, engine: AsyncEngine
    ) -> None:
        backend = PostgresStorageBackend(engine)  # no keyring at all
        with pytest.raises(ExecutionStorageConfigurationError):
            await backend.submit_or_get_job(
                make_tenant(),
                "verification-jobs",
                "key-outage-1",
                "hash-outage-1",
                make_request(),
                default_deadline(),
            )
        async with engine.connect() as conn:
            rows = (await conn.execute(select(job_execution_inputs.c.job_id))).all()
        assert rows == []

    async def test_keyring_with_no_active_key_fails_loudly_not_plaintext(
        self, engine: AsyncEngine
    ) -> None:
        """Keys are configured (decryption of existing rows would work) but
        no active key is set for *new* writes -- submission must still fail
        loudly, never pick an arbitrary key or fall back to plaintext."""
        keyring = ExecutionStorageKeyring({"v1": KEY_V1}, active_key_id=None)
        backend = PostgresStorageBackend(engine, execution_storage_keyring=keyring)
        with pytest.raises(ExecutionStorageConfigurationError):
            await backend.submit_or_get_job(
                make_tenant(),
                "verification-jobs",
                "key-outage-2",
                "hash-outage-2",
                make_request(),
                default_deadline(),
            )


class TestRotationPreservesOldCiphertextAccess:
    async def test_v1_encrypted_row_still_decrypts_after_v2_becomes_active(
        self, engine: AsyncEngine
    ) -> None:
        keyring_v1_active = ExecutionStorageKeyring({"v1": KEY_V1}, active_key_id="v1")
        backend_v1 = PostgresStorageBackend(engine, execution_storage_keyring=keyring_v1_active)
        tenant = make_tenant()
        job = await backend_v1.submit_or_get_job(
            tenant,
            "verification-jobs",
            "key-rotate-1",
            "hash-rotate-1",
            make_request(answer="encrypted under v1"),
            default_deadline(),
        )
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    select(job_execution_inputs.c.key_id).where(
                        job_execution_inputs.c.job_id == job.job_id
                    )
                )
            ).one()
        assert row.key_id == "v1"

        # "Rotation": a new backend instance (simulating a redeployed
        # worker) with v2 now active, but v1 still present -- ADR-020:
        # "rotation is not destruction."
        keyring_rotated = ExecutionStorageKeyring({"v1": KEY_V1, "v2": KEY_V2}, active_key_id="v2")
        backend_rotated = PostgresStorageBackend(engine, execution_storage_keyring=keyring_rotated)
        claimed = await backend_rotated.claim_due_job("worker-test", timedelta(minutes=5))
        assert claimed is not None
        assert claimed.request.answer == "encrypted under v1"

    async def test_new_writes_after_rotation_use_the_new_active_key(
        self, engine: AsyncEngine
    ) -> None:
        keyring_rotated = ExecutionStorageKeyring({"v1": KEY_V1, "v2": KEY_V2}, active_key_id="v2")
        backend = PostgresStorageBackend(engine, execution_storage_keyring=keyring_rotated)
        job = await backend.submit_or_get_job(
            make_tenant(),
            "verification-jobs",
            "key-rotate-2",
            "hash-rotate-2",
            make_request(),
            default_deadline(),
        )
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    select(job_execution_inputs.c.key_id).where(
                        job_execution_inputs.c.job_id == job.job_id
                    )
                )
            ).one()
        assert row.key_id == "v2"


class TestBackupRestoreCannotRevivedDestroyedAccess:
    async def test_restoring_an_already_expired_row_does_not_grant_access(
        self, engine: AsyncEngine, backend: PostgresStorageBackend
    ) -> None:
        """Simulates restoring a database backup taken *before* a row's
        expiry/deletion: the row comes back with its original (now-past)
        `expires_at` intact, since that field is immutable and set once at
        submission -- the restored row's own data protects it, not
        whatever cleanup happened to run against the live database in the
        meantime."""
        tenant = make_tenant()
        job = await backend.submit_or_get_job(
            tenant,
            "verification-jobs",
            "key-restore-1",
            "hash-restore-1",
            make_request(),
            default_deadline(),
        )
        async with engine.connect() as conn:
            backed_up_row = (
                await conn.execute(
                    select(job_execution_inputs).where(job_execution_inputs.c.job_id == job.job_id)
                )
            ).one()

        # The live row is deleted (e.g. normal terminal cleanup or the
        # expiry sweep) -- simulating time passing in the live system.
        async with engine.begin() as conn:
            await conn.execute(
                job_execution_inputs.delete().where(job_execution_inputs.c.job_id == job.job_id)
            )

        # "Restore from backup": reinsert the exact row as it was captured
        # above, but the backup was taken far enough in the past that its
        # own expires_at is now in the past relative to "now".
        stale_expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        async with engine.begin() as conn:
            await conn.execute(
                job_execution_inputs.insert().values(
                    tenant_id=backed_up_row.tenant_id,
                    job_id=backed_up_row.job_id,
                    key_id=backed_up_row.key_id,
                    payload_version=backed_up_row.payload_version,
                    nonce=backed_up_row.nonce,
                    ciphertext=backed_up_row.ciphertext,
                    created_at=backed_up_row.created_at,
                    expires_at=stale_expires_at,
                )
            )

        # A1 of the 2026-10-14 ADR-020 review: an expired input no longer
        # raises from inside the claim transaction -- `claim_due_job`
        # terminalizes the job directly and returns `None`.
        claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
        assert claimed is None


class TestMandatoryKeyRetirementDefeatsFullDatabaseRestore:
    """P1 of the 2026-10-14 ADR-020 closure review: a tombstone table lives
    in the same database as everything else, so restoring *every*
    application table together (not just `jobs`/`job_execution_inputs`)
    restores the tombstone's own absence too -- no purely in-database
    table can defend against that, by construction. The one thing that
    does NOT live in any restored table is the keyring itself (env
    configuration) -- the review's own suggested fix: a mandatory restore
    procedure that retires every key_id that might have encrypted a
    now-stale row before any worker/API process touches the restored
    database. These tests prove that procedure actually works when
    followed (`test_...`) and name, rather than hide, what happens when it
    is not (`test_without_key_retirement_...`, the closure review's own
    reproduction, kept permanent)."""

    async def test_retiring_the_key_before_restore_prevents_revival(
        self, engine: AsyncEngine, registry: PostgresRevocationRegistry
    ) -> None:
        keyring_v1 = ExecutionStorageKeyring({"v1": KEY_V1}, active_key_id="v1")
        backend = PostgresStorageBackend(engine, execution_storage_keyring=keyring_v1)
        tenant = make_tenant()
        job = await backend.submit_or_get_job(
            tenant,
            "verification-jobs",
            "key-retire-restore",
            "hash-retire-restore",
            make_request(),
            default_deadline(),
        )
        async with engine.connect() as conn:
            snapshot = {
                table.name: [dict(row) for row in (await conn.execute(select(table))).mappings()]
                for table in metadata.sorted_tables
            }
        claimed = await backend.claim_due_job("worker", timedelta(minutes=1))
        assert claimed is not None
        await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)
        status = await backend.get_job_status(tenant, job.job_id)
        assert status is not None and status.state.value == "SUCCEEDED"

        # Full-database restore: every application table, tombstone
        # included, rolled back to the pre-completion snapshot -- the
        # exact scenario the closure review reproduced.
        async with engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                await conn.execute(delete(table))
            for table in metadata.sorted_tables:
                if snapshot[table.name]:
                    await conn.execute(table.insert(), snapshot[table.name])

        # The mandatory restore procedure: retire key_id "v1" (the only key
        # any row in this restored database could have been encrypted
        # under) before bringing a worker back online against it. No
        # database state is touched by this step -- it is pure
        # configuration, which is exactly why it survives the restore.
        keyring_after_restore = ExecutionStorageKeyring({"v2": KEY_V2}, active_key_id="v2")
        backend_after_restore = PostgresStorageBackend(
            engine, execution_storage_keyring=keyring_after_restore
        )

        revived = await backend_after_restore.claim_due_job("restored-worker", timedelta(minutes=1))
        assert revived is None, "Retiring the key before restore must still prevent revival"
        status_after = await backend_after_restore.get_job_status(tenant, job.job_id)
        assert status_after is not None
        assert status_after.state.value == "FAILED", (
            "A row using a retired key must be terminalized, not left stuck"
        )

    async def test_without_key_retirement_full_database_restore_revives_input(
        self, engine: AsyncEngine, registry: PostgresRevocationRegistry
    ) -> None:
        """The closure review's own reproduction, kept permanent: skipping
        the mandatory key-retirement step (restoring with the keyring left
        unchanged) is the one scenario this fix does not and cannot close
        from inside a single database -- named explicitly, not hidden."""
        keyring = ExecutionStorageKeyring({"v1": KEY_V1}, active_key_id="v1")
        backend = PostgresStorageBackend(engine, execution_storage_keyring=keyring)
        tenant = make_tenant()
        job = await backend.submit_or_get_job(
            tenant,
            "verification-jobs",
            "key-no-retire-restore",
            "hash-no-retire-restore",
            make_request(),
            default_deadline(),
        )
        async with engine.connect() as conn:
            snapshot = {
                table.name: [dict(row) for row in (await conn.execute(select(table))).mappings()]
                for table in metadata.sorted_tables
            }
        claimed = await backend.claim_due_job("worker", timedelta(minutes=1))
        assert claimed is not None
        await process_claimed_job(claimed, backend, Verifier(), _fast_settings(), registry)

        async with engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                await conn.execute(delete(table))
            for table in metadata.sorted_tables:
                if snapshot[table.name]:
                    await conn.execute(table.insert(), snapshot[table.name])

        # Same unchanged keyring -- the mandatory procedure was skipped.
        revived = await backend.claim_due_job("restored-worker", timedelta(minutes=1))
        assert revived is not None and revived.job_id == job.job_id, (
            "Named limit: without key retirement, a full-database restore still revives "
            "a destroyed input -- this is why the retirement step is documented as "
            "mandatory, not optional, in REL-11's entry"
        )
