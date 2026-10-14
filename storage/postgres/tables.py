"""SQLAlchemy Core table definitions for G5's PostgreSQL storage backend.

Core (`Table`/`select`/`insert`), never the ORM/session layer -- ADR-019
SS2 forbids exposing ORM sessions through `StorageBackend`'s public
signatures, and using Core throughout removes the temptation to leak one.
Only `storage/postgres/*` may import this module (enforced by
`pyproject.toml`'s import-linter contracts); `storage/backend.py` and
everything above it knows nothing about these column names.

Four tables form the one atomic `StorageBackend` boundary ADR-019 SS2
requires: `jobs`, `idempotency_keys`, `evaluations`, `outbox_events`. Their
relationships are enforced by the transactions in
`storage/postgres/backend.py`, not by database-level foreign keys alone --
`commit_evaluation_and_outbox` must still hold even if a row were ever
inserted out of band.

`revoked_evaluators` is a separate, single-row-per-identity table backing
`storage.registry.RevocationRegistry` (G5's baseline revocation check,
`storage/postgres/registry.py`) -- deliberately not folded into the four
above, since a revocation lookup is not part of any job's atomic
commit/idempotency boundary and has its own reader (`worker.runner`, not
`api/`).
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

jobs = Table(
    "jobs",
    metadata,
    Column("job_id", String(128), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("project_id", String(128), nullable=False),
    Column("application_id", String(128), nullable=False),
    Column("route", String(128), nullable=False),
    Column("state", String(16), nullable=False),
    Column("request_json", JSONB, nullable=False),
    Column("deadline", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("attempt_count", Integer, nullable=False, server_default="0"),
    # Monotonic per-job counter bumped on every successful `claim_due_job`;
    # `renew_lease`/`commit_evaluation_and_outbox`/`fail_job_terminal`/
    # `release_for_retry` all require the caller's token to still match
    # (ADR-019 SS4.3's fencing requirement).
    Column("fencing_token", BigInteger, nullable=False, server_default="0"),
    Column("lease_owner", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    # QUEUED jobs are claimable once `available_at` (if set) has passed --
    # `release_for_retry`'s backoff+jitter delay before a retried attempt
    # becomes due again.
    Column("available_at", DateTime(timezone=True), nullable=True),
    Column("cancel_requested", Boolean, nullable=False, server_default="false"),
    Column("evaluation_id", String(128), nullable=True),
    Column("error_code", String(64), nullable=True),
    Column("error_message", Text, nullable=True),
    # P2 of the 2026-10-14 ADR-020 closure review: when `fail_job_terminal`
    # rejects a late outcome whose input expired/was destroyed between
    # claim and commit, the dispatch that already happened produced real,
    # safe-to-keep attempt/usage history (provider/model identity,
    # qualification, outcome, token counts -- never claim/evidence
    # content) that `FAILED`'s own columns have no room for. Nullable and
    # populated only on that specific path -- every other `fail_job_terminal`
    # caller passes nothing, same as before this column existed. ADR-020
    # section 4: "preserving safe metadata and available attempt/usage
    # history" on an operational terminal outcome, not just a factual one.
    Column("terminal_provenance_json", JSONB, nullable=True),
    # Supports `claim_due_job`'s `(state, lease_expires_at, available_at)`
    # predicate; without it, every claim attempt is a full table scan once
    # the jobs table is non-trivially sized.
    Index("ix_jobs_claimable", "state", "lease_expires_at", "available_at"),
    Index("ix_jobs_tenant", "tenant_id"),
)

idempotency_keys = Table(
    "idempotency_keys",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("route", String(128), primary_key=True),
    Column("project_id", String(128), primary_key=True),
    Column("idempotency_key", String(128), primary_key=True),
    Column("canonical_request_hash", String(64), nullable=False),
    Column("job_id", String(128), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

evaluations = Table(
    "evaluations",
    metadata,
    # docs/async-evaluation.md: "result persistence uses a unique
    # (tenant_id,evaluation_id) key". No UPDATE path ever targets this
    # table -- CONTRACTS.md: "Historical results are immutable; re-evaluation
    # creates a new evaluation_id."
    Column("tenant_id", String(128), primary_key=True),
    Column("evaluation_id", String(128), primary_key=True),
    Column("job_id", String(128), nullable=True),
    # Denormalized from the owning job so `get_result`'s tenant/project/
    # application authorization check does not depend on the `jobs` row
    # still existing -- job orchestration state and evaluation results may
    # end up on different retention schedules (storage.retention.
    # RetentionPolicy, unwired -- no caller purges either table yet).
    Column("project_id", String(128), nullable=False),
    Column("application_id", String(128), nullable=False),
    Column("result_json", JSONB, nullable=False),
    # sha256 hex of `result_json`'s canonical serialization; lets
    # `commit_evaluation_and_outbox` distinguish a retried identical commit
    # (no-op success) from a genuine conflicting rewrite (ADR-019 SS4.4)
    # without re-parsing/deep-comparing the stored JSON on every retry.
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

outbox_events = Table(
    "outbox_events",
    metadata,
    # CONTRACTS.md: "Event identity is stable across retries, and
    # consumers deduplicate on (tenant_id,event_id)."
    Column("tenant_id", String(128), primary_key=True),
    Column("event_id", String(128), primary_key=True),
    Column("evaluation_id", String(128), nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acked", Boolean, nullable=False, server_default="false"),
    Column("lease_owner", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Index("ix_outbox_claimable", "acked", "lease_expires_at"),
    UniqueConstraint("event_id", name="uq_outbox_event_id_global"),
)

revoked_evaluators = Table(
    "revoked_evaluators",
    metadata,
    # Presence of a row is the revocation itself -- there is no `state`
    # column here (unlike `core.qualification.QualificationRecord`'s full
    # lifecycle): this table answers exactly one question, "is this
    # identity revoked," not "what lifecycle state is it in."
    Column("provider_id", String(256), primary_key=True),
    Column("pinned_model_id", String(512), primary_key=True),
    Column("configuration_version", String(128), primary_key=True),
    Column("revoked_at", DateTime(timezone=True), nullable=False),
)

job_execution_inputs = Table(
    "job_execution_inputs",
    metadata,
    # ADR-020: DURABLE_ENCRYPTED's encrypted execution input, one row per
    # job, inserted atomically with the job itself (same transaction as
    # `jobs`/`idempotency_keys` in `submit_or_get_job`) and deleted
    # atomically with the job's terminal transition (same transaction as
    # `commit_evaluation_and_outbox`/`fail_job_terminal`). Never contains
    # plaintext -- `ciphertext` is `core.execution_storage`'s AES-256-GCM
    # output, `jobs.request_json` itself already stores only the governed
    # (redacted) projection for this mode, never the real content.
    Column("tenant_id", String(128), primary_key=True),
    Column("job_id", String(128), primary_key=True),
    Column("key_id", String(64), nullable=False),
    Column("payload_version", Integer, nullable=False),
    Column("nonce", LargeBinary, nullable=False),
    Column("ciphertext", LargeBinary, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Absolute; set once at insert from WorkerSettings.execution_input_ttl_seconds
    # and never updated afterward -- no retry or lease renewal may extend it
    # (ADR-020 section 4).
    Column("expires_at", DateTime(timezone=True), nullable=False),
    # Supports the independent expiry sweep's "every row past expires_at,
    # regardless of job state" scan (worker.expiry_cleanup) without a full
    # table scan.
    Index("ix_job_execution_inputs_expiry", "expires_at"),
)

execution_input_tombstones = Table(
    "execution_input_tombstones",
    metadata,
    # A4 of the 2026-10-14 ADR-020 review: restoring a `jobs`/
    # `job_execution_inputs` snapshot taken *before* a job's terminal
    # transition (e.g. a table-scoped backup restore) brings back a
    # pre-terminal `jobs` row and its still-unexpired ciphertext together,
    # self-consistently -- nothing in either restored table can tell
    # `claim_due_job` this job was already terminated once. This table is
    # the "revocation state outside the restored snapshot" the review asks
    # for: one append-only row per job that ever reached a terminal state,
    # inserted atomically with the terminal transition (same transaction as
    # `commit_evaluation_and_outbox`/`fail_job_terminal`'s existing
    # `job_execution_inputs` DELETE) and never deleted or updated again.
    # `claim_due_job` refuses to claim any job with a row here, regardless
    # of what the (possibly restored) `jobs`/`job_execution_inputs` rows
    # claim.
    #
    # This closes exactly the review's reproduced scenario -- a restore
    # that does not also happen to roll back this table. It does NOT
    # protect against a full-database point-in-time restore that rolls
    # back every table including this one to a snapshot taken before this
    # job's termination: that is a structural limit of a single
    # PostgreSQL instance (ADR-009's MVP baseline) and would need
    # revocation state in a genuinely separate system (an external
    # append-only audit log, or real per-job KMS-side revocation) -- named
    # here, not silently assumed solved.
    Column("tenant_id", String(128), primary_key=True),
    Column("job_id", String(128), primary_key=True),
    Column("terminated_at", DateTime(timezone=True), nullable=False),
)
