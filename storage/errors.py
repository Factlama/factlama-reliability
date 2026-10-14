"""Typed storage-layer errors (G5, ADR-019).

Every error here is adapter-independent: `storage/postgres/backend.py`
translates Postgres-specific failures (unique-violation, serialization
failure, connection loss) into one of these, so a caller in `api/` or a
future worker package never needs to know which concrete backend is
running. PR2 maps these onto `api/errors.py::ApiErrorCode` (e.g.
`JobConflict` -> `CONFLICT`); the names below are chosen so that mapping is
a lookup, not a judgment call.
"""

from typing import Any


class StorageError(Exception):
    """Base exception for storage-layer failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class JobConflict(StorageError):
    """`submit_or_get_job` saw the same `(tenant_id, route, project_id,
    idempotency_key)` with a different `canonical_request_hash`.

    API.md: "Reuse with a different body returns 409 CONFLICT." This is
    raised, not returned as a value, because a mismatched replay is a
    genuine caller error (reused key, different intent) rather than an
    expected branch of a normal submission -- unlike the matching-hash case,
    which returns the original `Job` as a plain value (an idempotent retry
    is success, not an error).
    """


class LeaseExpired(StorageError):
    """A caller's fencing token no longer matches the job's current lease.

    Raised by `renew_lease` and `commit_evaluation_and_outbox` once a newer
    worker has reclaimed the job (ADR-019 SS4.3: "after expiry/reclaim,
    stale workers cannot renew, update, cancel, or commit using an obsolete
    token"). The stale worker's own work is not wasted silently: the caller
    is expected to abandon its in-progress attempt rather than retry blind.
    """


class EvaluationConflict(StorageError):
    """A `commit_evaluation_and_outbox` call named an `evaluation_id` that
    already exists with *different* content.

    ADR-019 SS4.4: "duplicate identical commits are retry-safe; conflicting
    content under an immutable evaluation/event ID is rejected, not
    silently treated as a no-op." An identical duplicate commit is not an
    error -- the backend treats it as success without raising.
    """


class JobNotFound(StorageError):
    """No job exists for the given tenant-scoped identifier, or it exists
    but belongs to a different tenant (the two cases are indistinguishable
    to the caller by design -- CONTRACTS.md: "missing or inaccessible IDs
    return the same 404")."""


class EvaluationNotFound(StorageError):
    """No evaluation exists for the given tenant-scoped identifier, or it
    belongs to a different tenant. Same fail-safe indistinguishability as
    `JobNotFound`."""


class ExecutionStorageNotAuthorized(StorageError):
    """ADR-020: raised by `submit_or_get_job` when the request's effective
    `execution_storage.mode` is `NO_PERSISTENCE` -- the real backstop
    against durable persistence of raw inputs, independent of whatever the
    API layer already checked. No `jobs`/`idempotency_keys` row is created;
    this is raised before any insert, not after a partial write."""


class ExecutionInputExpired(StorageError):
    """ADR-020: raised by `commit_evaluation_and_outbox` when a
    `DURABLE_ENCRYPTED` job's encrypted input has passed its absolute
    expiry (or been destroyed) between claim and commit -- A5 of the
    2026-10-14 ADR-020 review: bounded retries/backoff can span real
    wall-clock time, and a claim valid when issued is not guaranteed to
    still be valid by the time a result is ready to commit. The computed
    result is refused outright -- never committed as a trusted factual
    verdict over input that is no longer verified.

    `claim_due_job` itself no longer raises this (A1 of the same review):
    an *already* expired input at claim time is instead terminalized
    directly within that same transaction, so a permanently-unusable job
    cannot roll back its own claim and get reselected forever -- see
    `PostgresStorageBackend._terminalize_unusable_claim`."""


class ExecutionInputCorrupted(StorageError):
    """ADR-020: previously raised directly by `claim_due_job` when a
    `DURABLE_ENCRYPTED` job's stored ciphertext failed authenticated
    decryption, or its `job_execution_inputs` row was missing outright.
    A1 of the 2026-10-14 ADR-020 review: raising from inside that
    transaction rolled back the claim along with it, leaving the job
    QUEUED and immediately reselected by the same oldest-first query
    forever -- `claim_due_job` now terminalizes this case directly
    instead (`PostgresStorageBackend._terminalize_unusable_claim`) and
    this exception type is kept only for any caller still matching on it;
    nothing in this codebase raises it anymore."""
