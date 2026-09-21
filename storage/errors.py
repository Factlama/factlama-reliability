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
