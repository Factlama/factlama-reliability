"""REL-12's HTTP API (API.md): `POST /v0.1/verifications` (sync, G3) plus
G5's durable async surface -- `POST /v0.1/verification-jobs`,
`GET /v0.1/verification-jobs/{job_id}`, `GET /v0.1/verifications/{evaluation_id}`,
`POST /v0.1/verification-jobs/{job_id}/cancel` -- now that
`storage.postgres` (G5 PR1) gives the async routes' idempotency/outbox
mechanics somewhere to live.

This is also `TenantContext`'s first real caller (see `schemas/tenancy.py`'s
docstring: "no authenticated ingress exists yet -- that is G3's sync API").
`authenticate()` below is the boundary that must call `TenantContext.authorizes()`
before a request reaches `Verifier.verify()`/the storage backend, not merely
echo a tenant_id it was handed -- closing REL-02's remaining "propagate
context through API" and "negative cross-tenant test against a real
boundary" items.

Async routes depend on `storage.backend.StorageBackend` only, never
`storage.postgres` directly -- `api/bootstrap.py` is the one composition-root
module that constructs the concrete Postgres backend
(`pyproject.toml`'s import-linter contract enforces this).
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.bootstrap import check_storage_ready, lifespan
from api.errors import ApiError, ApiErrorCode
from api.settings import Settings, get_settings
from core.verifier import Verifier, get_default_verifier
from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest
from storage.backend import StorageBackend
from storage.errors import JobConflict, JobNotFound
from storage.models import Job

logger = logging.getLogger(__name__)

app = FastAPI(title="FactLama Reliability API", version="0.1", lifespan=lifespan)


def get_verifier() -> Verifier:
    return get_default_verifier()


def get_storage_backend(request: Request) -> StorageBackend:
    """Populated by `api.bootstrap.lifespan` at process startup. Tests
    override this dependency directly (like `get_verifier`/`get_settings`),
    so they never depend on `app.state` or on lifespan having actually run."""
    backend: StorageBackend = request.app.state.storage_backend
    return backend


async def get_storage_readiness(request: Request) -> bool:
    """A live check, not just "did startup succeed once" -- overridden
    directly in tests for the same reason as `get_storage_backend`."""
    return await check_storage_ready(request.app)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def authenticate(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> TenantContext:
    """Establish trusted tenant identity from the `Authorization` header.

    Never trusts anything in the request body -- `VerificationRequest`
    itself already rejects a client-supplied `tenant_id` field (a defense in
    depth, not the primary boundary; this is).
    """
    api_key = _extract_bearer_token(authorization)
    tenant_context = settings.tenant_store.authenticate(api_key)
    if tenant_context is None:
        raise ApiError(ApiErrorCode.UNAUTHENTICATED, "missing or invalid API key")
    return tenant_context


@app.exception_handler(ApiError)
async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready(
    storage_ready: Annotated[bool, Depends(get_storage_readiness)],
) -> JSONResponse:
    if not storage_ready:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(status_code=200, content={"status": "ok"})


def _validation_error_message(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else str(first["msg"])


def _require_json_content_type(request: Request) -> None:
    """API.md: "JSON content type ... required." Checked before the body is
    even read -- a non-JSON payload should never reach `model_validate_json`,
    which would otherwise happily parse a JSON-shaped `text/plain` body."""
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            f"Content-Type must be application/json, got {content_type or '(none)'}",
        )


def _enforce_content_length_limit(request: Request, max_body_bytes: int) -> None:
    """A fast-path size check against the declared `Content-Length`, before
    reading the body. A malformed header must fail as a typed 400, never an
    uncontracted 500 -- the actual byte-count check after reading the body
    (`create_verification`) still catches a missing/lying header."""
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        declared_length = int(content_length)
    except ValueError as exc:
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            f"malformed Content-Length header: {content_length!r}",
        ) from exc
    if declared_length > max_body_bytes:
        raise ApiError(
            ApiErrorCode.PAYLOAD_TOO_LARGE,
            f"request body of {declared_length} bytes exceeds the {max_body_bytes}-byte limit",
        )


async def _read_body_within_limit(request: Request, max_body_bytes: int) -> bytes:
    """Stream the body, aborting as soon as more than `max_body_bytes` have
    actually been received.

    F3 of the 2026-09-21 G0-G4 validation report: `request.body()` buffers
    the *entire* payload into memory before any length check runs. When
    `Content-Length` is absent or understates the real size (both entirely
    under the client's control), `_enforce_content_length_limit`'s fast-path
    check above never fires, and the only remaining guard was a check
    performed *after* that full buffering -- by then the oversized payload
    was already fully resident in memory regardless.
    """
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > max_body_bytes:
            raise ApiError(
                ApiErrorCode.PAYLOAD_TOO_LARGE,
                f"request body exceeds the {max_body_bytes}-byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_verification_request(body: bytes) -> VerificationRequest:
    try:
        return VerificationRequest.model_validate_json(body)
    except ValidationError as exc:
        message = _validation_error_message(exc)
        if "Unsupported schema version" in message:
            code = ApiErrorCode.UNSUPPORTED_VERSION
        elif "policy_id cannot be resolved" in message:
            # CONTRACTS.md: "NOT_FOUND is used for inaccessible tenant-scoped
            # resources" -- a policy_id names exactly such a resource, not a
            # structurally malformed argument.
            code = ApiErrorCode.NOT_FOUND
        else:
            code = ApiErrorCode.INVALID_ARGUMENT
        raise ApiError(code, message) from exc


def _authorize_request_scope(
    tenant_context: TenantContext, verification_request: VerificationRequest
) -> None:
    if not tenant_context.authorizes(
        verification_request.project_id, verification_request.application_id
    ):
        logger.warning(
            "Tenant %s denied access to project=%s application=%s",
            tenant_context.tenant_id,
            verification_request.project_id,
            verification_request.application_id,
        )
        raise ApiError(
            ApiErrorCode.FORBIDDEN,
            "credential is not authorized for the requested project/application scope",
        )


@app.post("/v0.1/verifications")
async def create_verification(
    request: Request,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    settings: Annotated[Settings, Depends(get_settings)],
    verifier: Annotated[Verifier, Depends(get_verifier)],
) -> JSONResponse:
    _require_json_content_type(request)
    _enforce_content_length_limit(request, settings.max_body_bytes)

    body = await _read_body_within_limit(request, settings.max_body_bytes)
    verification_request = _parse_verification_request(body)
    _authorize_request_scope(tenant_context, verification_request)

    # verifier.verify() is a synchronous, potentially model-heavy call (a
    # local embedding/NLI evaluation is CPU-bound). Running it directly in
    # this coroutine would block the whole event loop for every other
    # in-flight request; dispatching it to Starlette's thread pool keeps the
    # loop free.
    result = await run_in_threadpool(
        verifier.verify, verification_request, tenant_id=tenant_context.tenant_id
    )
    # exclude_none: contracts/v0.1 requires several optional fields (e.g. an
    # unmeasured ScoreValue.value) to be absent, not present as null -- see
    # CONTRACTS.md: "null means known absent; omitted means not supplied."
    return JSONResponse(status_code=200, content=result.model_dump(mode="json", exclude_none=True))


def _canonical_request_hash(verification_request: VerificationRequest) -> str:
    """API.md: "Reuse with identical canonical request body returns the
    original job; reuse with a different body returns 409." Canonical means
    content-equivalent, not byte-identical -- key order/whitespace in the
    client's original bytes must not matter, so this hashes the parsed and
    re-sorted structure, not the raw request body."""
    canonical = json.dumps(
        verification_request.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_idempotency_key(idempotency_key: str | None) -> str:
    """API.md: "The client supplies Idempotency-Key for async submission
    (1-128 opaque characters)." Missing or out-of-range is a caller error,
    not a silently-generated key -- a generated key would defeat the whole
    point of client-supplied idempotency."""
    if idempotency_key is None or not (1 <= len(idempotency_key) <= 128):
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            "Idempotency-Key header is required and must be 1-128 characters",
        )
    return idempotency_key


def _job_status_body(job: Job) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "0.1",
        "job_id": job.job_id,
        "state": job.state.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "attempt_count": job.attempt_count,
    }
    if job.evaluation_id is not None:
        body["evaluation_id"] = job.evaluation_id
        body["result_url"] = f"/v0.1/verifications/{job.evaluation_id}"
    return body


@app.post("/v0.1/verification-jobs", status_code=202)
async def create_verification_job(
    request: Request,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    settings: Annotated[Settings, Depends(get_settings)],
    storage_backend: Annotated[StorageBackend, Depends(get_storage_backend)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    _require_json_content_type(request)
    _enforce_content_length_limit(request, settings.max_body_bytes)
    key = _require_idempotency_key(idempotency_key)

    body = await _read_body_within_limit(request, settings.max_body_bytes)
    verification_request = _parse_verification_request(body)
    _authorize_request_scope(tenant_context, verification_request)

    deadline = datetime.now(timezone.utc) + timedelta(seconds=settings.job_deadline_seconds)
    try:
        job = await storage_backend.submit_or_get_job(
            tenant_context,
            "verification-jobs",
            key,
            _canonical_request_hash(verification_request),
            verification_request,
            deadline,
        )
    except JobConflict as exc:
        raise ApiError(
            ApiErrorCode.CONFLICT,
            "Idempotency-Key was already used with a different request body",
        ) from exc

    return JSONResponse(status_code=202, content=_job_status_body(job))


@app.get("/v0.1/verification-jobs/{job_id}")
async def get_verification_job(
    job_id: str,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    storage_backend: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> JSONResponse:
    job = await storage_backend.get_job_status(tenant_context, job_id)
    if job is None:
        raise ApiError(ApiErrorCode.NOT_FOUND, "job not found")
    return JSONResponse(status_code=200, content=_job_status_body(job))


@app.get("/v0.1/verifications/{evaluation_id}")
async def get_verification_result(
    evaluation_id: str,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    storage_backend: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> JSONResponse:
    result = await storage_backend.get_result(tenant_context, evaluation_id)
    if result is None:
        raise ApiError(ApiErrorCode.NOT_FOUND, "evaluation not found")
    return JSONResponse(status_code=200, content=result.model_dump(mode="json", exclude_none=True))


@app.post("/v0.1/verification-jobs/{job_id}/cancel")
async def cancel_verification_job(
    job_id: str,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    storage_backend: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> JSONResponse:
    try:
        await storage_backend.request_cancel(tenant_context, job_id)
    except JobNotFound as exc:
        raise ApiError(ApiErrorCode.NOT_FOUND, "job not found") from exc

    job = await storage_backend.get_job_status(tenant_context, job_id)
    if job is None:
        # request_cancel just succeeded against this tenant/job_id; a
        # concurrent deletion is not a supported operation in this repo.
        raise ApiError(ApiErrorCode.INTERNAL, "job vanished immediately after cancellation")
    return JSONResponse(status_code=200, content=_job_status_body(job))
