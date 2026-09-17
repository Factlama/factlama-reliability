"""REL-12's synchronous verification API: `POST /v0.1/verifications`
(API.md), the API/worker half of LOW_LEVEL_IMPLEMENTATION.md's `api` unit.

Scope is deliberately synchronous-only. Async submission/status
(`POST /v0.1/verification-jobs`, REL-10) and its idempotency/outbox mechanics
need G5's durable persistence, which does not exist in this repo yet -- G3's
own acceptance criterion is "sync API", not the full API.md surface.

This is also `TenantContext`'s first real caller (see `schemas/tenancy.py`'s
docstring: "no authenticated ingress exists yet -- that is G3's sync API").
`authenticate()` below is the boundary that must call `TenantContext.authorizes()`
before a request reaches `Verifier.verify()`, not merely echo a tenant_id it
was handed -- closing REL-02's remaining "propagate context through API" and
"negative cross-tenant test against a real boundary" items.
"""

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.errors import ApiError, ApiErrorCode
from api.settings import Settings, get_settings
from core.verifier import Verifier, get_default_verifier
from schemas.tenancy import TenantContext
from schemas.verification import VerificationRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="FactLama Reliability API", version="0.1")


def get_verifier() -> Verifier:
    return get_default_verifier()


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
async def health_ready() -> dict[str, str]:
    # No durable dependency exists in this pass (no DB, no queue) -- ready is
    # equivalent to live until G5 gives this something real to check.
    return {"status": "ok"}


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


@app.post("/v0.1/verifications")
async def create_verification(
    request: Request,
    tenant_context: Annotated[TenantContext, Depends(authenticate)],
    settings: Annotated[Settings, Depends(get_settings)],
    verifier: Annotated[Verifier, Depends(get_verifier)],
) -> JSONResponse:
    _require_json_content_type(request)
    _enforce_content_length_limit(request, settings.max_body_bytes)

    body = await request.body()
    if len(body) > settings.max_body_bytes:
        raise ApiError(
            ApiErrorCode.PAYLOAD_TOO_LARGE,
            f"request body of {len(body)} bytes exceeds the {settings.max_body_bytes}-byte limit",
        )

    try:
        verification_request = VerificationRequest.model_validate_json(body)
    except ValidationError as exc:
        message = _validation_error_message(exc)
        code = (
            ApiErrorCode.UNSUPPORTED_VERSION
            if "Unsupported schema version" in message
            else ApiErrorCode.INVALID_ARGUMENT
        )
        raise ApiError(code, message) from exc

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

    # verifier.verify() is a synchronous, potentially model-heavy call (a
    # local embedding/NLI evaluation is CPU-bound). Running it directly in
    # this coroutine would block the whole event loop for every other
    # in-flight request; dispatching it to Starlette's thread pool keeps the
    # loop free.
    result = await run_in_threadpool(
        verifier.verify, verification_request, tenant_id=tenant_context.tenant_id
    )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))
