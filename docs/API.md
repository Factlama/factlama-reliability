# Reliability HTTP API v0.1 (target)

All routes are under `/v0.1`. Authentication establishes tenant scope; a project/application in the body is checked against it. JSON content type and `schema_version: "0.1"` are required. Error shape and [worked body](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md) are the shared specification. These routes are design targets, not implemented endpoints.

| Method/path | Behavior |
|---|---|
| `POST /v0.1/verifications` | Synchronous verification; 200 result, or 202 with job if server switches to async only when client requested `Prefer: respond-async` |
| `POST /v0.1/verification-jobs` | Durable async submission; 202 job handle |
| `GET /v0.1/verification-jobs/{job_id}` | Tenant-scoped status and result link; 200 |
| `GET /v0.1/verifications/{evaluation_id}` | Tenant-scoped result; 200 or 404 |
| `POST /v0.1/verification-jobs/{job_id}/cancel` | Best-effort cancellation; 202/200 state |

The client supplies `Idempotency-Key` for async submission (1–128 opaque characters). Its scope is tenant, route, project and key. Reuse with identical canonical request body returns the original job; reuse with a different body returns 409 `CONFLICT`. Retain the mapping at least as long as jobs/results are retained. `request_id` supports correlation but is not itself an idempotency key. A job is `QUEUED|RUNNING|SUCCEEDED|FAILED|CANCELLED`; `SUCCEEDED` may contain a completed or abstained verification result. `FAILED` is operational and has a typed error, never a factual verdict. Status responses include timestamps, attempt count, evaluation ID when assigned, and a result URL when available. Missing or inaccessible IDs return the same 404.

Limits, request deadlines, provider choices and retention are advertised in deployment configuration. Return 400 for malformed input, 401/403 for identity/authorization, 413 for oversized payload, 429 for quota, 503 for unavailable dependencies. `retryable` is true only when a retry may succeed; clients should use bounded exponential backoff. Synchronous provider timeout returns an abstained result if a logical evaluation completed with normalized evaluator failure, or 503 if no result could be recorded.
