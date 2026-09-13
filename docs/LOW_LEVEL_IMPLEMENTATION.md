# Reliability MVP: low-level implementation and review checklist

**Status: specification only; no runtime code is claimed.** Read the architecture [implementation map](https://github.com/Factlama/factlama-architecture/blob/main/IMPLEMENTATION_MAP.md) and [wire contract](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md) first. This file tells a reviewer what code must exist, how data moves, and what result to observe. `docs/implementation.md` remains the task-status ledger. Scope here is REL-01–12; REL-13–15 are later capabilities.

## Code units to build

| Unit | Owned behavior | Review in code |
|---|---|---|
| `contracts` | Typed v0.1 request/result, score, violation, provenance, job and event models; strict validators and JSON codecs | No provider/storage imports; enum and required-field tests |
| `auth` | Credential validation, trusted tenant/actor/project scope, internal worker context | Tenant never comes from untrusted body/header |
| `claims` | Accept explicit claims or split answer into atomic checkable claims, stable IDs and offsets | Deterministic fixture output and extractor version |
| `evidence` | Validate supplied evidence, resolve authorized references, map candidate evidence to claims | No similarity-as-truth; conflict/empty cases |
| `judge` | Provider port, one working adapter, strict normalized response parser, deadlines and error taxonomy | Adapter can be swapped without editing core |
| `verification` | Orchestrate stages and normalize claim findings | One result per request; failure becomes abstention |
| `scoring` | Pure v0.1 ratio formulas and aggregate verdict precedence | Golden table below; no model calls |
| `policy` | Ordered versioned rules and separate action | Same result + policy version gives same action |
| `jobs` | Durable submission, leases, bounded retries, cancellation, idempotency | Duplicate submission/delivery stays one evaluation |
| `repository` | Tenant-keyed evaluation, job, idempotency and outbox persistence | Every read/write has tenant predicate and constraint |
| `api` | Versioned routes, errors, health/readiness, request limits | OpenAPI or equivalent matches contract examples |
| `events` | Build and deliver metadata-only ReliabilityEvent | Stable event ID and eventual Observability visibility |

The suggested package names are ownership boundaries, not a requirement to create empty directories. Domain functions accept explicit inputs and return values; only adapters perform I/O. The API and worker call the same verification service. Dependency direction is `api/worker -> verification -> claims/evidence/judge port/scoring/policy`; implementations of judge, DB and event transport point inward.

## Request and result algorithm

1. Parse JSON and `schema_version`; reject unsupported versions and size/shape errors before provider calls. Authenticate to `TenantContext(tenant_id, actor_id, allowed_projects, scopes)` and authorize project/application. Ignore or reject a tenant field in the body; never adopt it.
2. Set `request_id`, `evaluation_id` and timestamps. Synchronous request creates one evaluation transaction; async submission first persists an idempotent job. Preserve `interaction_id`, W3C trace/span IDs and prompt/model identifiers exactly.
3. If `claims` is supplied, validate unique IDs and use it unchanged. Otherwise segment `answer` into atomic propositions, retaining source span and extractor version. No checkable claim produces `ABSTAINED/ABSTAIN`, not a spurious PASS.
4. Validate unique evidence IDs and exactly one of content/reference. Resolve references only through a tenant-approved resolver. Map explicit citations first, then bounded candidates. Pass conflicting candidates through. Empty/unavailable corpus yields `INSUFFICIENT_EVIDENCE`; a usable examined corpus with no support may yield `UNSUPPORTED`.
5. Select the tenant-approved judge/configuration version. Call the adapter once per claim (bounded concurrency) with candidate evidence, remaining deadline and cancellation. Validate response enum and evidence IDs against the request. Normalize timeout, rate limit, unavailable, malformed response and cancellation to typed failures; never map them to a factual verdict. A deterministic fixture provider is allowed for tests, but MVP needs one configured working external/local provider adapter.
6. Assemble one finding for every claim. If a technical failure prevents a complete evaluation, set `status=ABSTAINED`, `verdict=ABSTAIN` and affected scores `UNAVAILABLE`; retain sanitized stage errors/provenance. Otherwise calculate pure scores, then aggregate the factual verdict using the table below.
7. Evaluate ordered policy rules against the immutable factual result. Store `policy_decision` separately; it cannot rewrite a claim verdict. Persist permitted result metadata and an outbox event atomically. Return the result. Raw prompt/answer/evidence/judge text is not stored in metadata-only mode, including error logs and queue payloads after processing.

## Deterministic verdict and score table

| Findings (applicable claims) | `status/verdict` | Groundedness | Hallucination risk | Contradiction risk |
|---|---|---:|---:|---:|
| supported, supported | `COMPLETED/PASS` | 1 | 0 | 0 |
| supported, unsupported | `COMPLETED/PARTIAL` | 0.5 | 0.5 | 0 |
| supported, insufficient evidence | `COMPLETED/PARTIAL` | 0.5 | 0 | 0 |
| contradicted, supported | `COMPLETED/FAIL` | 0.5 | 0.5 | 0.5 |
| insufficient evidence only | `ABSTAINED/ABSTAIN` | 0 | 0 | 0 |
| no applicable claims | `ABSTAINED/ABSTAIN` | N/A | N/A | N/A |
| provider timeout on any claim | `ABSTAINED/ABSTAIN` | unavailable | unavailable | unavailable |

N/A is score `status=NOT_APPLICABLE` with no value; unavailable is `status=UNAVAILABLE` with no value. Scores are descriptive ratios, not calibrated probabilities. For all-insufficient evidence, the zero ratios describe checked claims but no factual answer is asserted. Each measured score carries method version `*-0.1`. `NOT_APPLICABLE` claims are excluded from denominators. The [scoring spec](scoring.md) owns formula details.

## Persistence and async mechanics

MVP PostgreSQL tables (names may vary, constraints may not): `evaluations(tenant_id,evaluation_id,request_id,status,verdict,result_json,created_at)`, `jobs(tenant_id,job_id,evaluation_id,state,attempt,lease_until,next_run_at,error_code,created_at,updated_at)`, `idempotency(tenant_id,route,project_id,key,body_hash,job_id,expires_at)`, `outbox(tenant_id,event_id,evaluation_id,payload_json,delivery_state,attempt,next_run_at)`. Primary/unique keys start with tenant ID. Index jobs by due state/time and evaluations by tenant/time. Use migrations, not startup `CREATE TABLE` side effects. Content governance may require separating result metadata from raw claim text; metadata-only storage must omit text fields and expose references/hashes where permitted. API result can still contain transient text immediately after evaluation.

Async transaction: insert idempotency mapping + job, or return existing job when canonical body hash matches; 409 on key/body conflict. Worker claims a due job with a lease and compare-and-set state transition. Lease expiry makes it claimable again; result insertion is unique on tenant/evaluation ID. Retry only typed transient failures under configured attempt and total-deadline limits. Terminal operational failure is `job=FAILED` with sanitized error; a completed evaluation that abstained is `job=SUCCEEDED` with `result.status=ABSTAINED`. Cancellation is best effort and terminal state cannot be overwritten by a late worker. Outbox sends at least once; stable `event_id` lets Observability deduplicate. No direct network call occurs inside the DB transaction.

## HTTP surface and examples

Implement exactly the target routes in [API.md](API.md). `POST /v0.1/verifications` returns a result on 200; `POST /v0.1/verification-jobs` returns 202 `{schema_version,job_id,evaluation_id,state,status_url}`. Status GET returns state, timestamps, attempt count, sanitized error and result URL when ready. A client cannot fetch another tenant's job/result even with its ID; respond 404 without revealing existence. Expose `/health/live`, `/health/ready` and build version. Publish OpenAPI and a runnable example using the [worked JSON](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md#worked-example); the sample requires only a nonproduction/local judge fixture or configured approved provider.

## Reviewer acceptance matrix

| Inspect/run | Expected observation |
|---|---|
| Submit worked request | Two claims, one supported and one unsupported, `PARTIAL`, groundedness 0.5, cited evidence ID and provenance |
| Submit contradiction fixture | Claim `CONTRADICTED`, overall `FAIL`, contradiction risk measured |
| Submit empty/ambiguous corpus | `INSUFFICIENT_EVIDENCE` and `ABSTAIN`, not PASS/FAIL |
| Force provider timeout/malformed JSON | `ABSTAINED/ABSTAIN` or typed operational error; no `UNSUPPORTED` fabricated |
| Repeat async key + same body, then changed body | Same job/evaluation, then 409 conflict |
| Deliver one job/event twice | One result and one counted reliability event |
| Query another tenant's job/result | 404 and no leaked content/existence |
| Run metadata-only mode and inspect DB/logs/queue | No raw prompt, answer, evidence or judge output persisted |
| Change provider adapter | Core/scoring/policy code unchanged; contract fixtures still pass |

The implementation is not complete because these words exist. It is complete only when the tests above run from a clean checkout and the status ledger is updated with evidence.
