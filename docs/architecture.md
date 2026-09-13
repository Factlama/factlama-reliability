# Reliability Architecture

FactLama Reliability is the semantic evaluation and decision engine for LLM, RAG, and agentic applications.

## Responsibilities
- claim extraction and normalization
- evidence mapping
- judge/evaluator abstraction
- groundedness, contradiction, citation, instruction, RAG, and agent/tool evaluation
- normalized scoring and verdicts
- policy decisions
- evaluator/model/configuration provenance

## Architectural boundaries
- Core logic must not depend directly on provider SDKs.
- Evidence access remains behind an abstraction.
- Public request/result contracts are versioned.
- Tenant identity is mandatory at API, async, storage, and query boundaries.
- Interaction-content persistence is optional.
- Reliability emits telemetry but does not own the observability storage/query/dashboard stack.

## Runtime flow
VerificationRequest -> validation -> claims -> evidence mapping -> JudgeProvider -> normalized findings -> scoring -> policy -> provenance -> persistence/telemetry -> VerificationResult.

## Enterprise characteristics
The engine must support bounded timeouts, cancellation, retry-safe async execution, idempotency, provider failure isolation, auditable provenance, schema evolution, and horizontal worker scaling.
