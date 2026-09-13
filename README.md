# FactLama Reliability

Start with [the low-level implementation checklist](docs/LOW_LEVEL_IMPLEMENTATION.md) to see the code units, algorithm and reviewer tests. Supporting specifications: [domain model](docs/domain-model.md), [claim engine](docs/claim-engine.md), [evidence engine](docs/evidence-engine.md), [judge provider](docs/judge-provider.md), [verification pipeline](docs/verification-pipeline.md), [scoring](docs/scoring.md), [policy](docs/policy-engine.md), [async execution](docs/async-evaluation.md), and [HTTP API](docs/API.md). The architecture repository's `CONTRACTS.md` owns shared wire semantics.

The FactLama Reliability repository contains the semantic evaluation and decision engine for LLM, RAG and agentic applications.

The system-level source of truth lives in `Factlama/factlama-architecture`. This repository implements those contracts and must not silently redefine them.

## Product responsibility

Reliability answers questions such as:

- Is an answer grounded in the supplied evidence?
- Which claims are supported, contradicted, unsupported or uncertain?
- Do citations support the claims they are attached to?
- Did the model follow instructions?
- Did an agent select and use tools correctly?
- What evaluator, model and configuration produced the result?
- What policy action should follow?

## Architectural rule

The Reliability Engine is the platform brain, not merely a wrapper around the FactLama SLM.

Core logic must remain evaluator-independent.

## Target module layout

```text
factlama-reliability/
├── schemas/
├── core/
│   ├── verification/
│   ├── claims/
│   ├── scoring/
│   ├── policies/
│   └── decisions/
├── evidence/
│   ├── router/
│   ├── retrievers/
│   ├── fusion/
│   └── sources/
├── judges/
│   ├── factlama/
│   ├── openai/
│   ├── anthropic/
│   ├── azure/
│   └── custom/
├── agents/
│   ├── tool_verification/
│   ├── trajectory/
│   └── agent_evaluation/
├── models/
│   └── interfaces/
├── api/
├── tests/
└── docs/
```

This is a target module layout, not permission to create every directory immediately. Create modules as the implementation reaches them.

## Core abstractions

### JudgeProvider

Conceptual interface:

```python
class JudgeProvider(Protocol):
    def evaluate(self, request: JudgeRequest, deadline: float, cancellation: CancellationToken) -> JudgeResult:
        ...
```

Providers may include enterprise/internal models, OpenAI, Anthropic, Azure, local open models and eventually the FactLama SLM. Reliability core must not branch on provider vendor.

### EvidenceRetriever

Evidence retrieval is optional and pluggable. Direct evidence supplied in the request is the MVP path. Vector retrieval is not an MVP requirement.

### Verification pipeline

Initial logical stages:

1. authenticate and establish trusted tenant/context IDs;
2. validate and normalize request within authorized scope;
3. segment output into verifiable claims;
4. map supplied evidence to claims;
5. execute configured evaluator/judge;
6. normalize provider-specific output;
7. create claim-level findings;
8. compute reliability dimensions;
9. derive aggregate verdict;
10. apply policy if configured;
11. record provenance;
12. emit telemetry and return/persist result.

## Public contract

### Verification request

Should include, as applicable:

- schema version;
- request/evaluation ID;
- tenant/project/application context;
- original question/input;
- answer/output;
- evidence/context;
- instructions;
- citations;
- tool executions;
- model/prompt metadata;
- evaluation mode/configuration.

### Evidence

Evidence must carry a stable identifier, content or reference, source metadata where available, and optional retrieval metadata. Evidence should not be treated as true merely because it is semantically similar.

### Claim verdicts

- `SUPPORTED`
- `CONTRADICTED`
- `UNSUPPORTED`
- `INSUFFICIENT_EVIDENCE`
- `NOT_APPLICABLE`

### Overall verdicts

- `PASS`
- `PARTIAL`
- `FAIL`
- `ABSTAIN`

### Score dimensions

- groundedness
- hallucination risk
- contradiction risk
- scope breach
- citation support
- instruction adherence
- tool correctness
- confidence alignment

Not every evaluator must populate every score. Missing/not-applicable dimensions must be represented explicitly rather than guessed.

### Violation codes

- `UNSUPPORTED_CLAIM`
- `CONTRADICTED_CLAIM`
- `HALLUCINATION`
- `SCOPE_BREACH`
- `INSTRUCTION_VIOLATION`
- `CITATION_MISMATCH`
- `CITATION_MISSING`
- `TOOL_SELECTION_ERROR`
- `TOOL_ARGUMENT_ERROR`
- `TOOL_RESULT_CONTRADICTION`
- `TOOL_RESULT_FABRICATION`
- `CONFIDENCE_MISMATCH`
- `INSUFFICIENT_EVIDENCE`

### Policy actions

- `PASS`
- `FAIL`
- `REGENERATE`
- `RETRIEVE_AGAIN`
- `SWITCH_MODEL`
- `ASK_USER`
- `BLOCK`
- `HUMAN_REVIEW`

## Provenance requirements

Every evaluation result should record enough information to explain how it was produced without requiring raw interaction content.

At minimum where available:

- evaluator/provider ID;
- evaluator version;
- model name/version;
- configuration/prompt version;
- policy version;
- evaluation mode;
- timestamps and latency;
- evidence references/IDs;
- request/evaluation correlation IDs.

## Scoring

The first scoring implementation must be transparent, deterministic and easy to replace. Do not hide logic in a second LLM prompt.

The v0.1 formulas and verdict precedence are specified in [scoring.md](docs/scoring.md). Future weighted composites require a separately named, versioned method, benchmark evidence and tests; missing dimensions are never assigned invented values.

## Evaluation modes

Initial API may expose:

- `FAST`: lowest cost/latency, limited checks;
- `STANDARD`: default balanced evaluation;
- `DEEP`: richer/multiple checks for high-risk or debugging scenarios.

The names describe intent, not hard-coded model vendors.

## Multi-tenancy and security

P0 requirements:

- tenant context required before persistence/query;
- no cross-tenant access;
- provider secrets never appear in public results;
- content capture can be disabled;
- metadata-only evaluation history remains useful;
- redaction hooks occur before optional content persistence;
- API/schema versions are explicit;
- async jobs use idempotency keys;
- failure messages avoid leaking credentials/internal secrets.

## MVP end product

Reliability MVP is complete when a caller can send a versioned request containing a question/output and supplied evidence, and receive a normalized result containing:

- claim-level findings;
- supported/contradicted/unsupported/insufficient-evidence outcomes;
- groundedness and hallucination-related signals;
- aggregate verdict;
- transparent score data;
- violations;
- evaluator provenance;
- correlation identifiers;
- stable serialized API contract.

The same logical evaluation must work synchronously and asynchronously, respect tenant boundaries, and emit telemetry consumable by FactLama Observability.

## Implementation plan and status

[`docs/implementation.md`](docs/implementation.md) is the sole status ledger for REL tasks. Its task IDs, acceptance criteria and `NOT_STARTED` state are authoritative. This README describes architecture and must not duplicate task numbering.

## Claude implementation instructions

For each task:

1. read this README and the relevant `factlama-architecture` sections;
2. inspect existing code before proposing new abstractions;
3. set the task status to `IN_PROGRESS`;
4. implement the smallest complete vertical slice;
5. add tests before marking complete;
6. run type/lint/test commands;
7. review dependency direction and tenant/security impact;
8. update docs/status;
9. set `COMPLETE` only when acceptance criteria pass.

Claude must not invent a new evaluator-specific public schema, put vendor SDK calls in core domain code, add a vector DB merely for similarity search, add distributed infrastructure without measured need, persist sensitive content by default, bypass tenant context, or call a partial happy-path implementation complete.

## Test scenarios

Minimum golden scenarios:

1. A factual claim directly supported by supplied evidence -> `SUPPORTED`.
2. A factual claim conflicts with supplied evidence -> `CONTRADICTED`.
3. A claim has no supporting evidence -> `UNSUPPORTED` or `INSUFFICIENT_EVIDENCE` according to contract rules.
4. Evidence is ambiguous -> evaluator abstains/uses insufficient evidence instead of fabricating certainty.
5. Multiple claims contain mixed support -> overall verdict is not incorrectly reduced to a binary pass.
6. Citation references wrong evidence -> citation mismatch signal.
7. Evaluator timeout/failure -> normalized operational failure with telemetry.
8. Same async idempotency key delivered repeatedly -> one logical evaluation.
9. Tenant A attempts to query tenant B evaluation -> rejected/no data.
10. Provider A and Provider B evaluate the same request -> public result remains structurally compatible.
11. Content capture disabled -> evaluation result/provenance still persists and remains queryable.
12. Malformed/oversized inputs -> deterministic validation behavior.

## Definition of done

A Reliability task is complete only when implementation, unit tests, integration/contract tests where applicable, failure-path tests, tenant/security verification, observability, lint/type checks, documentation and status updates are all complete.

## License

MIT. Third-party model weights, datasets and SDKs retain their own licenses and must be reviewed independently.
