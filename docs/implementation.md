# Reliability Implementation Plan

Status values: `NOT_STARTED`, `IN_PROGRESS`, `BLOCKED`, `COMPLETE`.

A task is COMPLETE only when code, tests, failure handling, tenant isolation, observability, documentation, and acceptance criteria are satisfied.

## REL-01 Foundation — STATUS: NOT_STARTED
- [ ] Create source module boundaries matching architecture.
- [ ] Add configuration model, logging, health, lint/type/test foundations.
- [ ] Add schema/API version conventions.
- [ ] Add dependency rules preventing core -> provider SDK coupling.

**Acceptance:** clean checkout builds/tests; module dependency rules are documented and enforceable.

## REL-02 Tenant and security context — STATUS: NOT_STARTED
- [ ] Tenant/project/application context model.
- [ ] Propagate context through API/services/async/persistence.
- [ ] Tenant-aware repository interfaces.
- [ ] Negative cross-tenant tests.
- [ ] Secret/provider credential isolation.

**Acceptance:** no persistence/query path is tenant-implicit; cross-tenant access fails safely.

## REL-03 Public contracts and provenance — STATUS: NOT_STARTED
- [ ] VerificationRequest and VerificationResult.
- [ ] Claim, Evidence, Citation, ToolExecution, Violation, Score models.
- [ ] Verdict/action enums.
- [ ] Evaluator/model/configuration/policy provenance.
- [ ] Serialization and compatibility tests.

## REL-04 Claim engine — STATUS: NOT_STARTED
- [ ] Accept explicit claims or extract claims from answer.
- [ ] Stable claim IDs.
- [ ] Preserve source offsets/references where practical.
- [ ] Unit and golden tests.

## REL-05 Evidence engine — STATUS: NOT_STARTED
- [ ] Direct supplied evidence support.
- [ ] EvidenceRetriever interface.
- [ ] Evidence normalization and mapping.
- [ ] Source metadata/provenance.
- [ ] Insufficient-evidence behavior.

## REL-06 Judge provider — STATUS: NOT_STARTED
- [ ] JudgeProvider interface.
- [ ] First working provider adapter.
- [ ] Bounded timeouts/cancellation.
- [ ] Normalize provider errors/timeouts.
- [ ] Provider version/config provenance.

## REL-07 Verification pipeline — STATUS: NOT_STARTED
- [ ] Claim/evidence evaluation orchestration.
- [ ] SUPPORTED/CONTRADICTED/UNSUPPORTED/INSUFFICIENT_EVIDENCE outcomes.
- [ ] Aggregate result.
- [ ] Synchronous API path.
- [ ] Deterministic golden tests.

## REL-08 Scoring — STATUS: NOT_STARTED
- [ ] Transparent score calculation.
- [ ] Groundedness, hallucination risk, contradiction risk.
- [ ] Explicit confidence/provenance.
- [ ] Calibration hook for future benchmark-driven scoring.

## REL-09 Policy engine — STATUS: NOT_STARTED
- [ ] Policy inputs and actions.
- [ ] Deterministic rule evaluation.
- [ ] PASS/FAIL/REGENERATE/RETRIEVE_AGAIN/SWITCH_MODEL/ASK_USER/BLOCK/HUMAN_REVIEW.
- [ ] Decision provenance.

## REL-10 Async evaluation — STATUS: NOT_STARTED
- [ ] Job lifecycle.
- [ ] Idempotency keys.
- [ ] Retry/backoff and terminal failure state.
- [ ] Duplicate-delivery tests.
- [ ] Worker telemetry.

## REL-11 Content governance — STATUS: NOT_STARTED
- [ ] Configurable capture modes.
- [ ] Redaction hook before persistence.
- [ ] Retention interfaces.
- [ ] Metadata-only operation tests.

## REL-12 API and examples — STATUS: NOT_STARTED
- [ ] Versioned sync API.
- [ ] Async submission/status API.
- [ ] Error contract.
- [ ] Example groundedness request.
- [ ] Example tenant-isolated usage.

## MVP end-product expectation

From a clean checkout, a developer can start the service, submit answer + supplied evidence, receive claim-level findings, scores, verdict, and provenance, repeat the same logical async request safely, and observe evaluation telemetry. The implementation must work without storing raw prompts/responses.

## Required test scenarios

1. Fully supported answer.
2. Partially supported answer.
3. Contradicted claim.
4. Unsupported claim.
5. Insufficient evidence.
6. Invalid request/schema version.
7. Provider timeout/failure.
8. Duplicate async delivery.
9. Cross-tenant read/write attempt.
10. Content capture disabled.
11. Stable serialization across supported schema versions.
12. Provenance present for every externally evaluated result.
