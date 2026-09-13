# Judge provider contract

The normalized `JudgeRequest`/`JudgeResult` port is defined in the [shared contract](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md). A provider adapter receives one claim, candidate evidence, deadline, cancellation and configuration. It returns one verdict with cited evidence IDs and provenance, or a typed failure. Provider-specific labels, confidence scales, JSON shapes and error codes are translated at the adapter boundary. Validate that returned evidence IDs were supplied; reject malformed or contradictory response shapes as `INVALID_RESPONSE`.

The registry selects an allowlisted tenant-approved provider and immutable configuration version. Credentials are resolved at call time from a secret reference; neither credentials nor raw provider payloads enter result/telemetry persistence. Apply a request deadline, bounded concurrency, rate limits and cancellation. Retry only transient errors within the same logical evaluation and deadline; never retry an invalid response indefinitely. A fallback provider is a policy decision recorded in provenance, not a hidden adapter behavior.

Contract tests should run the same supported, contradicted, unsupported, ambiguous, timeout, rate-limit and malformed-response fixtures against each adapter. A provider's confidence is not a calibrated FactLama score until benchmarked.
