# Verification pipeline

The synchronous path is: authenticate and establish tenant scope; validate schema, authorization and limits; assign/reuse logical evaluation identity; accept/extract claims; normalize and map supplied evidence; select provider/config; evaluate each claim with a bounded deadline; normalize findings; score; aggregate factual verdict; apply deterministic policy; persist allowed result metadata; emit `ReliabilityEvent`; return result. The async path runs the same pure pipeline after durable enqueue and returns a job handle immediately.

Each stage records its own version and timing. A stage cannot change tenant scope or correlation IDs. An individual claim may be `INSUFFICIENT_EVIDENCE` while others are judged. If the provider fails for any claim, the default v0.1 aggregate is `ABSTAINED/ABSTAIN` with unavailable score dimensions; partial technical failures must not be disguised as completed factual results. Successful claim findings may be retained for diagnosis but cannot drive a blocking factual decision unless a future version explicitly defines partial completion.

Cancellation stops external calls and marks a job `CANCELLED`; it does not create a verdict. Persistence and event emission use an outbox or equivalent atomic/recoverable handoff so completed results are eventually visible in Observability. Event delivery is at least once; event ID is stable for deduplication. The customer generation path never waits on optional telemetry export.

The [worked request/result](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md#worked-example) is a cross-repository contract fixture. Integration tests must verify that its evaluation ID appears in a trace projection, including delayed event delivery and duplicate events.
