"""G6 integration: a real `DeliverFn` (`worker.outbox_delivery.DeliverFn`)
that POSTs each `OutboxEvent`, serialized as `ReliabilityEvent` wire JSON
(`contracts/v0.1/schemas/reliability_event.schema.json`), to Observability's
authenticated internal ingress (`POST /internal/v0.1/reliability-events`).

This module only supplies the callable -- `run_outbox_delivery_loop`'s
claim/ack mechanics are unchanged and already proven at-least-once against
concurrent claimers (`tests/conformance/test_delivery_semantics.py`). Any
non-2xx response or transport failure raises `HttpOutboxDeliveryError`,
which the loop's existing `except Exception` handling leaves unacked for
redelivery -- the same policy `async-evaluation.md` already states:
"Outbox delivery of ReliabilityEvent is at least once with stable event
ID." There is deliberately no separate dead-letter path here: a response
the ingress can never accept (e.g. a genuine schema mismatch) will keep
retrying and keep logging, which is the right failure mode for a
producer/consumer contract bug -- it needs a human, not a silent drop.
"""

from typing import Any

import httpx

from storage.models import OutboxEvent
from worker.outbox_delivery import DeliverFn
from worker.settings import WorkerSettings

RELIABILITY_EVENT_SCHEMA_VERSION = "0.1"
INGRESS_PATH = "/internal/v0.1/reliability-events"


class HttpOutboxDeliveryError(RuntimeError):
    """Raised for anything `run_outbox_delivery_loop` must treat as an
    undelivered event: a non-2xx ingress response, or a transport failure
    reaching it at all."""


def event_to_reliability_event_json(event: OutboxEvent) -> dict[str, Any]:
    """`OutboxEvent` -> the wire JSON body `ReliabilityEvent.model_validate_json`
    on the Observability side expects. Mirrors
    `storage.postgres.backend._outbox_event_to_dict`'s field-for-field shape
    (kept independent rather than imported: `worker/` may depend on
    `storage.backend`'s port, not the concrete `storage.postgres` adapter --
    `pyproject.toml`'s import-linter contract)."""
    payload: dict[str, Any] = {
        "schema_version": RELIABILITY_EVENT_SCHEMA_VERSION,
        "event_id": event.event_id,
        "tenant_id": event.tenant_id,
        "evaluation_id": event.evaluation_id,
        "occurred_at": event.occurred_at.isoformat(),
        "status": event.status,
        "verdict": event.verdict,
        "scores": {name: score.model_dump(mode="json") for name, score in event.scores.items()},
        "evaluator_version": event.evaluator_version,
        "calibration_class": event.calibration_class,
        "qualification_status": event.qualification_status,
        "usage_summary": event.usage_summary.model_dump(mode="json"),
        "project_id": event.project_id,
        "application_id": event.application_id,
        "interaction_id": event.interaction_id,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "calibration_classes": list(event.calibration_classes),
    }
    return {k: v for k, v in payload.items() if v is not None}


def build_http_deliver(client: httpx.AsyncClient, settings: WorkerSettings) -> DeliverFn:
    """`client` is owned and closed by the caller (`worker/bootstrap.py`),
    not by this module -- so a test can pass one backed by
    `httpx.MockTransport`, and production code can share one client's
    connection pool across every delivery in the loop's lifetime."""

    async def deliver(event: OutboxEvent) -> bool:
        body = event_to_reliability_event_json(event)
        try:
            response = await client.post(
                INGRESS_PATH,
                json=body,
                headers={"Authorization": f"Bearer {settings.observability_workload_token}"},
            )
        except httpx.HTTPError as exc:
            raise HttpOutboxDeliveryError(
                f"event {event.event_id}: failed to reach the Observability ingress: {exc}"
            ) from exc
        if response.status_code != 200:
            raise HttpOutboxDeliveryError(
                f"event {event.event_id}: ingress rejected delivery with "
                f"HTTP {response.status_code}: {response.text}"
            )
        # A 200 alone does not prove the ingress accepted *this* event -- a
        # misconfigured base_url or an intermediary proxy can return 200
        # with unrelated content. Require the receipt to name this event's
        # own id and an accepted/duplicate outcome before acking (P2,
        # review-evidence/g6-2026-10-22/REVIEW.md).
        try:
            receipt = response.json()
        except ValueError as exc:
            raise HttpOutboxDeliveryError(
                f"event {event.event_id}: ingress returned 200 with a non-JSON body: "
                f"{response.text!r}"
            ) from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("event_id") != event.event_id
            or receipt.get("status") not in ("accepted", "duplicate")
        ):
            raise HttpOutboxDeliveryError(
                f"event {event.event_id}: ingress returned 200 with an unrecognized "
                f"receipt {receipt!r}"
            )
        return True

    return deliver
