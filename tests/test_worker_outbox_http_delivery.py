"""Fast, mocked-transport tests for `worker.outbox_http_delivery`'s
`DeliverFn`: request shape, success/duplicate acknowledgement, and the
retry-by-raising behavior `worker.outbox_delivery.run_outbox_delivery_loop`
depends on for every non-2xx/transport-failure case. The real cross-repo
behavior (an actual Observability ingress, real PostgreSQL persistence,
genuine duplicate delivery) is proven separately in
`tests/conformance/test_outbox_cross_repo_delivery.py`, not here.
"""

from datetime import datetime, timezone

import httpx
import pytest

from schemas.verification import OverallVerdict, ResultStatus, Usage
from storage.models import OutboxEvent
from worker.outbox_http_delivery import (
    INGRESS_PATH,
    HttpOutboxDeliveryError,
    build_http_deliver,
    event_to_reliability_event_json,
)
from worker.settings import WorkerSettings


def _settings(token: str = "shared-secret") -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.observability_workload_token = token
    return settings


def _event(event_id: str = "event-1") -> OutboxEvent:
    return OutboxEvent(
        event_id=event_id,
        tenant_id="tenant-a",
        evaluation_id="eval-1",
        occurred_at=datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc),
        status=ResultStatus.COMPLETED.value,
        verdict=OverallVerdict.PASS.value,
        scores={},
        evaluator_version="0.0.0",
        calibration_class="class-1",
        qualification_status="QUALIFIED",
        usage_summary=Usage(),
    )


def test_wire_json_matches_reliability_event_shape() -> None:
    body = event_to_reliability_event_json(_event())
    assert body["schema_version"] == "0.1"
    assert body["occurred_at"] == "2026-10-16T12:00:00+00:00"
    assert body["status"] == "COMPLETED"
    assert body["verdict"] == "PASS"
    # Unset optional fields (project_id, trace_id, ...) are omitted, not null.
    assert "project_id" not in body
    assert "trace_id" not in body


async def test_delivered_on_200_accepted() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"event_id": "event-1", "status": "accepted"})

    client = httpx.AsyncClient(
        base_url="http://observability.test", transport=httpx.MockTransport(handler)
    )
    deliver = build_http_deliver(client, _settings(token="shared-secret"))
    assert await deliver(_event()) is True
    await client.aclose()

    request = captured["request"]
    assert request.url.path == INGRESS_PATH
    assert request.headers["authorization"] == "Bearer shared-secret"


async def test_delivered_on_200_duplicate() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"event_id": "event-1", "status": "duplicate"})

    client = httpx.AsyncClient(
        base_url="http://observability.test", transport=httpx.MockTransport(handler)
    )
    deliver = build_http_deliver(client, _settings())
    assert await deliver(_event()) is True
    await client.aclose()


@pytest.mark.parametrize("status_code", [400, 401, 429, 503])
async def test_non_200_response_raises_for_retry(status_code: int) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": {"code": "X", "message": "nope"}})

    client = httpx.AsyncClient(
        base_url="http://observability.test", transport=httpx.MockTransport(handler)
    )
    deliver = build_http_deliver(client, _settings())
    with pytest.raises(HttpOutboxDeliveryError):
        await deliver(_event())
    await client.aclose()


async def test_transport_failure_raises_for_retry() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(
        base_url="http://observability.test", transport=httpx.MockTransport(handler)
    )
    deliver = build_http_deliver(client, _settings())
    with pytest.raises(HttpOutboxDeliveryError):
        await deliver(_event())
    await client.aclose()
