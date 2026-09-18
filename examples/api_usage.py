"""Runnable example of the REL-12 synchronous API (`POST /v0.1/verifications`,
docs/API.md), demonstrating a SUPPORTED, a CONTRADICTED and an ABSTAINED
response through the real HTTP layer -- authentication, request validation
and the JSON error/response envelope included, not just `core.verify()`
called directly (see `examples/basic_usage.py` for that library-level path).

Run with:
    pip install -e ".[dev]"
    python examples/api_usage.py

This drives the FastAPI app in-process via `TestClient` (the same mechanism
`tests/test_api.py` uses), so no separate server process is required; the
same requests work unchanged against a real `uvicorn` process on
`http://localhost:8000`.
"""

import json

from fastapi.testclient import TestClient

from api.app import app, get_verifier
from api.auth import TenantCredentialStore
from api.settings import Settings, get_settings
from core.verifier import Verifier
from judges.providers import RuleBasedProvider
from schemas.tenancy import TenantContext

API_KEY = "example-api-key"
TENANT_CONTEXT = TenantContext(tenant_id="example-tenant")


def _build_settings() -> Settings:
    settings = Settings.__new__(Settings)
    settings.tenant_store = TenantCredentialStore({API_KEY: TENANT_CONTEXT})
    settings.max_body_bytes = 1_000_000
    return settings


def _print_result(title: str, response) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"HTTP {response.status_code}")
    body = response.json()
    print(f"Verdict: {body.get('verdict')}")
    print(f"Status: {body.get('status')} (abstention_reason: {body.get('abstention_reason')})")
    for claim in body.get("claims", []):
        print(f"  - {claim['claim_id']}: {claim['verdict']} ({claim['rationale_code']})")
    print()


def main() -> None:
    # A deterministic, in-process RuleBasedProvider (ADR-011's T0 tier), not
    # a real vendor call -- this example is about the HTTP contract, not
    # judge accuracy.
    app.dependency_overrides[get_settings] = _build_settings
    app.dependency_overrides[get_verifier] = lambda: Verifier(model_provider=RuleBasedProvider())
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {API_KEY}"}

    supported_request = {
        "schema_version": "0.1",
        "request_id": "req_example_supported",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Paris is the capital of France.",
        "evidence": [
            {"evidence_id": "e1", "content": "Paris is the capital of France."},
        ],
    }
    _print_result(
        "Example 1: Supported",
        client.post("/v0.1/verifications", headers=headers, json=supported_request),
    )

    contradicted_request = {
        "schema_version": "0.1",
        "request_id": "req_example_contradicted",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Product X weighs 3.4 kg.",
        "evidence": [
            {"evidence_id": "e1", "content": "Product X weighs 2.4 kg."},
        ],
    }
    _print_result(
        "Example 2: Contradicted",
        client.post("/v0.1/verifications", headers=headers, json=contradicted_request),
    )

    abstained_request = {
        "schema_version": "0.1",
        "request_id": "req_example_abstained",
        "project_id": "proj-a",
        "application_id": "app-a",
        "answer": "Paris is the capital of France.",
        # No evidence supplied: the judge cannot confirm anything, so the
        # request abstains instead of guessing -- no optimistic default.
    }
    _print_result(
        "Example 3: Abstained (no evidence supplied)",
        client.post("/v0.1/verifications", headers=headers, json=abstained_request),
    )

    print("=" * 60)
    print("Example 4: Unauthenticated request (401, contract error envelope)")
    print("=" * 60)
    response = client.post("/v0.1/verifications", json=supported_request)
    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
