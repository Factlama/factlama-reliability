"""F1 of the 2026-09-21 G0-G4 validation report: test_contract_fixtures.py
only proves this repo's types can *parse* the canonical fixtures -- it never
proves this repo's own serialization of a `VerificationResult` still
validates against contracts/v0.1's schema. A normal supported result
serialized through `model_dump(mode="json")` produced 34 schema errors,
chiefly null optional fields the schema requires to be absent
(CONTRACTS.md: "null means known absent; omitted means not supplied").

This validates the actual producer output -- both a round-tripped canonical
fixture and a live API response -- against contracts/v0.1's own JSON Schema,
the same way contracts/validate.py does for the canonical examples
themselves.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from api.app import app, get_verifier
from api.auth import TenantCredentialStore
from api.settings import Settings, get_settings
from core.verifier import Verifier
from judges.providers import MockModelProvider
from schemas.tenancy import TenantContext
from schemas.verification import VerificationResult
from worker.outbox import build_outbox_event
from worker.outbox_http_delivery import event_to_reliability_event_json


def _contracts_dir() -> Path | None:
    env = os.environ.get("FACTLAMA_CONTRACTS_DIR")
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    local = Path(__file__).resolve().parents[2] / "factlama-architecture" / "contracts" / "v0.1"
    return local if local.is_dir() else None


def _running_in_ci() -> bool:
    """See tests/test_contract_fixtures.py's matching helper (F10 of the
    2026-09-21 G0-G4 validation report) -- the same "must fail loudly in CI,
    not silently skip" rule applies here."""
    return os.environ.get("CI", "").lower() in ("1", "true") or bool(
        os.environ.get("GITHUB_ACTIONS")
    )


CONTRACTS_DIR = _contracts_dir()

if CONTRACTS_DIR is None and _running_in_ci():
    raise RuntimeError(
        "FACTLAMA_CONTRACTS_DIR is unset or invalid in a CI run. This wire-output "
        "schema suite is a required G2/G3 acceptance gate (F1 of the 2026-09-21 "
        "G0-G4 validation report), not an optional local-dev check -- refusing to "
        "silently skip it. This usually means the 'Checkout factlama-architecture' "
        "step in ci.yml failed or was removed."
    )

pytestmark = pytest.mark.skipif(
    CONTRACTS_DIR is None,
    reason=(
        "contracts/v0.1 not found -- set FACTLAMA_CONTRACTS_DIR, or run inside the "
        "three-repo local workspace next to ../factlama-architecture"
    ),
)


def _validator_for(schema_filename: str) -> Draft202012Validator:
    schema_dir = CONTRACTS_DIR / "schemas"
    resources = []
    for schema_path in schema_dir.glob("*.schema.json"):
        contents = json.loads(schema_path.read_text())
        resource = Resource.from_contents(contents, default_specification=DRAFT202012)
        resources.append((resource.id(), resource))
    registry = Registry().with_resources(resources)
    schema = json.loads((schema_dir / schema_filename).read_text())
    return Draft202012Validator(schema, registry=registry)


def _verification_result_validator() -> Draft202012Validator:
    return _validator_for("verification_result.schema.json")


def _reliability_event_validator() -> Draft202012Validator:
    return _validator_for("reliability_event.schema.json")


def _assert_valid(validator: Draft202012Validator, instance: dict[str, Any]) -> None:
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        joined = "\n".join(
            f"  {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        pytest.fail(f"{len(errors)} schema error(s):\n{joined}")


def _result_fixtures() -> list[Path]:
    if CONTRACTS_DIR is None:
        return []
    directory = CONTRACTS_DIR / "examples" / "valid" / "verification_result"
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


class TestRoundTrippedFixturesValidate:
    """Parses each canonical PASS/PARTIAL/FAIL/ABSTAIN(x2)/FAILED/DISPUTED
    fixture into `VerificationResult`, re-serializes it exactly as the API
    does, and validates the output -- proving this repo's serializer, not
    just its parser, is contract-conformant for every status/verdict this
    pipeline's own producer output claims to cover."""

    @pytest.mark.parametrize("path", _result_fixtures(), ids=lambda p: p.stem)
    def test_round_tripped_output_validates(self, path: Path) -> None:
        validator = _verification_result_validator()
        result = VerificationResult(**json.loads(path.read_text()))
        serialized = result.model_dump(mode="json", exclude_none=True)
        _assert_valid(validator, serialized)


def test_at_least_one_fixture_was_actually_found() -> None:
    if CONTRACTS_DIR is None:
        pytest.skip("contracts/v0.1 not found")
    assert _result_fixtures()


class TestReliabilityEventOutputValidates:
    """ADR-022: the same producer-output-not-just-parser rule F1 established
    for `VerificationResult`, now applied to the `ReliabilityEvent` this
    repo's outbox actually emits -- including the new `claim_summary`
    projection, whose `ClaimFinding` is the one object in contracts/v0.1
    with `additionalProperties: false`. A producer that ever started
    attaching claim text or free-form rationale would fail here rather
    than silently publishing content to Observability.
    """

    @pytest.mark.parametrize("path", _result_fixtures(), ids=lambda p: p.stem)
    def test_outbox_event_json_validates_for_every_canonical_result(self, path: Path) -> None:
        validator = _reliability_event_validator()
        result = VerificationResult(**json.loads(path.read_text()))
        event = build_outbox_event(result)
        _assert_valid(validator, event_to_reliability_event_json(event))

    def test_emitted_claim_summary_carries_findings_and_no_content(self) -> None:
        """The canonical PARTIAL fixture has a supported and an unsupported
        claim, both with claim text -- so this proves the real projection is
        populated (not merely schema-valid by being empty) while the text
        itself never crosses the boundary."""
        validator = _reliability_event_validator()
        partial = next((p for p in _result_fixtures() if p.stem == "partial"), None)
        assert partial is not None, "expected a canonical 'partial' verification_result fixture"
        source = json.loads(partial.read_text())
        result = VerificationResult(**source)
        body = event_to_reliability_event_json(build_outbox_event(result))
        _assert_valid(validator, body)

        summary = body["claim_summary"]
        assert summary["total_claim_count"] == len(result.claims)
        assert summary["findings"], "expected at least one projected finding"
        for finding in summary["findings"].values():
            assert set(finding) <= {"status", "rationale_code"}

        # No claim's own text from the fixture appears anywhere in the body.
        serialized = json.dumps(body)
        for claim in result.claims:
            if claim.text:
                assert claim.text not in serialized


class TestLiveApiResponseValidates:
    """The actual HTTP response body, not a model constructed in-process --
    F1's reproduction was specifically that a normal supported result
    produced 34 errors through this exact path (`api/app.py`'s
    `model_dump`)."""

    @pytest.fixture
    def client(self) -> TestClient:
        settings = Settings.__new__(Settings)
        settings.tenant_store = TenantCredentialStore(
            {"key-1": TenantContext(tenant_id="tenant-one")}
        )
        settings.max_body_bytes = 1_000_000
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_verifier] = lambda: Verifier(
            model_provider=MockModelProvider()
        )
        try:
            yield TestClient(app)
        finally:
            app.dependency_overrides.clear()

    def test_supported_result_response_validates(self, client: TestClient) -> None:
        response = client.post(
            "/v0.1/verifications",
            json={
                "schema_version": "0.1",
                "request_id": "req-wire-schema-test",
                "project_id": "proj-a",
                "application_id": "app-a",
                "answer": "Paris is the capital of France.",
                "evidence": [{"evidence_id": "e1", "content": "Paris is the capital of France."}],
            },
            headers={"Authorization": "Bearer key-1"},
        )
        assert response.status_code == 200
        _assert_valid(_verification_result_validator(), response.json())
