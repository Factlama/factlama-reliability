"""G2 acceptance criterion: execute contracts/v0.1's canonical fixture suite
against this repository's own types, inside its own CI -- not merely this
repository's own unit/golden tests, which is all G0's earlier pass had.

`ReliabilityEvent` fixtures are deliberately not exercised here: this
repository has no ReliabilityEvent model yet (that is G5's outbox), so that
compatibility check belongs to Observability's OBS-03, not here.
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from judges.port import JudgeRequest, JudgeResult
from schemas.verification import VerificationRequest, VerificationResult


def _contracts_dir() -> Path | None:
    """Locate contracts/v0.1: an explicit env var (set by CI, which checks out
    factlama-architecture alongside this repo), or the local three-repo
    workspace layout (../factlama-architecture) for local development."""
    env = os.environ.get("FACTLAMA_CONTRACTS_DIR")
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    local = Path(__file__).resolve().parents[2] / "factlama-architecture" / "contracts" / "v0.1"
    return local if local.is_dir() else None


CONTRACTS_DIR = _contracts_dir()

pytestmark = pytest.mark.skipif(
    CONTRACTS_DIR is None,
    reason=(
        "contracts/v0.1 not found -- set FACTLAMA_CONTRACTS_DIR, or run inside the "
        "three-repo local workspace next to ../factlama-architecture"
    ),
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _valid_examples(*parts: str) -> list[Path]:
    if CONTRACTS_DIR is None:
        return []
    directory = CONTRACTS_DIR.joinpath("examples", "valid", *parts)
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _invalid_example(name: str) -> Path | None:
    if CONTRACTS_DIR is None:
        return None
    path = CONTRACTS_DIR / "examples" / "invalid" / name
    return path if path.is_file() else None


class TestValidVerificationRequestFixtures:
    @pytest.mark.parametrize("path", _valid_examples("verification_request"), ids=lambda p: p.stem)
    def test_parses_as_verification_request(self, path: Path) -> None:
        VerificationRequest(**_read_json(path))


class TestValidVerificationResultFixtures:
    """Covers success (pass/partial/fail), abstention (two reasons), a failed-
    internal-error case and a disputed case -- not just the success path."""

    @pytest.mark.parametrize("path", _valid_examples("verification_result"), ids=lambda p: p.stem)
    def test_parses_as_verification_result(self, path: Path) -> None:
        VerificationResult(**_read_json(path))


class TestValidJudgeRequestFixtures:
    @pytest.mark.parametrize("path", _valid_examples("judge_request"), ids=lambda p: p.stem)
    def test_parses_as_judge_request(self, path: Path) -> None:
        JudgeRequest(**_read_json(path))


class TestValidJudgeResultFixtures:
    @pytest.mark.parametrize("path", _valid_examples("judge_result"), ids=lambda p: p.stem)
    def test_parses_as_judge_result(self, path: Path) -> None:
        JudgeResult(**_read_json(path))


class TestInvalidFixturesAreRejected:
    """Every invalid fixture relevant to this repo's types must fail exactly
    the rule it was published to demonstrate -- a validator that doesn't
    fire would pass this fixture, silently regressing the contract."""

    def test_verification_request_with_tenant_id_is_rejected(self) -> None:
        path = _invalid_example("verification_request_with_tenant_id.json")
        assert path is not None
        with pytest.raises(ValidationError, match="must not carry tenant_id"):
            VerificationRequest(**_read_json(path))

    def test_verification_request_unknown_major_version_is_rejected(self) -> None:
        path = _invalid_example("verification_request_unknown_major_version.json")
        assert path is not None
        with pytest.raises(ValidationError, match="Unsupported schema version"):
            VerificationRequest(**_read_json(path))

    def test_judge_result_both_verdict_and_error_is_rejected(self) -> None:
        path = _invalid_example("judge_result_both_verdict_and_error.json")
        assert path is not None
        with pytest.raises(ValidationError, match="exactly one of verdict or error"):
            JudgeResult(**_read_json(path))

    def test_verification_result_supported_claim_without_evidence_is_rejected(self) -> None:
        path = _invalid_example("verification_result_supported_claim_without_evidence.json")
        assert path is not None
        with pytest.raises(ValidationError, match="requires at least one evidence_id"):
            VerificationResult(**_read_json(path))

    def test_verification_result_missing_provenance_is_rejected(self) -> None:
        path = _invalid_example("verification_result_missing_provenance.json")
        assert path is not None
        with pytest.raises(ValidationError):
            VerificationResult(**_read_json(path))


def test_at_least_one_fixture_of_each_kind_was_actually_found() -> None:
    """A path/glob bug that silently finds zero fixtures would make every
    parametrized test above vacuously pass -- fail loudly instead."""
    if CONTRACTS_DIR is None:
        pytest.skip("contracts/v0.1 not found")
    assert _valid_examples("verification_request")
    assert _valid_examples("verification_result")
    assert _valid_examples("judge_request")
    assert _valid_examples("judge_result")
    for name in [
        "verification_request_with_tenant_id.json",
        "verification_request_unknown_major_version.json",
        "judge_result_both_verdict_and_error.json",
        "verification_result_supported_claim_without_evidence.json",
        "verification_result_missing_provenance.json",
    ]:
        assert _invalid_example(name) is not None, name
