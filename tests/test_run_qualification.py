"""Tests for scripts/run_qualification.py (F8/F9 of the 2026-09-21 G0-G4
validation report): the end-to-end conformance -> agreement -> registry
runner, and its refusal to fabricate a leakage-control attestation.
"""

import json
from pathlib import Path

import pytest

from core.qualification import QualificationState
from scripts.run_qualification import _load_leakage_attestation, main

_DATASET_KWARGS = {
    "dataset_version": "harness-0.1",
    "dev_hash": "cafebabe" * 8,
    "held_out_hash": "deadbeef" * 8,
    "dev_fixture_count": 24,
    "held_out_fixture_count": 16,
}


def _write_attestation(path: Path, **overrides: object) -> None:
    record = {
        "attested_by": "Jane Reviewer",
        "date": "2026-09-21",
        **_DATASET_KWARGS,
        **overrides,
    }
    path.write_text(json.dumps(record))


class TestLoadLeakageAttestation:
    def test_no_path_means_not_attested(self) -> None:
        attested, record = _load_leakage_attestation(None, **_DATASET_KWARGS)
        assert attested is False
        assert record is None

    def test_matching_dataset_is_attested(self, tmp_path: Path) -> None:
        path = tmp_path / "attestation.json"
        _write_attestation(path)
        attested, record = _load_leakage_attestation(path, **_DATASET_KWARGS)
        assert attested is True
        assert record is not None
        assert record["attested_by"] == "Jane Reviewer"
        assert record["date"] == "2026-09-21"
        assert isinstance(record["file_hash"], str) and record["file_hash"]

    def test_missing_attested_by_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "attestation.json"
        record = {"date": "2026-09-21", **_DATASET_KWARGS}
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="attested_by"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)

    def test_missing_date_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "attestation.json"
        record = {"attested_by": "Jane Reviewer", **_DATASET_KWARGS}
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="date"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)

    def test_stale_held_out_hash_is_rejected(self, tmp_path: Path) -> None:
        """R6 of the 2026-09-21 re-audit: a genuine attestation whose
        held-out hash no longer matches the dataset this run actually
        scored must be refused, not silently accepted."""
        path = tmp_path / "attestation.json"
        _write_attestation(path, held_out_hash="stale" * 8)
        with pytest.raises(ValueError, match="held_out_hash"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)

    def test_stale_dev_hash_is_rejected(self, tmp_path: Path) -> None:
        """Follow-up re-audit finding (2026-09-21): editing public dev
        fixture *content* without changing dev_fixture_count must still
        invalidate a stale attestation -- leakage review compares both
        sets' content, not just the held-out side's."""
        path = tmp_path / "attestation.json"
        _write_attestation(path, dev_hash="stale" * 8)
        with pytest.raises(ValueError, match="dev_hash"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)

    def test_stale_dataset_version_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "attestation.json"
        _write_attestation(path, dataset_version="harness-0.0")
        with pytest.raises(ValueError, match="dataset_version"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)

    def test_mismatched_fixture_counts_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "attestation.json"
        _write_attestation(path, dev_fixture_count=1, held_out_fixture_count=1)
        with pytest.raises(ValueError, match="fixture_count"):
            _load_leakage_attestation(path, **_DATASET_KWARGS)


class TestMainEndToEnd:
    def test_rule_based_stops_at_conformance_passed_without_attestation(
        self, tmp_path: Path
    ) -> None:
        """No `--leakage-attestation-file` -> the registry state can never
        reach AGREEMENT_REPORTED, regardless of how the thresholds score."""
        out_path = tmp_path / "report.json"
        exit_code = main(["--provider", "rule-based", "--out", str(out_path)])
        assert exit_code == 0
        report = json.loads(out_path.read_text())
        assert report["registry_state"] != QualificationState.AGREEMENT_REPORTED.value
        assert any(
            "leakage-control attestation missing" in f
            for f in report["registry_threshold_failures"]
        )

    def test_unknown_provider_choice_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            main(["--provider", "not-a-real-provider"])

    def test_calibration_class_is_non_null_and_reflects_unqualified_status(
        self, tmp_path: Path
    ) -> None:
        """R7 / F9's lifecycle-reconciliation ask: unlike
        `scripts/run_agreement_harness.py`'s own report (always `null`,
        correctly, since it has no resulting qualification_status to
        hash), this report knows the real outcome -- without a real
        attestation, that outcome is honestly UNQUALIFIED, and the
        calibration class must say so, not stay null."""
        from core.provider_identity import configuration_fingerprint
        from core.scoring import derive_calibration_class
        from core.verifier import EVALUATOR_ID, EVALUATOR_VERSION
        from judges.providers import RuleBasedProvider

        out_path = tmp_path / "report.json"
        exit_code = main(["--provider", "rule-based", "--out", str(out_path)])
        assert exit_code == 0
        report = json.loads(out_path.read_text())

        assert report["calibration_class"] is not None
        expected = derive_calibration_class(
            evaluator_id=EVALUATOR_ID,
            evaluator_version=EVALUATOR_VERSION,
            provider_id="rule-based",
            pinned_model_id=report["registry_pinned_model_id"],
            configuration_version=configuration_fingerprint(RuleBasedProvider()),
            qualification_status="UNQUALIFIED",
        )
        assert report["calibration_class"] == expected

    def test_nli_report_carries_both_models_resolved_identity(self, tmp_path: Path) -> None:
        """R1/R5 of the 2026-09-21 re-audit's exact reproduction: the
        qualification runner previously reported
        `pinned_model_id="typeform/distilbert-base-uncased-mnli"` (the bare
        floating name, no revision) with `is_pinned` left at its `True`
        default, even though the primary model's revision was actually
        available and a second, unversioned model backed the verdict. Both
        must now be real."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("sentence_transformers")

        out_path = tmp_path / "report.json"
        exit_code = main(["--provider", "nli", "--out", str(out_path)])
        assert exit_code == 0
        report = json.loads(out_path.read_text())

        pinned = report["registry_pinned_model_id"]
        assert pinned is not None
        assert pinned.startswith("typeform/distilbert-base-uncased-mnli@")
        assert "+sentence-transformers/all-MiniLM-L6-v2@" in pinned
        assert report["registry_is_pinned"] is True
        assert report["pinned_model_id"] == pinned
        assert report["t0_conformance"] == {"checked": True, "failures": []}
