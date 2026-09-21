"""G4: evaluator qualification state machine and default-judge eligibility
(ADR-010/014/018, evaluator-registry.md). No storage, no Verifier wiring --
see core/qualification.py's own docstring for why."""

import pytest

from core.qualification import (
    ADR_018_MIN_LABEL_N,
    ADR_018_THRESHOLDS,
    IllegalTransitionError,
    QualificationRecord,
    QualificationState,
    approve_for_tenant,
    deprecate,
    evaluate_agreement_thresholds,
    is_default_eligible,
    mark_conformance_passed,
    report_agreement,
    revoke,
)


def _record(**overrides) -> QualificationRecord:
    defaults = {
        "provider_id": "rule-based",
        "pinned_model_id": "rule-based-v1",
        "configuration_version": "0.1",
    }
    defaults.update(overrides)
    return QualificationRecord(**defaults)


def _passing_per_label() -> dict:
    """A per_label section that clears every ADR-018 threshold, with all
    four labels well above ADR_018_MIN_LABEL_N -- used by tests that only
    care about lifecycle mechanics, not threshold edge cases."""
    return {
        "SUPPORTED": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "n": 6},
        "CONTRADICTED": {"precision": 0.85, "recall": 0.85, "f1": 0.85, "n": 6},
        "UNSUPPORTED": {"precision": 0.75, "recall": 0.75, "f1": 0.75, "n": 6},
        "INSUFFICIENT_EVIDENCE": {"precision": 0.75, "recall": 0.75, "f1": 0.75, "n": 6},
    }


def _passing_kwargs(**overrides) -> dict:
    """Every keyword `report_agreement()` needs to succeed -- per_label plus
    the held-out-inclusion and leakage-attestation evidence ADR-018/
    evaluator-agreement-harness.md require. Tests override only the one
    argument they're exercising."""
    defaults = {
        "dataset_version": "harness-0.1",
        "adversarial_flips": 0,
        "per_label": _passing_per_label(),
        "held_out_hash": "deadbeef" * 8,
        "held_out_fixture_count": 16,
        "leakage_attested": True,
    }
    defaults.update(overrides)
    return defaults


class TestLifecycleTransitions:
    def test_new_record_starts_registered(self) -> None:
        assert _record().state == QualificationState.REGISTERED

    def test_conformance_passed_from_registered(self) -> None:
        record = mark_conformance_passed(_record())
        assert record.state == QualificationState.CONFORMANCE_PASSED

    def test_conformance_cannot_be_marked_twice(self) -> None:
        record = mark_conformance_passed(_record())
        with pytest.raises(IllegalTransitionError):
            mark_conformance_passed(record)

    def test_agreement_reported_requires_conformance_first(self) -> None:
        with pytest.raises(IllegalTransitionError):
            report_agreement(_record(), **_passing_kwargs())

    def test_agreement_reported_with_zero_flips_advances(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, **_passing_kwargs())
        assert record.state == QualificationState.AGREEMENT_REPORTED
        assert record.dataset_version == "harness-0.1"
        assert record.threshold_failures == ()

    def test_agreement_reported_with_nonzero_flips_is_refused_not_raised(self) -> None:
        """ADR-013: any injection flip disqualifies -- a failed report is an
        expected outcome, not a programming error, so this must not raise."""
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs(adversarial_flips=1))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("adversarial" in reason for reason in result.threshold_failures)

    def test_tenant_approval_requires_agreement_reported(self) -> None:
        record = mark_conformance_passed(_record())
        with pytest.raises(IllegalTransitionError):
            approve_for_tenant(record, "tenant-a")

    def test_tenant_approval_advances_and_records_tenant(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, **_passing_kwargs())
        record = approve_for_tenant(record, "tenant-a")
        assert record.state == QualificationState.TENANT_APPROVED
        assert "tenant-a" in record.approved_tenant_ids

    def test_a_second_tenant_can_approve_an_already_approved_record(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, **_passing_kwargs())
        record = approve_for_tenant(record, "tenant-a")
        record = approve_for_tenant(record, "tenant-b")
        assert record.approved_tenant_ids == frozenset({"tenant-a", "tenant-b"})

    def test_deprecate_and_revoke_are_terminal(self) -> None:
        record = deprecate(_record())
        assert record.state == QualificationState.DEPRECATED
        with pytest.raises(IllegalTransitionError):
            deprecate(record)

    def test_revoke_is_idempotent(self) -> None:
        record = revoke(_record())
        record_again = revoke(record)
        assert record_again.state == QualificationState.REVOKED

    def test_cannot_revoke_a_deprecated_record(self) -> None:
        record = deprecate(_record())
        with pytest.raises(IllegalTransitionError):
            revoke(record)


class TestADR018QualificationThreshold:
    """ADR-018: an agreement report needs zero adversarial flips, all four
    labels independently reaching ADR_018_MIN_LABEL_N scored examples, and
    each one meeting its precision/recall bar."""

    def test_empty_per_label_is_refused_as_insufficient_sample_size(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs(per_label={}))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("insufficient sample size" in reason for reason in result.threshold_failures)

    def test_every_label_below_min_n_is_refused_as_insufficient_sample_size(self) -> None:
        record = mark_conformance_passed(_record())
        thin_per_label = {
            "SUPPORTED": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n": ADR_018_MIN_LABEL_N - 1}
        }
        result = report_agreement(record, **_passing_kwargs(per_label=thin_per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("insufficient sample size" in reason for reason in result.threshold_failures)

    def test_five_perfect_supported_examples_alone_cannot_qualify(self) -> None:
        """Regression for a real bypass: a report containing only 5 perfect
        SUPPORTED examples (no CONTRADICTED/UNSUPPORTED/INSUFFICIENT_EVIDENCE
        at all) must not reach AGREEMENT_REPORTED -- the other three verdicts
        were never agreement-tested."""
        record = mark_conformance_passed(_record())
        only_supported = {
            "SUPPORTED": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n": ADR_018_MIN_LABEL_N}
        }
        result = report_agreement(record, **_passing_kwargs(per_label=only_supported))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        failure_text = " ".join(result.threshold_failures)
        assert "CONTRADICTED" in failure_text
        assert "UNSUPPORTED" in failure_text
        assert "INSUFFICIENT_EVIDENCE" in failure_text

    def test_an_unscored_label_key_does_not_satisfy_sample_size(self) -> None:
        """A report carrying only a non-ADR-018 key (e.g. a stray/garbage
        label) with n>=5 must not be treated as sufficient sample size for
        any of the four real labels."""
        record = mark_conformance_passed(_record())
        per_label = {"NOT_A_REAL_LABEL": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n": 50}}
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("insufficient sample size" in reason for reason in result.threshold_failures)

    def test_a_label_below_its_precision_bar_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["SUPPORTED"] = {"precision": 0.5, "recall": 0.9, "f1": 0.6, "n": 6}
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("SUPPORTED: precision" in reason for reason in result.threshold_failures)

    def test_a_label_below_its_recall_bar_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["CONTRADICTED"] = {"precision": 0.9, "recall": 0.4, "f1": 0.55, "n": 6}
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("CONTRADICTED: recall" in reason for reason in result.threshold_failures)

    def test_a_thin_label_now_blocks_the_report(self) -> None:
        """Changed from the first ADR-018 pass: a label under
        ADR_018_MIN_LABEL_N must refuse the whole report, not be skipped --
        skipping let a report qualify without ever testing that label."""
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["INSUFFICIENT_EVIDENCE"] = {
            "precision": 0.1,
            "recall": 0.1,
            "f1": 0.1,
            "n": ADR_018_MIN_LABEL_N - 1,
        }
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("INSUFFICIENT_EVIDENCE" in reason for reason in result.threshold_failures)

    def test_evaluate_agreement_thresholds_ignores_labels_outside_the_scored_vocabulary(
        self,
    ) -> None:
        per_label = {"AMBIGUOUS": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 50}}
        assert evaluate_agreement_thresholds(per_label) == []

    def test_missing_held_out_hash_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs(held_out_hash=""))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("held-out set not included" in reason for reason in result.threshold_failures)

    def test_zero_held_out_fixtures_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs(held_out_fixture_count=0))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("held-out set not included" in reason for reason in result.threshold_failures)

    def test_missing_leakage_attestation_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs(leakage_attested=False))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any(
            "leakage-control attestation missing" in reason for reason in result.threshold_failures
        )

    def test_nan_precision_refuses_the_report_instead_of_silently_passing(self) -> None:
        """Regression: `float("nan") < 0.8` is `False` in Python, so the old
        `precision < min_precision` check let a NaN precision through as if
        it cleared the bar. A non-finite metric must refuse the report."""
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["SUPPORTED"] = {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": None,
            "n": 6,
        }
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("not a valid probability" in reason for reason in result.threshold_failures)

    def test_out_of_range_precision_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["CONTRADICTED"] = {"precision": 1.4, "recall": 0.9, "f1": 0.9, "n": 6}
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("not a valid probability" in reason for reason in result.threshold_failures)

    def test_all_four_labels_with_nan_metrics_does_not_reach_agreement_reported(self) -> None:
        """The exact scenario the reviewer reproduced: every label carries
        NaN precision/recall at a passing `n`. Must not qualify."""
        record = mark_conformance_passed(_record())
        per_label = {
            label: {"precision": float("nan"), "recall": float("nan"), "f1": None, "n": 6}
            for label in ADR_018_THRESHOLDS
        }
        result = report_agreement(record, **_passing_kwargs(per_label=per_label))
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert result.threshold_failures != ()

    def test_a_fully_passing_report_has_no_threshold_failures(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(record, **_passing_kwargs())
        assert result.state == QualificationState.AGREEMENT_REPORTED
        assert result.threshold_failures == ()


class TestDefaultEligibility:
    def _approved(self, **overrides) -> QualificationRecord:
        record = mark_conformance_passed(_record(**overrides))
        record = report_agreement(record, **_passing_kwargs())
        return approve_for_tenant(record, "tenant-a")

    def test_registered_provider_is_never_default_eligible(self) -> None:
        assert is_default_eligible(_record(), "tenant-a") is False

    def test_agreement_reported_but_not_tenant_approved_is_not_eligible(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, **_passing_kwargs())
        assert is_default_eligible(record, "tenant-a") is False

    def test_tenant_approved_for_a_different_tenant_is_not_eligible(self) -> None:
        record = self._approved()
        assert is_default_eligible(record, "tenant-other") is False

    def test_tenant_approved_and_pinned_is_eligible(self) -> None:
        record = self._approved()
        assert is_default_eligible(record, "tenant-a") is True

    def test_unpinned_alias_is_never_eligible_even_if_approved(self) -> None:
        record = self._approved(is_pinned=False)
        assert is_default_eligible(record, "tenant-a") is False

    def test_missing_required_compliance_tag_is_not_eligible(self) -> None:
        record = self._approved(compliance_tags=frozenset({"IN_PROCESS"}))
        assert is_default_eligible(record, "tenant-a", ["NO_EXTERNAL_EGRESS"]) is False

    def test_satisfied_compliance_tags_are_eligible(self) -> None:
        record = self._approved(compliance_tags=frozenset({"IN_PROCESS", "NO_EXTERNAL_EGRESS"}))
        assert is_default_eligible(record, "tenant-a", ["NO_EXTERNAL_EGRESS"]) is True

    def test_revoked_record_is_never_eligible(self) -> None:
        record = revoke(self._approved())
        assert is_default_eligible(record, "tenant-a") is False
