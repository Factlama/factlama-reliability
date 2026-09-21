"""G4: evaluator qualification state machine and default-judge eligibility
(ADR-010/014, evaluator-registry.md). No storage, no Verifier wiring --
see core/qualification.py's own docstring for why."""

import pytest

from core.qualification import (
    ADR_018_MIN_LABEL_N,
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
    """A per_label section that clears every ADR-018 threshold with n well
    above ADR_018_MIN_LABEL_N -- used by tests that only care about
    lifecycle mechanics, not threshold edge cases."""
    return {
        "SUPPORTED": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "n": 6},
        "CONTRADICTED": {"precision": 0.85, "recall": 0.85, "f1": 0.85, "n": 6},
        "UNSUPPORTED": {"precision": 0.75, "recall": 0.75, "f1": 0.75, "n": 6},
        "INSUFFICIENT_EVIDENCE": {"precision": 0.75, "recall": 0.75, "f1": 0.75, "n": 6},
    }


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
            report_agreement(
                _record(), dataset_version="harness-0.1", adversarial_flips=0, per_label={}
            )

    def test_agreement_reported_with_zero_flips_advances(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=0,
            per_label=_passing_per_label(),
        )
        assert record.state == QualificationState.AGREEMENT_REPORTED
        assert record.dataset_version == "harness-0.1"
        assert record.threshold_failures == ()

    def test_agreement_reported_with_nonzero_flips_is_refused_not_raised(self) -> None:
        """ADR-013: any injection flip disqualifies -- a failed report is an
        expected outcome, not a programming error, so this must not raise."""
        record = mark_conformance_passed(_record())
        result = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=1,
            per_label=_passing_per_label(),
        )
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("adversarial" in reason for reason in result.threshold_failures)

    def test_tenant_approval_requires_agreement_reported(self) -> None:
        record = mark_conformance_passed(_record())
        with pytest.raises(IllegalTransitionError):
            approve_for_tenant(record, "tenant-a")

    def test_tenant_approval_advances_and_records_tenant(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=0,
            per_label=_passing_per_label(),
        )
        record = approve_for_tenant(record, "tenant-a")
        assert record.state == QualificationState.TENANT_APPROVED
        assert "tenant-a" in record.approved_tenant_ids

    def test_a_second_tenant_can_approve_an_already_approved_record(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=0,
            per_label=_passing_per_label(),
        )
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
    """ADR-018: an agreement report needs zero adversarial flips, at least
    one label with ADR_018_MIN_LABEL_N scored examples, and every
    sufficiently-sampled label meeting its precision/recall bar."""

    def test_empty_per_label_is_refused_as_insufficient_sample_size(self) -> None:
        record = mark_conformance_passed(_record())
        result = report_agreement(
            record, dataset_version="harness-0.1", adversarial_flips=0, per_label={}
        )
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("insufficient sample size" in reason for reason in result.threshold_failures)

    def test_every_label_below_min_n_is_refused_as_insufficient_sample_size(self) -> None:
        record = mark_conformance_passed(_record())
        thin_per_label = {
            "SUPPORTED": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n": ADR_018_MIN_LABEL_N - 1}
        }
        result = report_agreement(
            record, dataset_version="harness-0.1", adversarial_flips=0, per_label=thin_per_label
        )
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("insufficient sample size" in reason for reason in result.threshold_failures)

    def test_a_label_below_its_precision_bar_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["SUPPORTED"] = {"precision": 0.5, "recall": 0.9, "f1": 0.6, "n": 6}
        result = report_agreement(
            record, dataset_version="harness-0.1", adversarial_flips=0, per_label=per_label
        )
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("SUPPORTED: precision" in reason for reason in result.threshold_failures)

    def test_a_label_below_its_recall_bar_refuses_the_report(self) -> None:
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["CONTRADICTED"] = {"precision": 0.9, "recall": 0.4, "f1": 0.55, "n": 6}
        result = report_agreement(
            record, dataset_version="harness-0.1", adversarial_flips=0, per_label=per_label
        )
        assert result.state == QualificationState.CONFORMANCE_PASSED
        assert any("CONTRADICTED: recall" in reason for reason in result.threshold_failures)

    def test_a_thin_label_does_not_block_a_report_carried_by_other_labels(self) -> None:
        """A label under ADR_018_MIN_LABEL_N is skipped, not failed -- it
        must not silently sink an otherwise-passing report."""
        record = mark_conformance_passed(_record())
        per_label = _passing_per_label()
        per_label["INSUFFICIENT_EVIDENCE"] = {
            "precision": 0.1,
            "recall": 0.1,
            "f1": 0.1,
            "n": ADR_018_MIN_LABEL_N - 1,
        }
        result = report_agreement(
            record, dataset_version="harness-0.1", adversarial_flips=0, per_label=per_label
        )
        assert result.state == QualificationState.AGREEMENT_REPORTED

    def test_evaluate_agreement_thresholds_ignores_labels_outside_the_scored_vocabulary(
        self,
    ) -> None:
        per_label = {"AMBIGUOUS": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 50}}
        assert evaluate_agreement_thresholds(per_label) == []


class TestDefaultEligibility:
    def _approved(self, **overrides) -> QualificationRecord:
        record = mark_conformance_passed(_record(**overrides))
        record = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=0,
            per_label=_passing_per_label(),
        )
        return approve_for_tenant(record, "tenant-a")

    def test_registered_provider_is_never_default_eligible(self) -> None:
        assert is_default_eligible(_record(), "tenant-a") is False

    def test_agreement_reported_but_not_tenant_approved_is_not_eligible(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(
            record,
            dataset_version="harness-0.1",
            adversarial_flips=0,
            per_label=_passing_per_label(),
        )
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
