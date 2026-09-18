"""G4: evaluator qualification state machine and default-judge eligibility
(ADR-010/014, evaluator-registry.md). No storage, no Verifier wiring --
see core/qualification.py's own docstring for why."""

import pytest

from core.qualification import (
    IllegalTransitionError,
    QualificationRecord,
    QualificationState,
    approve_for_tenant,
    deprecate,
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
            report_agreement(_record(), dataset_version="harness-0.1", adversarial_flips=0)

    def test_agreement_reported_with_zero_flips_advances(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=0)
        assert record.state == QualificationState.AGREEMENT_REPORTED
        assert record.dataset_version == "harness-0.1"

    def test_agreement_reported_with_nonzero_flips_is_refused_not_raised(self) -> None:
        """ADR-013: any injection flip disqualifies -- a failed report is an
        expected outcome, not a programming error, so this must not raise."""
        record = mark_conformance_passed(_record())
        result = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=1)
        assert result.state == QualificationState.CONFORMANCE_PASSED

    def test_tenant_approval_requires_agreement_reported(self) -> None:
        record = mark_conformance_passed(_record())
        with pytest.raises(IllegalTransitionError):
            approve_for_tenant(record, "tenant-a")

    def test_tenant_approval_advances_and_records_tenant(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=0)
        record = approve_for_tenant(record, "tenant-a")
        assert record.state == QualificationState.TENANT_APPROVED
        assert "tenant-a" in record.approved_tenant_ids

    def test_a_second_tenant_can_approve_an_already_approved_record(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=0)
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


class TestDefaultEligibility:
    def _approved(self, **overrides) -> QualificationRecord:
        record = mark_conformance_passed(_record(**overrides))
        record = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=0)
        return approve_for_tenant(record, "tenant-a")

    def test_registered_provider_is_never_default_eligible(self) -> None:
        assert is_default_eligible(_record(), "tenant-a") is False

    def test_agreement_reported_but_not_tenant_approved_is_not_eligible(self) -> None:
        record = mark_conformance_passed(_record())
        record = report_agreement(record, dataset_version="harness-0.1", adversarial_flips=0)
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
