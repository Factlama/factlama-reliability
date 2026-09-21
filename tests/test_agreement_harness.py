"""G4: the agreement-harness runner's scoring function. Exercises the
report shape and the adversarial-flip count against known providers --
this is not a claim that RuleBasedProvider or MockModelProvider are
qualified (see core/qualification.py); it only proves the runner scores
correctly against a provider whose behavior is already known.
"""

from judges.port import CancellationToken, JudgeRequest, JudgeResult
from judges.providers import RuleBasedProvider
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence
from scripts.run_agreement_harness import AgreementFixture, dev_fixtures, score_provider


class _AlwaysSupportedProvider:
    """A stub that always returns SUPPORTED, citing whatever evidence it
    was given -- used to prove the adversarial-flip counter actually
    counts, since every real adapter in this repo already resists it."""

    name = "always-supported"

    def evaluate(self, request: JudgeRequest, deadline: float, cancellation: CancellationToken):
        return JudgeResult(
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=[e.evidence_id for e in request.evidence],
        )


class TestScoreProvider:
    def test_report_has_expected_shape(self) -> None:
        report = score_provider(RuleBasedProvider(), dev_fixtures())

        assert report["report_version"] == "0.1"
        assert report["provider_id"] == "rule-based"
        assert set(report["per_label"]) == {
            "SUPPORTED",
            "CONTRADICTED",
            "UNSUPPORTED",
            "INSUFFICIENT_EVIDENCE",
        }
        assert report["adversarial"]["total"] == 3
        assert "adr_018_threshold_failures" in report

    def test_dev_set_covers_every_family_from_the_spec(self) -> None:
        families = {f.family for f in dev_fixtures()}
        assert families == {
            "direct_support",
            "paraphrased_support",
            "numerical_temporal_contradiction",
            "negation_contradiction",
            "irrelevant_evidence",
            "insufficient_evidence",
            "adversarial_injection",
            "ambiguous",
        }

    def test_ambiguous_family_is_excluded_from_per_label_scoring(self) -> None:
        """evaluator-agreement-harness.md: ambiguous fixtures are for
        calibration, not pass/fail scoring -- they must not inflate or
        deflate any label's n."""
        report = score_provider(RuleBasedProvider(), dev_fixtures())
        total_scored_n = sum(m["n"] for m in report["per_label"].values())
        ambiguous_count = sum(1 for f in dev_fixtures() if f.family == "ambiguous")
        assert ambiguous_count == 3
        assert total_scored_n == len(dev_fixtures()) - ambiguous_count
        assert sum(report["ambiguous_calibration"].values()) == ambiguous_count

    def test_perfect_provider_scores_perfectly_on_a_trivial_fixture_set(self) -> None:
        fixtures = [
            AgreementFixture(
                "trivial_support",
                "direct_support",
                Claim(claim_id="c1", text="X is true."),
                [Evidence(evidence_id="e1", content="X is true.")],
                ClaimVerdict.SUPPORTED,
            )
        ]

        class _Perfect:
            name = "perfect"

            def evaluate(self, request, deadline, cancellation):
                return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=["e1"])

        report = score_provider(_Perfect(), fixtures)
        assert report["per_label"]["SUPPORTED"] == {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "n": 1,
        }

    def test_adversarial_flip_is_counted_when_a_provider_trusts_injected_text(self) -> None:
        fixtures = [f for f in dev_fixtures() if f.family == "adversarial_injection"]
        report = score_provider(_AlwaysSupportedProvider(), fixtures)

        assert report["adversarial"]["total"] == len(fixtures)
        assert report["adversarial"]["flipped_to_supported"] == len(fixtures)

    def test_real_t0_adapter_resists_the_adversarial_fixture(self) -> None:
        """RuleBasedProvider must not flip -- this is the same guarantee
        tests/test_dispatch_gates.py::TestEvidenceInjectionDefense checks
        at the Verifier level, exercised here through the harness path."""
        fixtures = [f for f in dev_fixtures() if f.family == "adversarial_injection"]
        report = score_provider(RuleBasedProvider(), fixtures)

        assert report["adversarial"]["flipped_to_supported"] == 0

    def test_providers_with_different_tunable_config_get_different_configuration_version(
        self,
    ) -> None:
        """Regression: `pinned_model_id` alone (derived from `.name`) cannot
        distinguish two same-model providers configured with different
        tunable thresholds -- e.g. two `EmbeddingProvider`s at
        `support_threshold=0.70` vs `0.95` are different evaluators but
        previously produced identical report-identity fields."""

        class _ConfigurableStub:
            name = "stub:model-x"

            def __init__(self, support_threshold: float) -> None:
                self.support_threshold = support_threshold

            def evaluate(self, request, deadline, cancellation):
                return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=[])

        fixtures = [f for f in dev_fixtures() if f.family == "direct_support"][:1]
        report_a = score_provider(_ConfigurableStub(support_threshold=0.70), fixtures)
        report_b = score_provider(_ConfigurableStub(support_threshold=0.95), fixtures)
        report_a_again = score_provider(_ConfigurableStub(support_threshold=0.70), fixtures)

        assert report_a["pinned_model_id"] == report_b["pinned_model_id"]
        assert report_a["configuration_version"] is not None
        assert report_a["configuration_version"] != report_b["configuration_version"]
        assert report_a["configuration_version"] == report_a_again["configuration_version"]

    def test_provider_with_no_public_state_gets_no_configuration_version(self) -> None:
        fixtures = [f for f in dev_fixtures() if f.family == "adversarial_injection"]
        report = score_provider(_AlwaysSupportedProvider(), fixtures)
        assert report["configuration_version"] is None

    def test_resolved_revision_is_appended_to_pinned_model_id_when_available(self) -> None:
        """A provider exposing `resolved_revision` (EmbeddingProvider/
        NLIProvider, once loaded) gets that commit hash appended to
        `pinned_model_id` -- `model_name` alone may be a floating ref, not
        an immutable pin."""

        class _PinnedStub:
            name = "embedding:some/model"
            resolved_revision = "abc123deadbeef"

            def evaluate(self, request, deadline, cancellation):
                return JudgeResult(verdict=ClaimVerdict.SUPPORTED, evidence_ids=[])

        fixtures = [f for f in dev_fixtures() if f.family == "direct_support"][:1]
        report = score_provider(_PinnedStub(), fixtures)
        assert report["pinned_model_id"] == "some/model@abc123deadbeef"

    def test_missing_resolved_revision_leaves_pinned_model_id_unchanged(self) -> None:
        """A provider with no `resolved_revision` attribute at all (Mock,
        RuleBased) or one that returns None/empty is unaffected."""
        report = score_provider(RuleBasedProvider(), dev_fixtures())
        assert report["pinned_model_id"] is None

    def test_rule_based_providers_with_different_thresholds_are_distinguished(self) -> None:
        """RuleBasedProvider itself carries public `support_threshold`/
        `contradiction_threshold` state -- confirm the fingerprint actually
        distinguishes two real, differently-configured instances of it, not
        just the stub above."""
        fixtures = [f for f in dev_fixtures() if f.family == "direct_support"][:1]
        default_report = score_provider(RuleBasedProvider(), fixtures)
        retuned_report = score_provider(
            RuleBasedProvider(support_threshold=0.5, contradiction_threshold=0.9), fixtures
        )
        assert default_report["configuration_version"] is not None
        assert default_report["configuration_version"] != retuned_report["configuration_version"]
