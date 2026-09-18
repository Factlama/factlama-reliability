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
        assert report["adversarial"]["total"] == 1

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
        fixtures = [f for f in dev_fixtures() if f.family == "adversarial"]
        report = score_provider(_AlwaysSupportedProvider(), fixtures)

        assert report["adversarial"]["total"] == 1
        assert report["adversarial"]["flipped_to_supported"] == 1

    def test_real_t0_adapter_resists_the_adversarial_fixture(self) -> None:
        """RuleBasedProvider must not flip -- this is the same guarantee
        tests/test_dispatch_gates.py::TestEvidenceInjectionDefense checks
        at the Verifier level, exercised here through the harness path."""
        fixtures = [f for f in dev_fixtures() if f.family == "adversarial"]
        report = score_provider(RuleBasedProvider(), fixtures)

        assert report["adversarial"]["flipped_to_supported"] == 0
