"""EPIC-03's versioned routing profile (CONTRACTS.md): FAST/STANDARD/DEEP
must genuinely differ, and STANDARD must match this repo's pre-existing
global defaults exactly -- introducing per-mode profiles must not change
STANDARD's own behavior, since it is both the contract's documented and
the request schema's actual default mode.
"""

from core.budgets import MAX_CLAIMS_PER_REQUEST, MAX_EVIDENCE_PER_CLAIM
from core.routing_profiles import (
    effective_job_deadline_seconds,
    provider_set_for,
    routing_profile_for,
)
from schemas.verification import VerificationMode


class TestRoutingProfileDiffers:
    def test_fast_has_the_tightest_caps_and_shortest_deadline(self) -> None:
        fast = routing_profile_for(VerificationMode.FAST)
        standard = routing_profile_for(VerificationMode.STANDARD)
        deep = routing_profile_for(VerificationMode.DEEP)

        assert fast.max_claims < standard.max_claims < deep.max_claims
        assert (
            fast.max_evidence_per_claim
            < standard.max_evidence_per_claim
            < deep.max_evidence_per_claim
        )
        assert (
            fast.request_deadline_seconds
            < standard.request_deadline_seconds
            < deep.request_deadline_seconds
        )

    def test_standard_matches_this_repos_pre_existing_global_defaults(self) -> None:
        """Introducing per-mode profiles must not silently change
        STANDARD's own behavior -- it is this schema's own default mode
        (`VerificationMode.STANDARD`), so any drift here would be a
        regression for every caller that never set `mode` explicitly."""
        standard = routing_profile_for(VerificationMode.STANDARD)
        assert standard.max_claims == MAX_CLAIMS_PER_REQUEST
        assert standard.max_evidence_per_claim == MAX_EVIDENCE_PER_CLAIM
        assert standard.request_deadline_seconds == 60.0

    def test_version_strings_match_the_pre_existing_label_format(self) -> None:
        """`provenance.routing_profile_version` predates this module
        (`f"{mode.value.lower()}-0.1"`) -- this module must not change
        that string, only give it real behavior behind it."""
        assert routing_profile_for(VerificationMode.FAST).version == "fast-0.1"
        assert routing_profile_for(VerificationMode.STANDARD).version == "standard-0.1"
        assert routing_profile_for(VerificationMode.DEEP).version == "deep-0.1"

    def test_judgments_per_claim_is_fixed_at_one_for_every_mode(self) -> None:
        """CONTRACTS.md: "one in MVP" -- not a per-mode knob yet."""
        for mode in VerificationMode:
            assert routing_profile_for(mode).judgments_per_claim == 1


class TestProviderSet:
    def test_every_mode_resolves_to_the_same_single_local_provider(self) -> None:
        """This MVP has exactly one qualified local provider -- a real
        per-mode provider-set difference needs a second one (G10/EPIC-07),
        not built here. The dimension is real and checked, even though
        today it is trivially single-valued."""
        for mode in VerificationMode:
            assert provider_set_for(mode, local_provider_id="rule-based") == frozenset(
                {"rule-based"}
            )


class TestEffectiveJobDeadlineSeconds:
    """`api/app.py::create_verification_job`'s own async-job deadline
    computation (G9 independent review, 2026-11-01): the durable job API
    must apply the request's routing profile too, not only the flat
    `FACTLAMA_JOB_DEADLINE_SECONDS` operational ceiling -- a FAST job
    must not get DEEP's deadline just because it went through the async
    path instead of the sync one."""

    def test_fast_standard_deep_produce_genuinely_different_deadlines(self) -> None:
        ceiling = 300.0  # the operator default, well above every profile's own deadline
        fast = effective_job_deadline_seconds(ceiling, VerificationMode.FAST)
        standard = effective_job_deadline_seconds(ceiling, VerificationMode.STANDARD)
        deep = effective_job_deadline_seconds(ceiling, VerificationMode.DEEP)

        assert fast < standard < deep
        assert fast == routing_profile_for(VerificationMode.FAST).request_deadline_seconds
        assert standard == routing_profile_for(VerificationMode.STANDARD).request_deadline_seconds
        assert deep == routing_profile_for(VerificationMode.DEEP).request_deadline_seconds

    def test_the_operational_ceiling_can_only_tighten_never_loosen(self) -> None:
        """A ceiling tighter than DEEP's own 180s profile deadline must
        still win -- the profile can only narrow the operator's own
        configured ceiling, never widen past it."""
        tight_ceiling = 20.0
        assert effective_job_deadline_seconds(tight_ceiling, VerificationMode.DEEP) == tight_ceiling
        # FAST's own 15s profile deadline is still tighter than this
        # ceiling, so FAST wins on its own merits, not the ceiling's.
        assert effective_job_deadline_seconds(tight_ceiling, VerificationMode.FAST) == (
            routing_profile_for(VerificationMode.FAST).request_deadline_seconds
        )
