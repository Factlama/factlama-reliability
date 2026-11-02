"""EPIC-03's versioned routing profile (CONTRACTS.md: "`mode` selects a
versioned routing profile with claim/evidence caps, total deadline,
token/cost ceilings, provider set and judgments-per-claim"). Until this
module, `request.mode` was recorded only as a label
(`provenance.routing_profile_version`) -- `ANY_JUDGE_ARCHITECTURE_REVIEW.md`
named this exactly: "a field we accept and ignore." A G9 release-gate
review found this insufficient and asked for a minimal, deterministic MVP
binding: a real profile per mode whose claim/evidence caps and deadline
actually differ and are actually enforced, not merely a different label.

Scope, stated rather than silently assumed:

- Provider set: every profile names only this MVP's one local, unmetered
  provider (whatever `Verifier.model_provider.name` resolves to at
  dispatch time) -- a real multi-provider routing decision needs G4's
  qualification/registry machinery applied to more than one qualified
  adapter, which is G10/EPIC-07 territory, not this gate's. The field
  exists and is checked (so a future second provider has somewhere real
  to plug in) even though, with one provider, every mode currently
  resolves to the identical set.
- Judgments per claim: fixed at 1 for every profile, matching this
  pipeline's own existing one-claim-per-call invariant (ADR-015 defers
  multi-judge reconciliation) -- not a per-mode knob yet.
- Token/cost ceilings: `core.budgets`'s existing per-request
  `MAX_TOKENS_PER_REQUEST`/`MAX_COST_PER_REQUEST_USD` remain global, not
  per-mode -- no current adapter reports a non-default `Usage` to make a
  per-mode split observable or testable yet (see `core.budgets`'s own
  module docstring).
- Deadline enforcement: the synchronous `Verifier.verify()` path passes
  `request_deadline_seconds` as `dispatch_claims()`'s own
  `overall_deadline`. The async path's job-level deadline
  (`ClaimedJob.deadline`, read by `worker/runner.py`'s bounded-retry
  loop) is computed once at submission time, in
  `api/app.py::create_verification_job` -- `min(settings.
  job_deadline_seconds, routing_profile_for(mode).request_deadline_
  seconds)`, the same tightening-only composition `dispatch_claims()`
  itself already uses for the sync path's deadline and
  `_dispatch_with_bounded_retries` already uses for `deadline`/
  `input_expires_at`. A FAST job's durable deadline is therefore
  genuinely shorter than a DEEP job's, not just its sync-path deadline --
  `ADR-020`'s existing `execution_input_expires_at = min(TTL, deadline)`
  invariant (enforced in `submit_or_get_job`) means a tightened `deadline`
  here automatically tightens the execution-input expiry to match, with
  no separate change needed.
"""

from dataclasses import dataclass

from schemas.verification import VerificationMode

# STANDARD's values are exactly this repo's pre-existing global defaults
# (`core.budgets.MAX_CLAIMS_PER_REQUEST`/`MAX_EVIDENCE_PER_CLAIM`,
# `core.verifier.DEFAULT_REQUEST_TIMEOUT_SECONDS`) -- introducing per-mode
# profiles must not silently change STANDARD's own existing behavior,
# since it is both the documented and the request-schema default mode.
_STANDARD_MAX_CLAIMS = 50
_STANDARD_MAX_EVIDENCE_PER_CLAIM = 20
_STANDARD_REQUEST_DEADLINE_SECONDS = 60.0


@dataclass(frozen=True)
class RoutingProfile:
    """One mode's versioned routing profile (CONTRACTS.md). `version` is
    the exact string `provenance.routing_profile_version` already
    recorded before this module existed (`f"{mode.value.lower()}-0.1"`)
    -- unchanged, so no existing fixture or test asserting that string
    breaks."""

    version: str
    max_claims: int
    max_evidence_per_claim: int
    request_deadline_seconds: float
    judgments_per_claim: int = 1


def routing_profile_for(mode: VerificationMode) -> RoutingProfile:
    """The one lookup every caller uses -- never construct a
    `RoutingProfile` ad hoc, so there is exactly one place that defines
    what each mode means."""
    return _ROUTING_PROFILES[mode]


def effective_job_deadline_seconds(
    configured_ceiling_seconds: float, mode: VerificationMode
) -> float:
    """`api/app.py::create_verification_job`'s own async-job deadline
    computation, factored out so it is directly testable without a real
    backend. `configured_ceiling_seconds` is the operator's own
    `FACTLAMA_JOB_DEADLINE_SECONDS` -- an operational safety ceiling, not
    a per-mode budget. `min()` with the profile's own deadline: the
    profile can only tighten a job's budget below that ceiling, never
    loosen past it, the same tightening-only composition
    `dispatch_claims()` already uses for the sync path's deadline and
    `_dispatch_with_bounded_retries` already uses for
    `deadline`/`input_expires_at`."""
    return min(configured_ceiling_seconds, routing_profile_for(mode).request_deadline_seconds)


def provider_set_for(mode: VerificationMode, *, local_provider_id: str) -> frozenset[str]:
    """CONTRACTS.md's "provider set" dimension. Every MVP profile
    resolves to the same single-element set today (see this module's own
    docstring for why) -- a real per-mode `JudgeProvider` compliance/
    dispatch check against more than one provider is G10/EPIC-07 work."""
    del mode  # Every mode currently maps to the same one-provider set.
    return frozenset({local_provider_id})


_ROUTING_PROFILES: dict[VerificationMode, RoutingProfile] = {
    VerificationMode.FAST: RoutingProfile(
        version="fast-0.1",
        max_claims=10,
        max_evidence_per_claim=5,
        request_deadline_seconds=15.0,
    ),
    VerificationMode.STANDARD: RoutingProfile(
        version="standard-0.1",
        max_claims=_STANDARD_MAX_CLAIMS,
        max_evidence_per_claim=_STANDARD_MAX_EVIDENCE_PER_CLAIM,
        request_deadline_seconds=_STANDARD_REQUEST_DEADLINE_SECONDS,
    ),
    VerificationMode.DEEP: RoutingProfile(
        version="deep-0.1",
        max_claims=100,
        max_evidence_per_claim=40,
        request_deadline_seconds=180.0,
    ),
}
