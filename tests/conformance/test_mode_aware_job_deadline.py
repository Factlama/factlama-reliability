"""G9 independent review, 2026-11-01: a concrete blocker found that
`worker/runner.py`'s bounded-retry loop only ever bounds dispatch by
`claimed.deadline`/`claimed.input_expires_at` -- neither of which reflected
`request.mode`, since `ClaimedJob.deadline` is computed once at
submission time (`api/app.py::create_verification_job`) from a flat
`FACTLAMA_JOB_DEADLINE_SECONDS` ceiling, with no routing-profile input at
all. `core/routing_profiles.py` even documented this as a deliberately
deferred follow-up, which the review correctly rejected -- the gate's own
requirement is that caps *and* deadlines differ and are enforced for the
whole async path, not only the synchronous one.

This proves the real fix end to end against PostgreSQL: a FAST-mode job's
*stored* deadline, read back through the exact same `submit_or_get_job`/
`claim_due_job` round trip the real worker uses, is genuinely tighter than
a DEEP-mode job's -- not just that the pure `effective_job_deadline_
seconds()` function returns different numbers in isolation
(`tests/test_routing_profiles.py` already covers that).
"""

from datetime import datetime, timedelta, timezone

from core.routing_profiles import effective_job_deadline_seconds
from schemas.verification import VerificationMode
from storage.postgres.backend import PostgresStorageBackend
from tests.conformance.conftest import make_request, make_tenant

# Mirrors `api/settings.py::DEFAULT_JOB_DEADLINE_SECONDS` -- well above
# every profile's own deadline, so the profile (not this ceiling) is what
# actually determines each job's deadline below.
_OPERATOR_CEILING_SECONDS = 300.0


async def _submit_with_mode_aware_deadline(
    backend: PostgresStorageBackend, mode: VerificationMode, key: str
) -> datetime:
    """The exact computation `api/app.py::create_verification_job` performs
    -- reusing the same named function, not re-deriving the formula, so
    this test cannot silently drift from what production code actually
    does. Returns `submitted_at` -- the one piece of this computation the
    test itself cannot otherwise recover, needed below to recompute each
    job's real stored budget."""
    tenant = make_tenant()
    request = make_request(request_id=f"req-{key}", mode=mode)
    submitted_at = datetime.now(timezone.utc)
    deadline = submitted_at + timedelta(
        seconds=effective_job_deadline_seconds(_OPERATOR_CEILING_SECONDS, mode)
    )
    await backend.submit_or_get_job(
        tenant, "verification-jobs", key, f"hash-{key}", request, deadline
    )
    return submitted_at


async def test_fast_jobs_stored_deadline_is_tighter_than_deeps(
    backend: PostgresStorageBackend,
) -> None:
    fast_submitted_at = await _submit_with_mode_aware_deadline(
        backend, VerificationMode.FAST, key="fast-deadline"
    )
    deep_submitted_at = await _submit_with_mode_aware_deadline(
        backend, VerificationMode.DEEP, key="deep-deadline"
    )

    fast_claimed = await backend.claim_due_job("worker-fast", timedelta(minutes=5))
    deep_claimed = await backend.claim_due_job("worker-deep", timedelta(minutes=5))
    assert fast_claimed is not None
    assert deep_claimed is not None
    # `claim_due_job` claims oldest-due-first across both tenants; identify
    # each claimed row by its own request_id rather than assuming order.
    claimed_by_request_id = {
        fast_claimed.request.request_id: fast_claimed,
        deep_claimed.request.request_id: deep_claimed,
    }
    fast = claimed_by_request_id["req-fast-deadline"]
    deep = claimed_by_request_id["req-deep-deadline"]

    fast_budget = (fast.deadline - fast_submitted_at).total_seconds()
    deep_budget = (deep.deadline - deep_submitted_at).total_seconds()

    # Real stored-and-reclaimed values, allowing a little slack for the
    # wall-clock time this test itself took between computing
    # `submitted_at` and the deadline, and PostgreSQL's own timestamp
    # precision -- not exact-to-the-millisecond, but unmistakably
    # FAST's own ~15s budget versus DEEP's own ~180s one.
    assert fast_budget < 30.0
    assert deep_budget > 150.0
    assert fast_budget < deep_budget

    assert fast.deadline < deep.deadline
    assert fast.request.mode == VerificationMode.FAST
    assert deep.request.mode == VerificationMode.DEEP
