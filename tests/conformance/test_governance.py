"""ADR-019 SS4.7: "metadata-only capture, redaction-before-persistence,
retention/deletion, bounded reads, sanitized errors, lifecycle/health,
migration/upgrade, and backup/restore behavior remain intact."

The storage-boundary-only checks below (missing/inaccessible IDs fail
safely, reads are bounded by an explicit limit) predate `core.governance`;
`test_metadata_only_capture_mode_persists_no_claim_text_end_to_end` is this
file's actual "metadata-only capture" evidence -- proving a real worker run
persists the *governed* result, not merely that `core.governance`'s pure
function behaves correctly in isolation (`tests/test_governance.py`
already covers that). Retention/deletion has no implementation to test yet
(`storage.retention.RetentionPolicy` is an unwired interface, REL-11's own
named remaining gap)."""

from datetime import timedelta

import pytest

from core.verifier import Verifier
from schemas.claims import Claim, TextStatus
from schemas.evidence import Evidence
from schemas.policy import CaptureMode, CapturePolicy, Policy
from schemas.verification import VerificationRequest
from storage.errors import JobNotFound
from storage.postgres.backend import PostgresStorageBackend
from storage.postgres.registry import PostgresRevocationRegistry
from tests.conformance.conftest import (
    default_deadline,
    make_outbox_event,
    make_request,
    make_result,
    make_tenant,
)
from worker.runner import process_claimed_job
from worker.settings import WorkerSettings


async def test_missing_job_status_returns_none_not_an_error(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    assert await backend.get_job_status(tenant, "job_does_not_exist") is None


async def test_missing_result_returns_none_not_an_error(backend: PostgresStorageBackend) -> None:
    tenant = make_tenant()
    assert await backend.get_result(tenant, "eval_does_not_exist") is None


async def test_outbox_claim_never_exceeds_the_requested_limit(
    backend: PostgresStorageBackend,
) -> None:
    tenant = make_tenant()
    for i in range(5):
        await backend.submit_or_get_job(
            tenant,
            "verification-jobs",
            f"key-{i}",
            f"hash-{i}",
            make_request(request_id=f"req-{i}"),
            default_deadline(),
        )
        claimed = await backend.claim_due_job(f"worker-{i}", timedelta(minutes=5))
        assert claimed is not None
        await backend.commit_evaluation_and_outbox(
            claimed.job_id,
            claimed.fencing_token,
            make_result(f"eval-{i}"),
            make_outbox_event(f"event-{i}", f"eval-{i}"),
        )

    batch = await backend.claim_outbox_batch("consumer-1", timedelta(minutes=5), limit=2)

    assert len(batch) == 2


async def test_job_not_found_error_does_not_leak_a_different_tenants_data(
    backend: PostgresStorageBackend,
) -> None:
    owner = make_tenant(tenant_id="tenant-a")
    stranger = make_tenant(tenant_id="tenant-b")
    job = await backend.submit_or_get_job(
        owner, "verification-jobs", "key-1", "hash-1", make_request(), default_deadline()
    )

    with pytest.raises(JobNotFound) as exc_info:
        await backend.request_cancel(stranger, job.job_id)

    assert "tenant-a" not in str(exc_info.value)
    assert "proj-a" not in str(exc_info.value)


def _fast_settings() -> WorkerSettings:
    settings = WorkerSettings.__new__(WorkerSettings)
    settings.database_url = "unused-in-this-test"
    settings.worker_id = "worker-test"
    settings.lease_duration_seconds = 60.0
    settings.poll_interval_seconds = 0.01
    settings.max_attempts = 1
    settings.backoff_base_seconds = 0.01
    settings.backoff_max_seconds = 0.02
    settings.backoff_jitter_fraction = 0.0
    settings.outbox_batch_size = 20
    settings.outbox_lease_seconds = 60.0
    settings.outbox_poll_interval_seconds = 0.01
    return settings


async def test_metadata_only_capture_mode_persists_no_claim_text_end_to_end(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """The real REL-11 acceptance evidence: a worker run under
    `CaptureMode.METADATA_ONLY` (the default) must never persist claim text
    or rationale -- a later `get_result()` read (exactly what `GET
    /v0.1/verifications/{evaluation_id}` calls) must come back with
    `text_status=NOT_STORED` and `text=None`, per CONTRACTS.md."""
    tenant = make_tenant()
    policy = Policy(
        id="metadata-only-policy", capture=CapturePolicy(mode=CaptureMode.METADATA_ONLY)
    )
    request = VerificationRequest(
        request_id="req-governed",
        project_id="proj-a",
        application_id="app-a",
        answer="Paris is the capital of France.",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
        policy=policy,
    )
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-governed", "hash-governed", request, default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None

    verifier = Verifier()  # default RuleBasedProvider -- deterministic, no vendor model needed
    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.text_status == TextStatus.NOT_STORED
    assert claim.text is None
    assert claim.rationale is None
    # Non-content fields survive governance untouched.
    assert claim.claim_id == "c1"
    assert claim.verdict is not None


async def test_full_capture_mode_persists_claim_text_end_to_end(
    backend: PostgresStorageBackend,
    registry: PostgresRevocationRegistry,
) -> None:
    """The explicit-opt-in counterpart (ADR-008: "FULL requires explicit
    tenant opt-in") -- choosing FULL is that opt-in."""
    tenant = make_tenant()
    policy = Policy(id="full-capture-policy", capture=CapturePolicy(mode=CaptureMode.FULL))
    request = VerificationRequest(
        request_id="req-full",
        project_id="proj-a",
        application_id="app-a",
        answer="Paris is the capital of France.",
        claims=[Claim(claim_id="c1", text="Paris is the capital of France.")],
        evidence=[Evidence(evidence_id="e1", content="Paris is the capital of France.")],
        policy=policy,
    )
    await backend.submit_or_get_job(
        tenant, "verification-jobs", "key-full", "hash-full", request, default_deadline()
    )
    claimed = await backend.claim_due_job("worker-test", timedelta(minutes=5))
    assert claimed is not None

    verifier = Verifier()
    await process_claimed_job(claimed, backend, verifier, _fast_settings(), registry)

    status = await backend.get_job_status(tenant, claimed.job_id)
    assert status is not None
    assert status.evaluation_id is not None
    result = await backend.get_result(tenant, status.evaluation_id)
    assert result is not None
    claim = result.claims[0]
    assert claim.text_status == TextStatus.AVAILABLE
    assert claim.text == "Paris is the capital of France."
