"""Worker process configuration. Unlike `api/settings.py`,
`FACTLAMA_DATABASE_URL` is required here -- a worker with no database to
claim jobs against cannot do anything, so this fails at construction, not
at the first poll (ADR-019 SS3's "fail startup/readiness explicitly").
"""

import os

from core.execution_storage import (
    ExecutionStorageKeyring,
    parse_restore_mode,
    parse_retired_key_ids,
)

DEFAULT_LEASE_DURATION_SECONDS = 120.0
"""How long a claimed job's lease lasts before it is eligible for reclaim
by another worker (docs/async-evaluation.md). Must comfortably exceed one
bounded-retry sequence's worst-case wall-clock time
(`DEFAULT_REQUEST_TIMEOUT_SECONDS` per attempt, times `max_attempts`) --
`worker.runner` renews the lease between attempts, but a lease that were
shorter than one attempt's own deadline could expire mid-dispatch."""

DEFAULT_POLL_INTERVAL_SECONDS = 2.0
"""How long the main loop sleeps after finding no due job before polling
`claim_due_job` again."""

DEFAULT_MAX_ATTEMPTS = 2
"""Bounded retries (docs/async-evaluation.md): the primary attempt plus
this many additional attempts. `1` disables retry entirely -- the first
attempt's result is always final."""

DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 20.0
DEFAULT_BACKOFF_JITTER_FRACTION = 0.2

DEFAULT_OUTBOX_BATCH_SIZE = 20
DEFAULT_OUTBOX_LEASE_SECONDS = 60.0
DEFAULT_OUTBOX_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_OBSERVABILITY_INGRESS_TIMEOUT_SECONDS = 10.0

# ADR-020: how often the independent expiry-cleanup loop sweeps for
# job_execution_inputs rows past their absolute expires_at, regardless of
# the owning job's state -- this is what bounds retention even when
# workers stop entirely and a job never reaches a terminal state.
DEFAULT_EXECUTION_INPUT_SWEEP_INTERVAL_SECONDS = 30.0


class WorkerSettings:
    """Process-wide worker settings, read once from the environment."""

    def __init__(self) -> None:
        self.database_url = os.environ["FACTLAMA_DATABASE_URL"]
        self.worker_id = os.environ.get("FACTLAMA_WORKER_ID") or f"worker-{os.getpid()}"
        self.lease_duration_seconds = float(
            os.environ.get("FACTLAMA_WORKER_LEASE_SECONDS", DEFAULT_LEASE_DURATION_SECONDS)
        )
        self.poll_interval_seconds = float(
            os.environ.get("FACTLAMA_WORKER_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)
        )
        self.max_attempts = int(
            os.environ.get("FACTLAMA_WORKER_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
        )
        self.backoff_base_seconds = float(
            os.environ.get("FACTLAMA_WORKER_BACKOFF_BASE_SECONDS", DEFAULT_BACKOFF_BASE_SECONDS)
        )
        self.backoff_max_seconds = float(
            os.environ.get("FACTLAMA_WORKER_BACKOFF_MAX_SECONDS", DEFAULT_BACKOFF_MAX_SECONDS)
        )
        self.backoff_jitter_fraction = float(
            os.environ.get(
                "FACTLAMA_WORKER_BACKOFF_JITTER_FRACTION", DEFAULT_BACKOFF_JITTER_FRACTION
            )
        )
        self.outbox_batch_size = int(
            os.environ.get("FACTLAMA_WORKER_OUTBOX_BATCH_SIZE", DEFAULT_OUTBOX_BATCH_SIZE)
        )
        self.outbox_lease_seconds = float(
            os.environ.get("FACTLAMA_WORKER_OUTBOX_LEASE_SECONDS", DEFAULT_OUTBOX_LEASE_SECONDS)
        )
        self.outbox_poll_interval_seconds = float(
            os.environ.get(
                "FACTLAMA_WORKER_OUTBOX_POLL_INTERVAL_SECONDS",
                DEFAULT_OUTBOX_POLL_INTERVAL_SECONDS,
            )
        )
        self.execution_input_sweep_interval_seconds = float(
            os.environ.get(
                "FACTLAMA_WORKER_EXECUTION_INPUT_SWEEP_INTERVAL_SECONDS",
                DEFAULT_EXECUTION_INPUT_SWEEP_INTERVAL_SECONDS,
            )
        )
        # ADR-020: the worker needs this to decrypt a DURABLE_ENCRYPTED job's
        # input at claim time -- same env vars as api/settings.py, so both
        # processes resolve the same keys. An empty/missing keyring is valid
        # construction here too; `claim_due_job` fails loudly only if a
        # claimed row actually needs a key this keyring doesn't have.
        self.execution_storage_keyring = ExecutionStorageKeyring.from_env(
            os.environ.get("FACTLAMA_EXECUTION_STORAGE_KEYS"),
            os.environ.get("FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID"),
        )
        # ADR-020 section 5's mandatory restore procedure, enforced
        # mechanically at startup -- see
        # `core.execution_storage.enforce_retired_keys_are_absent`, called
        # from `worker/bootstrap.py` and `worker/expiry_cleanup_bootstrap.py`.
        # Empty/missing in normal operation.
        self.retired_execution_storage_key_ids = parse_retired_key_ids(
            os.environ.get("FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS")
        )
        # The operator's explicit restore declaration -- raises at
        # construction (not at use time) on an unrecognized value.
        self.execution_storage_restore_mode = parse_restore_mode(
            os.environ.get("FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE")
        )
        # G6 handoff: the base URL of Observability's internal ingress
        # (e.g. "https://observability.internal:8000"). Left unset, outbox
        # delivery falls back to `worker.outbox_delivery.log_only_deliver`
        # (`worker/bootstrap.py`) -- the same explicit, logged stand-in
        # used before G6's ingress existed, not a silent no-op. Set this
        # and `observability_workload_token` together; the ingress
        # authenticates every request against the matching
        # `FACTLAMA_RELIABILITY_WORKLOAD_TOKEN` on its own side.
        self.observability_ingress_url = os.environ.get("FACTLAMA_OBSERVABILITY_INGRESS_URL")
        self.observability_workload_token = os.environ.get("FACTLAMA_OBSERVABILITY_WORKLOAD_TOKEN")
        self.observability_ingress_timeout_seconds = float(
            os.environ.get(
                "FACTLAMA_OBSERVABILITY_INGRESS_TIMEOUT_SECONDS",
                DEFAULT_OBSERVABILITY_INGRESS_TIMEOUT_SECONDS,
            )
        )
