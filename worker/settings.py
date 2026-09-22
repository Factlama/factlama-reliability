"""Worker process configuration. Unlike `api/settings.py`,
`FACTLAMA_DATABASE_URL` is required here -- a worker with no database to
claim jobs against cannot do anything, so this fails at construction, not
at the first poll (ADR-019 SS3's "fail startup/readiness explicitly").
"""

import os

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
