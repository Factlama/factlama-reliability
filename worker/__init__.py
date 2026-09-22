"""G5's async worker (REL-10/docs/async-evaluation.md): claims durable jobs
from `storage.backend.StorageBackend`, runs bounded retries/fallback via
`core.verifier.Verifier`, reconciles cross-attempt claim disagreement into
`DISPUTED` (`worker.reconciliation`, ADR-015), and commits the result plus
its outbox event atomically. `worker/bootstrap.py` is this package's
composition root, mirroring `api/bootstrap.py`."""
