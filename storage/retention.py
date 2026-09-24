"""Retention interface (G5/REL-11, ADR-008). ADR-008 itself treats
retention/deletion as future work ("must cover replicas, indexes and
derived datasets *when implemented*") -- this is the interface the rule
above names, not a scheduler, purge job, or wiring into `worker.bootstrap`.
No caller exists yet, the same deliberate gap `core.evidence.
EvidenceRetriever` left for REL-05: building a real caller now would mean
guessing at an operational cadence (a cron? a worker-loop tick? an admin
endpoint?) with nothing yet to validate the choice against.

Scope: `jobs`/`evaluations`/`outbox_events` rows past a retention window --
not ADR-008's separate, explicitly-deferred `InteractionStore` archive.
"""

from datetime import datetime
from typing import Protocol


class RetentionPolicy(Protocol):
    """One backend-wide purge operation. Deliberately not tenant-scoped or
    per-data-class (ADR-008's eventual "per-data-class capture controls" is
    the full `InteractionStore`'s job) -- G5's own minimal scope is one
    global cutoff applied uniformly, matching how `storage.registry.
    RevocationRegistry` is also global rather than tenant-scoped.
    """

    async def purge_older_than(self, cutoff: datetime) -> int:
        """Permanently delete every terminal job, evaluation, and
        acknowledged outbox event created before `cutoff`. Returns the
        number of rows removed. Never touches a job still `QUEUED`/
        `RUNNING`, or an outbox event not yet acknowledged -- only rows a
        retention window, not active processing, should ever remove."""
        ...
