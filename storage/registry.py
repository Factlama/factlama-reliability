"""Baseline evaluator-revocation registry (G5, EXECUTION_PLAN.md's G5 gate
deliverable "revocation checks"; `evaluator-registry.md`: "REVOKED blocks
new dispatch immediately and is rechecked before result commit -- G5, once
there is a result-commit step to recheck against").

This is deliberately the narrow slice EXECUTION_PLAN.md scopes to G5, not
G10's full audited registry (`core.qualification`'s
`REGISTERED -> CONFORMANCE_PASSED -> AGREEMENT_REPORTED -> TENANT_APPROVED
-> {DEPRECATED, REVOKED}` state machine, transition auditing, tenant
approval, or default-judge eligibility): a minimal, persisted answer to
"has this exact (provider_id, pinned_model_id, configuration_version)
tuple been revoked", nothing else. G10 replaces the storage and extends the
lifecycle this checks against; it does not replace this check.

Revocation is global, not tenant-scoped (ADR-014: a revoked entry
"propagate[s] to any policy naming the provider as fallback" for every
tenant), so unlike `storage.backend.StorageBackend` this Protocol takes no
`TenantContext`.
"""

from typing import Protocol


class RevocationRegistry(Protocol):
    """One process-wide, atomic registry of revoked evaluator identities."""

    async def is_revoked(
        self, provider_id: str, pinned_model_id: str, configuration_version: str
    ) -> bool:
        """Whether this exact identity has been revoked. `False` for an
        identity that was never registered at all -- G4's own scope leaves
        every current T0 adapter unregistered by default
        (`core.qualification`), so "absent from this registry" must never
        be conflated with "revoked"."""
        ...

    async def revoke(
        self, provider_id: str, pinned_model_id: str, configuration_version: str
    ) -> None:
        """Record this identity as revoked. Idempotent: revoking an
        already-revoked identity is a no-op, not an error
        (`core.qualification.revoke`'s same rule)."""
        ...
