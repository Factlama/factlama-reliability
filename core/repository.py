"""Tenant-aware repository interface (REL-02).

No implementation exists yet -- there is no persistence in this repository
(G5 owns Reliability's PostgreSQL jobs/outbox). This `Protocol` exists now so
a future implementation has a contract to satisfy: every method is scoped by
an explicit `TenantContext`, so a tenant-implicit query path cannot be added
without visibly diverging from this interface.
"""

from typing import Protocol, TypeVar

from schemas.tenancy import TenantContext

T = TypeVar("T")


class TenantAwareRepository(Protocol[T]):
    """A key-value-shaped store scoped to an explicit tenant on every call."""

    def get(self, tenant: TenantContext, key: str) -> T | None:
        """Return the value for `key` within `tenant`'s scope, or None."""
        ...

    def put(self, tenant: TenantContext, key: str, value: T) -> None:
        """Store `value` under `key` within `tenant`'s scope."""
        ...

    def list(self, tenant: TenantContext) -> list[T]:
        """Return every value within `tenant`'s scope."""
        ...
