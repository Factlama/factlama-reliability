"""Tenant/project/application identity, established from authenticated ingress
context -- never a client-payload field (CONTRACTS.md; `VerificationRequest`'s
own validator already rejects a client-supplied `tenant_id`).

No authenticated ingress exists yet -- that is G3's sync API. Today,
`Verifier.verify(request, tenant_id=...)`'s trusted string argument remains
the actual boundary; `TenantContext` exists now so G3's API middleware has a
real type to construct and pass through, and so the authorization check
below (`authorizes()`) has something concrete to test against today, ahead
of a live API existing to call it from.
"""

from pydantic import BaseModel, ConfigDict, Field

from schemas.policy import ExecutionStorageMode


class TenantContext(BaseModel):
    """Authenticated tenant identity for one call, optionally narrowed to a
    specific project/application within that tenant.

    `project_id`/`application_id` unset means "authorized for the whole
    tenant"; set means the caller is authorized only for that scope.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(..., min_length=1)
    project_id: str | None = Field(None, description="Authorized project scope, if narrowed")
    application_id: str | None = Field(
        None, description="Authorized application scope, if narrowed"
    )
    execution_storage_grant: ExecutionStorageMode = Field(
        default=ExecutionStorageMode.NO_PERSISTENCE,
        description=(
            "ADR-020's trusted, operator-configured ceiling on this tenant's "
            "execution-storage permission -- never set from a request. A3 of "
            "the 2026-10-14 ADR-020 review: a request's own "
            "`policy.execution_storage.mode` selection must be intersected "
            "against this grant (`schemas.policy."
            "resolve_effective_execution_storage_mode`), not trusted alone, "
            "or any authenticated caller could grant itself durable storage "
            "just by changing its own request body."
        ),
    )

    def authorizes(self, project_id: str, application_id: str) -> bool:
        """Whether a request's project_id/application_id fall within what this
        context is authorized for.

        This is the "forged tenant fields ... fail safely" check from a
        typed context's perspective; it does not itself authenticate
        anything -- that is G3's API boundary. Once that boundary exists, it
        must call this before letting a request reach `Verifier.verify()`,
        not merely echo the tenant_id it was handed.
        """
        if self.project_id is not None and project_id != self.project_id:
            return False
        return self.application_id is None or application_id == self.application_id
