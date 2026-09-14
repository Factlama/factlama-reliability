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
