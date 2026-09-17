"""Credential validation and trusted tenant scope (LOW_LEVEL_IMPLEMENTATION.md's
`auth` unit): "Tenant never comes from untrusted body/header."

There is no tenant/credential database yet -- that is G5's persistence.
`TenantCredentialStore` is a deliberately minimal, config-driven
API-key-to-`TenantContext` map (one operator-managed secret, not a public
signup flow) so this repo's synchronous API has a real authenticated
boundary today rather than trusting a header verbatim. Replacing this with a
database-backed store later changes only this module, not any route.
"""

import json
from typing import Any

from schemas.tenancy import TenantContext


class TenantCredentialStore:
    """Maps an opaque API key to the `TenantContext` it authenticates."""

    def __init__(self, credentials: dict[str, TenantContext]) -> None:
        self._credentials = credentials

    @classmethod
    def from_json(cls, raw: str | None) -> "TenantCredentialStore":
        """Build a store from a JSON object string:
        `{"<api_key>": {"tenant_id": "...", "project_id": "...", "application_id": "..."}}`.
        `project_id`/`application_id` are optional narrowing scopes, same as
        `TenantContext` itself. Empty/missing input yields a store that
        authenticates nothing -- fail closed, never fail open.
        """
        if not raw:
            return cls({})

        parsed: dict[str, Any] = json.loads(raw)
        return cls({api_key: TenantContext(**scope) for api_key, scope in parsed.items()})

    def authenticate(self, api_key: str | None) -> TenantContext | None:
        """Return the authenticated `TenantContext` for a valid key, or None."""
        if not api_key:
            return None
        return self._credentials.get(api_key)
