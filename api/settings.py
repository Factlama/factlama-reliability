"""API configuration (REL-01's deferred config-model item; REL-12 is its
real home -- see docs/implementation.md's REL-01 entry). Environment-only:
there is no config service, and this repo has no other runtime to configure.
"""

import os
from functools import lru_cache

from api.auth import TenantCredentialStore

# One answer plus its supplied evidence is not expected to approach this;
# a request this large is already pathological before any cap on claim or
# evidence *count* (core.budgets) even applies.
DEFAULT_MAX_BODY_BYTES = 1_000_000


class Settings:
    """Process-wide API settings, read once from the environment."""

    def __init__(self) -> None:
        self.tenant_store = TenantCredentialStore.from_json(
            os.environ.get("FACTLAMA_TENANT_CREDENTIALS")
        )
        self.max_body_bytes = int(os.environ.get("FACTLAMA_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))


@lru_cache
def get_settings() -> Settings:
    """The default `Settings` singleton. Tests override this via FastAPI's
    `app.dependency_overrides`, never by mutating environment variables
    mid-process."""
    return Settings()
