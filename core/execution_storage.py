"""ADR-020: application-level authenticated encryption for `DURABLE_ENCRYPTED`
execution inputs. AES-256-GCM via `cryptography`'s `AESGCM` -- "do not
implement cryptographic primitives locally" (ADR-020 section 3).

Encrypts the **complete** `VerificationRequest` as one JSON blob rather than
a field-by-field blacklist (ADR-020 section 3: "Prefer encrypting the
complete execution input over a blacklist of selected fields") -- the exact
gap the 2026-10-13 G5 code reviews kept finding one field at a time (tool
payloads, then request-level `metadata`, each only discovered because the
previous governance pass used an allowlist instead).

Ciphertext is bound to `(tenant_id, job_id, payload_version)` as AEAD
associated data, so a substituted or replayed ciphertext from a different
job or tenant fails authentication outright, not merely fails to parse
(ADR-020 section 3: "bind ciphertext to tenant, job and payload version so
substitutions fail authentication").
"""

import base64
import json
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from schemas.verification import VerificationRequest

PAYLOAD_VERSION = 1
_NONCE_BYTES = 12  # AESGCM's standard nonce size (96 bits).
_KEY_BYTES = 32  # AES-256.


class ExecutionStorageConfigurationError(Exception):
    """Raised when `DURABLE_ENCRYPTED` is reachable but no usable key is
    configured. Fails loudly at construction/use time -- ADR-020 section 5:
    "No silent plaintext fallback is permitted on key-service failure.\""""


class ExecutionInputDecryptionFailed(Exception):
    """Raised when stored ciphertext fails authenticated decryption --
    tampered, wrong key, or bound to a different `(tenant_id, job_id)` than
    the caller is claiming. Wraps `cryptography`'s own `InvalidTag` so
    callers in `storage.postgres` never need to import `cryptography`
    directly just to catch this one condition."""


@dataclass(frozen=True)
class EncryptedPayload:
    """What gets persisted in `job_execution_inputs`: opaque ciphertext plus
    exactly the metadata needed to decrypt and re-authenticate it later --
    never the key itself."""

    key_id: str
    payload_version: int
    nonce: bytes
    ciphertext: bytes


def _associated_data(tenant_id: str, job_id: str, payload_version: int) -> bytes:
    return f"{tenant_id}:{job_id}:{payload_version}".encode()


def encrypt_request_payload(
    request: VerificationRequest,
    *,
    tenant_id: str,
    job_id: str,
    key_id: str,
    key: bytes,
) -> EncryptedPayload:
    """Encrypt the complete request as one JSON blob, bound to
    `(tenant_id, job_id, payload_version)`."""
    plaintext = request.model_dump_json(exclude_none=True).encode("utf-8")
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce, plaintext, _associated_data(tenant_id, job_id, PAYLOAD_VERSION)
    )
    return EncryptedPayload(
        key_id=key_id, payload_version=PAYLOAD_VERSION, nonce=nonce, ciphertext=ciphertext
    )


def decrypt_request_payload(
    payload: EncryptedPayload, *, tenant_id: str, job_id: str, key: bytes
) -> VerificationRequest:
    """Decrypt and reconstruct the real `VerificationRequest`.

    Raises `ExecutionInputDecryptionFailed` on any tamper, wrong key, or
    ciphertext bound to a different `(tenant_id, job_id)` -- callers
    (`storage.postgres.backend.claim_due_job`) translate that into
    `storage.errors.ExecutionInputCorrupted`, never silently falling back to
    empty/placeholder content as if it were real.
    """
    try:
        plaintext = AESGCM(key).decrypt(
            payload.nonce,
            payload.ciphertext,
            _associated_data(tenant_id, job_id, payload.payload_version),
        )
    except InvalidTag as exc:
        raise ExecutionInputDecryptionFailed(
            f"execution input for job {job_id!r} failed authenticated decryption "
            "(tampered, wrong key, or bound to a different tenant/job)"
        ) from exc
    return VerificationRequest.model_validate_json(plaintext)


class ExecutionStorageKeyring:
    """Config-driven `key_id -> raw key bytes` resolution, mirroring
    `api.auth.TenantCredentialStore.from_json`'s exact pattern. Supports
    multiple simultaneous `key_id`s so rotation adds a key without
    invalidating ciphertext encrypted under an older one (ADR-020 section 5:
    "rotation is not destruction") -- `active_key_id` is explicit
    configuration, never inferred by sorting key_id strings, which breaks
    silently past single-digit version numbers.
    """

    def __init__(self, keys: dict[str, bytes], active_key_id: str | None) -> None:
        if active_key_id is not None and active_key_id not in keys:
            raise ExecutionStorageConfigurationError(
                f"active_key_id={active_key_id!r} is not present in the configured keyring"
            )
        self._keys = keys
        self._active_key_id = active_key_id

    @classmethod
    def from_env(
        cls, keys_json: str | None, active_key_id: str | None
    ) -> "ExecutionStorageKeyring":
        """`keys_json`: a JSON object string `{"<key_id>": "<base64 32-byte key>"}`
        (`FACTLAMA_EXECUTION_STORAGE_KEYS`). `active_key_id`
        (`FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID`) names which configured
        key new writes should use. Empty/missing `keys_json` yields a
        keyring with no usable keys -- fail closed: `resolve()`/`active_key()`
        raise rather than this constructor guessing a default.
        """
        if not keys_json:
            return cls({}, active_key_id)
        parsed: dict[str, str] = json.loads(keys_json)
        keys = {key_id: base64.b64decode(encoded) for key_id, encoded in parsed.items()}
        for key_id, key in keys.items():
            if len(key) != _KEY_BYTES:
                raise ExecutionStorageConfigurationError(
                    f"execution-storage key {key_id!r} is {len(key)} bytes, expected "
                    f"{_KEY_BYTES} (AES-256)"
                )
        return cls(keys, active_key_id)

    def resolve(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError:
            raise ExecutionStorageConfigurationError(
                f"no execution-storage key configured for key_id={key_id!r}"
            ) from None

    def active_key(self) -> tuple[str, bytes]:
        """The `(key_id, key)` new writes should use. Raises
        `ExecutionStorageConfigurationError` if no active key is configured
        -- the caller (`DURABLE_ENCRYPTED` submission) must fail loudly
        rather than silently falling back to plaintext storage."""
        if self._active_key_id is None:
            raise ExecutionStorageConfigurationError(
                "DURABLE_ENCRYPTED execution storage requires an active execution-storage "
                "key (FACTLAMA_EXECUTION_STORAGE_ACTIVE_KEY_ID); none is configured"
            )
        return self._active_key_id, self.resolve(self._active_key_id)

    def has_any_key(self) -> bool:
        return bool(self._keys)

    def key_ids(self) -> frozenset[str]:
        return frozenset(self._keys)


def parse_retired_key_ids(env_value: str | None) -> frozenset[str]:
    """`FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS`: a comma-separated list
    of every `key_id` that could have encrypted a row in a backup an
    operator is about to restore from. Part of the mandatory restore
    procedure ADR-020 section 5 requires (see `enforce_retired_keys_are_absent`)
    -- empty/missing is the normal, non-restore case."""
    if not env_value:
        return frozenset()
    return frozenset(key_id.strip() for key_id in env_value.split(",") if key_id.strip())


RESTORE_MODE_NORMAL = "NORMAL"
RESTORE_MODE_RESTORE = "RESTORE"
_VALID_RESTORE_MODES = frozenset({RESTORE_MODE_NORMAL, RESTORE_MODE_RESTORE})


def parse_restore_mode(env_value: str | None) -> str:
    """`FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE`: an operator's explicit
    declaration of whether this process is starting against a database
    just restored from backup. `NORMAL` (missing/empty also default to
    this) is ordinary operation; `RESTORE` additionally requires a
    non-empty `FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS` manifest (see
    `enforce_retired_keys_are_absent`) -- declaring a restore without
    naming any key to retire would otherwise be a no-op bypass of the
    whole procedure. Any other value fails loudly at construction, rather
    than silently defaulting to `NORMAL` on a typo."""
    if not env_value:
        return RESTORE_MODE_NORMAL
    normalized = env_value.strip().upper()
    if normalized not in _VALID_RESTORE_MODES:
        raise ExecutionStorageConfigurationError(
            f"FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE={env_value!r} is not a "
            f"recognized restore mode (expected one of {sorted(_VALID_RESTORE_MODES)!r})"
        )
    return normalized


def enforce_retired_keys_are_absent(
    keyring: ExecutionStorageKeyring,
    retired_key_ids: frozenset[str],
    restore_mode: str = RESTORE_MODE_NORMAL,
) -> None:
    """Mechanical enforcement of ADR-020 section 5's mandatory restore
    procedure: the one piece of state that survives a full-database
    restore unchanged is the execution-storage keyring itself (environment
    configuration, not a database table) -- `claim_due_job` already relies
    on a retired key_id's genuine absence from the keyring to terminalize a
    revived row instead of decrypting it (`storage.postgres.backend`,
    P1 of the 2026-10-14 ADR-020 closure review). That only protects a
    restore if the operator actually removes the key; nothing previously
    stopped a process from starting against a restored database with the
    key still configured, or from declaring `RESTORE_MODE=RESTORE` without
    actually naming anything to retire.

    Call this at every process composition root (`api/bootstrap.py`,
    `worker/bootstrap.py`, `worker/expiry_cleanup_bootstrap.py`) before
    constructing a storage backend or serving any request, with
    `restore_mode` already validated by `parse_restore_mode`:

    - `RESTORE_MODE_RESTORE` with an empty `retired_key_ids` refuses to
      start outright -- an operator declaring a restore must name every
      key_id that could have encrypted a row in the restored backup, not
      merely assert that a restore happened.
    - Any `retired_key_ids` still present in the live keyring refuses to
      start, regardless of `restore_mode` -- this is the actual
      enforcement mechanism and must not be gameable by omitting the
      restore-mode declaration.
    - `RESTORE_MODE_NORMAL` with no `retired_key_ids` declared is always a
      no-op: ordinary, non-restore startup is unaffected.
    """
    if restore_mode == RESTORE_MODE_RESTORE and not retired_key_ids:
        raise ExecutionStorageConfigurationError(
            "FACTLAMA_EXECUTION_STORAGE_RESTORE_MODE=RESTORE requires a non-empty "
            "FACTLAMA_EXECUTION_STORAGE_RETIRED_KEY_IDS manifest -- an operator "
            "declaring a restore must name every key_id that could have encrypted "
            "a row in the restored backup, not merely assert that a restore happened"
        )
    still_present = retired_key_ids & keyring.key_ids()
    if still_present:
        raise ExecutionStorageConfigurationError(
            "execution-storage keyring still contains key_id(s) marked for "
            f"mandatory post-restore retirement: {sorted(still_present)!r}. ADR-020 "
            "section 5's restore procedure requires every key_id that could have "
            "encrypted a row in the restored backup to be removed from "
            "FACTLAMA_EXECUTION_STORAGE_KEYS before any API/worker process touches "
            "the restored database -- refusing to start rather than risk reviving a "
            "destroyed input."
        )
