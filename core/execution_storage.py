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
