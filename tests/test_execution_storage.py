"""ADR-020: pure-Python tests for `core.execution_storage` -- no DB, no
async, no real key-management infrastructure, just the cryptographic
primitive and keyring config resolution in isolation."""

import base64
import json

import pytest

from core.execution_storage import (
    RESTORE_MODE_NORMAL,
    RESTORE_MODE_RESTORE,
    ExecutionInputDecryptionFailed,
    ExecutionStorageConfigurationError,
    ExecutionStorageKeyring,
    decrypt_request_payload,
    encrypt_request_payload,
    enforce_retired_keys_are_absent,
    parse_restore_mode,
    parse_retired_key_ids,
)
from schemas.verification import VerificationRequest

KEY_A = b"\x01" * 32
KEY_B = b"\x02" * 32


def _request(answer: str = "Paris is the capital of France.") -> VerificationRequest:
    return VerificationRequest(
        request_id="req-1", project_id="proj-a", application_id="app-a", answer=answer
    )


class TestEncryptDecryptRoundTrip:
    def test_round_trip_reconstructs_the_same_request(self) -> None:
        request = _request()
        payload = encrypt_request_payload(
            request, tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        decrypted = decrypt_request_payload(
            payload, tenant_id="tenant-a", job_id="job-1", key=KEY_A
        )
        assert decrypted.answer == request.answer
        assert decrypted.request_id == request.request_id

    def test_ciphertext_never_contains_the_plaintext_answer(self) -> None:
        request = _request(answer="PRIVATE_SENTINEL_VALUE")
        payload = encrypt_request_payload(
            request, tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        assert b"PRIVATE_SENTINEL_VALUE" not in payload.ciphertext

    def test_nonce_differs_across_encryptions_of_the_same_content(self) -> None:
        request = _request()
        first = encrypt_request_payload(
            request, tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        second = encrypt_request_payload(
            request, tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        assert first.nonce != second.nonce
        assert first.ciphertext != second.ciphertext


class TestTamperAndSubstitutionFailClosed:
    def test_tampered_ciphertext_fails_authenticated_decryption(self) -> None:
        payload = encrypt_request_payload(
            _request(), tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        tampered_bytes = bytearray(payload.ciphertext)
        tampered_bytes[0] ^= 0xFF
        from dataclasses import replace

        tampered = replace(payload, ciphertext=bytes(tampered_bytes))
        with pytest.raises(ExecutionInputDecryptionFailed):
            decrypt_request_payload(tampered, tenant_id="tenant-a", job_id="job-1", key=KEY_A)

    def test_wrong_key_fails_authenticated_decryption(self) -> None:
        payload = encrypt_request_payload(
            _request(), tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        with pytest.raises(ExecutionInputDecryptionFailed):
            decrypt_request_payload(payload, tenant_id="tenant-a", job_id="job-1", key=KEY_B)

    def test_ciphertext_substituted_from_a_different_job_fails(self) -> None:
        """The whole point of binding (tenant_id, job_id) as associated
        data: a ciphertext genuinely encrypted under the right key, for a
        *different* job, must not decrypt as if it belonged here."""
        payload = encrypt_request_payload(
            _request(), tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        with pytest.raises(ExecutionInputDecryptionFailed):
            decrypt_request_payload(payload, tenant_id="tenant-a", job_id="job-2", key=KEY_A)

    def test_ciphertext_substituted_from_a_different_tenant_fails(self) -> None:
        payload = encrypt_request_payload(
            _request(), tenant_id="tenant-a", job_id="job-1", key_id="v1", key=KEY_A
        )
        with pytest.raises(ExecutionInputDecryptionFailed):
            decrypt_request_payload(payload, tenant_id="tenant-b", job_id="job-1", key=KEY_A)


class TestExecutionStorageKeyring:
    def _keys_json(self, **keys: bytes) -> str:
        return json.dumps(
            {key_id: base64.b64encode(key).decode("ascii") for key_id, key in keys.items()}
        )

    def test_resolves_a_configured_key(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id="v1")
        assert keyring.resolve("v1") == KEY_A

    def test_unknown_key_id_raises_configuration_error(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id="v1")
        with pytest.raises(ExecutionStorageConfigurationError):
            keyring.resolve("v999")

    def test_active_key_is_explicit_not_inferred_by_sorting(self) -> None:
        """v1/v2 lexicographic sort happens to match version order, but the
        active key must come from explicit configuration, not inference --
        this is what keeps rotation correct past v9/v10."""
        keyring = ExecutionStorageKeyring.from_env(
            self._keys_json(v1=KEY_A, v2=KEY_B), active_key_id="v1"
        )
        key_id, key = keyring.active_key()
        assert key_id == "v1"
        assert key == KEY_A

    def test_rotation_keeps_the_old_key_resolvable(self) -> None:
        """Adding v2 as active must not make v1-encrypted ciphertext
        unresolvable -- ADR-020: "rotation is not destruction.\""""
        keyring = ExecutionStorageKeyring.from_env(
            self._keys_json(v1=KEY_A, v2=KEY_B), active_key_id="v2"
        )
        assert keyring.resolve("v1") == KEY_A
        assert keyring.active_key() == ("v2", KEY_B)

    def test_missing_active_key_id_raises_on_use_not_at_construction(self) -> None:
        """A keyring with keys but no active_key_id configured is valid for
        *decrypting* existing rows (resolve() by key_id still works); only
        choosing a key for a *new* write requires an active key."""
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id=None)
        assert keyring.resolve("v1") == KEY_A
        with pytest.raises(ExecutionStorageConfigurationError):
            keyring.active_key()

    def test_active_key_id_not_in_keyring_fails_at_construction(self) -> None:
        with pytest.raises(ExecutionStorageConfigurationError):
            ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id="v2")

    def test_empty_configuration_has_no_usable_key(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(None, active_key_id=None)
        assert keyring.has_any_key() is False
        with pytest.raises(ExecutionStorageConfigurationError):
            keyring.active_key()

    def test_wrong_key_length_is_rejected_at_construction(self) -> None:
        short_key_json = json.dumps({"v1": base64.b64encode(b"too-short").decode("ascii")})
        with pytest.raises(ExecutionStorageConfigurationError):
            ExecutionStorageKeyring.from_env(short_key_json, active_key_id="v1")

    def test_key_ids_reports_every_configured_key(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(
            self._keys_json(v1=KEY_A, v2=KEY_B), active_key_id="v2"
        )
        assert keyring.key_ids() == frozenset({"v1", "v2"})


class TestParseRetiredKeyIds:
    def test_missing_or_empty_is_an_empty_set(self) -> None:
        assert parse_retired_key_ids(None) == frozenset()
        assert parse_retired_key_ids("") == frozenset()

    def test_parses_a_comma_separated_list_ignoring_surrounding_whitespace(self) -> None:
        assert parse_retired_key_ids("v1, v2 ,v3") == frozenset({"v1", "v2", "v3"})


class TestParseRestoreMode:
    def test_missing_or_empty_defaults_to_normal(self) -> None:
        assert parse_restore_mode(None) == RESTORE_MODE_NORMAL
        assert parse_restore_mode("") == RESTORE_MODE_NORMAL

    def test_recognizes_restore_case_insensitively_and_trims_whitespace(self) -> None:
        assert parse_restore_mode("RESTORE") == RESTORE_MODE_RESTORE
        assert parse_restore_mode("restore") == RESTORE_MODE_RESTORE
        assert parse_restore_mode("  Restore  ") == RESTORE_MODE_RESTORE
        assert parse_restore_mode("NORMAL") == RESTORE_MODE_NORMAL

    def test_unrecognized_value_raises_rather_than_silently_defaulting(self) -> None:
        with pytest.raises(ExecutionStorageConfigurationError, match="RESTOER"):
            parse_restore_mode("RESTOER")  # typo, must not silently become NORMAL


class TestMandatoryRestoreRetirementGate:
    """ADR-020 section 5's mandatory restore procedure, enforced
    mechanically: a `key_id` an operator has declared retired (because it
    could have encrypted a row in a backup they are restoring from) must
    actually be absent from the live keyring before any process is allowed
    to start against the restored database, and declaring
    `RESTORE_MODE=RESTORE` without naming anything to retire must not be a
    no-op bypass of that procedure."""

    def test_no_retired_keys_declared_is_a_no_op(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id="v1")
        enforce_retired_keys_are_absent(keyring, frozenset())

    def test_retired_key_already_absent_from_keyring_passes(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v2=KEY_B), active_key_id="v2")
        enforce_retired_keys_are_absent(keyring, frozenset({"v1"}))

    def test_retired_key_still_present_refuses_to_start(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(
            self._keys_json(v1=KEY_A, v2=KEY_B), active_key_id="v2"
        )
        with pytest.raises(ExecutionStorageConfigurationError, match="v1"):
            enforce_retired_keys_are_absent(keyring, frozenset({"v1"}))

    def test_restore_mode_with_empty_manifest_refuses_to_start(self) -> None:
        """Declaring a restore without naming any key to retire would
        otherwise let an operator "confirm" the procedure happened without
        actually doing anything -- the manifest must be non-empty."""
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v1=KEY_A), active_key_id="v1")
        with pytest.raises(ExecutionStorageConfigurationError, match="RESTORE_MODE=RESTORE"):
            enforce_retired_keys_are_absent(keyring, frozenset(), RESTORE_MODE_RESTORE)

    def test_restore_mode_with_retired_key_already_absent_succeeds(self) -> None:
        """The positive path: operator declares the restore, names the
        retired key, and has actually removed it -- startup proceeds."""
        keyring = ExecutionStorageKeyring.from_env(self._keys_json(v2=KEY_B), active_key_id="v2")
        enforce_retired_keys_are_absent(keyring, frozenset({"v1"}), RESTORE_MODE_RESTORE)

    def test_restore_mode_with_retired_key_still_present_refuses_to_start(self) -> None:
        keyring = ExecutionStorageKeyring.from_env(
            self._keys_json(v1=KEY_A, v2=KEY_B), active_key_id="v2"
        )
        with pytest.raises(ExecutionStorageConfigurationError, match="v1"):
            enforce_retired_keys_are_absent(keyring, frozenset({"v1"}), RESTORE_MODE_RESTORE)

    def _keys_json(self, **keys: bytes) -> str:
        return json.dumps(
            {key_id: base64.b64encode(key).decode("ascii") for key_id, key in keys.items()}
        )
