"""Shared provider-identity primitives for ADR-010's calibration-class inputs.

Both the live `Verifier` (core.verifier) and the offline agreement harness
(scripts/run_agreement_harness.py) must derive a judge attempt's identity
from the *same* underlying signals -- a real adapter instance's resolved
model revision and tunable configuration (e.g. `EmbeddingProvider.
support_threshold`), not just its `.name`, which collapses two
differently-configured instances of the same model into one identity. F2 of
the 2026-09-21 G0-G4 validation report: two real `EmbeddingProvider`
instances at support thresholds 0.70 and 0.95 previously produced the same
`configuration_version`/`calibration_class` because only `.name` (constant
across both) and a hardcoded default fed `derive_calibration_class()`.
"""

import hashlib
import json

from judges.port import JudgeProvider

#: Used when a provider exposes no public tunable state to fingerprint
#: (`Mock`/`RuleBasedProvider` today) -- a stable, honest baseline, not a
#: fabricated version string.
DEFAULT_CONFIGURATION_VERSION = "0.1"


def model_id(provider: JudgeProvider) -> str | None:
    """The base model name from a vendor adapter's "<kind>:<model_name>"
    `.name` convention (judges/vendor_adapters.py) -- e.g.
    "sentence-transformers/all-MiniLM-L6-v2". `None` for a provider with no
    underlying pinned model (`Mock`/`RuleBasedProvider` have no colon in
    `.name`), never guessed.
    """
    return provider.name.split(":", 1)[1] if ":" in provider.name else None


def pinned_model_version(provider: JudgeProvider) -> str | None:
    """The immutable resolved model revision, when the adapter can report
    one (`EmbeddingProvider`/`NLIProvider` expose `resolved_revision` once
    their model has loaded) -- `model_id()` alone may be a floating ref (a
    branch/tag), not an immutable pin. `None` before the model loads, or for
    a provider with no such property.
    """
    return getattr(provider, "resolved_revision", None)


def full_pinned_model_id(provider: JudgeProvider) -> str | None:
    """`model_id()` plus its resolved revision (`"<model>@<revision>"` when
    available, bare `model_id()` otherwise), plus a `"+<secondary>@<rev>"`
    suffix when the provider exposes a *second* decision-producing model
    via the duck-typed `secondary_model_id`/`secondary_resolved_revision`
    attributes (e.g. `NLIProvider`'s relatedness-check model, R1/R5 of the
    2026-09-21 re-audit) -- so a composite evaluator whose verdicts depend
    on two models cannot silently keep the identity of a single-model one.
    `None` when the provider has no underlying model at all (Mock/
    RuleBasedProvider).

    This is the one place both the live report (`scripts/
    run_agreement_harness.py`) and the qualification runner (`scripts/
    run_qualification.py`) must compute a provider's pinned identity --
    computing it twice, differently, is exactly how R1 of the re-audit
    found the qualification runner silently discarding the resolved
    revision the agreement report had already computed correctly.
    """
    base = model_id(provider)
    if base is None:
        return None
    revision = pinned_model_version(provider)
    pinned = f"{base}@{revision}" if revision else base

    secondary_name = getattr(provider, "secondary_model_id", None)
    if secondary_name:
        secondary_revision = getattr(provider, "secondary_resolved_revision", None)
        secondary = (
            f"{secondary_name}@{secondary_revision}" if secondary_revision else secondary_name
        )
        pinned = f"{pinned}+{secondary}"
    return pinned


def is_fully_pinned(provider: JudgeProvider) -> bool:
    """Whether every decision-producing model this provider's verdicts
    depend on has resolved to an immutable revision. `True` for a provider
    with no underlying model at all (Mock/RuleBasedProvider -- pinned by
    construction, nothing to resolve); otherwise `False` if the primary
    model's revision is unresolved, or if a secondary model
    (`secondary_model_id`) is declared but its own
    `secondary_resolved_revision` is unresolved. A qualification record
    built from an unpinned identity must not default to `is_pinned=True`
    (R1 of the 2026-09-21 re-audit).
    """
    if model_id(provider) is None:
        return True
    if pinned_model_version(provider) is None:
        return False
    secondary_name = getattr(provider, "secondary_model_id", None)
    return not (secondary_name and getattr(provider, "secondary_resolved_revision", None) is None)


def configuration_fingerprint(provider: JudgeProvider) -> str | None:
    """A real, non-fabricated identity signal for whatever tunable state the
    provider exposes as public instance attributes -- e.g. two
    `EmbeddingProvider`s with the same `model_name` but different
    `support_threshold`/`contradiction_threshold` are different evaluators
    and must not collapse to the same calibration class. Derived only from
    the provider's own public (`vars()`, non-underscore, non-callable)
    state, not invented: a provider with no such state (`Mock`/
    `RuleBasedProvider` today) gets `None`, not a fabricated version string.
    """
    public_state = {
        key: value
        for key, value in vars(provider).items()
        if not key.startswith("_") and not callable(value)
    }
    if not public_state:
        return None
    encoded = json.dumps(public_state, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def registry_identity(provider: JudgeProvider) -> tuple[str, str, str]:
    """`(provider_id, pinned_model_id, configuration_version)` -- the exact
    tuple `core.qualification.QualificationRecord` keys on, computed the one
    shared way every other identity-consuming call site in this module
    already does (`full_pinned_model_id`/`configuration_fingerprint`), so a
    revocation registered against a report's identity (G4) and a lookup
    performed against a runtime attempt's identity (G5,
    `storage.registry.RevocationRegistry`) cannot silently diverge -- the
    same class of bug F2/R1 of the 2026-09-21 re-audit found and fixed for
    calibration-class derivation.
    """
    provider_id = provider.name
    base_model_id = model_id(provider)
    pinned_model_id = full_pinned_model_id(provider) or base_model_id or provider_id
    configuration_version = configuration_fingerprint(provider) or DEFAULT_CONFIGURATION_VERSION
    return provider_id, pinned_model_id, configuration_version
