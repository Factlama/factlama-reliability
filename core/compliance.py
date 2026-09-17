"""Baseline provider-compliance gate (G3 scope).

A static comparison between a policy's declared requirements and the
dispatched provider's own declared compliance tags -- not a lookup against
G10's evaluator registry, which does not exist yet (see EXECUTION_PLAN.md's
gate-mapping note: "a baseline NO_COMPLIANT_PROVIDER check belongs here too,
filtering the request's configured/allowed provider set against
tenant-declared compliance attributes"). Checked once per request, before any
claim is dispatched -- the provider choice itself doesn't vary per claim.
"""


def is_provider_compliant(provider_tags: frozenset[str], required_tags: list[str]) -> bool:
    """Whether a provider's declared tags satisfy every tag a policy requires."""
    return set(required_tags).issubset(provider_tags)
