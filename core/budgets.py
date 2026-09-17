"""Pre-dispatch request-scoped budgets (ADR-012).

Bounds one synchronous request's own claim/evidence fan-out before any judge
is dispatched. Deliberately request-scoped only: a durable per-tenant-per-
window token/cost ceiling across retries and fallback belongs to G5, once
retries/fallback exist to spend against -- see EXECUTION_PLAN.md's gate-
mapping note on REL-04/G3 vs G5. Token/cost ceilings for a single request are
not enforced here yet either: no provider in this repo reports usage ahead of
dispatch to check against (ADR-012's cost/latency plumbing is `Usage`, read
only after a call completes).
"""

MAX_CLAIMS_PER_REQUEST = 50
MAX_EVIDENCE_PER_CLAIM = 20


def check_request_budget(
    claim_count: int,
    evidence_count: int,
    max_claims: int = MAX_CLAIMS_PER_REQUEST,
    max_evidence: int = MAX_EVIDENCE_PER_CLAIM,
) -> str | None:
    """Return a violation reason if the request exceeds its pre-dispatch
    budget, else None. Checked once per request, before any claim is
    dispatched to a judge -- not per claim.
    """
    if claim_count > max_claims:
        return f"{claim_count} claims exceeds MAX_CLAIMS_PER_REQUEST={max_claims}"
    if evidence_count > max_evidence:
        return f"{evidence_count} evidence items exceeds MAX_EVIDENCE_PER_CLAIM={max_evidence}"
    return None
