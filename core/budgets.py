"""Pre-dispatch and running request-scoped budgets (ADR-012).

Bounds one synchronous request's own claim/evidence fan-out before any judge
is dispatched, and bounds its cumulative judge usage/cost as attempts
complete. Deliberately request-scoped only: a durable per-tenant-per-window
token/cost ceiling across retries and fallback belongs to G5, once
retries/fallback exist to spend against -- see EXECUTION_PLAN.md's gate-
mapping note on REL-04/G3 vs G5.

Token/cost ceilings cannot be checked *before* a judge call the way the
claim/evidence fan-out caps are: no provider in this repo reports usage
ahead of dispatch (ADR-012's `Usage` is read only after a call completes).
Two checks cover this together: `check_usage_reservation` runs before each
dispatch and reserves a conservative worst-case-per-call ceiling against the
running total, so a call is never even started once the reservation alone
would exceed budget; `check_usage_budget` then runs after every attempt,
against the real running total, as a circuit breaker that stops dispatching
further claims once the request's own spend is already over budget. Neither
is a true pre-dispatch gate in the claim/evidence sense: the reservation is
a placeholder bound, not a real per-call estimate, and cannot stop an
unusually large first call (there is no usage-so-far yet to reserve
against) or one that itself blows past the reservation -- doing that needs
a real per-call estimate and a `JudgeRequest` field a provider honors as an
output/token cap, both gated on a metered adapter existing to bound
(REL-13). No current T0 adapter (Mock, RuleBased, Embedding, NLI) reports a
non-default `Usage`, so neither ceiling is reachable with today's adapters;
they exist so a metered adapter (REL-13) is bounded from the day it is
added, not retrofitted.
"""

from schemas.verification import Attempt, Cost, Usage

MAX_CLAIMS_PER_REQUEST = 50
MAX_EVIDENCE_PER_CLAIM = 20

MAX_TOKENS_PER_REQUEST = 200_000
MAX_COST_PER_REQUEST_USD = 5.0

# The only currency `MAX_COST_PER_REQUEST_USD` is denominated in. There is no
# documented FX conversion table in this repo (ADR-012: "never sum
# currencies" -- extended here to never *compare* across them either), so a
# MEASURED cost reported in any other currency cannot be checked against this
# ceiling at all; it is treated the same as an UNAVAILABLE cost for that
# purpose, never compared, never silently coerced into dollars.
ENFORCED_COST_CURRENCY = "USD"

# Worst-case reservation for one not-yet-dispatched judge call, used as a
# pre-dispatch ceiling (see `check_usage_reservation`). No T0 adapter in this
# repo reports usage ahead of a call, so there is no real per-call estimate
# to reserve -- these are conservative placeholders, not a measured bound.
MAX_TOKENS_PER_JUDGE_CALL = 20_000
MAX_COST_PER_JUDGE_CALL_USD = 0.50


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


def check_usage_budget(
    usage: Usage,
    max_tokens: int = MAX_TOKENS_PER_REQUEST,
    max_cost_usd: float = MAX_COST_PER_REQUEST_USD,
) -> str | None:
    """Return a violation reason if a request's usage-so-far exceeds its
    per-request ceiling, else None.

    `usage` is the running summary (`summarize_usage`) of every attempt made
    in this request so far. Cost is only checked when it is `MEASURED` in
    `ENFORCED_COST_CURRENCY` -- an `UNAVAILABLE` cost is never treated as
    zero, so it cannot be compared against a ceiling (ADR-012: "unknown
    price is UNAVAILABLE not zero"), and a cost measured in a different
    currency is exactly as uncomparable: with no documented conversion,
    comparing its raw amount to a USD ceiling is not "checking budget," it
    is reading 6 JPY as exceeding a $5.00 limit.
    """
    tokens = (
        usage.total_tokens
        if usage.total_tokens is not None
        else (usage.input_tokens or 0) + (usage.output_tokens or 0)
    )
    if tokens > max_tokens:
        return f"{tokens} tokens exceeds MAX_TOKENS_PER_REQUEST={max_tokens}"
    if (
        usage.cost.status == "MEASURED"
        and usage.cost.amount is not None
        and usage.cost.currency == ENFORCED_COST_CURRENCY
        and usage.cost.amount > max_cost_usd
    ):
        return (
            f"cost {usage.cost.amount:.4f} {usage.cost.currency} exceeds "
            f"MAX_COST_PER_REQUEST_USD={max_cost_usd:.2f}"
        )
    return None


def check_usage_reservation(
    usage_so_far: Usage,
    max_tokens: int = MAX_TOKENS_PER_REQUEST,
    max_cost_usd: float = MAX_COST_PER_REQUEST_USD,
    max_tokens_per_call: int = MAX_TOKENS_PER_JUDGE_CALL,
    max_cost_per_call_usd: float = MAX_COST_PER_JUDGE_CALL_USD,
) -> str | None:
    """Return a violation reason if dispatching one more judge call could
    push this request over its ceiling, else None. Checked BEFORE dispatch,
    against the running usage-so-far summary -- unlike `check_usage_budget`,
    which is checked after.

    No T0 adapter in this repo reports usage ahead of a call (this module's
    top docstring), so there is no real per-call estimate to reserve
    against. This reserves a conservative worst-case-per-call ceiling
    (`max_tokens_per_call`/`max_cost_per_call_usd`) instead: it bounds how
    far a single additional call can be allowed to push the running total
    before it is even dispatched, closing the gap where one huge call could
    otherwise blow past the entire per-request allowance before
    `check_usage_budget` gets a chance to run afterward.

    This is a placeholder ceiling, not a real per-call estimate or an
    actual cap on what a dispatched call can spend -- it cannot stop an
    unusually large *first* call (there is no usage-so-far yet to reserve
    against) or a call that itself blows past the reservation. Closing that
    residual gap needs two things this repo does not have yet: a real
    per-call cost/token estimate (from prompt/evidence size and a
    provider's documented limits) and a `JudgeRequest` field a provider
    honors as an output/token cap -- both are `JudgeProvider` port changes
    gated on a metered T0/T1 adapter existing to bound (REL-13), not
    something to retrofit onto today's unmetered Mock/RuleBased/Embedding/
    NLI adapters.
    """
    tokens_so_far = usage_so_far.total_tokens or 0
    if tokens_so_far + max_tokens_per_call > max_tokens:
        return (
            f"{tokens_so_far} tokens so far + reserved {max_tokens_per_call} for the "
            f"next call exceeds MAX_TOKENS_PER_REQUEST={max_tokens}"
        )
    if (
        usage_so_far.cost.status == "MEASURED"
        and usage_so_far.cost.amount is not None
        and usage_so_far.cost.currency == ENFORCED_COST_CURRENCY
        and usage_so_far.cost.amount + max_cost_per_call_usd > max_cost_usd
    ):
        return (
            f"cost {usage_so_far.cost.amount:.4f} so far + reserved "
            f"{max_cost_per_call_usd:.2f} for the next call exceeds "
            f"MAX_COST_PER_REQUEST_USD={max_cost_usd:.2f}"
        )
    return None


def _sum_optional(values: list[int | float | None]) -> float | None:
    """Sum the values that are actually known, or None if none are.

    `Usage`'s own docstring: "Known usage summed across attempts" -- a
    dimension no attempt reported must stay unset, never default to 0.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _effective_tokens(usage: Usage) -> int | None:
    """One attempt's own normalized token total: its reported `total_tokens`
    when present, else its input+output sum, else unknown (`None`).

    Summing raw `total_tokens` across attempts as one running dimension and
    raw `input_tokens` as a separate one -- the previous approach -- lets
    one attempt's `total_tokens` and a *different* attempt's input-only
    report pass as two independent, non-overlapping counts, when they are
    really two incompatible views of the same per-attempt quantity. An
    attempt reporting `total_tokens=10_000` and another reporting only
    `input_tokens=200_000` must contribute at least 210,000 to the running
    total, not 10,000 -- normalizing per attempt, before summing across
    attempts, is the only way to get that right.
    """
    if usage.total_tokens is not None:
        return usage.total_tokens
    if usage.input_tokens is not None or usage.output_tokens is not None:
        return (usage.input_tokens or 0) + (usage.output_tokens or 0)
    return None


def _usd_measured_costs(attempts: list[Attempt]) -> list[Cost]:
    """Every attempt's cost that is `MEASURED` in `ENFORCED_COST_CURRENCY`,
    tracked independent of what any other attempt's cost is or isn't.

    This is deliberately currency-*specific*, not "any single currency the
    measured costs happen to share": `MAX_COST_PER_REQUEST_USD` can only
    ever be enforced in USD, so a request that spent 1 JPY and then $6 must
    still show its known $6 -- a differently-priced attempt sitting
    alongside it must not blank out or dilute that known USD total any
    more than an UNAVAILABLE one would.
    """
    return [
        a.usage.cost
        for a in attempts
        if a.usage.cost.status == "MEASURED"
        and a.usage.cost.amount is not None
        and a.usage.cost.currency == ENFORCED_COST_CURRENCY
    ]


def summarize_usage(attempts: list[Attempt]) -> Usage:
    """Sum known usage across a request's judge attempts (ADR-012).

    Token counts sum whatever attempts actually reported, treating an
    unreported dimension as absent rather than zero; the `total_tokens`
    field sums each attempt's own normalized total (`_effective_tokens`),
    not a same-name-only pooling of whichever attempts happened to report
    `total_tokens` directly.

    Cost sums whatever attempts report a `MEASURED` cost specifically in
    `ENFORCED_COST_CURRENCY` (`_usd_measured_costs`) -- an attempt with no
    cost data, or cost measured in some other currency, contributes nothing
    and is simply excluded, the same as an unreported token dimension; it
    must never erase or dilute the known USD total that *other* attempts
    did report (never a blanket `UNAVAILABLE` just because attempts
    disagree on currency -- that is precisely the bug where a $6 known
    spend hid behind an unrelated 1 JPY attempt). Currencies are still never
    summed together (ADR-012): a non-USD cost simply doesn't count toward
    this total at all, rather than being coerced into it. Call
    `describe_cost_completeness` alongside this to tell a caller whether
    the reported amount is the complete cost or a known partial one.
    """
    input_tokens = _sum_optional([a.usage.input_tokens for a in attempts])
    output_tokens = _sum_optional([a.usage.output_tokens for a in attempts])
    total_tokens = _sum_optional([_effective_tokens(a.usage) for a in attempts])
    latency_ms = _sum_optional([a.usage.latency_ms for a in attempts])

    usd_costs = _usd_measured_costs(attempts)
    if usd_costs:
        cost = Cost(
            status="MEASURED",
            amount=sum(c.amount for c in usd_costs if c.amount is not None),
            currency=ENFORCED_COST_CURRENCY,
            pricing_version=usd_costs[0].pricing_version,
        )
    else:
        cost = Cost(status="UNAVAILABLE")

    return Usage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        cost=cost,
        latency_ms=latency_ms,
    )


def describe_cost_completeness(attempts: list[Attempt]) -> str | None:
    """Human-readable note for when `summarize_usage`'s cost is a known
    partial total, not the complete picture -- e.g. "$2.00 USD known; 1
    call's cost unavailable" -- so a caller never mistakes a known-partial
    amount for the full cost of the request.

    Returns `None` when there is nothing to flag: either no attempt has a
    known USD cost at all (the summary is already plainly `UNAVAILABLE`,
    not a number that could be mistaken for complete), or every attempt's
    cost is already folded into the known USD total with nothing excluded.
    """
    usd_costs = _usd_measured_costs(attempts)
    if not usd_costs:
        return None

    excluded = len(attempts) - len(usd_costs)
    if excluded == 0:
        return None

    unavailable_count = sum(1 for a in attempts if a.usage.cost.status == "UNAVAILABLE")
    other_currency_count = excluded - unavailable_count
    known_amount = sum(c.amount for c in usd_costs if c.amount is not None)

    parts = [f"${known_amount:.2f} {ENFORCED_COST_CURRENCY} known"]
    if unavailable_count == 1:
        parts.append("1 call's cost unavailable")
    elif unavailable_count > 1:
        parts.append(f"{unavailable_count} calls' cost unavailable")
    if other_currency_count == 1:
        parts.append(f"1 call's cost was in a non-{ENFORCED_COST_CURRENCY} currency, excluded")
    elif other_currency_count > 1:
        parts.append(
            f"{other_currency_count} calls' cost was in a non-{ENFORCED_COST_CURRENCY} "
            "currency, excluded"
        )
    return "; ".join(parts)
