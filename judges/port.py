"""Judge provider port: the vendor-neutral JudgeProvider abstraction.

`JudgeProvider.evaluate()` is the bounded port from CONTRACTS.md: it takes
exactly one claim and bounded evidence, a deadline and a cancellation token,
and returns one normalized verdict or a typed error. Claim segmentation
(`extract_claims`) is deliberately not part of this interface -- that is
`core.claims`'s job, not a judge's.

This module must never import a vendor model SDK (ADR-004): reference
implementations with no vendor dependency live in `judges.providers`,
and vendor-coupled adapters live in `judges.vendor_adapters`. Both import
from here, never the other way around -- import-linter enforces that
`schemas`/`core` never transitively reach a vendor SDK through this port.
"""

import re
import time
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.claims import Claim, ClaimVerdict, RationaleCode
from schemas.evidence import Evidence
from schemas.instruction import Instruction
from schemas.policy import Policy
from schemas.verification import Usage


class JudgeErrorCode(str, Enum):
    """Typed provider/dispatch error taxonomy (CONTRACTS.md).

    BUDGET_EXHAUSTED and NO_COMPLIANT_PROVIDER are pre-dispatch gate outcomes
    (checked once per request by `core.budgets`/`core.compliance`, not per
    claim) rather than a live provider's own failure mode; they are part of
    this enum because CONTRACTS.md's provider/dispatch error vocabulary
    includes them and `Attempt.error` stores a value from this same set.
    REVOKED is not produced by this pipeline yet -- it needs G10's registry.
    """

    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CANCELLED = "CANCELLED"
    CONFIGURATION = "CONFIGURATION"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_COMPLIANT_PROVIDER = "NO_COMPLIANT_PROVIDER"
    REVOKED = "REVOKED"


class CancellationToken:
    """A minimal cooperative cancellation flag for a single evaluation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class JudgeError(BaseModel):
    """A typed failure from a judge provider. Never a fabricated factual verdict."""

    model_config = ConfigDict(frozen=True)

    code: JudgeErrorCode
    message: str


class JudgeRequest(BaseModel):
    """One claim plus bounded evidence, submitted to a JudgeProvider."""

    model_config = ConfigDict(frozen=True)

    claim: Claim
    evidence: list[Evidence] = Field(default_factory=list)
    configuration_version: str = "0.1"


class JudgeResult(BaseModel):
    """A normalized judge outcome: either a verdict, or a typed error, never both."""

    model_config = ConfigDict(frozen=True)

    verdict: ClaimVerdict | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    rationale_code: RationaleCode | None = None
    reason: str | None = None
    usage: Usage | None = None
    error: JudgeError | None = None

    @model_validator(mode="after")
    def validate_verdict_xor_error(self) -> "JudgeResult":
        """Enforce the port's stated invariant: a verdict, or an error, never both, never neither."""
        if (self.verdict is None) == (self.error is None):
            raise ValueError("JudgeResult must set exactly one of verdict or error")
        return self


def validate_judge_result(result: JudgeResult, request: JudgeRequest) -> JudgeResult:
    """Downgrade a malformed judge response to a typed error before it reaches core.

    CONTRACTS.md: "SUPPORTED without a cited evidence ID is INVALID_RESPONSE." A
    provider must not be trusted to enforce this itself, nor to cite only
    evidence IDs that were actually part of its request -- a citation to an ID
    the judge was never given is not a real citation and must not be trusted
    any more than citing nothing at all. Both are checked once, here, for
    every provider's output.
    """
    if result.error is not None or result.verdict != ClaimVerdict.SUPPORTED:
        return result

    valid_evidence_ids = {e.evidence_id for e in request.evidence}
    if not result.evidence_ids or any(eid not in valid_evidence_ids for eid in result.evidence_ids):
        return JudgeResult(
            error=JudgeError(
                code=JudgeErrorCode.INVALID_RESPONSE,
                message="SUPPORTED verdict cites no evidence ID, or an evidence ID absent from the request",
            )
        )
    return result


_CITATION_OVERLAP_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "and",
    "or",
    "by",
    "with",
    "that",
    "this",
    "it",
    "as",
    "be",
    "for",
    "from",
    "its",
    "has",
    "have",
    "had",
    "not",
}


#: F7 of the 2026-09-21 G0-G4 validation report: a *purely* lexical overlap
#: check downgrades any claim/evidence pair that shares zero raw tokens, even
#: when they are a legitimate paraphrase (e.g. "The physician purchased an
#: automobile." vs. "The doctor bought a car." -- zero shared tokens, but the
#: same claim). CONTRACTS.md is explicit that this is not allowed: "legitimate
#: paraphrase is not rejected solely for lacking shared words." Each group
#: below is a small, hand-curated, versioned set of near-synonyms mapped to a
#: shared canonical token before overlap is computed -- deliberately
#: non-exhaustive: this guard's job is catching evidence that shares
#: *nothing at all* with the claim, not proving semantic equivalence (that
#: remains the judge's job, not this deterministic guard's). Additive: a new
#: group requires a version note here, the same convention CONTRACTS.md uses
#: for its own versioned vocabularies.
_CITATION_OVERLAP_SYNONYMS_V1: tuple[frozenset[str], ...] = (
    frozenset({"physician", "physicians", "doctor", "doctors", "gp"}),
    frozenset({"automobile", "automobiles", "car", "cars", "vehicle", "vehicles"}),
    frozenset(
        {"purchase", "purchased", "purchases", "buy", "buys", "bought", "acquire", "acquired"}
    ),
    frozenset({"company", "companies", "firm", "firms", "corporation", "corporations", "business"}),
    frozenset(
        {"began", "begin", "start", "started", "commence", "commenced", "founded", "founding"}
    ),
    frozenset({"ceo", "chief", "executive"}),
    frozenset({"employee", "employees", "staff", "worker", "workers"}),
    frozenset({"large", "big", "sizable", "substantial"}),
    frozenset({"small", "tiny", "minor"}),
    frozenset({"increase", "increased", "rise", "rose", "grew", "grow", "growth"}),
    frozenset({"decrease", "decreased", "fall", "fell", "declined", "decline", "drop", "dropped"}),
)

_CITATION_OVERLAP_SYNONYM_CANONICAL: dict[str, str] = {
    word: min(group) for group in _CITATION_OVERLAP_SYNONYMS_V1 for word in group
}


def _content_words(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with common stopwords removed, then
    canonicalized through `_CITATION_OVERLAP_SYNONYMS_V1` so a recognized
    near-synonym pair counts as overlap."""
    tokens = {
        w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _CITATION_OVERLAP_STOPWORDS
    }
    return {_CITATION_OVERLAP_SYNONYM_CANONICAL.get(w, w) for w in tokens}


def apply_citation_support_check(
    result: JudgeResult,
    request: JudgeRequest,
) -> tuple[JudgeResult, bool]:
    """Run CONTRACTS.md's conservative, non-model overlap/support check on a
    SUPPORTED verdict's cited evidence.

    This runs after `validate_judge_result()` has already rejected a
    nonexistent evidence ID as INVALID_RESPONSE; here every cited ID is real,
    but a judge can still confidently cite genuine evidence that has nothing
    to do with the claim. Deliberately weak: it only downgrades when a claim
    and every one of its cited evidence texts share *zero* non-trivial
    content words -- a clearly fabricated citation, not merely a paraphrase.
    "Lexical overlap alone is never proof, and legitimate paraphrase is not
    rejected solely for lacking shared words" (CONTRACTS.md), so any nonzero
    overlap passes unchanged.

    Returns the (possibly downgraded) result and whether it downgraded.
    """
    if result.error is not None or result.verdict != ClaimVerdict.SUPPORTED:
        return result, False

    claim_words = _content_words(request.claim.text)
    if not claim_words:
        return result, False

    evidence_by_id = {e.evidence_id: e for e in request.evidence}
    has_support = any(
        claim_words & _content_words(evidence_by_id[eid].content or "")
        for eid in result.evidence_ids
        if eid in evidence_by_id
    )
    if has_support:
        return result, False

    downgraded = result.model_copy(
        update={
            "verdict": ClaimVerdict.INSUFFICIENT_EVIDENCE,
            "rationale_code": RationaleCode.CITATION_UNSUPPORTED,
            "reason": "Cited evidence shares no content with the claim (conservative overlap check failed)",
        }
    )
    return downgraded, True


_INJECTION_PATTERNS = [
    re.compile(
        r"\bignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE
    ),
    re.compile(r"\bdisregard\s+(the\s+)?(claim|verdict|rationale|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bmark\s+(this|the)\s+claim\s+as\s+(supported|true|verified)\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(verdict\s*[:=]?\s*)?supported\b", re.IGNORECASE),
    re.compile(
        r"\byou\s+(are|must)\s+(now\s+)?(a|acting as|the)\s+(judge|verifier|evaluator)\b",
        re.IGNORECASE,
    ),
]


def detect_evidence_injection(result: JudgeResult, request: JudgeRequest) -> list[str]:
    """Flag cited evidence containing judge-directive-style language.

    This is defense-in-depth, not the primary guard: `validate_judge_result()`
    already rejects a SUPPORTED verdict citing no real evidence ID, and
    `apply_citation_support_check()` already downgrades one whose cited
    evidence shares no content with the claim. A judge that ignored an
    injection attempt entirely (the expected case, since evidence is passed
    as untrusted data, never concatenated into instructions -- CONTRACTS.md)
    still produces a correct verdict; this only surfaces that an attempt was
    present in evidence the judge actually cited, for CONTRACTS.md's reserved
    `EVIDENCE_INJECTION_SUSPECTED` violation code. Returns the cited evidence
    IDs that matched, or an empty list.
    """
    if result.error is not None or not result.evidence_ids:
        return []

    evidence_by_id = {e.evidence_id: e for e in request.evidence}
    hits: list[str] = []
    for eid in result.evidence_ids:
        evidence = evidence_by_id.get(eid)
        if evidence is None or not evidence.content:
            continue
        if any(pattern.search(evidence.content) for pattern in _INJECTION_PATTERNS):
            hits.append(eid)
    return hits


def bounded_check(deadline: float, cancellation: CancellationToken) -> JudgeError | None:
    """Shared pre-dispatch deadline/cancellation check for every provider."""
    if cancellation.is_cancelled:
        return JudgeError(code=JudgeErrorCode.CANCELLED, message="cancelled before dispatch")
    if time.monotonic() > deadline:
        return JudgeError(code=JudgeErrorCode.TIMEOUT, message="deadline exceeded before dispatch")
    return None


class JudgeProvider(ABC):
    """Abstract base class for judge providers behind the bounded JudgeProvider port.

    Providers may include enterprise/internal models, OpenAI, Anthropic,
    Azure, local open models and eventually the FactLama SLM. Reliability
    core must not branch on provider vendor.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    def evaluate(
        self,
        request: JudgeRequest,
        deadline: float,
        cancellation: CancellationToken,
    ) -> JudgeResult:
        """Evaluate exactly one claim against bounded evidence.

        Args:
            request: The claim and evidence to evaluate.
            deadline: A `time.monotonic()`-based absolute deadline.
            cancellation: Cooperative cancellation for this evaluation.

        Returns:
            JudgeResult with a normalized verdict, or a typed error.
        """
        ...

    def evaluate_instruction(
        self,
        answer: str,
        instruction: Instruction,
    ) -> tuple[float, str | None]:
        """Evaluate how well the answer follows an instruction.

        Not part of the formal JudgeProvider port (CONTRACTS.md's port is
        claim-evaluation only) -- an optional extra capability some
        providers offer. Defaults to assumed compliance.
        """
        return 1.0, None

    def evaluate_scope(
        self,
        answer: str,
        policy: Policy,
    ) -> tuple[float, list[str]]:
        """Evaluate scope compliance.

        Not part of the formal JudgeProvider port -- an optional extra
        capability. Defaults to no breach.
        """
        return 0.0, []

    @property
    def compliance_tags(self) -> frozenset[str]:
        """Static compliance attributes this provider declares.

        A G3-scoped baseline (`core.compliance.is_provider_compliant`), not a
        lookup against G10's evaluator registry -- there is no tenant
        approval/audit trail yet, only a static tag comparison against
        whatever a policy declares it requires. Default: no attributes
        declared, so any policy with at least one `required_provider_compliance`
        tag rejects this provider until it overrides this property.
        """
        return frozenset()
