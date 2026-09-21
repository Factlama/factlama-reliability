"""G4's one-command agreement runner (`evaluator-agreement-harness.md`).

Produces one versioned JSON report for one `(provider_id, pinned_model_id,
configuration_version, dataset_version)` tuple. This is the *agreement*
half only, against real `JudgeProvider` adapters -- REL-13's conformance
fixture suite (well-formed responses under timeout/rate-limit/malformed-
response) does not exist in this codebase yet (REL-13 is NOT_STARTED), so
this script cannot run it and its report says so explicitly
(`conformance_checked: false`) rather than implying it happened.

By default this scores the *public development* fixture set only
(`tests/fixtures/agreement/dev/*.json`, checked into this repo) -- the
FactLama-held-out set lives outside this repo's tree entirely (see
`evaluator-agreement-harness.md` and `factlama-private/README.md` in the
workspace root) so a provider author cannot memorize an answer key by
reading this file or its fixtures. A report produced against only the dev
set is informative for development but is not by itself sufficient for
`AGREEMENT_REPORTED` (`evaluator-registry.md` requires the held-out set
too; `core.qualification.report_agreement()` refuses a report with no
`held_out_hash`/`held_out_fixture_count`); this script does not claim
otherwise.

Run with:
    python scripts/run_agreement_harness.py --provider rule-based
    python scripts/run_agreement_harness.py --provider embedding \\
        --held-out-dir ../factlama-private/agreement-held-out
    python scripts/run_agreement_harness.py --provider nli \\
        --held-out-dir ../factlama-private/agreement-held-out

`--provider embedding`/`--provider nli` need their vendor extras installed
(`pip install -e '.[embeddings,nli]'`); `mock`/`rule-based` need nothing
beyond this repo, and importing this module never pulls in
sentence-transformers/transformers/torch unless one of the vendor
providers is actually selected.

Passing this script's own exit code is not a qualification decision --
`core.qualification.report_agreement()` is what actually records
`AGREEMENT_REPORTED`, gated by ADR-018's per-label thresholds plus
held-out inclusion and leakage attestation, and only a human/process
feeding it a report that includes the held-out set and a real attestation
should call it expecting a real qualification outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.qualification import evaluate_agreement_thresholds
from judges.port import CancellationToken, JudgeProvider, JudgeRequest
from judges.providers import MockModelProvider, RuleBasedProvider
from schemas.claims import Claim, ClaimVerdict
from schemas.evidence import Evidence

DATASET_VERSION = "harness-0.1"
DEFAULT_DEADLINE_SECONDS = 5.0
DEV_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "agreement" / "dev"
)

#: Families whose expected_verdict is a graded answer, per
#: evaluator-agreement-harness.md. "ambiguous" is deliberately excluded --
#: reasonable evaluators may disagree, so it is reported for calibration
#: only, never scored into precision/recall/F1.
_SCORED_LABELS = (
    ClaimVerdict.SUPPORTED,
    ClaimVerdict.CONTRADICTED,
    ClaimVerdict.UNSUPPORTED,
    ClaimVerdict.INSUFFICIENT_EVIDENCE,
)
_CALIBRATION_ONLY_FAMILY = "ambiguous"


@dataclass(frozen=True)
class AgreementFixture:
    fixture_id: str
    family: str
    claim: Claim
    evidence: list[Evidence]
    expected_verdict: ClaimVerdict


def _load_fixtures_from_dir(directory: Path) -> list[AgreementFixture]:
    """Load every `*.json` file in `directory` as a list of fixture dicts
    (one file per family, per tests/fixtures/agreement/dev/README.md), in a
    stable sorted-by-filename order so a report's fixture ordering (and
    therefore latency-percentile ordering) is reproducible."""
    fixtures: list[AgreementFixture] = []
    for path in sorted(directory.glob("*.json")):
        entries = json.loads(path.read_text())
        for entry in entries:
            fixtures.append(
                AgreementFixture(
                    fixture_id=entry["fixture_id"],
                    family=entry["family"],
                    claim=Claim(**entry["claim"]),
                    evidence=[Evidence(**e) for e in entry["evidence"]],
                    expected_verdict=ClaimVerdict(entry["expected_verdict"]),
                )
            )
    return fixtures


def dev_fixtures() -> list[AgreementFixture]:
    """The public development set: `tests/fixtures/agreement/dev/*.json`,
    one hand-authored file per fixture family (see that directory's own
    README for authorship/review provenance). 24 fixtures across 8
    families -- a real, usable dev set, not a substitute for the
    FactLama-held-out set `report_agreement()` also requires."""
    return _load_fixtures_from_dir(DEV_FIXTURES_DIR)


def held_out_fixtures(directory: Path) -> list[AgreementFixture]:
    """Load the FactLama-held-out set from a directory outside this repo's
    tree (see `factlama-private/README.md`). Same file shape as the dev
    set; `directory` typically points at `agreement-held-out/` containing
    `held_out.json`."""
    return _load_fixtures_from_dir(directory)


def _held_out_content_hash(directory: Path) -> str:
    """sha256 over every `*.json` file's bytes in `directory`, sorted by
    filename, so a later silent edit to the held-out set is detectable
    against a report that cites an earlier hash
    (evaluator-agreement-harness.md)."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def score_provider(
    provider: JudgeProvider,
    fixtures: list[AgreementFixture],
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> dict:
    """Run every fixture once and compute per-label precision/recall/F1,
    the adversarial-flip count, a calibration-only breakdown for the
    `ambiguous` family, and latency percentiles. This is the "agreement"
    half of evaluator-agreement-harness.md only -- conformance (well-formed
    responses under timeout/rate-limit/malformed-response) is not run here;
    REL-13, which would own that suite, is NOT_STARTED in this codebase.
    The returned report's `conformance_checked` is `False` for this reason,
    not a placeholder.
    """
    counts: dict[str, dict[str, int]] = {label.value: _empty_counts() for label in _SCORED_LABELS}
    adversarial_total = 0
    adversarial_flips = 0
    latencies_ms: list[float] = []
    calibration_verdicts: dict[str, int] = {}

    for fixture in fixtures:
        deadline = time.monotonic() + deadline_seconds
        started = time.monotonic()
        result = provider.evaluate(
            JudgeRequest(claim=fixture.claim, evidence=fixture.evidence),
            deadline,
            CancellationToken(),
        )
        latencies_ms.append((time.monotonic() - started) * 1000)

        actual = result.verdict if result.error is None else None

        if fixture.family == "adversarial_injection":
            adversarial_total += 1
            if actual == ClaimVerdict.SUPPORTED:
                adversarial_flips += 1

        if fixture.family == _CALIBRATION_ONLY_FAMILY:
            key = actual.value if actual is not None else "ERROR"
            calibration_verdicts[key] = calibration_verdicts.get(key, 0) + 1
            continue

        expected_key = fixture.expected_verdict.value
        if actual == fixture.expected_verdict:
            counts[expected_key]["tp"] += 1
        else:
            counts[expected_key]["fn"] += 1
            if actual is not None and actual.value in counts:
                counts[actual.value]["fp"] += 1

    per_label = {}
    for label, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": tp + fn,
        }

    latencies_ms.sort()

    def _percentile(p: float) -> float | None:
        if not latencies_ms:
            return None
        idx = min(len(latencies_ms) - 1, round(p * (len(latencies_ms) - 1)))
        return latencies_ms[idx]

    threshold_failures = evaluate_agreement_thresholds(per_label)

    # `.name` is the base model-identity signal a JudgeProvider currently
    # exposes (CONTRACTS.md's port has no separate pinned_model_id/
    # configuration_version property). Vendor adapters format it as
    # "<kind>:<model_name>" (judges/vendor_adapters.py), so the part after
    # the colon is a real, non-fabricated pinned_model_id; providers with no
    # colon (Mock, RuleBased) have no underlying pinned model, so it is
    # honestly None rather than guessed.
    pinned_model_id = provider.name.split(":", 1)[1] if ":" in provider.name else None
    # `model_name` alone may be a floating ref (a branch/tag), not an
    # immutable pin -- append the resolved commit hash when the adapter can
    # report one (EmbeddingProvider/NLIProvider expose `resolved_revision`
    # after loading; getattr() so providers without the attribute, or that
    # can't resolve one, are unaffected rather than erroring).
    resolved_revision = getattr(provider, "resolved_revision", None)
    if pinned_model_id is not None and resolved_revision:
        pinned_model_id = f"{pinned_model_id}@{resolved_revision}"

    return {
        "report_version": "0.1",
        "provider_id": provider.name,
        "pinned_model_id": pinned_model_id,
        "configuration_version": _configuration_fingerprint(provider),
        # `core.scoring.derive_calibration_class()` needs a `qualification_status`
        # as one of its inputs -- this report is itself the evidence that
        # feeds `report_agreement()`'s qualification decision, so at the
        # point this report is generated the qualification_status it would
        # need does not exist yet. Computing it here would mean guessing
        # that input rather than reporting a real one; None is correct, not
        # a placeholder for "not implemented yet."
        "calibration_class": None,
        "dataset_version": DATASET_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # REL-13's conformance fixture suite (success/ambiguity/timeout/
        # rate-limit/malformed-response) does not exist in this codebase yet
        # (REL-13 is NOT_STARTED) -- this script runs the agreement half
        # only and must not imply conformance was checked.
        "conformance_checked": False,
        "per_label": per_label,
        "adversarial": {
            "total": adversarial_total,
            "flipped_to_supported": adversarial_flips,
        },
        "ambiguous_calibration": calibration_verdicts,
        "latency_ms": {"p50": _percentile(0.5), "p95": _percentile(0.95)},
        # No fixture provider reports usage today (REL-07's usage-plumbing
        # note in factlama-reliability/docs/implementation.md) -- UNAVAILABLE
        # is honest; it is not silently omitted or defaulted to zero cost.
        "usage": {"total_tokens": None, "cost": {"status": "UNAVAILABLE"}},
        "adr_018_threshold_failures": threshold_failures,
    }


def _configuration_fingerprint(provider: JudgeProvider) -> str | None:
    """A real, non-fabricated identity signal for whatever tunable state the
    provider exposes as public instance attributes -- e.g. two
    `EmbeddingProvider`s with the same `model_name` but different
    `support_threshold`/`contradiction_threshold` are different evaluators
    and must not produce reports with an identical `pinned_model_id` and no
    other distinguishing field. Derived only from the provider's own public
    (`vars()`, non-underscore, non-callable) state, not invented: `Mock`/
    `RuleBasedProvider` carry no such state today and get `None`, not a
    fabricated version string.
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


def _embedding_provider() -> JudgeProvider:
    from judges.vendor_adapters import EmbeddingProvider

    return EmbeddingProvider()


def _nli_provider() -> JudgeProvider:
    from judges.vendor_adapters import NLIProvider

    return NLIProvider()


#: Factories, not instances -- embedding/nli are only imported (pulling in
#: sentence-transformers/transformers/torch) when actually selected, so
#: `--provider mock` or `--provider rule-based` never requires those extras.
_PROVIDER_FACTORIES = {
    "mock": MockModelProvider,
    "rule-based": RuleBasedProvider,
    "embedding": _embedding_provider,
    "nli": _nli_provider,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(_PROVIDER_FACTORIES), default="rule-based")
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument(
        "--held-out-dir",
        type=Path,
        default=None,
        help="Path to the FactLama-held-out fixture set (outside this repo's tree, "
        "e.g. ../factlama-private/agreement-held-out). Omit to score the dev set only.",
    )
    args = parser.parse_args(argv)

    try:
        provider = _PROVIDER_FACTORIES[args.provider]()
    except ImportError as exc:
        print(
            f"--provider {args.provider} needs its vendor extras installed "
            f"(pip install -e '.[embeddings,nli]'): {exc}",
            file=sys.stderr,
        )
        return 1

    fixtures = dev_fixtures()
    report = score_provider(provider, fixtures)
    report["fixture_sources"] = {"dev": len(fixtures), "held_out": 0}

    if args.held_out_dir is not None:
        ho_fixtures = held_out_fixtures(args.held_out_dir)
        combined = score_provider(provider, fixtures + ho_fixtures)
        combined["fixture_sources"] = {"dev": len(fixtures), "held_out": len(ho_fixtures)}
        combined["held_out_hash"] = _held_out_content_hash(args.held_out_dir)
        report = combined

    json.dump(report, args.out, indent=2)
    args.out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
