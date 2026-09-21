"""G4's one-command conformance/agreement runner (`evaluator-agreement-harness.md`).

Produces one versioned JSON report for one `(provider_id, pinned_model_id,
configuration_version, dataset_version)` tuple. By default this scores the
*public development* fixture set only
(`tests/fixtures/agreement/dev/*.json`, checked into this repo) -- the
FactLama-held-out set lives outside this repo's tree entirely (see
`evaluator-agreement-harness.md` and `factlama-private/README.md` in the
workspace root) so a provider author cannot memorize an answer key by
reading this file or its fixtures. A report produced against only the dev
set is informative for development but is not by itself sufficient for
`AGREEMENT_REPORTED` (`evaluator-registry.md` requires the held-out set
too); this script does not claim otherwise.

Run with:
    python scripts/run_agreement_harness.py --provider rule-based
    python scripts/run_agreement_harness.py --provider rule-based \\
        --held-out-dir ../factlama-private/agreement-held-out

Passing this script's own exit code is not a qualification decision --
`core.qualification.report_agreement()` is what actually records
`AGREEMENT_REPORTED`, gated by ADR-018's per-label thresholds, and only a
human/process feeding it a report that includes the held-out set should
call it expecting a real qualification outcome.
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
    half of evaluator-agreement-harness.md -- conformance (well-formed
    responses under failure modes) is a separate, prior check this script
    does not repeat (REL-13/G4's conformance suite owns it).
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

    return {
        "report_version": "0.1",
        "provider_id": provider.name,
        "dataset_version": DATASET_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_label": per_label,
        "adversarial": {
            "total": adversarial_total,
            "flipped_to_supported": adversarial_flips,
        },
        "ambiguous_calibration": calibration_verdicts,
        "latency_ms": {"p50": _percentile(0.5), "p95": _percentile(0.95)},
        "adr_018_threshold_failures": threshold_failures,
    }


_PROVIDERS = {
    "mock": MockModelProvider,
    "rule-based": RuleBasedProvider,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(_PROVIDERS), default="rule-based")
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument(
        "--held-out-dir",
        type=Path,
        default=None,
        help="Path to the FactLama-held-out fixture set (outside this repo's tree, "
        "e.g. ../factlama-private/agreement-held-out). Omit to score the dev set only.",
    )
    args = parser.parse_args(argv)

    provider = _PROVIDERS[args.provider]()
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
