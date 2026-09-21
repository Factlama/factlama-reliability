"""G4's end-to-end qualification runner: conformance, then agreement, then
(only with a real recorded human attestation) the registry transition
itself.

F8 of the 2026-09-21 G0-G4 validation report asked for two things: a T0
conformance subset (`core.conformance`) explicitly assigned to G4, and at
least one T0 adapter that actually earns qualification under ADR-018's
thresholds without weakening them. This script demonstrates both end to
end for one `(provider, dataset)` pair:

1. Runs `core.conformance.run_t0_conformance_suite()` against the provider.
   Any failure stops here -- `mark_conformance_passed()` is never called on
   evidence that didn't earn it.
2. Runs the same agreement scoring `scripts/run_agreement_harness.py` does
   (dev set, plus the FactLama-held-out set if `--held-out-dir` is given).
3. Calls `core.qualification.report_agreement()`. This only succeeds --
   moving the record to `AGREEMENT_REPORTED` -- if every ADR-018 threshold
   is met *and* a real leakage-control attestation is on file
   (`--leakage-attestation-file`, see below). Absent that file, this script
   still runs steps 1-2 and reports exactly which ADR-018 thresholds (if
   any) remain unmet -- a real, honest status, not a placeholder.

F9 of the same report: "human label/leakage review remains unperformed."
That review is a human action this script cannot perform or fake on its
own initiative -- it does not set `leakage_attested=True` by default, and
does not accept a bare `--leakage-attested` boolean flag a caller could
pass casually. It only accepts a JSON file recording who attested and when
(`factlama-private/README.md`'s own convention: "a dated note ... do not
treat its mere existence as satisfying that requirement"), e.g.:

    {"attested_by": "<repository owner's name>", "date": "2026-09-21",
     "dataset_version": "harness-0.1", "dev_hash": "<sha256 this run's
     public tests/fixtures/agreement/dev/ actually hashes to>",
     "held_out_hash": "<sha256 this run's --held-out-dir actually hashes
     to>", "dev_fixture_count": 24, "held_out_fixture_count": 16,
     "note": "reviewed factlama-private/agreement-held-out/held_out.json
     for leakage against the public dev set"}

R6 of the 2026-09-21 re-audit: the five identity fields
(`dataset_version`/`dev_hash`/`held_out_hash`/`dev_fixture_count`/
`held_out_fixture_count`) must exactly match the run this file is passed
to -- a genuine attestation from an earlier release is refused, not
silently accepted, if the dataset has since changed. Follow-up re-audit
finding (2026-09-21, "leakage attestation is only partially
dataset-bound"): `dev_hash` closes the gap where public fixture *content*
could be edited without changing `dev_fixture_count`, silently preserving
approval even though leakage review compares both the public and
held-out sets, not just their sizes. Run once without
`--leakage-attestation-file` first to read the actual `dev_hash`/
`held_out_hash` this tool computes (in the JSON report's own `dev_hash`/
`held_out_hash` fields) before a human reviewer writes the attestation
file referencing them.

Run with:
    python scripts/run_qualification.py --provider nli \\
        --held-out-dir ../factlama-private/agreement-held-out
    python scripts/run_qualification.py --provider nli \\
        --held-out-dir ../factlama-private/agreement-held-out \\
        --leakage-attestation-file ../factlama-private/leakage_attestation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from core.conformance import run_t0_conformance_suite
from core.provider_identity import (
    DEFAULT_CONFIGURATION_VERSION,
    configuration_fingerprint,
    full_pinned_model_id,
    is_fully_pinned,
)
from core.qualification import (
    IllegalTransitionError,
    QualificationRecord,
    QualificationState,
    mark_conformance_passed,
    report_agreement,
)
from core.scoring import derive_calibration_class
from core.verifier import EVALUATOR_ID, EVALUATOR_VERSION
from schemas.verification import QualificationStatus
from scripts.run_agreement_harness import (
    _PROVIDER_FACTORIES,
    DATASET_VERSION,
    DEV_FIXTURES_DIR,
    _fixture_dir_content_hash,
    dev_fixtures,
    held_out_fixtures,
    score_provider,
)


def _load_leakage_attestation(
    path: Path | None,
    *,
    dataset_version: str,
    dev_hash: str,
    held_out_hash: str,
    dev_fixture_count: int,
    held_out_fixture_count: int,
) -> tuple[bool, dict[str, object] | None]:
    """`(leakage_attested, attestation_record)`. `leakage_attested` is only
    ever `True` when `path` names a real, readable file whose
    `attested_by`/`date` are non-empty **and** whose `dataset_version`/
    `dev_hash`/`held_out_hash`/`dev_fixture_count`/`held_out_fixture_count`
    fields exactly match the run actually being reported on.

    R6 of the 2026-09-21 re-audit: a genuine attestation from a prior
    release must not silently approve a since-modified dataset -- without
    this check, an old `attested_by`/`date` file with no other content
    would pass this function's earlier version regardless of whether the
    held-out set it once reviewed still matches the one this run just
    scored against. This function never fabricates a match; it only
    compares and refuses on any mismatch.

    Follow-up re-audit finding (2026-09-21): the held-out set was bound by
    content hash, but the public dev set was only bound by fixture
    *count* -- editing existing public fixture content (without adding or
    removing one) left `dev_fixture_count` unchanged, so a stale
    attestation would still match even though leakage review compares
    both sets' actual content, not just the held-out side's. `dev_hash`
    closes that gap the same way `held_out_hash` already closes it for
    the held-out set.

    `attestation_record` (when attested) carries a sha256 of the
    attestation file's own bytes, so the report itself preserves a
    reference to exactly which attestation backed it -- not only printed
    to stderr, where a reader of the JSON report alone would never see it.
    """
    if path is None:
        return False, None
    raw = path.read_text()
    data = json.loads(raw)
    attested_by = data.get("attested_by")
    date = data.get("date")
    if not attested_by or not date:
        raise ValueError(
            f"{path} is missing a non-empty 'attested_by' and/or 'date' field -- "
            "not a valid leakage-control attestation record"
        )

    mismatches = []
    if data.get("dataset_version") != dataset_version:
        mismatches.append(
            f"dataset_version: attestation has {data.get('dataset_version')!r}, "
            f"this run is {dataset_version!r}"
        )
    if data.get("dev_hash") != dev_hash:
        mismatches.append(
            f"dev_hash: attestation has {data.get('dev_hash')!r}, "
            f"this run's public dev set hashes to {dev_hash!r}"
        )
    if data.get("held_out_hash") != held_out_hash:
        mismatches.append(
            f"held_out_hash: attestation has {data.get('held_out_hash')!r}, "
            f"this run's held-out set hashes to {held_out_hash!r}"
        )
    if data.get("dev_fixture_count") != dev_fixture_count:
        mismatches.append(
            f"dev_fixture_count: attestation has {data.get('dev_fixture_count')!r}, "
            f"this run has {dev_fixture_count}"
        )
    if data.get("held_out_fixture_count") != held_out_fixture_count:
        mismatches.append(
            f"held_out_fixture_count: attestation has "
            f"{data.get('held_out_fixture_count')!r}, this run has {held_out_fixture_count}"
        )
    if mismatches:
        raise ValueError(
            f"{path} does not match the dataset this run actually scored -- a stale "
            "attestation cannot approve a since-modified dataset "
            "(evaluator-agreement-harness.md; R6 of the 2026-09-21 re-audit):\n  "
            + "\n  ".join(mismatches)
        )

    return True, {
        "attested_by": attested_by,
        "date": date,
        "file_hash": hashlib.sha256(raw.encode()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(_PROVIDER_FACTORIES), required=True)
    parser.add_argument(
        "--held-out-dir",
        type=Path,
        default=None,
        help="Path to the FactLama-held-out fixture set, e.g. "
        "../factlama-private/agreement-held-out. Required for any real chance "
        "of meeting ADR-018's minimum sample size.",
    )
    parser.add_argument(
        "--leakage-attestation-file",
        type=Path,
        default=None,
        help="Path to a JSON file recording a real human leakage-control review "
        "(attested_by, date, dataset_version, dev_hash, held_out_hash, "
        "dev_fixture_count, held_out_fixture_count -- refused if any "
        "dataset-identity field doesn't match this run). Omit unless that review "
        "has actually happened -- see this script's own module docstring.",
    )
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
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

    # Runs `provider.evaluate()` against real requests, which for a vendor
    # adapter is also what loads its model(s) (R1/R5 of the 2026-09-21
    # re-audit: `NLIProvider._get_model()` now eagerly loads its secondary
    # relatedness model alongside the primary one) -- identity below is
    # therefore built *after* this, never before, so `full_pinned_model_id()`
    # can report a real resolved revision instead of silently falling back
    # to a floating model name.
    conformance_failures = run_t0_conformance_suite(provider)
    if conformance_failures:
        print(f"CONFORMANCE FAILED for {provider.name}:", file=sys.stderr)
        for failure in conformance_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    # R1 of the 2026-09-21 re-audit: this previously used `model_id()`
    # (bare, un-resolved model name) and left `is_pinned` at its `True`
    # default regardless of whether a real revision was ever resolved --
    # `full_pinned_model_id()`/`is_fully_pinned()` are the same shared
    # identity primitives `score_provider()` below now uses too, so record
    # and report can never silently disagree on identity again.
    record = QualificationRecord(
        provider_id=provider.name,
        pinned_model_id=full_pinned_model_id(provider) or provider.name,
        configuration_version=configuration_fingerprint(provider) or DEFAULT_CONFIGURATION_VERSION,
        is_pinned=is_fully_pinned(provider),
    )
    record = mark_conformance_passed(record)

    # Bound the same way as held_out_hash below (R6 of the 2026-09-21
    # re-audit's follow-up finding): editing existing public fixture
    # *content* without changing fixture count must still invalidate a
    # stale attestation, since leakage review compares both sets' actual
    # content, not just the held-out side's.
    dev_hash = _fixture_dir_content_hash(DEV_FIXTURES_DIR)
    fixtures = dev_fixtures()
    held_out_hash = ""
    held_out_count = 0
    if args.held_out_dir is not None:
        ho_fixtures = held_out_fixtures(args.held_out_dir)
        fixtures = fixtures + ho_fixtures
        held_out_hash = _fixture_dir_content_hash(args.held_out_dir)
        held_out_count = len(ho_fixtures)

    # R1: no longer overwritten with the record's fields below -- both are
    # now derived from the same `full_pinned_model_id()`/
    # `configuration_fingerprint()` calls, so there is nothing left to
    # reconcile. (`record.configuration_version` can differ from
    # `report["configuration_version"]` in exactly one case: a provider
    # with no public tunable state, where the dataclass field's `str` type
    # forces `DEFAULT_CONFIGURATION_VERSION` as a fallback but the JSON
    # report is free to stay `null` -- not a re-introduction of R1's bug,
    # a type constraint the report itself isn't under.)
    report = score_provider(provider, fixtures)
    # R1: this report's own `conformance_checked` field is `score_provider()`'s
    # (REL-13's full ecosystem-tier suite, still NOT_STARTED -- correctly
    # `false`). This run's *own* T0 conformance result is a different,
    # narrower thing and must not be conflated with that field; state it
    # explicitly instead of leaving a bare `false` next to a
    # `CONFORMANCE_PASSED` registry state with no explanation in the JSON
    # itself.
    report["t0_conformance"] = {"checked": True, "failures": conformance_failures}

    leakage_attested, attestation_record = _load_leakage_attestation(
        args.leakage_attestation_file,
        dataset_version=DATASET_VERSION,
        dev_hash=dev_hash,
        held_out_hash=held_out_hash,
        dev_fixture_count=len(dev_fixtures()),
        held_out_fixture_count=held_out_count,
    )

    try:
        record = report_agreement(
            record,
            dataset_version=DATASET_VERSION,
            adversarial_flips=report["adversarial"]["flipped_to_supported"],
            per_label=report["per_label"],
            held_out_hash=held_out_hash,
            held_out_fixture_count=held_out_count,
            leakage_attested=leakage_attested,
        )
    except IllegalTransitionError as exc:
        print(f"unexpected: {exc}", file=sys.stderr)
        return 1

    print(f"provider: {provider.name}", file=sys.stderr)
    print(f"pinned_model_id: {record.pinned_model_id}", file=sys.stderr)
    print(f"is_pinned: {record.is_pinned}", file=sys.stderr)
    print("conformance: PASSED (0 failures)", file=sys.stderr)
    attestation_summary = (
        f"{attestation_record['attested_by']} on {attestation_record['date']}"
        if attestation_record
        else "NOT PROVIDED"
    )
    print(f"leakage attestation: {attestation_summary}", file=sys.stderr)
    print(f"registry state: {record.state.value}", file=sys.stderr)
    if record.threshold_failures:
        print("agreement/registry blockers:", file=sys.stderr)
        for failure in record.threshold_failures:
            print(f"  - {failure}", file=sys.stderr)
    else:
        print("no agreement/registry blockers.", file=sys.stderr)

    # R7 of the 2026-09-21 re-audit / F9's own "reconcile the chosen
    # lifecycle with the harness specification" ask: `score_provider()`'s
    # `calibration_class` is always `null` (it has no `qualification_status`
    # to hash -- see that function's own comment on why). This report,
    # unlike that one, knows the actual resulting registry state, so it can
    # compute a real, non-null, addressable calibration class -- ADR-010's
    # own definition ("QUALIFIED only after a published agreement report")
    # makes this mapping direct, not invented: `AGREEMENT_REPORTED` (or
    # later) is `QUALIFIED`, anything earlier is `UNQUALIFIED`. Before a
    # real leakage attestation exists, every record this script can
    # produce stays at `CONFORMANCE_PASSED`, so this is honestly
    # `UNQUALIFIED` today -- F9's "a current UNQUALIFIED class can exist"
    # option, not a placeholder.
    qualification_status_for_class = (
        QualificationStatus.QUALIFIED.value
        if record.state
        in (QualificationState.AGREEMENT_REPORTED, QualificationState.TENANT_APPROVED)
        else QualificationStatus.UNQUALIFIED.value
    )
    report["calibration_class"] = derive_calibration_class(
        evaluator_id=EVALUATOR_ID,
        evaluator_version=EVALUATOR_VERSION,
        provider_id=provider.name,
        pinned_model_id=record.pinned_model_id,
        configuration_version=record.configuration_version,
        qualification_status=qualification_status_for_class,
    )
    report["registry_state"] = record.state.value
    report["registry_pinned_model_id"] = record.pinned_model_id
    report["registry_is_pinned"] = record.is_pinned
    report["registry_threshold_failures"] = list(record.threshold_failures)
    report["fixture_sources"] = {
        "dev": len(dev_fixtures()),
        "held_out": held_out_count,
    }
    report["dev_hash"] = dev_hash
    if held_out_hash:
        report["held_out_hash"] = held_out_hash
    # R6: preserved in the report itself, not only printed to stderr, so a
    # reader of the JSON alone can see exactly which attestation (if any)
    # this registry state relied on.
    report["leakage_attestation"] = attestation_record

    json.dump(report, args.out, indent=2)
    args.out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
