# Public development fixture set

`evaluator-agreement-harness.md`'s public development set. One JSON file per
fixture family; each entry is `{fixture_id, family, claim, evidence,
expected_verdict}`, loaded by `scripts/run_agreement_harness.py::dev_fixtures()`.

Authored and labeled 2026-09-21 by a Claude Code session together with the
repository owner, not by an independent third-party dataset reviewer. This is
a real, usable fixture set for the runner and for a provider author's own
self-testing -- it is not the "hand-reviewed" bar `evaluator-agreement-harness.md`
means when it describes leakage-control review of the held-out set, which
requires a human reviewer attesting per release, separate from whoever wrote
the fixtures. State that separation honestly rather than treating this set's
existence as satisfying it.

Do not add expected answers for the FactLama-held-out set here. Do not
paraphrase a held-out fixture into this directory, or vice versa -- see the
held-out set's own README for the leakage-control rule this boundary exists
to enforce.

The `ambiguous` family is excluded from per-label precision/recall/F1
scoring (`evaluator-agreement-harness.md`: "used for calibration, not
pass/fail scoring"). Its `expected_verdict` records the harness's own
resolution (treat genuine disagreement as `INSUFFICIENT_EVIDENCE`) for
calibration reporting only, not as a graded answer key.
