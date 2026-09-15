# Codex context guide

The source of truth is `../factlama-architecture/factlama-reliability/` in the three-repo workspace. In a standalone clone, use the same paths in `https://github.com/Factlama/factlama-architecture/tree/main/factlama-reliability`. Use local files when available.

Load context for the task, not the whole documentation set:

1. Identify the REL task ID from the request or affected code. Find that task's heading in `docs/implementation.md` and read its section, including acceptance criteria. If the task is unclear, scan headings and code first.
2. Read the matching subsystem spec only: contracts/domain (`docs/domain-model.md`), claims (`docs/claim-engine.md`), evidence (`docs/evidence-engine.md`), judges (`docs/judge-provider.md`), pipeline (`docs/verification-pipeline.md`), scoring (`docs/scoring.md`), policy (`docs/policy-engine.md`), async (`docs/async-evaluation.md`), or API (`docs/API.md`). Use `docs/LOW_LEVEL_IMPLEMENTATION.md` only for the relevant module or acceptance checklist. Use the component `README.md` only for an unfamiliar product boundary or module layout.
3. For work in migrated `schemas/`, `core/`, or `judges/`, check the relevant sections of `CURRENT_IMPLEMENTATION.md` and `docs/PROTOTYPE_MIGRATION_NOTES.md` before assuming prototype behavior meets the target contract.
4. Read the relevant gate in `../factlama-architecture/EXECUTION_PLAN.md` when the work changes build order or crosses repositories. Read the relevant section of `../factlama-architecture/CONTRACTS.md` and, for exact wire shapes, the affected file under `contracts/v0.1/` when touching public schemas, tenancy, provenance, or compatibility. Read related ADRs only when their decision applies.
5. Inspect the affected code and tests before editing. Expand to other documents when a concrete dependency or conflict calls for them; avoid loading entire plans, contracts, or unrelated subsystem specs by default.

The central `docs/implementation.md` owns REL status. Keep contract changes coordinated with architecture ADRs. Test the relevant success, failure, and tenant boundaries; update the central docs and task status only after acceptance criteria pass.
