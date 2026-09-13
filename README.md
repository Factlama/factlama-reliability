# FactLama Reliability

Reliability contains the Python verification prototype in `schemas/`, `core/`, and `judges/`, with tests and examples. Its request/result wire shape, judge provider port, and scoring formulas were revised 2026-09-13 to align with the accepted [shared contract](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md), but it is still not fully compliant (no async execution, budgets, or evaluator qualification registry).

The authoritative component documentation is in [factlama-architecture/factlama-reliability](https://github.com/Factlama/factlama-architecture/tree/main/factlama-reliability). In the three-repo local workspace, open `../factlama-architecture/factlama-reliability/README.md`. Start with its `CURRENT_IMPLEMENTATION.md` and `docs/PROTOTYPE_MIGRATION_NOTES.md` before modifying the prototype.

Run the existing tests with `.venv/bin/python -m pytest -q` when a local virtual environment exists, or install the `dev` extra into your own environment first. Do not mark a REL task complete solely because the prototype tests pass.
