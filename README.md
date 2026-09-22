# FactLama Reliability

Reliability contains the Python verification prototype in `schemas/`, `core/`, and `judges/`, a synchronous and async HTTP API in `api/`, a durable PostgreSQL storage layer in `storage/` and an async worker in `worker/` (both G5), with tests and examples. Its request/result wire shape, judge provider port, and scoring formulas were revised 2026-09-13 to align with the accepted [shared contract](https://github.com/Factlama/factlama-architecture/blob/main/CONTRACTS.md), but it is still not fully compliant (no durable per-tenant budgets or evaluator qualification registry).

The authoritative component documentation is in [factlama-architecture/factlama-reliability](https://github.com/Factlama/factlama-architecture/tree/main/factlama-reliability). In the three-repo local workspace, use `../factlama-architecture/factlama-reliability/`. Follow [CLAUDE.md](CLAUDE.md) or [CODEX.md](CODEX.md) to select the task section and relevant prototype notes.

Run the existing tests with `.venv/bin/python -m pytest -q` when a local virtual environment exists, or install the `dev` extra into your own environment first. Do not mark a REL task complete solely because the prototype tests pass.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the signed-off commit and pull-request workflow. FactLama Reliability is licensed under the [Apache License 2.0](LICENSE); third-party SDKs, model weights and datasets retain their own licenses.

## Running the API locally

```bash
pip install -e ".[api]"
export FACTLAMA_TENANT_CREDENTIALS='{"local-dev-key":{"tenant_id":"local-dev"}}'
uvicorn api.app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/v0.1/verifications \
  -H 'Authorization: Bearer local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "schema_version": "0.1",
    "request_id": "req_example_1",
    "project_id": "demo-project",
    "application_id": "demo-app",
    "answer": "Paris is the capital of France.",
    "evidence": [{"evidence_id": "e1", "content": "Paris is the capital of France."}]
  }'
```

`FACTLAMA_TENANT_CREDENTIALS` is a JSON object mapping opaque API keys to `{tenant_id, project_id?, application_id?}` -- there is no credential database, so this is operator-managed configuration, not a signup flow. See [API.md](https://github.com/Factlama/factlama-architecture/blob/main/factlama-reliability/docs/API.md) for the full route/error contract.

The API's async routes (`POST /v0.1/verification-jobs` and friends) and `/health/ready` also require `FACTLAMA_DATABASE_URL` (a `postgresql+asyncpg://` URL); apply the migrations first with `alembic upgrade head`.

## Running the worker locally

```bash
export FACTLAMA_DATABASE_URL='postgresql+asyncpg://user:pass@localhost:5432/factlama'
alembic upgrade head
python -m worker
```

The worker claims durable jobs submitted through `POST /v0.1/verification-jobs`, runs bounded retries/fallback, reconciles cross-attempt claim disagreement into `DISPUTED` results (ADR-015), and delivers committed outbox events (currently log-only -- there is no Observability ingress yet, G6). Configuration is environment-driven; see `worker/settings.py` for the full list (`FACTLAMA_WORKER_*`).
