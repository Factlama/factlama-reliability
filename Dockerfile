# OBS-12/G9 Docker MVP. One image, three processes selected by `command:`
# in docker-compose.yml (api / worker / expiry-cleanup) -- they share the
# same dependency set and source tree, so a single image avoids building
# and keeping three near-identical images in sync.
#
# `[dev]` is installed, not a slimmer runtime extra, because `fastapi`/
# `uvicorn`/`httpx` are only declared under this repo's `dev` extras today
# (pyproject.toml has no separate "api runtime" group) -- a real,
# pre-existing packaging gap, named here rather than silently worked
# around with a hand-maintained duplicate dependency list that could drift
# from pyproject.toml's own.
FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
