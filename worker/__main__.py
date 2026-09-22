"""Process entrypoint: `python -m worker`. Mirrors `uvicorn api.app:app`
being the API's own entrypoint -- this is the worker's."""

import asyncio
import logging

from worker.bootstrap import run


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
