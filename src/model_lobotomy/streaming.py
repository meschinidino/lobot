"""Shared NDJSON streaming (MOD phase 8).

Every NDJSON route returns ndjson(...). It closes the generator
when the client leaves; see server history for why this matters.
"""
from __future__ import annotations

import time

from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool


def close_when_done(gen) -> None:
    for _ in range(2400):
        try:
            gen.close()
            return
        except ValueError:
            time.sleep(0.05)


def ndjson(gen, cancel) -> StreamingResponse:
    async def body():
        finished = False
        try:
            async for chunk in iterate_in_threadpool(gen):
                yield chunk
            finished = True
        finally:
            if not finished:
                cancel.set()
                await run_in_threadpool(close_when_done, gen)
    return StreamingResponse(body(), media_type="application/x-ndjson")
