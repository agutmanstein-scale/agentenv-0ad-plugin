"""Serve an env over real HTTP on a free port (uvicorn), as agent-env's server provider does in its container."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn

from agentenv_zeroad.engine import free_port


@asynccontextmanager
async def served(env) -> AsyncIterator[str]:
    """The env's app served at the yielded base URL until the block ends."""
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(env.create_app().streamable_http_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await serving
