"""A pool of long-lived engines for training games (spec item 5).

Starting an engine takes seconds, and `/reset` replaces its game in 1.1-1.8 s, so a training run starts N engines
once and reuses each for game after game: `lease()` waits for a free engine and yields a handle a `LiveMatch` plays on
(its `close()` gives the engine back instead of stopping it).

What the pool does that a one-off `start_engine` does not:
- it picks every port itself, from a range it scans (`free_ports`), all before any engine starts, so two engines can
  never be handed the same port and a port another process grabbed in between is caught by `start_engine`'s own check;
- it runs `reap_orphans` once, when it starts, instead of before each engine's start (the launchers it makes have
  `reap=False`);
- each engine has its own replay folder: a local engine gets its own HOME under `home_root` (a temporary folder the
  pool removes when it closes, unless one is given), a docker engine has its own container filesystem, and the
  replays an engine wrote for an earlier game are deleted when it is leased, so `copy_replay` can only return the
  replay of the game that just ran on that engine, and the caller's own replays are never touched;
- an engine whose game failed can be marked `broken`; it is restarted on its port the next time it is leased, so a
  wedged engine costs one game, not the pool. An engine that fails to restart stays broken, so the next lease tries
  again, and the failure reaches only the game that leased it.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentenv_zeroad.engine import (
    DEFAULT_IMAGE,
    PORT_STOP,
    Engine,
    EngineOptions,
    RunningEngine,
    free_ports,
    reap_orphans,
    start_engine,
)

log = logging.getLogger(__name__)

BASE_PORT = 16000

StartOne = Callable[[EngineOptions, Path | None], Awaitable[RunningEngine]]


@dataclass
class _Slot:
    index: int
    options: EngineOptions
    home: Path | None
    running: RunningEngine
    broken: bool = False


class PooledEngine:
    """An engine on loan from the pool, shaped as `match.EngineHandle`. `close()` does nothing: the match is done with
    the engine but it stays leased, so the runner can still mark it broken; leaving `lease()` returns it."""

    def __init__(self, pool: EnginePool, slot: _Slot):
        self._pool = pool
        self._slot = slot
        self.engine: Engine = slot.running.engine
        self.port = slot.options.port

    def mark_broken(self) -> None:
        """The game on this engine failed in a way that may have wedged it: restart it before its next game."""
        self._slot.broken = True

    async def close(self) -> None:
        return None

    async def copy_replay(self, dest: Path) -> Path | None:
        if not self._pool.keep_replays:
            return None
        return await self._slot.running.copy_replay(dest)


class EnginePool:
    """`size` engines started once (`async with EnginePool(...)`, or `start()` and `close()`), leased one game at a
    time."""

    def __init__(
        self,
        size: int,
        *,
        mode: Literal["local", "docker"] = "docker",
        image: str = DEFAULT_IMAGE,
        base_port: int = BASE_PORT,
        port_stop: int = PORT_STOP,
        home_root: Path | None = None,
        keep_replays: bool = False,
        ready_timeout: float = 120.0,
        start_one: StartOne | None = None,
    ):
        if size < 1:
            raise ValueError("an engine pool needs at least one engine")
        self.size = size
        self.mode = mode
        self.image = image
        self.base_port = base_port
        self.port_stop = port_stop
        self.home_root = home_root
        self.keep_replays = keep_replays
        self.ready_timeout = ready_timeout
        self._start_one = start_one or self._start_engine
        self._free: asyncio.Queue[_Slot] = asyncio.Queue()
        self._slots: list[_Slot] = []
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> EnginePool:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Reap dead owners' containers once, pick the ports, and start every engine concurrently."""
        if self.mode == "docker":
            await reap_orphans()
        elif self.home_root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="zeroad-engines-")
            self.home_root = Path(self._tmp.name)
        ports = free_ports(self.size, start=self.base_port, stop=self.port_stop)
        started = await asyncio.gather(*(self._open(i, port) for i, port in enumerate(ports)), return_exceptions=True)
        failed = [r for r in started if isinstance(r, BaseException)]
        self._slots = [r for r in started if isinstance(r, _Slot)]
        if failed:
            await self.close()
            raise failed[0]
        for slot in self._slots:
            self._free.put_nowait(slot)

    async def close(self) -> None:
        slots, self._slots = self._slots, []
        self._free = asyncio.Queue()
        await asyncio.gather(*(slot.running.close() for slot in slots), return_exceptions=True)
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp, self.home_root = None, None

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[PooledEngine]:
        """Wait for a free engine, yield it with its old replays gone, and give it back at the end."""
        slot = await self._free.get()
        try:
            try:
                if slot.broken:
                    await self._restart(slot)
                await slot.running.clear_replays()
            except Exception:
                slot.broken = True
                raise
            yield PooledEngine(self, slot)
        finally:
            self._free.put_nowait(slot)

    async def _open(self, index: int, port: int) -> _Slot:
        options = EngineOptions(port=port)
        home = self.home_root / f"engine-{index}" if self.home_root is not None and self.mode == "local" else None
        return _Slot(index, options, home, await self._start_one(options, home))

    async def _restart(self, slot: _Slot) -> None:
        log.warning("restarting engine %d on port %d after a failed game", slot.index, slot.options.port)
        await asyncio.gather(slot.running.close(), return_exceptions=True)
        slot.running = await self._start_one(slot.options, slot.home)
        slot.broken = False

    async def _start_engine(self, options: EngineOptions, home: Path | None) -> RunningEngine:
        return await start_engine(
            options,
            mode=self.mode,
            image=self.image,
            ready_timeout=self.ready_timeout,
            docker={"reap": False},
            home=home,
        )
