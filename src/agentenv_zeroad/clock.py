"""The game clock: steps the engine on a fixed wall-clock schedule so the game runs in real time (spec §4.2).

Brains never block the game. They queue commands with `queue()` (sync, non-blocking) whenever they like; each step
sends whatever is queued at that moment. State goes out to subscribers through 1-slot mailboxes that keep only the
newest state, so a slow reader drops stale states instead of holding the clock up.

Schedule: slot k is due at `t0 + k * period`, period = 200 ms / speed. If a step ends after the next slot was due,
that is an overrun: it is logged, counted, the worst lateness is kept for results, and the schedule restarts from now,
so the clock never fires a burst of steps to catch up.

A step that fails (the engine errors or times out) puts its commands back at the front of their queues, so nothing a
brain ordered is lost, and the error ends `run()`; `stop()` re-raises it. Retrying is the owner's call: the match
decides whether a dead engine forfeits or restarts.

Realtime mode (`realtime=True`, the patched `-rl-realtime` visual engine): the engine advances sim time from its own
frame loop and `/step` only posts commands and returns state. The loop is identical; only the post cadence stays at
200 ms because speed is then a sim-rate setting on the engine, not a step rate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agentenv_zeroad.engine import TURN_MS, GameEngine

log = logging.getLogger(__name__)

MIN_SPEED = 1.0
MAX_SPEED = 2.0


class StateMailbox:
    """Holds only the newest state for one subscriber; `put` never blocks and never fails."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)

    def put(self, state: dict[str, Any]) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(state)

    async def get(self) -> dict[str, Any]:
        return await self._queue.get()

    def get_nowait(self) -> dict[str, Any] | None:
        return None if self._queue.empty() else self._queue.get_nowait()


class GameClock:
    """Steps `engine` every `200 ms / speed` of wall time, flushing queued per-player commands into each step."""

    def __init__(
        self,
        engine: GameEngine,
        *,
        speed: float = 1.0,
        realtime: bool = False,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be in [{MIN_SPEED}, {MAX_SPEED}], got {speed}")
        self.engine = engine
        self.speed = speed
        self.realtime = realtime
        self.period = TURN_MS / 1000 if realtime else TURN_MS / 1000 / speed
        self._now = now
        self._sleep = sleep
        self._queues: dict[int, deque[dict[str, Any]]] = {}
        self._mailboxes: list[StateMailbox] = []
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.latest_state: dict[str, Any] | None = None
        self.steps = 0
        self.overruns = 0
        self.worst_overrun_s = 0.0
        self.last_step_s = 0.0

    def queue(self, player: int, command: Mapping[str, Any]) -> None:
        """Queue one command for `player`; it goes out with the next step."""
        self._queues.setdefault(int(player), deque()).append(dict(command))

    def pending(self, player: int) -> int:
        return len(self._queues.get(int(player), ()))

    def subscribe(self) -> StateMailbox:
        mailbox = StateMailbox()
        self._mailboxes.append(mailbox)
        if self.latest_state is not None:
            mailbox.put(self.latest_state)
        return mailbox

    def unsubscribe(self, mailbox: StateMailbox) -> None:
        if mailbox in self._mailboxes:
            self._mailboxes.remove(mailbox)

    def metrics(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "overruns": self.overruns,
            "worst_overrun_ms": round(self.worst_overrun_s * 1000, 1),
            "speed": self.speed,
            "realtime": self.realtime,
        }

    async def run(self, max_steps: int | None = None) -> None:
        """Step until `stop()` or until `max_steps` more steps have run."""
        target = None if max_steps is None else self.steps + max_steps
        t0 = self._now()
        slot = 0
        try:
            while not self._stopping and (target is None or self.steps < target):
                wait = t0 + slot * self.period - self._now()
                if wait > 0:
                    await self._sleep(wait)
                if self._stopping:
                    break
                await self.step_once()
                slot += 1
                lateness = self._now() - (t0 + slot * self.period)
                if lateness > 0:
                    log.warning(
                        "clock overrun: step %d ended %.0f ms late (step took %.0f ms, slot is %.0f ms)",
                        self.steps,
                        lateness * 1000,
                        self.last_step_s * 1000,
                        self.period * 1000,
                    )
                    self.overruns += 1
                    self.worst_overrun_s = max(self.worst_overrun_s, lateness)
                    t0, slot = self._now(), 0
        finally:
            self._stopping = False

    async def step_once(self) -> dict[str, Any]:
        """Send every queued command in one `/step` and publish the resulting state."""
        commands = self._drain()
        started = self._now()
        try:
            state = await self.engine.step(commands)
        except BaseException:
            self._requeue(commands)
            raise
        self.last_step_s = self._now() - started
        self.steps += 1
        self.latest_state = state
        for mailbox in self._mailboxes:
            mailbox.put(state)
        return state

    def start(self, max_steps: int | None = None) -> asyncio.Task[None]:
        if self._task is not None and not self._task.done():
            raise RuntimeError("clock already running")
        self._stopping = False
        self._task = asyncio.create_task(self.run(max_steps))
        return self._task

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            await self._task
            self._task = None

    def _drain(self) -> list[tuple[int, dict[str, Any]]]:
        commands: list[tuple[int, dict[str, Any]]] = []
        for player in sorted(self._queues):
            pending = self._queues[player]
            while pending:
                commands.append((player, pending.popleft()))
        return commands

    def _requeue(self, commands: list[tuple[int, dict[str, Any]]]) -> None:
        """Put commands from a failed step back ahead of anything queued since, keeping their order."""
        for player, command in reversed(commands):
            self._queues.setdefault(player, deque()).appendleft(command)
