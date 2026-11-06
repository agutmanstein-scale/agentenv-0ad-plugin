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

`LockstepClock` (training games, `train/`) replaces the wall-clock schedule with game time. It steps the engine back to
back as fast as it answers, and every `tick_turns` turns (5 = one game second) it runs a boundary:

1. `observe()`: the match reads the engine's view and snapshot (not at turn 0, which `setup` already observed);
2. every pending decision whose deadline `T + L` has been reached is settled: the clock blocks until that seat's model
   has answered (or `brain_timeout_s` wall seconds have passed, which records a missed decision and moves on), then
   the seat's orders are applied;
3. `act()`: the executors tick, so the orders just applied turn into commands that go out with the next step;
4. each seat brain that is not already waiting on a decision is asked whether one falls due at this game time; if so
   its brief is frozen right now and its model call starts, while the clock keeps stepping.

A decision therefore always spans exactly `latency_ms` of game time (a multiple of the tick), however long the model
takes, and the game never advances past `T + L` before the seat's reply is in. Seats' decisions overlap freely (one
pending decision per seat). Game logic reads only game time; wall time is used for the missed-reply guard and the
metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentenv_zeroad.engine import TURN_MS, GameEngine
from agentenv_zeroad.grading import BRAIN_TIMEOUT_S

log = logging.getLogger(__name__)

MIN_SPEED = 1.0
MAX_SPEED = 8.0


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


TICK_TURNS = 5  # one game second
DEFAULT_LATENCY_MS = 5_000  # a decision's orders land this long after its brief (25 turns)


class Decision(Protocol):
    """One seat decision in flight: the model call (`answer`), then its orders (`apply`) or its absence (`miss`)."""

    async def answer(self) -> None:
        """The model call. Runs as a task while the game keeps stepping; never raises for a failed call."""
        ...

    async def apply(self) -> None:
        """Hand the answer's orders to the seat. Called once, at the decision's deadline."""
        ...

    async def miss(self, reason: str) -> None:
        """Record that no usable answer arrived (it never came, or the task failed)."""
        ...


class SeatBrain(Protocol):
    """What the lockstep clock asks of a seat's brain at each boundary."""

    slot: str

    async def begin(self, game_ms: int) -> Decision | None:
        """Freeze the seat's brief and return a decision if one falls due at `game_ms`; None if not."""
        ...


@dataclass
class _InFlight:
    brain: SeatBrain
    decision: Decision
    task: asyncio.Task[None]
    started: float  # loop time (wall) when the model call began
    frozen_ms: int
    due_ms: int


class LockstepClock(GameClock):
    """Steps `engine` back to back and runs the seat boundary every `tick_turns` turns, in game time (see module
    docstring). `observe` and `act` are the match's two halves of a seat tick: `observe` returns False when it failed,
    which skips `act` and starts no new decision at that boundary. `done` ends the run once the match is over."""

    def __init__(
        self,
        engine: GameEngine,
        *,
        observe: Callable[[], Awaitable[bool]],
        act: Callable[[], Awaitable[None]],
        done: Callable[[], bool],
        brains: Sequence[SeatBrain] = (),
        latency_ms: int = DEFAULT_LATENCY_MS,
        tick_turns: int = TICK_TURNS,
        brain_timeout_s: float = BRAIN_TIMEOUT_S,
    ):
        super().__init__(engine)
        tick_ms = tick_turns * TURN_MS
        if tick_turns < 1 or latency_ms < tick_ms or latency_ms % tick_ms:
            raise ValueError(f"latency_ms must be a positive multiple of the {tick_ms} ms tick, got {latency_ms}")
        self.latency_ms = latency_ms
        self.tick_turns = tick_turns
        self.brain_timeout_s = brain_timeout_s
        self._observe, self._act, self._done = observe, act, done
        self._brains = list(brains)
        self._inflight: list[_InFlight] = []
        self.commands: list[tuple[int, int, dict[str, Any]]] = []
        self.decisions: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.blocked_wall_s = 0.0

    @property
    def game_ms(self) -> int:
        return self.steps * TURN_MS

    def add_brain(self, brain: SeatBrain) -> None:
        self._brains.append(brain)

    def metrics(self) -> dict[str, Any]:
        return {
            **super().metrics(),
            "lockstep": True,
            "latency_ms": self.latency_ms,
            "tick_turns": self.tick_turns,
            "decisions": self.counts["started"],
            "applied": self.counts["applied"],
            "missed": self.counts["missed"],
            "blocked": self.counts["blocked"],
            "blocked_wall_s": round(self.blocked_wall_s, 3),
        }

    async def run(self, max_steps: int | None = None) -> None:
        """Step until `stop()`, until the match is done, or until `max_steps` more steps have run."""
        target = None if max_steps is None else self.steps + max_steps
        try:
            while not self._stopping and not self._done() and (target is None or self.steps < target):
                if self.steps % self.tick_turns == 0:
                    await self._boundary()
                    if self._stopping or self._done():
                        break
                await self.step_once()
        finally:
            self._stopping = False
            await self._abandon()

    async def step_once(self) -> dict[str, Any] | None:
        """Send every queued command in one `/step`; the state is decoded only if something subscribed to it."""
        commands = self._drain()
        started = self._now()
        try:
            if self._mailboxes:
                state = await self.engine.step(commands)
            else:
                await self.engine.step_raw(commands)
                state = None
        except BaseException:
            self._requeue(commands)
            raise
        self.last_step_s = self._now() - started
        self.commands.extend((self.steps, player, command) for player, command in commands)
        self.steps += 1
        if state is not None:
            self.latest_state = state
            for mailbox in self._mailboxes:
                mailbox.put(state)
        return state

    async def stop(self) -> None:
        self._stopping = True
        for pending in self._inflight:
            pending.task.cancel()
        await super().stop()

    async def _boundary(self) -> None:
        now = self.game_ms
        observed = True
        if self.steps > 0:
            observed = await self._observe()
        await self._settle_due(now)
        if self._stopping:
            return
        if observed:
            if self.steps > 0:
                await self._act()
            if not self._done():
                await self._begin(now)

    async def _settle_due(self, now: int) -> None:
        for pending in sorted((p for p in self._inflight if p.due_ms <= now), key=lambda p: p.due_ms):
            try:
                await self._settle(pending, now)
            finally:
                self._inflight.remove(pending)
            if self._stopping:
                return

    async def _settle(self, pending: _InFlight, now: int) -> None:
        loop = asyncio.get_running_loop()
        waited_from = loop.time()
        if not pending.task.done():
            self.counts["blocked"] += 1
            remaining = max(0.0, self.brain_timeout_s - (waited_from - pending.started))
            await asyncio.wait({pending.task}, timeout=remaining)
            self.blocked_wall_s += loop.time() - waited_from
        row = {"slot": pending.brain.slot, "frozen_ms": pending.frozen_ms, "due_ms": pending.due_ms}
        if self._stopping:
            return
        if not pending.task.done():
            pending.task.cancel()
            await asyncio.gather(pending.task, return_exceptions=True)
            await self._miss(pending, row, f"no reply within {self.brain_timeout_s:g} s")
        elif pending.task.cancelled() or pending.task.exception() is not None:
            error = pending.task.exception() if not pending.task.cancelled() else "cancelled"
            await self._miss(pending, row, f"the decision failed: {error!r}")
        else:
            await pending.decision.apply()
            self.counts["applied"] += 1
            self.decisions.append({**row, "applied_ms": now, "missed": False})

    async def _miss(self, pending: _InFlight, row: dict[str, Any], reason: str) -> None:
        log.warning("seat %s: decision frozen at %d ms missed: %s", pending.brain.slot, pending.frozen_ms, reason)
        await pending.decision.miss(reason)
        self.counts["missed"] += 1
        self.decisions.append({**row, "applied_ms": None, "missed": True})

    async def _begin(self, now: int) -> None:
        waiting = {p.brain.slot for p in self._inflight}
        for brain in self._brains:
            if brain.slot in waiting:
                continue
            try:
                decision = await brain.begin(now)
            except Exception:  # noqa: BLE001 - one seat's failed brief costs it this decision, not the game
                log.exception("seat %s: could not begin a decision", brain.slot)
                continue
            if decision is None:
                continue
            task = asyncio.create_task(decision.answer())
            started = asyncio.get_running_loop().time()
            self._inflight.append(_InFlight(brain, decision, task, started, now, now + self.latency_ms))
            self.counts["started"] += 1

    async def _abandon(self) -> None:
        pending, self._inflight = self._inflight, []
        for item in pending:
            item.task.cancel()
        await asyncio.gather(*(item.task for item in pending), return_exceptions=True)
