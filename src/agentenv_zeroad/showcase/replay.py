"""Replay mode (A2, the insurance for a live render that can't run): each game is played headless, then its replay is
rendered on the stream, labelled as a replay.

The pieces, none of which the live session path uses:
- `start_headless_engine`: the engine a game is played on. The runner's container has no docker socket, so it is the
  container's own pyrogenesis through the local launcher: nonvisual, with no `-rl-paced` and no `-rl-realtime`, and
  `no_state_step` as training sets it. That binary is the rt build plus the paced patch, which does nothing without its
  flag, so the game is the 13-3's engine behaviour.
- `replay_args` and `start_replay_process`: pyrogenesis showing a finished game's `commands.txt`. The flag is
  `-replay-visual=<path to commands.txt>` (0.28 main.cpp: the path must be a file, not the replay's folder).
- `ReplayClock`: where the overlay's game time comes from during a replay. `start()` is called once the replay's first
  frame is up: `ReplayHandle.first_frame` waits for the director mod's session-start line in the replay engine's own
  log, which comes after the map has loaded, not when the window is placed. `WallReplayClock` then counts wall time at
  1x. It cannot see a stalled or slowed render or a replay that runs faster than 1x, and the log line is only the
  first frame's proxy; the spike swaps the source without touching anything else.
- `ReplayTimeline`: the overlay's seats and scoreboard at a game time, from the game's own `results.json`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from agentenv_zeroad.engine import (
    DOCKER_PYROGENESIS,
    EngineOptions,
    LocalLauncher,
    RunningEngine,
    free_port,
    start_engine,
    stop_process,
)
from agentenv_zeroad.live import CIV_NAMES, ENGINE_LOG, GAME, RECENT_ORDERS, _model_label, color_hex, engine_args
from agentenv_zeroad.state import clock_text

HEADLESS_READY_TIMEOUT_S = 180.0
FIRST_FRAME_LINE = "zeroad-director: viewing as player"  # the director mod's line when its session starts
FIRST_FRAME_TIMEOUT_S = 90.0  # map load and the director's start
FIRST_FRAME_POLL_S = 0.25
DEFAULT_LATENCY_MS = 5000
REPLAY_TAIL_S = 3.0  # the last frame stays up this long, so the result is on screen before the table changes
EARLY_EXIT_SLACK_MS = 10_000  # a replay engine gone this far before the game's end is a failed render
TIME_ZONE = "America/New_York"


# ---- the headless game -----------------------------------------------------------------------------------------------
def headless_options() -> EngineOptions:
    """Training's engine options: nonvisual, not paced, not realtime."""
    return EngineOptions(port=free_port())


async def start_headless_engine(home: Path) -> RunningEngine:
    """The engine for one replay-mode game. `home` is its own HOME, kept between games so the engine's caches stay; the
    replays it holds are removed first, so the replay copied out afterwards can only be this game's."""
    await LocalLauncher(headless_options(), home=home).clear_replays()
    return await start_engine(
        headless_options(), mode="local", home=home, ready_timeout=HEADLESS_READY_TIMEOUT_S, no_state_step=True
    )


# ---- the replay on screen --------------------------------------------------------------------------------------------
def replay_args(commands: Path, game: tuple[int, int] = GAME) -> list[str]:
    """pyrogenesis arguments for the visual replay of `commands`: the stream's window and director mod
    (`live.engine_args`) over the public mod, and no RL interface."""
    return ["-mod=public", *engine_args(low_graphics=False, game=game), f"-replay-visual={commands}"]


class ReplayHandle(Protocol):
    """What the session needs of a running replay engine."""

    def exited(self) -> bool: ...

    async def first_frame(self, timeout_s: float) -> float | None:
        """Wait until the replay's first frame is up: the `time.monotonic()` at which it was first seen, or None if it
        is not up within `timeout_s` or the engine exits first."""
        ...

    async def stop(self) -> None: ...


class ReplayProcess:
    """A running replay engine. Its first frame is when the director mod has logged its session start in the engine's
    own `mainlog.html` (under its HOME) since this process began."""

    def __init__(self, proc: asyncio.subprocess.Process, home: Path, started_at: float) -> None:
        self._proc, self._log, self._started_at = proc, home / ENGINE_LOG, started_at

    def exited(self) -> bool:
        return self._proc.returncode is not None

    def _logged(self) -> bool:
        try:
            if self._log.stat().st_mtime < self._started_at - 1:
                return False  # an earlier engine's log
            return FIRST_FRAME_LINE in self._log.read_text(errors="replace")
        except OSError:
            return False

    async def first_frame(self, timeout_s: float) -> float | None:
        deadline = time.monotonic() + timeout_s
        while not self._logged():
            if self.exited() or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(FIRST_FRAME_POLL_S)
        return time.monotonic()

    async def stop(self) -> None:
        await stop_process(self._proc)


async def start_replay_process(commands: Path, display: str, binary: str | None = None) -> ReplayHandle:
    """Start pyrogenesis on `display` showing `commands`."""
    binary = binary or shutil.which("pyrogenesis") or DOCKER_PYROGENESIS
    started_at = time.time()
    proc = await asyncio.create_subprocess_exec(
        binary,
        *replay_args(commands),
        env={**os.environ, "DISPLAY": display},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return ReplayProcess(proc, Path(os.environ.get("HOME", "~")).expanduser(), started_at)


class ReplayClock(Protocol):
    """The game time the replay on screen has reached. `start()` is called when its first frame is up, with the time
    `ReplayHandle.first_frame` saw it (not when its window is placed, nor when the session got round to it)."""

    def start(self, at: float | None = None) -> None: ...

    def game_ms(self) -> int: ...


class WallReplayClock:
    """Wall time since `start()`, at 1x; 0 before it. `start(at=...)` counts from that time on the clock's own time
    base (`time.monotonic` by default, as `first_frame` reports) instead of from now."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._began: float | None = None

    def start(self, at: float | None = None) -> None:
        self._began = self._now() if at is None else at

    def game_ms(self) -> int:
        return 0 if self._began is None else max(0, int((self._now() - self._began) * 1000))


def et_clock(epoch: float) -> str:
    """HH:MM in New York, for the header's "replay · played 14:05 ET"."""
    return datetime.fromtimestamp(epoch, ZoneInfo(TIME_ZONE)).strftime("%H:%M")


# ---- the overlay's data for a replay ---------------------------------------------------------------------------------
class ReplayTimeline:
    """The overlay's game block at any game time, from a game's `results.json`: each seat's numbers from the latest
    `timeline` row (every 10 s) at or before the time, its recent orders from its logged `turns` once they landed, and
    its outcome only once the game's own end is reached. There is no hold and no thinking timer in a replay."""

    def __init__(self, results: Mapping[str, Any]) -> None:
        self._seats: Mapping[str, Mapping[str, Any]] = results.get("seats") or {}
        self._rows: list[Mapping[str, Any]] = sorted(results.get("timeline") or [], key=lambda r: r["t_ms"])
        self.end_ms = int(float(results.get("game_time_s") or 0) * 1000)
        self.cap_ms = int(float(results.get("time_cap_s") or 0) * 1000)
        latency = int((results.get("lockstep") or {}).get("latency_ms") or DEFAULT_LATENCY_MS)
        applied = {
            (str(d["slot"]), d["frozen_ms"]): d.get("applied_ms")
            for d in (results.get("lockstep") or {}).get("decisions") or []
        }
        self._orders: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for slot, seat in self._seats.items():
            landed = []
            for turn in seat.get("turns") or []:
                began = int(turn["began_ms"])
                at = applied.get((slot, began), began + latency)
                if at is None:
                    continue  # a missed decision's orders never went out
                landed += [(int(at), order) for order in turn.get("orders") or []]
            self._orders[slot] = sorted(landed, key=lambda item: item[0])

    def state_at(self, game_ms: int) -> dict[str, Any]:
        over = game_ms >= self.end_ms
        row = next((r for r in reversed(self._rows) if r["t_ms"] <= game_ms), None)
        seats = [
            self._seat(slot, seat, (row or {}).get("players", {}).get(slot), game_ms, over)
            for slot, seat in self._seats.items()
        ]
        board = sorted(seats, key=lambda s: (-s["score"], s["player"]))
        return {
            "clock": clock_text(game_ms),
            "game_ms": game_ms,
            "cap": clock_text(self.cap_ms),
            "time_cap_ms": self.cap_ms,
            "status": "over" if over else "playing",
            "detail": None,
            "seats": seats,
            "scoreboard": [
                {"slot": s["slot"], "name": s["name"], "score": s["score"], "color": s["color"]} for s in board
            ],
        }

    def _seat(
        self, slot: str, seat: Mapping[str, Any], row: Mapping[str, Any] | None, game_ms: int, over: bool
    ) -> dict[str, Any]:
        row = row or {}
        player = int(seat.get("player", 0))
        kind = str(seat.get("kind", ""))
        landed = [(at, order) for at, order in self._orders.get(slot, []) if at <= game_ms]
        return {
            "slot": slot,
            "player": player,
            "name": seat.get("name"),
            "model": seat.get("model") or _model_label(kind),
            "kind": kind,
            "reflex": False,
            "civ": seat.get("civ"),
            "civ_name": CIV_NAMES.get(str(seat.get("civ")), seat.get("civ")),
            "color": color_hex(player),
            "state": row.get("state") or "active",
            "outcome": seat.get("outcome") if over else None,
            "score": round(row.get("total") or 0),
            "resources": None,  # the timeline keeps only their sum
            "pop": row.get("pop"),
            "pop_cap": row.get("pop_cap"),
            "army": row.get("army"),
            "phase": row.get("phase"),
            "posture": None,
            "economy": None,
            "recent": [_order_line(at, o) for at, o in landed[-RECENT_ORDERS:]][::-1],
            "say": None,
            "thinking_s": None,
            "last_call_s": None,
            "calls": 0,
        }


def _order_line(at: int, order: Mapping[str, Any]) -> dict[str, Any]:
    args = json.dumps(order.get("arguments") or {}, separators=(",", ":"))
    line = f"{order.get('tool')}({args})"
    return {"at": clock_text(at), "who": "strategist", "by": "strategist", "line": line, "status": order.get("status")}
