"""Replay mode of the showcase session (A2): `_ReplayRunner` is the session runner with the games played headless and
their replays shown afterwards.

One producer task plays the games headless (`replay.start_headless_engine`, unpaced, as training does) one after
another, and one consumer renders their replays in the order played, at most one game ahead of the one on screen. A
game's row is claimed, classified and booked exactly as in live mode, and written to `session.json` when the *game*
finishes; the overlay's table (and the intermission's "next") reads a separate list of the rows whose replay has ended,
so it never shows a result before the picture does.

A replay counts as failed when its engine can't be started or shown, exits before the game's end, or a game that
isn't an engine no-result has no replay file. `RENDER_FAILURES_IN_A_ROW` of them end the session as "render", so a
broken renderer doesn't keep spending on games nobody sees; a whole replay resets the count.

Since one more replay is owed after the game being admitted, the power-off margin and the budget check's planned hours
carry a game cap plus an intermission on top of live mode's.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_zeroad.live import Broadcast
from agentenv_zeroad.showcase.overlay import showcase_state
from agentenv_zeroad.showcase.replay import (
    EARLY_EXIT_SLACK_MS,
    FIRST_FRAME_TIMEOUT_S,
    REPLAY_TAIL_S,
    ReplayClock,
    ReplayHandle,
    ReplayTimeline,
    WallReplayClock,
    et_clock,
)
from agentenv_zeroad.showcase.results import TIME_CAP_MINUTES, GameRow, Session
from agentenv_zeroad.showcase.session import (
    PLANNED_GAME_HOURS,
    STOPPED_BY_HAND,
    WATCH_EVERY_S,
    SessionDeps,
    SessionOptions,
    _Runner,
    _Stopped,
    display_name,
)
from agentenv_zeroad.showcase.specs import ShowcaseConfig
from agentenv_zeroad.train.spec import GameSpec

log = logging.getLogger(__name__)

RENDER_FAILURES_IN_A_ROW = 3


@dataclass
class _OnScreen:
    """The replay being shown: its overlay data and the clock its game time is read from."""

    timeline: ReplayTimeline
    clock: ReplayClock


@dataclass
class _Screen:
    """What the overlay shows (the producer's own `_Live` is the game being played, not this)."""

    phase: str = "starting"
    opponent: str = "…"
    game_no: int = 1
    round_no: int = 1
    played_at: str | None = None
    replay: _OnScreen | None = None
    intermission: dict[str, Any] | None = None


@dataclass(frozen=True)
class _Played:
    """A finished headless game waiting for its replay: its row, its results and where its replay is, if it has one."""

    row: GameRow
    results: dict[str, Any]
    commands: Path | None


class _Shown(Session):
    """The rows whose replay has ended, for the overlay. It is not the session and can never write its file."""

    def save(self) -> None:
        raise RuntimeError("the shown rows are not the session")


class _ReplayRunner(_Runner):
    def __init__(
        self, opts: SessionOptions, cfg: ShowcaseConfig, deps: SessionDeps, broadcast: Broadcast | None
    ) -> None:
        if deps.start_headless is None or deps.start_replay is None:
            raise ValueError("replay mode needs deps.start_headless and deps.start_replay")
        super().__init__(opts, cfg, deps, broadcast)
        self.pace_s = None  # a headless game is training's
        self.screen = _Screen()
        self.shown = _Shown(opts.out / "session.json", rotation=self.names)
        self.producing = True  # the producer may still play games
        self.unshown = 0  # games finished and not yet shown
        self.render_failures = 0

    # ---- what differs from live mode -------------------------------------------------------------------------------
    async def _open_engine(self) -> None:
        assert self.deps.start_headless is not None
        self.running = await self._until_stop(self.deps.start_headless())

    def _owed_s(self) -> float:
        """What one more replay takes after the game being admitted: a game cap and an intermission."""
        return TIME_CAP_MINUTES * 60 + self.opts.intermission_s

    def _margin_s(self) -> float:
        return super()._margin_s() + self._owed_s()

    def _planned_hours(self) -> float:
        return PLANNED_GAME_HOURS + self._owed_s() / 3600

    async def _refusal(self, opponent: str) -> tuple[str, str] | None:
        if not self.stop.is_set() and self.render_failures >= RENDER_FAILURES_IN_A_ROW:
            return "render", f"{self.render_failures} replays in a row could not be shown"
        return await super()._refusal(opponent)

    def _state(self) -> dict[str, Any]:
        screen = self.screen
        on_screen = screen.replay if screen.phase == "playing" else None
        return showcase_state(
            self.shown,
            None,
            phase=screen.phase,
            opponent=screen.opponent,
            game_no=screen.game_no,
            round_no=screen.round_no,
            hold=None,
            mode="replay",
            played_at=screen.played_at,
            intermission=screen.intermission,
            dropped=sorted(self.shown.dropped()),
            base=on_screen.timeline.state_at(on_screen.clock.game_ms()) if on_screen is not None else None,
        )

    # ---- the pipeline ----------------------------------------------------------------------------------------------
    async def _run(self) -> None:
        self.shown.rows = list(self.session.rows)
        slots = asyncio.Semaphore(2)  # the game on screen and one played ahead of it
        queue: asyncio.Queue[_Played | None] = asyncio.Queue()
        producer = asyncio.create_task(self._produce(queue, slots))
        try:
            await self._consume(queue, slots)
        except BaseException:
            producer.cancel()
            raise
        finally:
            (outcome,) = await asyncio.gather(producer, return_exceptions=True)
            self.screen.phase = "ended"
        if isinstance(outcome, Exception):
            raise outcome  # a failed save, classify or ledger write is the run's failure, as in live mode
        if self.stop.is_set():
            self._end("stopped", STOPPED_BY_HAND)

    async def _produce(self, queue: asyncio.Queue[_Played | None], slots: asyncio.Semaphore) -> None:
        """Play the games headless, one after another, until the session ends; `None` tells the consumer it has."""
        try:
            while True:
                try:
                    await self._until_stop(slots.acquire())
                except _Stopped:
                    return self._end("stopped", STOPPED_BY_HAND)
                next_game = await self._next_game()
                if next_game is None:
                    return
                played = await self._headless_game(*next_game)
                self.unshown += 1
                queue.put_nowait(played)
                if self.stop.is_set():
                    return self._end("stopped", STOPPED_BY_HAND)
                if (verdict := self._verdict(played.row)) is not None:
                    return self._end(*verdict)
        finally:
            self.producing = False
            queue.put_nowait(None)

    async def _headless_game(self, start: int, opponent: str) -> _Played:
        spec, pending = self._claim(start, opponent)
        booked = False
        try:
            results = await self._run_game(spec)
            row = self._conclude(spec, pending, results)
            booked = True
        finally:
            await self._release(spec, booked)
        return _Played(row, results, self._commands(spec))

    def _commands(self, spec: GameSpec) -> Path | None:
        commands = self.opts.out / "games" / spec.game_id / "commands.txt"
        return commands if commands.is_file() else None

    async def _consume(self, queue: asyncio.Queue[_Played | None], slots: asyncio.Semaphore) -> None:
        """Show the replays in the order the games were played."""
        while True:
            try:
                played = await self._until_stop(queue.get())
                if played is None:
                    return
                self.unshown -= 1
                await self._show_replay(played)
            except _Stopped:
                return
            slots.release()

    async def _show_replay(self, played: _Played) -> None:
        """Render the game's replay, then add its row to the table and hold the intermission card. A render that fails
        is logged and counted, and the game is still counted (there is just no picture of it)."""
        row, screen = played.row, self.screen
        screen.opponent, screen.game_no = row.opponent, len(self.shown.rows) + 1
        screen.round_no = len(self.shown.rows) // len(self.names) + 1
        screen.played_at, screen.phase, screen.replay = et_clock(row.played_at), "starting", None
        whole = await self._render(played)
        if whole is not None:
            self.render_failures = 0 if whole else self.render_failures + 1
        self.shown.rows.append(row)
        screen.replay = None
        screen.phase = "intermission"
        upcoming = self._next_opponent(after=row.opponent, session=self.shown)
        done = not self.producing and self.unshown == 0
        screen.intermission = self._card(
            row, "end of session" if done or upcoming is None else display_name(upcoming), self.shown.table()
        )
        try:
            await self._wait_intermission()
        finally:
            screen.intermission = None
            screen.phase = "starting"

    async def _render(self, played: _Played) -> bool | None:
        """Show the replay: True if it played whole, False if it failed, None if there was nothing to show for a game
        that failed on the engine (which neither counts for nor against the renderer)."""
        row, screen = played.row, self.screen
        if played.commands is None:
            log.error("game %s has no replay to show", row.game_id)
            return None if row.blame == "engine" else False
        assert self.deps.start_replay is not None
        process: ReplayHandle | None = None
        try:
            clock = self.deps.replay_clock() if self.deps.replay_clock is not None else WallReplayClock(self.deps.now)
            timeline = ReplayTimeline(played.results)
            screen.replay = _OnScreen(timeline, clock)
            process = await self._until_stop(self.deps.start_replay(played.commands))
            await self._until_stop(self._show())
            first = await self._until_stop(process.first_frame(FIRST_FRAME_TIMEOUT_S))
            if not first:
                log.warning("no first frame of the replay of %s within %g s", row.game_id, FIRST_FRAME_TIMEOUT_S)
            clock.start()
            screen.phase = "playing"
            slack_ms = 0 if first else int(REPLAY_TAIL_S * 1000)
            return await self._until_stop(self._until_over(process, clock, timeline.end_ms, slack_ms))
        except _Stopped:
            raise
        except Exception as e:  # noqa: BLE001 - a replay that can't be shown must not end the session
            log.error("the replay of %s failed: %s", row.game_id, type(e).__name__)
            return False
        finally:
            if process is not None:
                with contextlib.suppress(Exception):
                    await process.stop()

    async def _until_over(self, process: ReplayHandle, clock: ReplayClock, end_ms: int, slack_ms: int = 0) -> bool:
        """Wait for the replay's game time to pass the game's end, the tail and any `slack_ms`; False if its engine
        exits well before the end instead."""
        last_ms = end_ms + int(REPLAY_TAIL_S * 1000) + slack_ms
        while clock.game_ms() < last_ms:
            if process.exited():
                early = clock.game_ms() < end_ms - EARLY_EXIT_SLACK_MS
                if early:
                    log.warning("the replay engine exited at %d ms of a %d ms game", clock.game_ms(), end_ms)
                return not early
            await self.deps.sleep(min(WATCH_EVERY_S, (last_ms - clock.game_ms()) / 1000))
        return True
