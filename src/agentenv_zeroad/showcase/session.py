"""The showcase session: one game at a time under training's lockstep rules, until the hours, the games, the budget or
the power-off say stop.

`run_session` plays the rotation (Opus, Astra, Petra-3, ...) one game after another. Before each game it picks the
next opponent that isn't dropped, a fresh held-out start no earlier panel or game used, and checks the stop rules, the
box's budget and the power-off margin. The policy plays `local/policy-v<N>` (`mix.repin`), every other seat is as the
13-3's panel built it. A game is classified (`results.classify`), added to `session.json`, and followed by an
intermission with the finished engine left up but unstepped, so the last frame stays on screen.

A game also ends as "no result" when a seat's calls fail `FAILURES_IN_A_ROW` times in a row. That is read from the
seats' own `sampler_stats()` (the counters `classify` reads), once a second: `failed_calls` counts calls that raised,
never a reply that came too late (the clock cancels that call and misses the decision), so slow replies never count.

SIGTERM stops the game in play as no result ("stopped by hand"), saves `session.json` and returns, so the script that
runs the session can upload and power the box off.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import secrets
import signal
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from agentenv_zeroad.engine import RunningEngine
from agentenv_zeroad.live import Broadcast
from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.showcase.overlay import OverlayServer, hold_of, showcase_state
from agentenv_zeroad.showcase.results import FAILURES_IN_A_ROW, GameRow, Session, classify
from agentenv_zeroad.showcase.specs import ShowcaseConfig, next_starts, used_seeds
from agentenv_zeroad.train.config import PanelConfig, RunConfig
from agentenv_zeroad.train.game import LockstepMatch, play_game
from agentenv_zeroad.train.mix import repin
from agentenv_zeroad.train.panels import UNSET_URL, game_at, position
from agentenv_zeroad.train.spec import GameSpec, SeatSpec

log = logging.getLogger(__name__)

# What each frontier seat of the 13-3's games could spend: the box's ZEROAD_MATCH_CAP_USD (live/box/s2-run.sh exports
# 9 unless set; run1-frontier.toml has no game_cap_usd), which the orchestrator gave build_model as cap_usd.
FRONTIER_SEAT_CAP_USD = 9.0
PLANNED_GAME_HOURS = 0.75
ENGINE_FAILURES_IN_A_ROW = 3
WATCH_EVERY_S = 1.0
POLICY_FAILURES_IN_A_ROW = 2
STOPPED_BY_HAND = "stopped by hand"
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)  # a Ctrl-C is a stop by hand too
CRASHED = "the box or the runner failed"
NON_RUN_CONFIGS = frozenset({"sequence.toml", "sft-panels.toml"})  # settings and a template, not run configs
NO_FAILURES = {"policy": 0, "opponent": 0}
OVERLAY_PORT = 18766

ModelFor = Callable[[SeatSpec, str], StrategistModel]


@dataclass(frozen=True)
class SessionOptions:
    """How a session runs. `rotation` names the opponents (a teacher of the panel, or `petra-<n>`); `policy_url` is the
    box's vLLM and `teacher_url` the LiteLLM proxy the teachers are reached through."""

    out: Path
    rotation: tuple[str, ...]
    hours: float = 6.0
    max_games: int | None = None
    mode: Literal["live", "replay"] = "live"
    pace_s: float = 0.2
    intermission_s: float = 120.0
    min_poweroff_margin_s: float = 2700.0
    seat_cap_usd: float = FRONTIER_SEAT_CAP_USD
    policy_url: str = "http://127.0.0.1:8000"
    teacher_url: str = UNSET_URL


@dataclass(frozen=True)
class SessionDeps:
    """What the session needs from the box; tests fake them. `budget_ok` is `budget.sh check`, `preflight` returns
    the problems found (one check per model of the rotation but the dropped opponents' ids it is given, run before
    every game), `record_spend(what, usd)` books a
    game's spend estimate in the box ledger `budget.sh check` reads."""

    start_engine: Callable[[], Awaitable[RunningEngine]]
    budget_ok: Callable[[float, float], Awaitable[bool]]
    poweroff_left_s: Callable[[], float]
    model_for: ModelFor
    preflight: Callable[[Collection[str]], Awaitable[list[str]]]
    record_spend: Callable[[str, float], None]
    now: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]


class _Stopped(Exception):
    """SIGTERM arrived while the engine was starting or the game was being set up."""


class _HeldEngine:
    """The engine handle a match gets: `close()` does nothing, so the finished game's last frame stays up through the
    intermission; the session closes the real engine afterwards."""

    def __init__(self, running: RunningEngine):
        self.engine = running.engine
        self._running = running

    async def close(self) -> None:
        return None

    async def copy_replay(self, dest: Path) -> Path | None:
        return await self._running.copy_replay(dest)


@dataclass
class _Live:
    """What the overlay shows right now."""

    phase: str = "starting"
    opponent: str = ""
    match: LockstepMatch | None = None
    intermission: dict[str, Any] | None = None
    game_no: int = 1
    round_no: int = 1
    watch: asyncio.Task[None] | None = field(default=None, repr=False)


def display_name(opponent: str) -> str:
    """The name on the board: `opus` is Opus, `petra-3` is Petra-3."""
    return opponent.capitalize()


def game_spec(
    cfg: ShowcaseConfig, run_cfg: RunConfig, panel: PanelConfig, opts: SessionOptions, start: int, opponent: str
) -> GameSpec:
    """The policy's game at held-out index `start` against `opponent`: the 13-3's spec, with the policy on the adapter
    version the config names, served by `opts.policy_url`."""
    spec = game_at(run_cfg, panel, start, opponent, opts.teacher_url)
    return repin(spec, cfg.adapter_version, opts.policy_url)


def run_configs(rules_config: Path) -> list[Path]:
    """The configs beside the rules config whose panels' seeds a showcase start must avoid: every `*.toml` there but
    `NON_RUN_CONFIGS`. One that fails to load raises later, since a skipped config could let a seen start through."""
    found = [p for p in sorted(rules_config.parent.glob("*.toml")) if p.name not in NON_RUN_CONFIGS]
    if rules_config not in found:
        raise ValueError(f"the rules config {rules_config.name} is not a run config beside it")
    return found


async def run_session(
    opts: SessionOptions, cfg: ShowcaseConfig, deps: SessionDeps, broadcast: Broadcast | None
) -> Session:
    """Play the session and return it (also saved in `opts.out/session.json`)."""
    if opts.mode == "replay":
        raise NotImplementedError("replay mode is not built yet")
    return await _Runner(opts, cfg, deps, broadcast).run()


class _Runner:
    def __init__(
        self, opts: SessionOptions, cfg: ShowcaseConfig, deps: SessionDeps, broadcast: Broadcast | None
    ) -> None:
        self.opts, self.cfg, self.deps, self.broadcast = opts, cfg, deps, broadcast
        self.run_cfg = cfg.run_config()
        self.panel = next(p for p in self.run_cfg.panels if p.name == cfg.panel)
        self.names = tuple(display_name(o) for o in opts.rotation)
        self.session = Session(opts.out / "session.json", rotation=self.names)
        self.stop = asyncio.Event()
        self.live = _Live()
        self.broadcast_started = False
        self.played = 0
        self.engine_failures = 0
        self.started_at = 0.0
        self.tools: list[Any] = []
        self.token = ""
        self.running: RunningEngine | None = None
        self.earlier: set[int] | None = None

    async def run(self) -> Session:
        from agentenv_zeroad.server import tool_specs

        for opponent in self.opts.rotation:
            if not opponent.startswith("petra-") and opponent not in {t.name for t in self.panel.teachers}:
                raise ValueError(f"the rotation names {opponent!r}, which is no teacher of panel {self.panel.name}")
        self.tools = await tool_specs()
        self.session.load()
        self.session.rows = [
            replace(r, result="no_result", reason=CRASHED, blame="engine") if r.result == "pending" else r
            for r in self.session.rows
        ]
        self.session.ended = self.session.ended_why = None
        self.started_at = self.deps.now()
        loop = asyncio.get_running_loop()
        handled: list[signal.Signals] = []
        for sig in STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self.stop.set)
                handled.append(sig)
        overlay = self._start_overlay()
        try:
            await self._run()
        finally:
            self.live.phase = "ended"
            self.session.save()
            await self._shutdown(overlay, handled)
        return self.session

    def _start_overlay(self) -> OverlayServer | None:
        if self.broadcast is None:
            return None
        self.token = secrets.token_urlsafe(24)
        server = OverlayServer(self._state, self.token, port=OVERLAY_PORT)
        server.start()
        return server

    async def _shutdown(self, overlay: OverlayServer | None, handled: list[signal.Signals]) -> None:
        for sig in handled:
            asyncio.get_running_loop().remove_signal_handler(sig)
        if self.broadcast is not None and self.broadcast_started:
            with contextlib.suppress(Exception):
                await self.broadcast.stop()
        if overlay is not None:
            overlay.stop()

    def _state(self) -> dict[str, Any]:
        live = self.live
        return showcase_state(
            self.session,
            live.match,
            phase=live.phase,
            opponent=live.opponent,
            game_no=live.game_no,
            round_no=live.round_no,
            hold=hold_of(live.match) if live.match is not None else None,
            mode=self.opts.mode,
            played_at=None,
            intermission=live.intermission,
            dropped=sorted(self.session.dropped()),
        )

    def _end(self, code: str, why: str) -> None:
        self.session.ended, self.session.ended_why = code, why
        log.info("session ends: %s (%s)", code, why)

    async def _run(self) -> None:
        while True:
            opponent = self._next_opponent()
            if opponent is None:
                return self._end("opponents", "every opponent in the rotation was dropped")
            if (refusal := await self._refusal(opponent)) is not None:
                return self._end(*refusal)
            try:
                problems = await self._until_stop(self.deps.preflight(self._dropped_ids()))
            except _Stopped:
                return self._end("stopped", STOPPED_BY_HAND)
            if problems:
                return self._end("preflight", "; ".join(problems))
            try:
                start = next_starts(self.run_cfg, self.cfg.first_index, self._used(), 1)[0]
            except ValueError as e:
                return self._end("starts", str(e))
            row = await self._game(start, opponent)
            if self.stop.is_set():
                return self._end("stopped", STOPPED_BY_HAND)
            self.engine_failures = self.engine_failures + 1 if row.blame == "engine" else 0
            if self.engine_failures >= ENGINE_FAILURES_IN_A_ROW:
                return self._end("engine", f"{self.engine_failures} games in a row failed on the engine")
            recent = self.session.rows[-POLICY_FAILURES_IN_A_ROW:]
            if len(recent) == POLICY_FAILURES_IN_A_ROW and all(r.blame == "policy" for r in recent):
                return self._end("policy", f"{POLICY_FAILURES_IN_A_ROW} games in a row failed on the policy's calls")

    def _used(self) -> set[int]:
        if self.earlier is None:
            siblings = sorted(self.opts.out.parent.glob("*/session.json"))
            self.earlier = used_seeds(run_configs(self.cfg.rules_config), siblings)
        return self.earlier | self.session.seeds() | self._started_here()

    def _started_here(self) -> set[int]:
        """The seeds of the starts whose game folder exists in this session's folder, whatever became of the game."""
        found = set()
        for folder in (self.opts.out / "games").glob(f"{self.panel.name}-*"):
            if match := re.match(rf"{re.escape(self.panel.name)}-(\d+)-", folder.name):
                found.add(position(self.run_cfg, "held_out", int(match.group(1))).seed)
        return found

    def _dropped_ids(self) -> set[str]:
        dropped = self.session.dropped()
        return {o for o, name in zip(self.opts.rotation, self.names, strict=True) if name in dropped}

    def _next_opponent(self) -> str | None:
        """The rotation's opponent after the last game's, skipping dropped ones (as the rotation's id, not its name)."""
        names, dropped = self.names, self.session.dropped()
        last = self.session.rows[-1].opponent if self.session.rows else None
        first = (names.index(last) + 1) % len(names) if last in names else 0
        for k in range(len(names)):
            i = (first + k) % len(names)
            if names[i] not in dropped:
                return self.opts.rotation[i]
        return None

    async def _refusal(self, opponent: str) -> tuple[str, str] | None:
        opts = self.opts
        if self.stop.is_set():
            return "stopped", STOPPED_BY_HAND
        if opts.max_games is not None and self.played >= opts.max_games:
            return "max_games", f"{self.played} games played"
        if self.deps.now() - self.started_at >= opts.hours * 3600:
            return "hours", f"{opts.hours:g} hours passed"
        left = self.deps.poweroff_left_s()
        if left < opts.min_poweroff_margin_s:
            return (
                "poweroff",
                f"{left / 60:.0f} min to the power-off, a game needs {opts.min_poweroff_margin_s / 60:.0f}",
            )
        planned = 0.0 if opponent.startswith("petra-") else opts.seat_cap_usd
        if not await self.deps.budget_ok(planned, PLANNED_GAME_HOURS):
            return "budget", f"the budget guard refused a game of ${planned:g} and {PLANNED_GAME_HOURS} h"
        return None

    async def _game(self, start: int, opponent: str) -> GameRow:
        spec = game_spec(self.cfg, self.run_cfg, self.panel, self.opts, start, opponent)
        name = display_name(opponent)
        slot = str(next(n for n, seat in enumerate(spec.seats, 1) if seat.trainable))
        rows = len(self.session.rows)
        self.live = _Live("starting", name, None, None, rows + 1, rows // len(self.names) + 1)
        pending = GameRow(
            index=self.session.next_index(),
            game_id=spec.game_id,
            opponent=name,
            seed=spec.seed,
            policy_slot=slot,
            result="pending",
            how=None,
            reason=None,
            failed=dict(NO_FAILURES),
            decisions=dict(NO_FAILURES),
            played_at=time.time(),
        )
        self.session.add(
            pending
        )  # claimed before anything starts: a crash leaves this row, so the start isn't replayed
        self.session.save()
        self.running = None
        booked = False
        try:
            results = await self._run_game(spec)
            row = classify(results, index=pending.index, opponent=name, policy_slot=slot)
            if self.stop.is_set():
                row = replace(row, result="no_result", how=None, reason=STOPPED_BY_HAND, blame=None)
            self.session.rows[-1] = row
            self.session.save()
            self.played += 1
            self._book(spec, results)
            booked = True
            if not self.stop.is_set():
                await self._intermission(row)
        finally:
            if not booked:
                self._book(spec, None)
            if self.running is not None:
                with contextlib.suppress(Exception):
                    await self.running.close()
        return row

    async def _run_game(self, spec: GameSpec) -> dict[str, Any]:
        """The game's results; a failure (or a stop before the match began) is a failed game's results."""
        try:
            self.running = await self._until_stop(self.deps.start_engine())
            await self._until_stop(self._show(self.running))
            return await self._play(spec, self.running)
        except _Stopped:
            failure = "stopped"
        except Exception as e:  # noqa: BLE001 - one game's failure is its no result, not the session's end
            log.error("game %s failed: %s", spec.game_id, type(e).__name__)
            failure = type(e).__name__
        return {"status": "failed", "ended_by": "failed", "detail": f"{failure}: ", "spec": spec.to_json()}

    async def _until_stop(self, work: Awaitable[Any]) -> Any:
        """`work`'s result, or `_Stopped` (with `work` cancelled) if SIGTERM comes first."""
        task = asyncio.ensure_future(work)
        stopper = asyncio.ensure_future(self.stop.wait())
        try:
            await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            raise _Stopped
        finally:
            stopper.cancel()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _book(self, spec: GameSpec, results: dict[str, Any] | None) -> None:
        """The game's spend into the box ledger: its estimate from the results, or the caps of its frontier seats when
        there are none (the game failed, or was interrupted), as run-match.sh books a match it can't read."""
        usd = (results or {}).get("spend_usd_est")
        if usd is None:
            usd = self.opts.seat_cap_usd * sum(1 for s in spec.seats if s.kind == "llm" and not s.trainable)
        self.deps.record_spend(spec.game_id, float(usd))

    async def _show(self, running: RunningEngine) -> None:
        if self.broadcast is None:
            return
        if self.broadcast_started:
            await self.broadcast.attach_game()
            return
        await self.broadcast.open_display()
        self.broadcast_started = True  # so a stop or a failure part-way still stops Chrome and ffmpeg
        try:
            await self.broadcast.start(f"http://127.0.0.1:{OVERLAY_PORT}/live?stream&token={self.token}")
        except BaseException:
            self.broadcast_started = False
            with contextlib.suppress(Exception):
                await self.broadcast.stop()
            raise

    async def _play(self, spec: GameSpec, running: RunningEngine) -> dict[str, Any]:
        game = asyncio.create_task(
            play_game(
                spec,
                _HeldEngine(running),
                self.opts.out / "games" / spec.game_id,
                tools=self.tools,
                model_for=self.deps.model_for,
                pace_s=self.opts.pace_s,
                on_match=self._on_match,
            )
        )
        stopper = asyncio.create_task(self.stop.wait())
        try:
            await asyncio.wait({game, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if not game.done():
                await self._stop_match(game)
            try:
                return await game
            except asyncio.CancelledError:
                if self.stop.is_set() and game.cancelled():
                    raise _Stopped from None
                raise
        finally:
            for task in (stopper, self.live.watch):
                if task is not None:
                    task.cancel()
            game.cancel()

    def _on_match(self, match: LockstepMatch) -> None:
        self.live.match, self.live.phase = match, "playing"
        self.live.watch = asyncio.create_task(self._watch_failures(match))

    async def _stop_match(self, game: asyncio.Task[dict[str, Any]]) -> None:
        if self.live.match is None:
            game.cancel()  # still setting up: play_game stops its match and re-raises
            await asyncio.gather(game, return_exceptions=True)
        else:
            await self.live.match.stop(STOPPED_BY_HAND)

    async def _watch_failures(self, match: LockstepMatch) -> None:
        """Stop the game as no result once a seat's last FAILURES_IN_A_ROW calls all failed."""
        seen = {slot: model.sampler_stats() for slot, model in match.models.items()}
        streak = dict.fromkeys(match.models, 0)
        while not match.over:
            await self.deps.sleep(WATCH_EVERY_S)
            for slot, model in match.models.items():
                now = model.sampler_stats()
                answered = now["calls"] - seen[slot]["calls"]
                failed = now["failed_calls"] - seen[slot]["failed_calls"]
                seen[slot] = now
                # within one second a seat finishes at most one call; if both counters moved, count the worst order
                streak[slot] = failed if answered else streak[slot] + failed
                if streak[slot] >= FAILURES_IN_A_ROW:
                    match.extra["showcase"] = {"ended_early": {"slot": slot, "failures": FAILURES_IN_A_ROW}}
                    await match.stop("ended early")
                    return

    async def _intermission(self, row: GameRow) -> None:
        final = self.opts.hours * 3600 <= self.deps.now() - self.started_at or (
            self.opts.max_games is not None and self.played >= self.opts.max_games
        )
        upcoming = self._next_opponent()
        won, lost = {"win": ("Qwen", row.opponent), "loss": (row.opponent, "Qwen")}.get(row.result, (None, None))
        self.live.match, self.live.phase = None, "intermission"
        self.live.intermission = {
            "last": {
                "winner": won,
                "loser": lost,
                "how": row.how,
                "result": row.result if row.result in ("draw", "no_result") else "win",
                "reason": row.reason,
            },
            "next": "end of session" if final or upcoming is None else display_name(upcoming),
            "table": self.session.table(),
        }
        sleeper = asyncio.create_task(self.deps.sleep(self.opts.intermission_s))
        stopper = asyncio.create_task(self.stop.wait())
        try:
            await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            sleeper.cancel()
            stopper.cancel()
            self.live.intermission = None
