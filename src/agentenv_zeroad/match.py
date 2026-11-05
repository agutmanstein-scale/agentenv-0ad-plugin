"""One live 0 A.D. match: the engine, its wall-clock schedule, every seat's executor, the bots, and how it ends.

The clock (U1) steps the engine on its own 200 ms / speed schedule. Beside it, one loop ticks every `tick_s` (the env
sets one game second: a wall second over the speed):
it reads the executor view and the fog-limited snapshot (U2), ticks each agent and bot seat's executor (U3), lets
the bots read their snapshot and give orders, feeds the metrics and event trackers, and checks the end. Brains never
touch this loop: their MCP tool calls only put orders in a seat's book (`submit`) or read the latest snapshot
(`brief`), so a slow model never delays a step or a tick (spec §4.2).

Seats: `agent` seats are played by brains through MCP and driven by the executor; `bot` seats by the scripted bot
through the same executor; `replay` seats by an order replay (`replay.py`) through the same executor; `petra` seats by
the engine's own AI. Fog (spec §4.3, U2's carry-forward note): right
after the reset each player's own LOS reveal flag is turned off, so Petra and every snapshot see only what that
player sees, while the observer still sees all.

The match ends by conquest (the engine's own player states), or at the time cap, where the highest score among the
players still standing wins and a tie on top draws (`grading`); a player the engine already defeated stays lost. The
score is the engine's official total or, with `score_rule="net_worth"`, each player's net worth (`js/snapshot.js`); an
unset rule follows what the agents' prompt says (`grading.resolve_score_rule`). A
`time_cap_s` of 0 means no time cap: the game then ends at the safety cap (`safety_cap_s`), decided the same way and
recorded as `ended_by` "safety_cap". The cap is checked every seat-loop second even when that second's engine reads
fail, on the last known scores, and a loop that fails `MAX_FAILED_TICKS` seconds in a row fails the match, so it can't
run unbounded.
At the end the clock stops, the replay is copied out of the engine and the results file is written.

Brains (U5): a seat whose plan sets `reflex="luna"` gets a Luna reflex (`reflex.py`), its own task beside the clock and
the seat loop, which asks the Decisions API about the seat's open tactical decisions once a second. Its credentials
come from the env's own environment (LITELLM_BASE_URL / LITELLM_API_KEY) or from the seat's brain at run time
(`set_reflex_credentials`). Brains report their model calls (with spend) and when the seat is waiting on them; both
land in the seat's metrics and results.

Rendered matches (U6, `live.py`): a match given a `Broadcast` starts it once set up (the overlay beside the game,
recording), and stops it first when it ends, so the MP4 is finished before the engine goes; it becomes the match's
`recording` file. For the overlay each seat keeps when its brain began thinking (its brief read or a wait begun,
until the model call is reported).

Who changed what (U5c): each seat keeps **one** action record, `SeatState.actions`, of every order that reached its
book and who gave it: its brain through the order tools (resolved to `strategist` or `luna` by the model the brain
reports), its Luna reflex (one entry per answer that changed something, with the reflex's line for viewers), the
scripted bot, or the autopilot carrying out a default the seat's Luna player held past its deadline
(`record_hold`, `autopilot.held_default`; shown as "autopilot (Luna held)", never credited to Luna). results.json
(`actions`, `action_counts`), the brief footer, the strategist's brief and the overlay all read it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import ValidationError

from agentenv_zeroad import grading
from agentenv_zeroad.army import ArmyReport
from agentenv_zeroad.autopilot import held_default
from agentenv_zeroad.bots import ScriptedBot
from agentenv_zeroad.brief import json_brief, text_brief
from agentenv_zeroad.brief_v2 import BriefContext, affordable_text, net_worth, reference_text, text_brief_v2
from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.decisions import DecisionsClient
from agentenv_zeroad.engine import TURN_MS, GameEngine
from agentenv_zeroad.executor import Executor, build_executors
from agentenv_zeroad.interface import DEFAULT_INTERFACE, Interface
from agentenv_zeroad.metrics import MatchMetrics, percentile
from agentenv_zeroad.orders import OrderResult, _Order, call_text, rejected
from agentenv_zeroad.reflex import Decider, Reflex, ReflexAction, open_decisions
from agentenv_zeroad.replay import ReplayController, ReplaySpec
from agentenv_zeroad.roles import RoleCatalog
from agentenv_zeroad.scenario import AiBehavior, Seat, build_scenario
from agentenv_zeroad.spend import SpendGuard, results_spend
from agentenv_zeroad.state import RESOURCES, EventTracker, Snapshot, SnapshotQuery, clock_text
from agentenv_zeroad.view import ExecutorView, ViewQuery

if TYPE_CHECKING:
    from agentenv_zeroad.live import Broadcast

log = logging.getLogger(__name__)

SeatKind = Literal["agent", "bot", "replay", "petra"]
TICK_S = 1.0
MAX_FAILED_TICKS = 30
TOOL_LOG_LIMIT = 2000
TIMELINE_EVERY_MS = 10_000  # results' timeline: every player's score and essentials this often (game time)
TURNS_KEPT = 1000  # logged brain turns a seat keeps for results (a 12-hour match at 25 s is under 1,800)
REFLEX_RECENT = 3  # reflex and autopilot actions the strategist's brief lists
ACTIONS_KEPT = 500
SAFETY_CAP_S = 45 * 60  # the longest a game with no time cap runs
NO_TIME_LIMIT = "No time limit: the game is won by conquest."
HOLD_REPEAT_MS = 30_000  # a held default is carried out at most this often per decision
Who = Literal["brain", "reflex", "autopilot", "bot"]


def js(name: str) -> str:
    return resources.files("agentenv_zeroad.js").joinpath(name).read_text()


class EngineHandle(Protocol):
    """A started engine: its client, how to stop it, and how to copy its replay out."""

    engine: GameEngine

    async def close(self) -> None: ...

    async def copy_replay(self, dest: Path) -> Path | None: ...


@dataclass(frozen=True)
class SeatPlan:
    slot: str
    player: int
    kind: SeatKind
    civ: str
    name: str
    ai_difficulty: int = 3
    reflex: str | None = None
    reflex_max_usd: float | None = None
    interface: Interface = DEFAULT_INTERFACE
    ai_behavior: AiBehavior = "balanced"
    starting_resources: int | None = None  # experimental (scenario.py)
    population_limit: int | None = None  # experimental (scenario.py)
    replay: str | None = None  # a replay seat's spec file (replay.py)


@dataclass(frozen=True)
class MatchConfig:
    seats: tuple[SeatPlan, ...]
    map_name: str = "mainland"
    biome: str = "generic/temperate"
    placement: str = "circle"
    size: int = 192
    seed: int = 7
    speed: float = 1.0
    time_cap_s: float = 40 * 60  # 0: no time cap, the game ends at `safety_cap_s`
    safety_cap_s: float = SAFETY_CAP_S
    score_rule: grading.ScoreRule | None = None
    """None: the rule the agent seats' prompt names (v2 net worth, else official); see grading.resolve_score_rule."""
    draw_margin: float = 0.0
    fog: bool = True
    tick_s: float = TICK_S
    ai_seed: int | None = None  # Petra's seed; the map seed by default
    ceasefire_minutes: int = 0
    starting_resources: int | None = None  # experimental (scenario.py)

    def __post_init__(self) -> None:
        agents = (s.interface for s in self.seats if s.kind == "agent")
        object.__setattr__(self, "score_rule", grading.resolve_score_rule(self.score_rule, agents))

    @property
    def cap_s(self) -> float:
        """Game seconds until the match is decided by score: the time cap, or the safety cap when there is none."""
        return self.time_cap_s if self.time_cap_s > 0 else self.safety_cap_s

    def scenario(self) -> dict[str, Any]:
        seats = [
            Seat(
                s.name,
                s.civ,
                controller="petra" if s.kind == "petra" else "external",
                ai_difficulty=s.ai_difficulty,
                ai_behavior=s.ai_behavior,
                starting_resources=s.starting_resources,
                population_limit=s.population_limit,
            )
            for s in self.seats
        ]
        return build_scenario(
            seats,
            map_name=self.map_name,
            biome=self.biome,
            placement=self.placement,
            size=self.size,
            seed=self.seed,
            ai_seed=self.ai_seed,
            ceasefire_minutes=self.ceasefire_minutes,
            starting_resources=self.starting_resources,
        )


@dataclass
class ToolCall:
    tool: str
    seconds: float
    status: str
    game_ms: int


@dataclass(frozen=True)
class Action:
    """One change to a seat's standing orders and who made it."""

    game_ms: int
    by: Who
    what: str  # the order(s) as tool calls
    status: str
    kind: str  # the tool; the reflex's question:answer; held:<decision> for the autopilot
    line: str | None = None  # a short line for viewers (reflex, autopilot); None: `what` says it
    held_by: str | None = None  # the brain that held the decision the autopilot carried out


@dataclass
class SeatState:
    plan: SeatPlan
    executor: Executor | None = None
    bot: ScriptedBot | ReplayController | None = None
    brain_calls: list[grading.BrainCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_count: int = 0
    reflex: Reflex | None = None
    reflex_guard: SpendGuard | None = None
    brain_spend: dict[str, Any] = field(default_factory=lambda: {"usd_est": 0.0, "input_tokens": 0, "output_tokens": 0})
    brain_model: str | None = None
    actions: deque[Action] = field(default_factory=lambda: deque(maxlen=ACTIONS_KEPT))
    action_counts: Counter[str] = field(default_factory=Counter)
    holds: dict[str, int] = field(default_factory=dict)  # decision -> game ms of the last held default carried out
    thinking_since: float | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    brief_stats: dict[str, int] = field(default_factory=dict)  # the combat totals at the last v2 brief
    reference: str | None = None  # the v2 cost tables, written once so every ask gets the same text
    kills_seen: int = 0  # the engine's enemy_units_killed at the last tick
    kills_home: int = 0  # kills made while enemy soldiers were within reach of the seat's base
    kills_away: int = 0  # every other kill
    enemies_near: bool = False  # enemy soldiers were within reach of the base at the last tick

    @property
    def brain_is_luna(self) -> bool:
        return "luna" in (self.brain_model or "").lower()

    def who(self, action: Action) -> str:
        """`by` with a brain resolved to `strategist` or `luna` (by the model its brain reports)."""
        if action.by != "brain":
            return action.by
        if self.brain_model is None:
            return "brain"
        return "luna" if self.brain_is_luna else "strategist"

    def label(self, action: Action) -> str:
        """Who, as viewers read it: "Luna", "strategist", "bot", "autopilot (Luna held)"."""
        if action.by == "autopilot":
            return f"autopilot ({action.held_by} held)" if action.held_by else "autopilot"
        return {"luna": "Luna", "reflex": "Luna"}.get(self.who(action), self.who(action))

    def action_json(self, action: Action) -> dict[str, Any]:
        return {
            "t": clock_text(action.game_ms),
            "time_ms": action.game_ms,
            "by": self.who(action),
            "who": self.label(action),
            "kind": action.kind,
            "what": action.what,
            "status": action.status,
            "line": action.line or action.what,
            **({"held_by": action.held_by} if action.held_by else {}),
        }

    def record(self, action: Action) -> None:
        self.actions.append(action)
        self.action_counts[action.by] += 1

    @property
    def graded_by_calls(self) -> bool:
        return self.plan.kind == "agent"

    def moves(self) -> tuple[int, int]:
        """The brain calls as the match protocol's moves; a seat whose brain reported none, by its tool calls."""
        if self.brain_calls:
            return grading.moves(self.brain_calls)
        return self.tool_count, 0

    def calls_for_grading(self) -> list[grading.BrainCall]:
        if self.brain_calls:
            return self.brain_calls
        return [grading.BrainCall(seconds=0.0)] * self.tool_count


class LiveMatch:
    """A match on one engine. `setup()` resets the engine; `start()` runs it; it ends on its own, or by `stop()`."""

    def __init__(
        self,
        handle: EngineHandle,
        config: MatchConfig,
        *,
        out_dir: Path,
        now: Callable[[], float] = time.monotonic,
        catalog: RoleCatalog | None = None,
        placer: Any = None,
        decider_for: Callable[[SeatPlan, str, str, SpendGuard], Decider] | None = None,
        environ: Mapping[str, str] | None = None,
        broadcast: Broadcast | None = None,
        clock: GameClock | None = None,
    ):
        self.handle = handle
        self.broadcast = broadcast
        self.engine = handle.engine
        self.config = config
        self.out_dir = out_dir
        self._now = now
        self._catalog = catalog
        self._placer = placer
        self._decider_for = decider_for or _decisions_client
        self._environ = os.environ if environ is None else environ
        self._reflex_tasks: list[asyncio.Task[None]] = []
        # A rendered match runs on the -rl-realtime engine, which advances the game from its own frame loop.
        self.clock = clock or GameClock(self.engine, speed=config.speed, realtime=broadcast is not None)
        self.seats = {s.slot: SeatState(s) for s in config.seats}
        self.by_player = {s.player: s.slot for s in config.seats}
        self.metrics = MatchMetrics(sorted(self.by_player))
        self.trackers: dict[int, EventTracker] = {}
        self.snapshot: Snapshot | None = None
        self.scores: dict[str, dict[str, Any]] = {}  # net_worth: None until read; net_worth_stale: reads since
        self.states: dict[str, str] = {}
        self.outcomes: dict[str, grading.Outcome] | None = None
        self.ended_by: str | None = None
        self.failed: str | None = None
        self.fog: dict[str, Any] | None = None
        self.started_wall: float | None = None
        self.ended_wall: float | None = None
        self.ticks = 0
        self.timeline: list[dict[str, Any]] = []
        self.tick_errors = 0
        self.failed_ticks = 0
        self.files: dict[str, Path] = {}
        self._snapshots: SnapshotQuery | None = None
        self._view: ViewQuery | None = None
        self._clock_task: asyncio.Task[None] | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._ended = asyncio.Event()
        self._finalized = asyncio.Lock()
        self._final = False
        self._final_task: asyncio.Task[None] | None = None

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    async def setup(self) -> None:
        """Reset the engine to this match (replay on), set fog, and build the executors and bots."""
        await self.engine.reset(self.config.scenario(), save_replay=True)
        if self.config.fog:
            self.fog = await self.engine.evaluate(js("fog.js"))
        driven = sorted(s.plan.player for s in self.seats.values() if s.plan.kind in ("agent", "bot", "replay"))
        self._snapshots = SnapshotQuery(self.engine, sorted(self.by_player))
        if driven:
            executors, self._view = await build_executors(
                self.engine,
                self.clock,
                driven,
                players=sorted(self.by_player),
                catalog=self._catalog,
                placer=self._placer,
                interfaces={s.plan.player: s.plan.interface for s in self.seats.values()},
            )
            for p, executor in executors.items():
                seat = self.seats[self.by_player[p]]
                seat.executor = executor
                if seat.plan.kind == "bot":
                    seat.bot = ScriptedBot(executor, submit=self._scripted_submit(seat))
                elif seat.plan.kind == "replay":
                    assert seat.plan.replay is not None, f"seat {seat.plan.slot} is a replay seat without a spec"
                    spec = ReplaySpec.load(Path(seat.plan.replay))
                    seat.bot = ReplayController(executor, spec, submit=self._scripted_submit(seat))
                if seat.plan.reflex == "luna":
                    self._add_reflex(seat)
        self.trackers = {s.plan.player: EventTracker(s.plan.player) for s in self.seats.values()}
        await self._tick()

    def _scripted_submit(self, seat: SeatState) -> Callable[[_Order], OrderResult]:
        return lambda order: self._submit_logged(seat, order, by="bot")

    def start(self) -> None:
        """Start the clock and the seat loop; the match runs until it ends or `stop()`."""
        if self._clock_task is not None:
            return
        self.started_wall = self._now()
        self._clock_task = asyncio.create_task(self.clock.run())
        self._loop_task = asyncio.create_task(self._loop())
        self._reflex_tasks = [asyncio.create_task(s.reflex.run()) for s in self.seats.values() if s.reflex is not None]

    async def start_broadcast(self, overlay_url: str) -> None:
        """Put the overlay beside the rendered game and start recording (a rendered match only)."""
        if self.broadcast is not None:
            await self.broadcast.start(overlay_url)

    def now(self) -> float:
        return self._now()

    @property
    def cap_ms(self) -> int:
        return int(self.config.cap_s * 1000)

    @property
    def started(self) -> bool:
        return self._clock_task is not None

    @property
    def over(self) -> bool:
        return self._ended.is_set()

    async def wait_end(self) -> None:
        await self._ended.wait()
        await self.finalize()

    async def stop(self, reason: str = "stopped") -> None:
        """End the match where it stands (cancelled by the harness) and release the engine."""
        if not self.over:
            self.ended_by = reason
            self._ended.set()
        await self.finalize()

    async def finalize(self) -> None:
        """Once: stop the clock and loop, copy the replay out, write the results file, stop the engine."""
        async with self._finalized:
            if self._final:
                return
            self._final = True
            self.ended_wall = self._now()
            await self._halt()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.broadcast is not None:
                try:
                    recording = await self.broadcast.stop()
                except Exception as e:  # noqa: BLE001 - a lost recording must not lose the results
                    log.warning("stopping the broadcast failed: %r", e)
                    recording = None
                if recording is not None:
                    self.files["recording"] = recording
                if (engine_log := getattr(self.broadcast, "engine_log", None)) is not None:
                    self.files["engine_log"] = engine_log
            try:
                replay = await self.handle.copy_replay(self.out_dir / "commands.txt")
            except Exception as e:  # noqa: BLE001 - a missing replay must not lose the results
                log.warning("copying the replay failed: %r", e)
                replay = None
            if replay is not None:
                self.files["replay"] = replay
            results = self.out_dir / "results.json"
            results.write_text(json.dumps(self.results(), indent=2))
            self.files["results"] = results
            await self.handle.close()

    async def _halt(self) -> None:
        for task in self._reflex_tasks:
            task.cancel()
        await asyncio.gather(*self._reflex_tasks, return_exceptions=True)
        for seat in self.seats.values():
            if seat.reflex is not None:
                await seat.reflex.stop()
        if self._loop_task is not None and self._loop_task is not asyncio.current_task():
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        if self._clock_task is not None:
            await self.clock.stop()
            await asyncio.gather(self._clock_task, return_exceptions=True)

    # ---- the seat loop -------------------------------------------------------------------------------------------
    async def _loop(self) -> None:
        while not self.over:
            await asyncio.sleep(self.config.tick_s)
            if self._clock_task is not None and self._clock_task.done() and not self.over:
                error = self._clock_task.exception()
                self._fail(f"the engine stopped stepping: {error!r}" if error else "the clock stopped")
                break
            try:
                await self._tick()
                self.failed_ticks = 0
            except Exception as e:  # noqa: BLE001 - a bad tick is retried next second; a dead engine fails the clock
                self._tick_failed(e)
        if self.over:
            self._final_task = asyncio.get_running_loop().create_task(self.finalize())

    def _tick_failed(self, error: Exception) -> None:
        """Count a failed tick: the time cap is still checked on the last known scores, and too many in a row fail the
        match, so it can't run unbounded. Call from inside the `except` that caught `error`."""
        self.tick_errors += 1
        self.failed_ticks += 1
        log.exception("match tick failed")
        if self.failed_ticks >= MAX_FAILED_TICKS:
            self._fail(f"the match loop failed {self.failed_ticks} ticks in a row: {error!r}")
        else:
            self._check_end(max(self.time_ms, self.clock.steps * TURN_MS))

    async def _tick(self) -> None:
        view = await self._observe()
        await self._act(view)

    async def _observe(self) -> ExecutorView | None:
        """Read the executors' view and the fog-limited snapshot at this moment."""
        assert self._snapshots is not None
        view: ExecutorView | None = await self._view() if self._view is not None else None
        self.snapshot = await self._snapshots()
        return view

    async def _act(self, view: ExecutorView | None) -> None:
        """Tick every driven seat's executor and bot on what `_observe` read, then track the match and check its end."""
        snap = self.snapshot
        assert snap is not None
        for seat in self.seats.values():
            if seat.executor is None or view is None or seat.plan.player not in view.seats:
                continue
            me = snap.players.get(seat.plan.player)
            try:
                await seat.executor.tick(view.seats[seat.plan.player], me)
                if seat.bot is not None and me is not None:
                    seat.bot.tick(me, snap.time_ms)
            except Exception:  # noqa: BLE001 - one seat's bad tick must not stop the others
                self.tick_errors += 1
                log.exception("seat %s: tick failed", seat.plan.slot)
        self.metrics.observe(snap)
        self._count_kills(snap)
        for tracker in self.trackers.values():
            tracker.update(snap)
        outcome = await self.engine.evaluate(js("outcome.js"))
        self._read_outcome(outcome or {}, snap)
        self._sample_timeline(snap)
        self.ticks += 1
        self._check_end(snap.time_ms)

    def _count_kills(self, snap: Snapshot) -> None:
        """Split each seat's new kills into home and away. The engine counts kills without saying where, so a kill is
        'home' when enemy soldiers were within reach of the seat's base (`enemies_near_base`) at this tick or the last
        one, which is when a defence fights; anything else is 'away'."""
        for seat in self.seats.values():
            me = snap.players.get(seat.plan.player)
            if me is None:
                continue
            total = int(me.stats.get("enemy_units_killed", 0))
            new, seat.kills_seen = max(0, total - seat.kills_seen), total
            near = bool(me.enemies_near_base)
            if new:
                if near or seat.enemies_near:
                    seat.kills_home += new
                else:
                    seat.kills_away += new
            seat.enemies_near = near

    def _read_outcome(self, raw: Mapping[str, Any], snap: Snapshot) -> None:
        """Each player's state and score: the engine's parts and total, and the net worth the snapshot read."""
        for p, row in raw.items():
            slot = self.by_player.get(int(p))
            if slot is None:
                continue
            self.states[slot] = str(row.get("state", "active"))
            scores = {k: float(row.get(k, 0)) for k in ("total", "economy", "military", "exploration")}
            seen = snap.players.get(int(p))
            before = self.scores.get(slot, {})
            if seen is not None and seen.net_worth is not None:
                scores["net_worth"], scores["net_worth_stale"] = float(seen.net_worth), 0
            else:  # carried forward, and counted: None until the first reading, then `net_worth_stale` reads missed
                scores["net_worth"] = before.get("net_worth")
                scores["net_worth_stale"] = int(before.get("net_worth_stale", 0)) + 1
            self.scores[slot] = scores
        fallen = frozenset(s.plan.player for s in self.seats.values() if self.states.get(s.plan.slot) == "defeated")
        for seat in self.seats.values():
            if seat.executor is not None:
                seat.executor.fallen = fallen

    def _sample_timeline(self, snap: Snapshot) -> None:
        """Every `TIMELINE_EVERY_MS` of game time, each player's engine score and essentials (no fog: the whole
        match as the observer sees it, for training's value estimate and the progress plots, never for a brain)."""
        if self.timeline and snap.time_ms - self.timeline[-1]["t_ms"] < TIMELINE_EVERY_MS:
            return
        players = {}
        for slot, seat in self.seats.items():
            me = snap.players.get(seat.plan.player)
            row: dict[str, Any] = {"state": self.states.get(slot), **self.scores.get(slot, {})}
            if me is not None:
                row |= {
                    "pop": me.pop,
                    "pop_cap": me.pop_cap,
                    "army": me.army,
                    "gatherers": sum(me.gatherers.values()),
                    "resources": sum(me.resources.values()),
                    "villagers": me.units.get("villager", 0),
                    "phase": me.phase,
                    "buildings": len(me.buildings),
                    "kills": int(me.stats.get("enemy_units_killed", 0)),
                    "lost": int(me.stats.get("units_lost", 0)),
                    "buildings_lost": int(me.stats.get("buildings_lost", 0)),
                }
            players[slot] = row
        self.timeline.append({"t_ms": snap.time_ms, "players": players})

    def _check_end(self, time_ms: int) -> None:
        if self.over:
            return
        decided = grading.conquest_outcomes(self.states) if len(self.states) == len(self.seats) else None
        if decided is not None:
            self._end(decided, "conquest")
        elif time_ms >= self.cap_ms:
            key = grading.SCORE_KEYS[self.config.score_rule]
            standing = [slot for slot in self.seats if self.states.get(slot) != "defeated"]
            measure = {slot: self.scores.get(slot, {}).get(key) or 0.0 for slot in standing}
            outcomes: dict[str, grading.Outcome] = dict.fromkeys(self.seats, "lost")
            outcomes |= grading.time_cap_outcomes(measure, draw_margin=self.config.draw_margin)
            self._end(outcomes, "time_cap" if self.config.time_cap_s > 0 else "safety_cap")

    def _end(self, outcomes: dict[str, grading.Outcome], ended_by: str) -> None:
        self.outcomes, self.ended_by = outcomes, ended_by
        log.info("match over (%s) at %s: %s", ended_by, self.game_clock, outcomes)
        self._ended.set()

    def _fail(self, detail: str) -> None:
        self.failed = detail
        self.ended_by = "failed"
        log.error("match failed: %s", detail)
        self._ended.set()

    # ---- what the env reads ---------------------------------------------------------------------------------------
    @property
    def time_ms(self) -> int:
        return self.snapshot.time_ms if self.snapshot is not None else 0

    @property
    def game_clock(self) -> str:
        return clock_text(self.time_ms)

    def detail(self) -> str | None:
        if self.failed:
            return self.failed
        if self.outcomes is None:
            return None
        winners = [self.seats[s].plan.name for s, o in self.outcomes.items() if o == "won"]
        if self.ended_by == "conquest":
            return f"{', '.join(winners) or 'nobody'} won by conquest at {self.game_clock}"
        measure = "net worth" if self.config.score_rule == "net_worth" else "score"
        cap = "safety cap" if self.ended_by == "safety_cap" else "time cap"
        if winners:
            return f"{', '.join(winners)} won on {measure} at the {cap}, {self.game_clock}"
        drawn = [self.seats[s].plan.name for s, o in self.outcomes.items() if o == "drawn"]
        return f"a draw between {', '.join(drawn)} at the {cap}, {self.game_clock}"

    # ---- seats: orders, briefs and timings ------------------------------------------------------------------------
    def submit(self, slot: str, order: _Order) -> OrderResult:
        seat = self.seats[slot]
        if seat.executor is None:
            return rejected("not_driven", f"seat {slot} is not driven by orders.")
        return self._submit_logged(seat, order, by="brain")

    def read_brief(self, slot: str, *, compact: bool = False, started: float | None = None) -> str:
        """The `get_brief` tool call: the seat's brief (or the compact JSON one), recorded as a tool call."""
        started = time.perf_counter() if started is None else started
        text = self.compact_brief(slot) if compact else self.brief(slot)
        self.record_tool_call(slot, "get_brief", time.perf_counter() - started, "accepted")
        return text

    def run_order(
        self,
        slot: str,
        tool: str,
        build: Callable[[], _Order],
        *,
        closed: bool = False,
        started: float | None = None,
    ) -> str:
        """One order tool call as a seat's brain gets its answer: the order `build` makes, submitted and recorded as a
        tool call, as the JSON result ending in the brief footer. `closed`: the match is being played out."""
        started = time.perf_counter() if started is None else started
        if self.over or closed:
            result = rejected("match_over", f"The match is over: {self.detail() or 'it is being played out'}.")
        else:
            try:
                result = self.submit(slot, build())
            except ValidationError as e:
                result = rejected(
                    "bad_arguments", "; ".join(err["msg"] for err in e.errors()), suggest="get_brief() for the state"
                )
        self.record_tool_call(slot, tool, time.perf_counter() - started, result.status)
        return json.dumps({**result.to_json(), "footer": self.footer(slot)})

    def _submit_logged(
        self, seat: SeatState, order: _Order, *, by: Who, line: str | None = None, held_by: str | None = None
    ) -> OrderResult:
        """Submit to the seat's executor and keep it in the seat's action record (a `say` is a line, not a change)."""
        assert seat.executor is not None
        result = seat.executor.submit(order)
        if order.kind != "say":
            kind = f"held:{'phase' if order.kind == 'advance_phase' else 'attack'}" if held_by else order.kind
            seat.record(Action(self.time_ms, by, call_text(order), result.status, kind, line, held_by))
        return result

    def _record_reflex(self, seat: SeatState, action: ReflexAction) -> None:
        seat.record(Action(action.time_ms, "reflex", action.what, action.status, action.kind, action.line))

    def brief(self, slot: str) -> str:
        player = self.seats[slot].plan.player
        if self.snapshot is None or player not in self.snapshot.players:
            return "The match has not produced a snapshot yet; try again in a second."
        if self.seats[slot].plan.interface == "v2" and self.seats[slot].executor is not None:
            return self._brief_v2(slot)
        names = {s.plan.player: s.plan.name for s in self.seats.values()}
        events = self.trackers[player].drain() if player in self.trackers else []
        orders = self.seats[slot].executor.book.summary() if self.seats[slot].executor else {}
        lines = [text_brief(self.snapshot, player, events, names=names)]
        if self.config.time_cap_s > 0:
            left = max(0, self.cap_ms - self.time_ms)
            lines.append(f"Time left before the score decides: {clock_text(left)}.")
        else:
            lines.append(NO_TIME_LIMIT)
        if orders:
            lines.append("Standing orders: " + json.dumps(orders, separators=(", ", ": ")))
        seat = self.seats[slot]
        if helped := self._helpers(seat):
            recent = "; ".join(
                f"[{clock_text(a.game_ms)}] {seat.label(a)}: {a.line or a.what}" for a in helped[-REFLEX_RECENT:]
            )
            n = seat.action_counts["reflex"] + seat.action_counts["autopilot"]
            lines.append(f"Your Luna reflexes and autopilot acted {n} times; recently: {recent}.")
        lines.append(self.footer(slot))
        return "\n".join(lines)

    def _brief_v2(self, slot: str) -> str:
        """The v2 brief: players by number only, own score and army, what is affordable, the executor's give-ups."""
        seat = self.seats[slot]
        player, executor = seat.plan.player, seat.executor
        assert self.snapshot is not None and executor is not None
        me = self.snapshot.players[player]
        tracker = self.trackers.get(player)
        events = tracker.drain() if tracker else []
        totals = {k: int(me.stats.get(k, 0)) for k in ("enemy_units_killed", "units_lost")}
        before, seat.brief_stats = seat.brief_stats, totals
        view = executor.last_seat
        book = executor.book
        ctx = BriefContext(
            cap_ms=self.cap_ms if self.config.time_cap_s > 0 else 0,
            scores=self.scores.get(slot, {}),
            army=executor.army.report(executor, view) if view is not None else ArmyReport({}, "none"),
            orders=book.army.describe(),
            killed=totals["enemy_units_killed"] - before.get("enemy_units_killed", 0),
            lost=totals["units_lost"] - before.get("units_lost", 0),
            villager_target=book.villager_target,
            affordable=affordable_text(executor.roles, view, me.resources, executor._phase_tech(view))
            if view is not None
            else "",
            standing=_standing(book.summary()),
            net_worth=net_worth(self.snapshot, player),
            notices=executor.drain_notices(),
        )
        return text_brief_v2(self.snapshot, player, events, ctx)

    def reference(self, slot: str) -> str:
        """The seat's static cost tables for its prompt prefix (interface v2; empty for a v1 seat)."""
        seat = self.seats[slot]
        if seat.plan.interface != "v2" or seat.executor is None:
            return ""
        if seat.reference is None:
            seat.reference = reference_text(seat.executor.roles, seat.executor.last_seat)
        return seat.reference

    def compact_brief(self, slot: str) -> str:
        """The JSON brief Luna's questions are built from (spec §4.3), with the standing orders and match status."""
        player = self.seats[slot].plan.player
        if self.snapshot is None or player not in self.snapshot.players:
            return json.dumps({"ready": False, "over": self.over})
        events = self.trackers[player].drain() if player in self.trackers else []
        executor = self.seats[slot].executor
        brief = json_brief(self.snapshot, player, events)
        brief["state"] = self.snapshot.players[player].state
        brief["orders"] = executor.book.summary() if executor else {}
        v2 = self.seats[slot].plan.interface == "v2"
        brief["players"] = {
            str(s.plan.player): f"player {s.plan.player}" if v2 else s.plan.name for s in self.seats.values()
        }
        brief["cap"] = clock_text(self.cap_ms)
        brief["cap_ms"] = self.cap_ms
        brief["over"] = self.over
        if executor is not None and executor.last_seat is not None:
            tech = executor._phase_tech(executor.last_seat)
            if tech is not None:
                brief["next_phase"] = {"tech": tech[0], "ready": tech[1], "cost": tech[2]}
        return json.dumps(brief, separators=(",", ":"))

    def footer(self, slot: str) -> str:
        """One line every tool result ends with: clock, the seat's essentials, pending events, how the match ends."""
        player = self.seats[slot].plan.player
        # With no time cap the safety cap is internal: the seat is told there is no limit, never a countdown.
        limit = f"/ cap {clock_text(self.cap_ms)}" if self.config.time_cap_s > 0 else "· no time limit"
        clock = f"{self.game_clock} {limit}"
        if self.snapshot is None or player not in self.snapshot.players:
            return f"[{clock}]"
        me = self.snapshot.players[player]
        tracker = self.trackers.get(player)
        pending = tracker.peek() if tracker else []
        urgent = sum(1 for e in pending if e.urgent)
        res = " ".join(f"{r[0]}{me.resources[r]}" for r in me.resources)
        status = "over" if self.over else "playing"
        seat = self.seats[slot]
        luna = ""
        if helped := self._helpers(seat):
            last, n = helped[-1], seat.action_counts["reflex"] + seat.action_counts["autopilot"]
            line = (last.line or last.what)[:60]
            luna = f" · {n} reflex/autopilot actions, last {clock_text(last.game_ms)} {seat.label(last)}: {line}"
        return (
            f"[{clock} · {status} · pop {me.pop}/{me.pop_cap} · {res} · {me.phase} · "
            f"army {me.army} · {len(pending)} new events ({urgent} urgent){luna}: get_brief()]"
        )

    @staticmethod
    def _helpers(seat: SeatState) -> list[Action]:
        """The seat's actions its brain did not give itself: the reflex's and the autopilot's."""
        return [a for a in seat.actions if a.by in ("reflex", "autopilot")]

    def record_tool_call(self, slot: str, tool: str, seconds: float, status: str) -> None:
        seat = self.seats[slot]
        # The overlay's thinking timer: a brain reads its brief right before each model call. A seat's first call only
        # marks it ready (both shipped brains make it before the start), so it starts no turn.
        ready_mark = seat.tool_count == 0
        seat.tool_count += 1
        turn = tool == "get_brief" and not ready_mark and self.started and not self.over
        if turn and seat.thinking_since is None and seat.plan.kind == "agent":
            seat.thinking_since = self._now()
        if len(seat.tool_calls) < TOOL_LOG_LIMIT:
            seat.tool_calls.append(ToolCall(tool, round(seconds, 4), status, self.time_ms))

    def record_brain_call(
        self,
        slot: str,
        call: grading.BrainCall,
        *,
        usd: float | None = None,
        tokens: tuple[int, int] | None = None,
        model: str | None = None,
        turn: Mapping[str, Any] | None = None,
    ) -> None:
        seat = self.seats[slot]
        seat.brain_calls.append(call)
        if turn is not None and len(seat.turns) < TURNS_KEPT:
            seat.turns.append({**turn, "reported_ms": call.game_ms, "seconds": call.seconds, "model": model})
        seat.thinking_since = None
        metrics = self.metrics.seats[seat.plan.player]
        metrics.record_decision(call.seconds, missed=call.missed)
        if usd is not None:
            seat.brain_spend["usd_est"] = round(seat.brain_spend["usd_est"] + usd, 6)
        if tokens is not None:
            seat.brain_spend["input_tokens"] += tokens[0]
            seat.brain_spend["output_tokens"] += tokens[1]
        if model:
            seat.brain_model = model

    # ---- brains: the Luna reflex, waiting and status (U5) ---------------------------------------------------------
    def _add_reflex(self, seat: SeatState) -> None:
        assert seat.executor is not None
        executor = seat.executor
        seat.reflex_guard = SpendGuard(
            f"reflex seat {seat.plan.slot}", cap_usd=seat.plan.reflex_max_usd, log_path=self.out_dir / "spend.jsonl"
        )
        seat.reflex = Reflex(
            seat.plan.player,
            executor.book,
            executor.submit,  # recorded once per answer, by `on_action`
            lambda: self.snapshot,
            self.metrics.seats[seat.plan.player],
            on_action=lambda action: self._record_reflex(seat, action),
        )
        base, key = self._environ.get("LITELLM_BASE_URL"), self._environ.get("LITELLM_API_KEY")
        if base and key:
            seat.reflex.decider = self._decider_for(seat.plan, base, key, seat.reflex_guard)

    def set_reflex_credentials(self, slot: str, base_url: str, api_key: str) -> bool:
        """The seat's brain hands its model endpoint to the seat's reflex (held in memory only, never written out).
        False when the seat has no reflex."""
        seat = self.seats[slot]
        if seat.reflex is None or seat.reflex_guard is None:
            return False
        if seat.reflex.decider is None and not seat.reflex_guard.stopped:
            seat.reflex.decider = self._decider_for(seat.plan, base_url, api_key, seat.reflex_guard)
        return True

    def record_waiting(self, slot: str, state: str) -> None:
        """The seat's brain began or ended a wait: a question the empire can't act on until the brain answers."""
        seat = self.seats[slot]
        metrics = self.metrics.seats[seat.plan.player]
        now = self.time_ms / 1000
        if state == "begin":
            if seat.thinking_since is None:
                seat.thinking_since = self._now()
            metrics.begin_waiting(now)
        else:
            seat.thinking_since = None
            metrics.end_waiting(now)

    def record_hold(self, slot: str, decision: str) -> dict[str, Any]:
        """The seat's brain held a decision (phase, attack) it was asked about. Past the plan's deadline the autopilot
        carries out the default (`held_default`), logged as the autopilot's on the brain's hold; before it, or when
        the default would change nothing, nothing happens. At most once per `HOLD_REPEAT_MS` per decision."""
        seat = self.seats[slot]
        me = self.snapshot.players.get(seat.plan.player) if self.snapshot is not None else None
        if seat.executor is None or me is None or not self.started or self.over or me.state != "active":
            return {"applied": False, "reason": "no match in play for this seat"}
        last = seat.holds.get(decision)
        if last is not None and self.time_ms - last < HOLD_REPEAT_MS:
            return {"applied": False, "reason": "carried out moments ago"}
        book = seat.executor.book
        standing = {s.plan.player: s for s in self.seats.values() if self.states.get(s.plan.slot) != "defeated"}
        enemies = {
            p: (s.plan.name, self.scores.get(s.plan.slot, {}).get("total", 0.0))
            for p, s in standing.items()
            if p != seat.plan.player
        }
        picked = held_default(
            decision,
            time_ms=self.time_ms,
            cap_ms=self.cap_ms,
            phase=me.phase,
            advancing=book.advance_phase,
            army_action=book.army.action,
            soldiers=me.army,
            enemies=enemies,
            threat=me.enemies_near_base,
        )
        if picked is None:
            return {"applied": False, "reason": "not past the deadline, or nothing to change"}
        order, line = picked
        seat.holds[decision] = self.time_ms
        held_by = "Luna" if seat.brain_is_luna else seat.plan.name
        result = self._submit_logged(seat, order, by="autopilot", line=line, held_by=held_by)
        return {"applied": result.ok, "order": call_text(order), "status": result.status}

    def status(self, slot: str) -> dict[str, Any]:
        """What a brain polls between calls: whether the match runs, the game time, the seat's pending events, and
        which decisions are `open` (the brain counts as waited on only while one is). Polling it marks the seat's
        waits as reported."""
        seat = self.seats[slot]
        player = seat.plan.player
        # A brain polls status between turns, never inside a model call: a turn that read its brief and then made no
        # reported call (the spend cap stopped the brain, or there was nothing to decide) ends here.
        seat.thinking_since = None
        # A brain that polls status gates its waits on `open` (both shipped brains do), so no wait is a measured zero;
        # a brain that only reports its calls says nothing about waits, and its seat's waiting_share stays None.
        self.metrics.seats[player].waits_reported = True
        tracker = self.trackers.get(player)
        pending = tracker.peek() if tracker else []
        me = self.snapshot.players.get(player) if self.snapshot is not None else None
        book = seat.executor.book if seat.executor is not None else None
        return {
            "started": self.started,
            "over": self.over,
            "defeated": self.states.get(slot) == "defeated",
            "game_ms": self.time_ms,
            "time_cap_ms": self.cap_ms,
            "events": len(pending),
            "urgent": sum(1 for e in pending if e.urgent),
            "open": open_decisions(me, book, pending),
        }

    # ---- results ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _petra_settings(plan: SeatPlan) -> dict[str, Any]:
        if plan.kind != "petra":
            return {}
        return {"ai_difficulty": plan.ai_difficulty, "ai_behavior": plan.ai_behavior}

    def _combat(self, player: int) -> dict[str, int] | None:
        """Units and buildings lost and enemy units killed, from the engine's statistics in the last snapshot."""
        me = self.snapshot.players.get(player) if self.snapshot is not None else None
        if me is None or not me.stats:
            return None
        keys = ("units_lost", "enemy_units_killed", "buildings_lost", "units_trained")
        seat = next((s for s in self.seats.values() if s.plan.player == player), None)
        split = {"kills_home": seat.kills_home, "kills_away": seat.kills_away} if seat is not None else {}
        return {**{k: int(me.stats.get(k, 0)) for k in keys}, **split}

    def results(self) -> dict[str, Any]:
        clock = self.clock.metrics()
        metrics = self.metrics.summary(clock)
        wall = (self.ended_wall or self._now()) - self.started_wall if self.started_wall is not None else 0.0
        seats = {}
        for slot, seat in self.seats.items():
            outcome = (self.outcomes or {}).get(slot)
            made, missed = seat.moves()
            calls = seat.calls_for_grading()
            seats[slot] = {
                "player": seat.plan.player,
                "kind": seat.plan.kind,
                "civ": seat.plan.civ,
                "name": seat.plan.name,
                "interface": seat.plan.interface,
                "reference": self.reference(slot),  # the cost tables the policy's prompt carried (v2; "" for v1)
                **self._petra_settings(seat.plan),
                "state": self.states.get(slot),
                "outcome": outcome,
                "score": self.scores.get(slot),
                "grade": grading.seat_score(outcome, calls, graded_by_calls=seat.graded_by_calls)
                if self.outcomes is not None
                else None,
                "forfeited": seat.graded_by_calls and grading.forfeited(calls),
                "moves": {"made": made, "missed": missed},
                "brain_calls": [c.__dict__ for c in seat.brain_calls],
                "tool_calls": _tool_summary(seat.tool_calls, seat.tool_count),
                "metrics": metrics.get(seat.plan.player),
                "orders": seat.executor.book.summary() if seat.executor else None,
                "said": list(seat.executor.book.said) if seat.executor else [],
                "failures": [f.__dict__ for f in seat.executor.failures] if seat.executor else [],
                "rejections": len(seat.executor.rejections) if seat.executor else 0,
                "model": seat.brain_model,
                "combat": self._combat(seat.plan.player),
                "actions": [seat.action_json(a) for a in seat.actions],
                "action_counts": {by: seat.action_counts[by] for by in ("brain", "reflex", "autopilot", "bot")},
                "reflex": seat.reflex.stats.summary(self.metrics.seats[seat.plan.player].match_s)
                if seat.reflex is not None
                else None,
                "spend": {
                    "brain": dict(seat.brain_spend),
                    **({"reflex": seat.reflex_guard.summary()} if seat.reflex_guard is not None else {}),
                },
                **({"turns": seat.turns} if seat.turns else {}),
            }
        return {
            "status": "failed" if self.failed else "finished" if self.outcomes is not None else "cancelled",
            "detail": self.detail(),
            "ended_by": self.ended_by,
            "game_time_s": round(self.time_ms / 1000, 1),
            "wall_s": round(wall, 1),
            "speed": self.config.speed,
            "time_cap_s": self.config.time_cap_s,
            "safety_cap_s": self.config.safety_cap_s,
            "score_rule": self.config.score_rule,
            "ai_seed": self.config.ai_seed,
            "ceasefire_minutes": self.config.ceasefire_minutes,
            "map": {"name": self.config.map_name, "size": self.config.size, "seed": self.config.seed},
            "fog": self.fog,
            "clock": clock,
            "ticks": self.ticks,
            "tick_errors": self.tick_errors,
            "spend_usd_est": round(results_spend({"seats": seats}), 6),
            "rendered": self.broadcast is not None,
            "seats": seats,
            "timeline": self.timeline,
        }


def _standing(summary: dict[str, Any]) -> str:
    """The v2 brief's standing orders in a line: the split, what is queued, to build and to research (the villager
    target and the army posture are in the brief's own lines)."""
    split = "/".join(str(summary["economy"][r]) for r in RESOURCES)
    parts = [f"economy {split} (food/wood/stone/metal)"]
    if summary["train"]:
        parts.append("train " + ", ".join(f"{n} {role}" for role, n in summary["train"].items()))
    if summary["build"]:
        parts.append("build " + ", ".join(summary["build"]))
    if summary["research"]:
        parts.append("research " + ", ".join(summary["research"]))
    if summary["advance_phase"]:
        parts.append("advancing phase")
    return "; ".join(parts) + "."


def _tool_summary(calls: list[ToolCall], count: int) -> dict[str, Any]:
    seconds = [c.seconds for c in calls]
    by_tool: dict[str, int] = {}
    for c in calls:
        by_tool[c.tool] = by_tool.get(c.tool, 0) + 1
    return {
        "n": count,
        "by_tool": by_tool,
        "seconds": {"p50": percentile(seconds, 50), "p95": percentile(seconds, 95), "max": max(seconds, default=None)},
        "calls": [c.__dict__ for c in calls],
    }


def _decisions_client(plan: SeatPlan, base_url: str, api_key: str, guard: SpendGuard) -> Decider:
    return DecisionsClient(base_url, api_key, guard)


StartEngine = Callable[[], Awaitable[EngineHandle]]
