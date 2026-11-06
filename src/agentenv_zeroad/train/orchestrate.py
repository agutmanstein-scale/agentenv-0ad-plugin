"""The orchestrator of a post-training run: one process on the training box that keeps the games flowing, the learner
fed and the run inside its budget, from a run config (`train/config.py`).

    python -m agentenv_zeroad.train.orchestrate run --config run.toml [--play-only]
    python -m agentenv_zeroad.train.orchestrate tools            # the policy's v2 order tools, for --known-tools

What it does, each on its own interval:

- Games. It keeps `concurrency` games in flight on the engine pool (`runner.run_game`) from `mix.GameFactory`'s stream,
  every one pinned to the adapter version current when it starts and to the vLLM instance serving that version with the
  fewest games. When a game ends it reads its results.json, hands it to the credit function (`Credit.rows`) and writes
  the rows it returns into ROOT/inbox atomically (a `.jsonl.tmp` file renamed to a unique `.jsonl`), where the learner
  reads them. A game with no results.json, or one credit fails on, is counted, written to games.jsonl and costs the
  run nothing else.
- Adapters. It watches ROOT/adapters for versions the learner publishes and keeps both vLLM instances serving the
  newest, every version an in-flight game or panel pins and the configured snapshots; it unloads the rest
  (`serving.py`). Versions that nothing wants (not the hourly, champion, pinned, configured or current ones) are
  deleted from the disk once a sync has taken their manifest. A game only starts on a version once every up instance
  holds it, so "current" never runs ahead of what can serve it. Adapter loads that keep failing alarm and stop the run.
- vLLM watchdog. The same pass reads each instance's /health and served models and notes its outages. An instance that
  is down gets no new games; one that comes back (a restarted container has forgotten its runtime-loaded adapters)
  gets its adapter set loaded again before it takes a game. A game on an instance that has been down for two checks (or
  that has already had a request fail) is cancelled at once. A game is void when a model request failed because of the
  server: it overlaps an outage of that instance, or it was a connect error, a 5xx or a model-not-found. Its files move
  to ROOT/voided, no rows are credited, and it is replayed (same map, civs, opponent and seeds) up to `max_requeues`
  times. A request that failed on its own (a read timeout on a healthy server) leaves the game standing: the
  commander records a missed decision and the game is credited as played.
- Hard-state labels (off unless `[hard_states] enabled`, `train/hardstate.py`). It tracks the policy's rolling win rate
  against each opponent; against one below the threshold (every group of games is all losses, so RL has no signal) it
  samples up to `per_hour` decisions an hour from those games, has the calibrated teacher relabel them under an hourly
  dollar cap (the ledger counts it) and writes them to the learner's inbox as rows flagged sft, for the learner's
  auxiliary cross-entropy term. The windows and the hour's counters are saved in ROOT/orchestrator/hard.json.
- Tripwires. It tails ROOT/health.jsonl (`control.TripWatch`): a trip sends the learner a rollback to the last green
  checkpoint (which also halves its learning rate); two trips within `trip_window_h` pause the learner, upload and stop.
  No new health record or adapter for `learner_stall_minutes` while `stall_rows` rows wait in the inbox alarms and
  stops the run too.
- Budget. Every `budget_interval_s`: box hours (the box's, `budget.BoxTime`) at `box_usd_per_hour` plus the box ledger;
  at `budget_usd` the run stops; so it does after `duration_hours` of run time. Below `min_free_gb` of disk it stops
  starting games and alarms until space comes back.
- Load. One asyncio loop drives every engine's HTTP and every vLLM call (credit runs in a thread), so it is a single
  core's worth of Python for the whole run. Every minute status.json's `loop` records this process's CPU (`cpu_cores`:
  1.0 is a saturated core) and the loop's lag (`lag_ms` p50/p95/max, `train/loopstats.py`). A p95 of tens of ms or a
  `cpu_cores` near 1 means the engines wait on this loop; the follow-up is then to shard the games into worker processes
  the way `runner.py` does (one pool of engines per worker), which is not done yet.
- Durability. Every `sync_interval_s`, as a background task, it syncs the run folder to the configured S3 prefix with
  the instance role (`sync.py`): games, decision rows, health, every adapter's manifest, hourly and champion adapters,
  the latest healthy checkpoint, the run config. The hourly adapter is the version current at each hour of the run; the
  champion is `ROOT/champion.json`'s `version`.
- Panels. At each panel's `at_hours` it plays the pre-registered configs (a JSONL of game specs whose `trainable` seats
  are pinned to the panel's version, or the current one) ahead of the training games, into ROOT/panels/NAME. A panel's
  games are never credited, never reach the inbox and are never part of the training board. A `teacher_calibration`
  panel has no spec file: `calibration.py` makes its games (frontier teachers against Petra-3 and Petra-4 at the
  policy's lockstep latency, recorded in ROOT/panels/NAME/specs.jsonl); no adapter is involved, so it runs on the base
  model's instance slot and under the frontier spend caps like any frontier seat.
- Restarts. Everything that must not start over is saved in ROOT/orchestrator after the step that changes it: the
  tripwatch (`tripwatch.json`), credit (`credit.json`), the game stream's random state (`factory.json`), and the
  panels, queues, games in flight and counts (`state.json`, with `keep.json`'s hourly adapters). A restarted process
  loads them, moves the game folders the old one left half done to ROOT/voided and replays those games.
- Stopping. On completion, budget, a second trip or SIGTERM (and on an unexpected error: `run` always ends in `finish`):
  no new games; a finished run drains the games in flight (up to `drain_timeout_s`) while the other reasons cancel them
  (recorded as `stopped`, not voided or replayed); the learner is told to stop (unless `--play-only`) and given
  `learner_grace_s` to read it; the panels and queues left over are written down; the final upload runs (retried); and
  with `shutdown = true` the box powers off, unless the upload failed and an S3 prefix is configured, when it stays up
  so the results are not lost with the disk.

The orchestrator's own files are in ROOT/orchestrator (clock.json, keep.json, status.json, final.json, config.toml ...).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from agentenv_zeroad.llm import DEFAULT_TIMEOUT_S, StrategistModel
from agentenv_zeroad.spend import SpendCapReached, append_ledger, results_spend
from agentenv_zeroad.train import budget
from agentenv_zeroad.train.calibration import LITELLM_URL_ENV, panel_specs
from agentenv_zeroad.train.config import PanelConfig, RunConfig, check_run, load_config
from agentenv_zeroad.train.control import Action, ControlWriter, TripWatch
from agentenv_zeroad.train.game import ModelFactory, build_model
from agentenv_zeroad.train.loopstats import LoopMonitor
from agentenv_zeroad.train.mix import GameFactory, repin
from agentenv_zeroad.train.runner import GameOutcome
from agentenv_zeroad.train.serving import Adapters, Fleet, Instance
from agentenv_zeroad.train.spec import GameSpec
from agentenv_zeroad.train.sync import Syncer

if TYPE_CHECKING:
    from agentenv_zeroad.train.hardstate import HardStates

log = logging.getLogger(__name__)

DRAIN_REASONS = ("duration",)
SYNC_ATTEMPTS = 3
HOUSEKEEPING_S = 60.0  # disk, learner liveness, adapter pruning
LOOP_STATS_S = 60.0  # the process's CPU and event-loop lag into status.json
STEP_FAILURES_TO_STOP = 3  # consecutive failures of the budget check before the run stops rather than go unwatched
DOWN_POLLS_TO_CANCEL = 2  # checks an instance must be down before a game on it with no failed request is cancelled
DEFAULT_MIN_FREE_GB = 8.0

Play = Callable[[GameSpec, Path, ModelFactory], Awaitable[GameOutcome]]


class Credit(Protocol):
    """What the orchestrator needs of credit (`train.credit.Credit` has it): a finished game's results.json in, the
    learner's inbox rows out (each a dict `trainer/learner.py parse_row` accepts). A credit that also has
    `state_dict()` / `load_state_dict()` is saved after every game and loaded at a restart."""

    def rows(self, results: dict[str, Any]) -> list[dict[str, Any]]: ...


class PlayOnly:
    """No credit: games are played and kept but nothing is trained on (a baseline or panel run)."""

    def rows(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        return []


def is_server_error(error: BaseException) -> bool:
    """Whether a failed model request says the server (not the request) is at fault: it could not be reached, it
    answered 5xx, or it does not have the model (a vLLM that restarted has forgotten its adapters)."""
    if isinstance(error, httpx.ConnectError | httpx.ConnectTimeout):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code >= 500 or error.response.status_code == 404
    return False


@dataclass
class Pending:
    """A game waiting to be (re)started: a void game's spec to replay, or a panel game."""

    spec: GameSpec
    kind: str  # "train" or "panel"
    base_id: str  # the game id of the first attempt; a replay's is `BASE-rN`
    panel: str | None = None
    version: int | None = None  # the pinned version of a panel game
    attempt: int = 0

    @property
    def game_id(self) -> str:
        return f"{self.base_id}-r{self.attempt}" if self.attempt else self.base_id

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_json(),
            "kind": self.kind,
            "base_id": self.base_id,
            "panel": self.panel,
            "version": self.version,
            "attempt": self.attempt,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Pending:
        return cls(
            GameSpec.from_json(raw["spec"]), raw["kind"], raw["base_id"], raw["panel"], raw["version"], raw["attempt"]
        )


@dataclass
class Game:
    spec: GameSpec
    kind: str
    version: int
    url: str
    pending: Pending | None = None
    panel: str | None = None
    models: list[StrategistModel] = field(default_factory=list)
    failures: list[tuple[float, bool]] = field(default_factory=list)  # (when, the server's fault by its own error)
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    stopping: bool = False  # cancelled because the run is ending, not because its vLLM went down

    @property
    def failed_calls(self) -> int:
        return sum(m.stats["failed_calls"] for m in self.models)

    @property
    def spent_usd(self) -> float:
        return sum(m.guard.spent_usd for m in self.models)

    def as_pending(self) -> Pending:
        return self.pending or Pending(self.spec, self.kind, self.spec.game_id, self.panel, None)


@dataclass
class PanelRun:
    config: PanelConfig
    version: int
    remaining: int
    games: int
    finished: int = 0
    voided: int = 0

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "config"}


class Orchestrator:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        fleet: Fleet,
        play: Play,
        credit: Credit,
        syncer: Syncer,
        http: httpx.AsyncClient,
        now: Callable[[], float] = time.time,
        power_off: Callable[[], None] | None = None,
        learner: bool = True,
        free_gb: Callable[[], float] | None = None,
        hard_states: HardStates | None = None,
    ):
        self.cfg, self.fleet, self.play, self.credit, self.syncer, self.http = cfg, fleet, play, credit, syncer, http
        self.now = now
        self.power_off = power_off or _power_off
        self.learner = learner
        self.hard = hard_states
        self._hard_task: asyncio.Task[None] | None = None
        self.root = cfg.root
        self.odir = self.root / "orchestrator"
        self.free_gb = free_gb or self._free_gb
        self.min_free_gb = _first(cfg.min_free_gb, _env_float("ZEROAD_MIN_FREE_GB"), DEFAULT_MIN_FREE_GB)
        self.game_cap_usd = _first(cfg.game_cap_usd, _env_float("ZEROAD_MATCH_CAP_USD"), None)
        self.adapters = Adapters(self.root, cfg.servers.keep_versions)
        self.control = ControlWriter(self.root / "control")
        self.trips = TripWatch(
            self.root / "health.jsonl",
            self.root / "checkpoints",
            cfg.trip_window_h * 3600,
            cfg.trip_limit,
            state_path=self.odir / "tripwatch.json",
        )
        self.clock = budget.RunClock(self.odir / "clock.json", now())
        self.boxtime = budget.BoxTime(cfg.boxtime_path) if cfg.boxtime_path else None
        self.factory = GameFactory(cfg, start_index=self._existing_games())
        self.current = 0
        self.in_flight: dict[str, Game] = {}
        self.requeue: deque[Pending] = deque()
        self.panel_queue: deque[Pending] = deque()
        self.panels: dict[str, PanelRun] = {}
        self.hourly: dict[int, int] = {}
        self.counts: Counter[str] = Counter()
        self.alarms: dict[str, str] = {}
        self.stop_reason: str | None = None
        self.disk_paused = False
        self._wake = asyncio.Event()
        self._credit_lock = asyncio.Lock()
        self._last: dict[str, float] = {"sync": now()}  # the first sync is one interval after the start, not before it
        self._failing: Counter[str] = Counter()
        self._sync_task: asyncio.Task[None] | None = None
        self._progress: tuple[int, int] = (-1, -1)
        self._progress_at = now()
        self._waiting_since: float | None = None
        self.monitor = LoopMonitor()
        self.loop_stats: dict[str, Any] = {}
        self.last_spend: budget.Spend | None = None
        self._restore()

    # ---- the loop ------------------------------------------------------------------------------------------------
    async def run(self) -> str:
        """Run until a stop reason, then stop cleanly (always: an error in the loop is a stop reason too); returns the
        reason."""
        self.odir.mkdir(parents=True, exist_ok=True)
        self.monitor.start()
        try:
            await self._step("serving", self.refresh_serving)
            while self.stop_reason is None:
                await self.tick()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), self.cfg.poll_s)
                self._wake.clear()
        except asyncio.CancelledError:
            self.request_stop("cancelled")
            raise
        except Exception as e:  # noqa: BLE001 - whatever it was, the learner, the upload and the power-off still happen
            log.exception("the orchestrator loop failed")
            self.request_stop(f"crash: {type(e).__name__}: {str(e)[:200]}")
        finally:
            await self.monitor.stop()
            await self.finish(self.stop_reason or "crash")
        return self.stop_reason or "crash"

    async def tick(self) -> None:
        """One pass of everything that is due; the tests drive this directly. Each step is guarded: one that raises is
        logged and counted, and the others still run."""
        now = self.now()
        if self._due("panels", now, self.cfg.poll_s):
            await self._step("panels", self.start_due_panels)
        if self._due("serving", now, self.cfg.poll_s):
            await self._step("serving", self._serve_pass)
        if self._due("health", now, self.cfg.poll_s):
            await self._step("health", lambda: self.poll_health(now))
        if self._due("budget", now, self.cfg.budget_interval_s):
            await self._step("budget", lambda: self.check_budget(now))
        if self._due("loop", now, LOOP_STATS_S):
            await self._step("loop", lambda: self.record_loop(now))
        if self._due("housekeeping", now, HOUSEKEEPING_S):
            await self._step("housekeeping", lambda: self.housekeeping(now))
        if self.counts["started"] and self._due("sync", now, self.cfg.sync_interval_s):
            self._start_sync()
        if self.hard is not None and self._due("hard", now, self.cfg.hard_states.interval_s):
            self._start_hard_labels(int(self.clock.hours(now)))
        await self._step("fill", self.fill)

    async def _serve_pass(self) -> None:
        await self.refresh_serving()
        self.cancel_games_on_down_instances()

    async def _step(self, name: str, fn: Callable[[], Any]) -> None:
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - one broken step must not take the others, or the run, down
            self.counts[f"step_error_{name}"] += 1
            self._failing[name] += 1
            log.exception("step %s failed (%d in a row)", name, self._failing[name])
            if name == "budget" and self._failing[name] >= STEP_FAILURES_TO_STOP:
                self.request_stop("the budget check keeps failing: the run is not being watched")
        else:
            self._failing[name] = 0

    def _due(self, name: str, now: float, every: float) -> bool:
        if now - self._last.get(name, float("-inf")) < every:
            return False
        self._last[name] = now
        return True

    def request_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            log.warning("stopping the run: %s", reason)
            self.stop_reason = reason
        self._wake.set()

    def alarm(self, key: str, message: str | None) -> None:
        """Raise (message) or clear (None) a named alarm; status.json carries the open ones."""
        if message is None:
            if self.alarms.pop(key, None) is not None:
                log.warning("alarm %s cleared", key)
        elif self.alarms.get(key) != message:
            self.alarms[key] = message
            log.error("ALARM %s: %s", key, message)

    # ---- restarts ------------------------------------------------------------------------------------------------
    def _read(self, name: str) -> Any:
        try:
            return json.loads((self.odir / name).read_text())
        except FileNotFoundError:
            return None
        except ValueError:
            log.error("%s is unreadable: starting that part over", self.odir / name)
            return None

    def _restore(self) -> None:
        """Load what a previous process of this run saved (all of it optional)."""
        keep = self._read("keep.json")
        if keep:
            self.hourly = {int(h): int(v) for h, v in keep.get("hourly", {}).items()}
        factory = self._read("factory.json")
        if factory:
            self.factory.load_state_dict(factory)
        credit = self._read("credit.json")
        if credit and hasattr(self.credit, "load_state_dict"):
            try:
                self.credit.load_state_dict(credit)
            except (ValueError, KeyError, TypeError) as e:
                log.error("credit state not restored (%s): its baselines start over", e)
                self.counts["credit_state_lost"] += 1
        hard = self._read("hard.json")
        if hard and self.hard is not None:
            try:
                self.hard.load_state_dict(hard)
            except (ValueError, KeyError, TypeError) as e:
                log.error("hard-state windows not restored (%s): they start over", e)
        critic = self._read("critic.json")
        holder = getattr(self.credit, "critic", None)
        if critic and hasattr(holder, "load_state_dict"):
            try:
                holder.load_state_dict(critic)
            except (ValueError, KeyError, TypeError) as e:
                log.error("critic state not restored (%s)", e)
        state = self._read("state.json")
        if state:
            self._restore_state(state)
        self._set_aside_unfinished()

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.counts = Counter(state.get("counts", {}))
        self.counts["restarts"] += 1
        configs = {p.name: p for p in self.cfg.panels}
        for name, saved in state.get("panels", {}).items():
            if name in configs:
                self.panels[name] = PanelRun(configs[name], **saved)
                if saved["remaining"] > 0:
                    self.adapters.pin(saved["version"])
        self.panel_queue = deque(Pending.from_json(x) for x in state.get("panel_queue", []))
        self.requeue = deque(Pending.from_json(x) for x in state.get("requeue", []))
        for raw in reversed(
            state.get("in_flight", [])
        ):  # games the old process was playing: replay them as a new attempt
            old = Pending.from_json(raw)
            pending = replace(old, attempt=old.attempt + 1)
            if pending.attempt > self.cfg.max_requeues:
                self.counts["dropped"] += 1
                continue
            (self.panel_queue if pending.kind == "panel" else self.requeue).appendleft(pending)
            self.counts["interrupted"] += 1

    def _set_aside_unfinished(self) -> None:
        """A game folder with no results.json was cut off by a stop or a crash: it goes to ROOT/voided."""
        folders = [self.root / "games", *sorted((self.root / "panels").glob("*/"))]
        for folder in folders:
            for game in sorted(folder.iterdir()) if folder.is_dir() else ():
                if game.is_dir() and not (game / "results.json").exists():
                    self._move_to_voided(game)
                    self.counts["interrupted_folders"] += 1

    def _move_to_voided(self, folder: Path) -> None:
        voided = self.root / "voided"
        voided.mkdir(parents=True, exist_ok=True)
        target, n = voided / folder.name, 1
        while target.exists():
            target, n = voided / f"{folder.name}.{n}", n + 1
        shutil.move(str(folder), str(target))

    def save_state(self) -> None:
        in_flight = [g.as_pending().to_json() for g in self.in_flight.values()]
        self._write(
            "state.json",
            {
                "counts": dict(self.counts),
                "panels": {n: p.to_json() for n, p in self.panels.items()},
                "panel_queue": [p.to_json() for p in self.panel_queue],
                "requeue": [p.to_json() for p in self.requeue],
                "in_flight": in_flight,
            },
        )

    def save_credit(self) -> None:
        if hasattr(self.credit, "state_dict"):
            self._write("credit.json", self.credit.state_dict())

    async def save_critic(self) -> None:
        holder = getattr(self.credit, "critic", None)
        if hasattr(holder, "state_dict"):
            async with self._credit_lock:
                state = await asyncio.to_thread(holder.state_dict)
            self._write("critic.json", state)

    # ---- serving: adapters and the watchdog ----------------------------------------------------------------------
    async def refresh_serving(self) -> None:
        """Read every instance, make them serve the wanted adapters, and move `current` to the newest version they all
        hold. The previous current stays loaded until the new one is everywhere."""
        for inst in await self.fleet.check(self.now()):
            log.info("vLLM %s is %s, serving adapters %s", inst.url, "up" if inst.up else "down", sorted(inst.loaded))
        on_disk = self.adapters.on_disk()
        newest = on_disk[-1] if on_disk else 0
        await self.fleet.reconcile(self.adapters.wanted(newest, self.current))
        everywhere = self.fleet.newest_everywhere(on_disk)
        if everywhere > self.current:
            log.info("adapter v%d is current", everywhere)
            self.current = everywhere
        worst = self.fleet.worst_load_failures()
        if worst >= self.cfg.max_load_failures:
            stuck = [f"{url} v{v}" for (url, v), n in self.fleet.load_failures.items() if n >= worst]
            self.alarm("adapter_loads", f"{worst} loads in a row failed: {', '.join(stuck)}")
            self.request_stop(f"adapter loads keep failing ({', '.join(stuck)}): is /train mounted at the run's root?")
        else:
            self.alarm("adapter_loads", None)

    def _instance(self, url: str) -> Instance | None:
        return next((i for i in self.fleet.instances if i.url == url), None)

    def cancel_games_on_down_instances(self) -> None:
        """A game on an instance that is down can't go on: end it now (it is void) instead of letting it play out. One
        whose request has already failed goes at the first check; one that has not waits for a second, so a single
        slow /health does not cost the games on a busy server."""
        for game in self.in_flight.values():
            inst = self._instance(game.url)
            if inst is None or inst.up or game.cancelled or game.task is None:
                continue
            if game.failed_calls or inst.down_polls >= DOWN_POLLS_TO_CANCEL:
                log.warning("cancelling %s: its vLLM %s is down", game.spec.game_id, game.url)
                game.cancelled = True
                game.task.cancel()

    # ---- games ---------------------------------------------------------------------------------------------------
    def _existing_games(self) -> int:
        """One past the highest game number already on disk, in games/ or voided/."""
        pattern = re.compile(rf"^{re.escape(self.cfg.run_id)}-g(\d+)")
        found = (
            pattern.match(p.name) for folder in (self.root / "games", self.root / "voided") if folder.is_dir()
            for p in folder.iterdir()
        )  # fmt: skip
        return max((int(m.group(1)) + 1 for m in found if m), default=0)

    def _pick(self, version: int) -> Instance | None:
        load = Counter(g.url for g in self.in_flight.values())
        return min(self.fleet.serving(version), key=lambda i: load[i.url], default=None)

    def fill(self) -> None:
        """Start games until `concurrency` are in flight, panel games first, then replays, then new training games. A
        box short of disk starts none."""
        while self.stop_reason is None and not self.disk_paused and len(self.in_flight) < self.cfg.concurrency:
            if not (self._start_pending(self.panel_queue) or self._start_pending(self.requeue) or self._start_new()):
                return

    def _start_new(self) -> bool:
        inst = self._pick(self.current)
        if inst is None:
            return False
        spec = self.factory.next(self.current, inst.url)
        self._write("factory.json", self.factory.state_dict())
        self._launch(Game(spec, "train", self.current, inst.url))
        return True

    def _start_pending(self, queue: deque[Pending]) -> bool:
        for pending in queue:
            version = pending.version if pending.version is not None else self.current
            inst = self._pick(version)
            if inst is None:
                continue
            queue.remove(pending)
            spec = repin(pending.spec, version, inst.url, game_id=pending.game_id)
            self._launch(Game(spec, pending.kind, version, inst.url, pending=pending, panel=pending.panel))
            return True
        return False

    def _out_dir(self, game: Game) -> Path:
        return self.root / "panels" / game.panel if game.panel else self.root / "games"

    def _launch(self, game: Game) -> None:
        self.adapters.pin(game.version)
        self.in_flight[game.spec.game_id] = game
        self.counts["started"] += 1

        def model_for(seat: Any, slot: str) -> StrategistModel:
            model = build_model(seat, slot, http=self.http, cap_usd=self.game_cap_usd)
            self._track_failures(model, game)
            game.models.append(model)
            return model

        game.task = asyncio.create_task(self._play(game, model_for))
        self.save_state()
        log.info("started %s (%s, v%d on %s)", game.spec.game_id, game.kind, game.version, game.url)

    def _track_failures(self, model: StrategistModel, game: Game) -> None:
        """Note when each of the game's model requests failed and whether the error itself blames the server."""
        call = model.call

        async def tracked(*args: Any, **kwargs: Any) -> Any:
            try:
                return await call(*args, **kwargs)
            except SpendCapReached:
                raise
            except Exception as e:
                game.failures.append((self.now(), is_server_error(e)))
                raise

        model.call = tracked  # type: ignore[method-assign]

    async def _play(self, game: Game, model_for: ModelFactory) -> None:
        out_dir = self._out_dir(game)
        try:
            try:
                outcome = await self.play(game.spec, out_dir, model_for)
            except asyncio.CancelledError:
                if not game.cancelled:
                    raise
                if game.stopping:
                    outcome = GameOutcome(game.spec.game_id, "stopped", detail="the run ended")
                else:
                    outcome = GameOutcome(game.spec.game_id, "cancelled", detail="its vLLM went down")
            except Exception as e:  # noqa: BLE001 - one game's failure must not stop the run
                log.exception("game %s raised", game.spec.game_id)
                outcome = GameOutcome(game.spec.game_id, "failed", detail=f"{type(e).__name__}: {str(e)[:300]}")
            try:
                await self._finish_game(game, out_dir, outcome)
            except Exception:  # noqa: BLE001 - bookkeeping that fails is logged; the game's slot is freed below
                log.exception("finishing %s failed", game.spec.game_id)
                self.counts["finish_errors"] += 1
        finally:
            self.adapters.release(game.version)
            self.in_flight.pop(game.spec.game_id, None)
            self.save_state()
            self._wake.set()

    def void_reason(self, game: Game) -> str | None:
        """Why this game's failed model requests void it, or None when it stands: one whose own error blames the server
        (connect error, 5xx, model not found) or that happened while its instance was in an outage that was seen."""
        inst = self._instance(game.url)
        for when, server_fault in game.failures:
            if server_fault:
                return "a request failed with a server error"
            if inst is not None and inst.overlaps_outage(when, margin_s=self.cfg.poll_s):
                return "a request failed during an outage"
        return None

    async def _finish_game(self, game: Game, out_dir: Path, outcome: GameOutcome) -> None:
        rows, error, results = 0, None, None
        reason = None if outcome.status == "stopped" else self.void_reason(game)
        if outcome.status == "cancelled" and reason is None:
            reason = "its vLLM went down"
        if outcome.status == "stopped":
            self.counts[f"{game.kind}_stopped"] += 1
        elif reason is not None:
            self._void(game, out_dir, outcome, reason)
        elif outcome.status == "finished":
            results, error = await self._read_results(out_dir, game)
            if results is None:
                self.counts[f"{game.kind}_failed"] += 1
                self._panel_done(game, finished=False)
            else:
                if game.kind == "train":
                    rows, error = await self._credit_game(game, results)
                self.counts[f"{game.kind}_finished"] += 1
                self.counts["missed_calls"] += game.failed_calls
                self._panel_done(game, finished=True)
        else:
            self.counts[f"{game.kind}_failed"] += 1
            log.warning("game %s %s: %s", game.spec.game_id, outcome.status, outcome.detail)
            self._panel_done(game, finished=False)
        self._record_spend(game, results, void=reason is not None)
        self._ledger_line(out_dir, game, outcome, rows, error=error, void_reason=reason)

    async def _read_results(self, out_dir: Path, game: Game) -> tuple[dict[str, Any] | None, str | None]:
        path = out_dir / game.spec.game_id / "results.json"
        try:
            return json.loads(await asyncio.to_thread(path.read_text)), None
        except (OSError, ValueError) as e:
            self.counts["results_missing"] += 1
            log.error("game %s finished but its results.json is unusable: %s", game.spec.game_id, e)
            return None, f"results.json: {type(e).__name__}: {str(e)[:200]}"

    async def _credit_game(self, game: Game, results: dict[str, Any]) -> tuple[int, str | None]:
        try:
            async with self._credit_lock:
                return await asyncio.to_thread(self._credit, game.spec.game_id, results), None
        except Exception as e:  # noqa: BLE001 - credit.rows raises its own errors; none of them may end the run
            self.counts["credit_failed"] += 1
            log.exception("credit failed for %s", game.spec.game_id)
            return 0, f"credit: {type(e).__name__}: {str(e)[:200]}"

    def _credit(self, game_id: str, results: dict[str, Any]) -> int:
        rows = self.credit.rows(results)
        write_rows(self.root / "inbox", game_id, rows)
        self.counts["rows"] += len(rows)
        self.save_credit()
        if self.hard is not None:
            if self.hard.observe(results, game_id):
                self.counts["hard_games"] += 1
            self._write("hard.json", self.hard.state_dict())
        return len(rows)

    def _record_spend(self, game: Game, results: dict[str, Any] | None, *, void: bool) -> None:
        """The game's frontier spend into the box ledger, whatever became of the game: a void or failed game was paid
        for all the same. The larger of the results' figure and what the seats' guards counted."""
        spent = max(results_spend(results) if results else 0.0, game.spent_usd)
        if spent:
            append_ledger(self.cfg.ledger_path, f"{game.kind} {game.spec.game_id}{' (void)' if void else ''}", spent)

    def _void(self, game: Game, out_dir: Path, outcome: GameOutcome, reason: str) -> None:
        """A game whose model requests failed: keep its files aside, replay it unless it has used its tries."""
        self.counts["voided"] += 1
        folder = out_dir / game.spec.game_id
        if folder.exists():
            self._move_to_voided(folder)
        pending = game.pending or Pending(game.spec, game.kind, game.spec.game_id)
        if pending.attempt >= self.cfg.max_requeues:
            log.error("dropping %s: void %d times", game.spec.game_id, pending.attempt + 1)
            self.counts["dropped"] += 1
            self._panel_done(game, finished=False)
            return
        log.warning(
            "void %s (%s; %d model requests failed, %s): replaying it",
            game.spec.game_id, reason, game.failed_calls, outcome.status,
        )  # fmt: skip
        replay = replace(pending, attempt=pending.attempt + 1)
        (self.panel_queue if game.panel else self.requeue).appendleft(replay)
        self.counts["requeued"] += 1
        if game.panel and game.panel in self.panels:
            self.panels[game.panel].voided += 1

    def _ledger_line(
        self, out_dir: Path, game: Game, outcome: GameOutcome, rows: int, *, error: str | None, void_reason: str | None
    ) -> None:
        line = {**asdict(outcome), "kind": game.kind, "version": game.version, "url": game.url, "rows": rows}
        line["failed_calls"] = game.failed_calls
        if void_reason:
            line["void"] = void_reason
        if error:
            line["error"] = error
        self._append_game_line(out_dir, line)

    def _append_game_line(self, out_dir: Path, line: dict[str, Any]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "games.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")

    # ---- tripwires and budget ------------------------------------------------------------------------------------
    def poll_health(self, now: float) -> None:
        for action in self.trips.poll(now):
            self.apply_trip(action)

    def apply_trip(self, action: Action) -> None:
        if action.kind == "rollback" and action.version is not None:
            self.counts["rollbacks"] += 1
            self.control.send({"op": "rollback", "version": action.version})
            log.warning("rolling the learner back to version %d", action.version)
        else:
            self.counts["pauses"] += 1
            self.request_stop(f"tripwire: {[t.get('message') for t in action.trips]}")

    def check_budget(self, now: float) -> budget.Spend:
        spent = budget.spend(self.clock, now, self.cfg.box_usd_per_hour, self.cfg.ledger_path, self.boxtime)
        self.clock.save(now)
        self._copy_box_files()
        hour = int(self.clock.hours(now))
        if hour >= 1 and self.current and hour not in self.hourly:
            self.hourly[hour] = self.current
            self._write("keep.json", {"hourly": self.hourly})
        log.info("budget: %s", spent.line(self.cfg.budget_usd))
        self.last_spend = spent
        self.write_status(now, spent)
        if spent.total >= self.cfg.budget_usd:
            self.request_stop(f"budget: {spent.line(self.cfg.budget_usd)}")
        elif self.clock.hours(now) >= self.cfg.duration_hours:
            self.request_stop("duration")
        return spent

    def _copy_box_files(self) -> None:
        """The box's ledger and box-time files live outside ROOT; a copy goes where the sync takes it."""
        for source in (self.cfg.ledger_path, self.cfg.boxtime_path):
            if source is not None and source.exists() and source.parent != self.root:
                with contextlib.suppress(OSError):
                    self.odir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, self.odir / f"box-{source.name}")

    def record_loop(self, now: float) -> None:
        """Once a minute: this process's CPU and event-loop lag into status.json, so a saturated orchestrator shows."""
        self.loop_stats = self.monitor.snapshot()
        if self.last_spend is not None:
            self.write_status(now, self.last_spend)

    def write_status(self, now: float, spent: budget.Spend) -> None:
        self._write(
            "status.json",
            {
                "time": now,
                "hours": round(spent.box_hours, 3),
                "run_hours": round(self.clock.hours(now), 3),
                "spend": {"box": round(spent.box_usd, 2), "model": round(spent.model_usd, 2)},
                "current": self.current,
                "in_flight": {g: f"{x.kind} v{x.version}" for g, x in self.in_flight.items()},
                "counts": dict(self.counts),
                "instances": [{"url": i.url, "up": i.up, "adapters": sorted(i.loaded)} for i in self.fleet.instances],
                "panels": {n: {"version": p.version, "remaining": p.remaining} for n, p in self.panels.items()},
                "hard_states": self.hard.status(int(self.clock.hours(now))) if self.hard is not None else None,
                "alarms": dict(self.alarms),
                "paused_for_disk": self.disk_paused,
                "sync": {"last_ok": self.syncer.last_ok, "failures": self.syncer.failures},
                "loop": self.loop_stats,
                "stop_reason": self.stop_reason,
            },
        )

    def _write(self, name: str, value: dict[str, Any]) -> None:
        self.odir.mkdir(parents=True, exist_ok=True)
        tmp = self.odir / f".{name}.tmp"
        tmp.write_text(json.dumps(value))
        os.replace(tmp, self.odir / name)

    # ---- housekeeping: disk, the learner's pulse, old adapters --------------------------------------------------
    async def housekeeping(self, now: float) -> None:
        self.check_disk()
        self.check_learner(now)
        if self._due("critic", now, 600.0):
            await self._step("critic", self.save_critic)
        await self._step("prune", self.prune_adapters)

    def _free_gb(self) -> float:
        folder = self.root
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        return shutil.disk_usage(folder).free / 1e9

    def check_disk(self) -> None:
        free = self.free_gb()
        if free < self.min_free_gb:
            self.disk_paused = True
            self.alarm("disk", f"{free:.1f} GB free is below {self.min_free_gb:g} GB: no new games until there is room")
        else:
            self.disk_paused = False
            self.alarm("disk", None)

    def _inbox_rows(self) -> int:
        inbox = self.root / "inbox"
        rows = 0
        for path in inbox.glob("*.jsonl") if inbox.is_dir() else ():
            with contextlib.suppress(OSError), path.open("rb") as f:
                rows += sum(1 for _ in f)
        return rows

    def check_learner(self, now: float) -> None:
        """Alarm and stop when the learner has made no progress (no health record, no adapter) for
        `learner_stall_minutes` while `stall_rows` rows wait in the inbox: it is down or wedged, and the games being
        played train nothing."""
        if not self.learner:
            return
        health = self.root / "health.jsonl"
        on_disk = self.adapters.on_disk()
        sig = (health.stat().st_size if health.exists() else 0, on_disk[-1] if on_disk else 0)
        if sig != self._progress:
            self._progress, self._progress_at = sig, now
        rows = self._inbox_rows()
        if rows < self.cfg.stall_rows:
            self._waiting_since = None
            self.alarm("learner", None)
            return
        if self._waiting_since is None:
            self._waiting_since = now
        quiet_s = now - max(self._progress_at, self._waiting_since)
        if quiet_s > self.cfg.learner_stall_minutes * 60:
            message = f"{rows} rows wait and the learner has shown no progress for {quiet_s / 60:.0f} min"
            self.alarm("learner", message)
            self.request_stop(f"learner stalled: {message}")

    def prune_adapters(self) -> None:
        """Delete the published versions nothing wants: not the newest or current, a pinned, hourly, champion or
        configured version, one a vLLM still holds, or one whose manifest a sync has not yet taken."""
        on_disk = self.adapters.on_disk()
        if not on_disk:
            return
        keep = (
            self.keep_adapters(final=True) | set(self.adapters.pins) | self.adapters.keep | {on_disk[-1], self.current}
        )
        keep |= {p.version for p in self.cfg.panels if p.version is not None}
        for inst in self.fleet.instances:
            keep |= inst.loaded | inst.misrooted
        synced = self.syncer.last_start_ok
        for version in on_disk:
            folder = self.adapters.dir / f"v{version}"
            if version in keep or version > self.current:
                continue
            if self.syncer.uri is not None and (synced is None or folder.stat().st_mtime > synced):
                continue
            shutil.rmtree(folder, ignore_errors=True)
            self.counts["pruned"] += 1
            log.info("pruned adapter v%d", version)

    # ---- panels --------------------------------------------------------------------------------------------------
    def start_due_panels(self) -> None:
        hours = self.clock.hours(self.now())
        for panel in self.cfg.panels:
            if panel.name in self.panels or hours < panel.at_hours:
                continue
            calibration = panel.kind == "teacher_calibration"
            version = 0 if calibration else panel.version if panel.version is not None else self.current
            if version and version not in self.adapters.on_disk():
                log.error("panel %s: adapter v%d is not on disk; skipping it", panel.name, version)
                self.panels[panel.name] = PanelRun(panel, version, 0, 0)
                self.save_state()
                continue
            specs = panel_specs(self.cfg, panel)
            if calibration:
                self._write_panel_specs(panel, specs)
            self.adapters.pin(version)
            self.panels[panel.name] = PanelRun(panel, version, len(specs), len(specs))
            for spec in specs:
                base_id = f"{panel.name}-v{version}-{spec.game_id}"
                self.panel_queue.append(Pending(spec, "panel", base_id, panel.name, version))
            self.save_state()
            log.info("panel %s: %d games on v%d", panel.name, len(specs), version)

    def _write_panel_specs(self, panel: PanelConfig, specs: Sequence[GameSpec]) -> None:
        """A generated panel's specs, kept beside its games: what was played stays on record."""
        folder = self.root / "panels" / panel.name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "specs.jsonl").write_text("".join(json.dumps(s.to_json()) + "\n" for s in specs))

    def _panel_done(self, game: Game, *, finished: bool) -> None:
        run = self.panels.get(game.panel or "")
        if run is None:
            return
        run.remaining -= 1
        run.finished += finished
        if run.remaining == 0:
            self.adapters.release(run.version)
            self._write_panel_summary(run)

    def _write_panel_summary(self, run: PanelRun, *, stopped: bool = False) -> None:
        folder = self.root / "panels" / run.config.name
        folder.mkdir(parents=True, exist_ok=True)
        summary = {"name": run.config.name, "version": run.version, "games": run.games, "finished": run.finished}
        summary["voided"] = run.voided
        if stopped:
            summary |= {"stopped": True, "unfinished": run.remaining}
        (folder / "summary.json").write_text(json.dumps(summary))
        log.info(
            "panel %s %s: %d of %d games finished",
            run.config.name,
            "stopped" if stopped else "done",
            run.finished,
            run.games,
        )

    # ---- durability and the end ----------------------------------------------------------------------------------
    def keep_adapters(self, *, final: bool = False) -> set[int]:
        champion = self.root / "champion.json"
        kept = set(self.hourly.values())
        if champion.exists():
            with contextlib.suppress(ValueError, KeyError, TypeError):
                kept.add(int(json.loads(champion.read_text())["version"]))
        if final and self.current:
            kept.add(self.current)
        return kept - {0}

    async def sync_now(self, *, final: bool = False) -> bool:
        return await self.syncer.sync(adapters=self.keep_adapters(final=final), checkpoint=self.trips.latest_healthy())

    def _start_sync(self) -> None:
        """Sync in the background: a slow or hung upload must not hold up the games or the watchdog."""
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_task_body())

    async def _sync_task_body(self) -> None:
        try:
            await self.sync_now()
        except Exception:  # noqa: BLE001
            log.exception("the background sync failed")

    def _start_hard_labels(self, hour: int) -> None:
        """One labelling pass in the background: a slow teacher must not hold up the games or the watchdog."""
        if self._hard_task is None or self._hard_task.done():
            self._hard_task = asyncio.create_task(self._hard_labels(hour))

    async def _hard_labels(self, hour: int) -> None:
        assert self.hard is not None
        try:
            rows, info = await self.hard.label(hour)
            if rows:
                write_rows(self.root / "inbox", f"hard-{hour:03d}-{self.hard.passes:05d}", rows)
                self.counts["hard_rows"] += len(rows)
            self.counts["hard_asked"] += info["asked"]
            self._write("hard.json", self.hard.state_dict())
            if info["asked"]:
                log.info("hard-state labels: %s", info)
        except Exception:  # noqa: BLE001 - labels are an extra: a failed pass is counted and the run goes on
            self.counts["hard_errors"] += 1
            log.exception("a hard-state labelling pass failed")

    async def join_hard(self) -> None:
        if self._hard_task is not None:
            await asyncio.shield(self._hard_task)

    async def join_sync(self) -> None:
        """Wait for a sync in progress (tests, and the final upload, which must not overlap it)."""
        if self._sync_task is not None:
            await asyncio.shield(self._sync_task)

    async def finish(self, reason: str) -> None:
        """No new games (`fill` is already off); drain or cancel the ones in flight, stop the learner, write down what
        was left, upload, and power off if the config says so and the results are safe. Every stage is attempted even
        if an earlier one fails."""
        if reason in DRAIN_REASONS:
            await self._stage("drain", self._drain(self.cfg.drain_timeout_s))
        await self._stage("cancel", self._cancel_all())
        if self.learner:
            await self._stage("stop the learner", self._stop_learner())
        await self._stage("save critic", self.save_critic())
        await self._stage("hard-state labels", self.join_hard())
        now = self.now()
        spent: budget.Spend | None = None
        with contextlib.suppress(Exception):
            spent = self.check_budget(now)
        self._record_stop()
        await self._stage("sync in progress", self.join_sync())
        ok = False
        for attempt in range(SYNC_ATTEMPTS):
            self._write(
                "final.json", {"reason": reason, "time": now, "spent_usd": round(spent.total, 2) if spent else None}
            )
            ok = await self.sync_now(final=True)
            if ok or self.syncer.uri is None or attempt == SYNC_ATTEMPTS - 1:
                break
            await asyncio.sleep(self.cfg.sync_retry_s)
        log.info("run stopped (%s); final upload %s", reason, "done" if ok else "FAILED or not configured")
        if not self.cfg.shutdown:
            return
        if self.syncer.uri is not None and not ok:
            log.critical(
                "NOT powering off: the final upload to %s failed and the run's files are only on this box's disk "
                "(train.sh status; aws s3 sync by hand, then stop it)", self.syncer.uri,
            )  # fmt: skip
            self.alarm("final_upload", "the final upload failed; the box stays up")
            self._write("final.json", {"reason": reason, "time": now, "upload_failed": True})
            return
        self.power_off()

    async def _stage(self, name: str, work: Awaitable[Any]) -> None:
        try:
            await work
        except Exception:  # noqa: BLE001 - the later stages of a stop must still run
            log.exception("stopping: %s failed", name)

    def _record_stop(self) -> None:
        """Write down what a stop left: the games that never started (dropped) and the panels that did not finish."""
        for queue in (self.panel_queue, self.requeue):
            for pending in queue:
                self.counts["dropped_at_stop"] += 1
                out_dir = self.root / "panels" / pending.panel if pending.panel else self.root / "games"
                self._append_game_line(
                    out_dir,
                    {"game_id": pending.game_id, "status": "dropped", "kind": pending.kind, "detail": "the run ended"},
                )
        for run in self.panels.values():
            if run.remaining > 0:
                self._write_panel_summary(run, stopped=True)
        self.save_state()

    async def _drain(self, timeout_s: float) -> None:
        tasks = [g.task for g in self.in_flight.values() if g.task]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s)

    async def _cancel_all(self) -> None:
        games = list(self.in_flight.values())
        for game in games:
            game.cancelled = game.stopping = True
            if game.task:
                game.task.cancel()
        await asyncio.gather(*(g.task for g in games if g.task), return_exceptions=True)

    async def _stop_learner(self) -> None:
        self.control.send({"op": "stop"})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cfg.learner_grace_s
        while self.control.pending() and loop.time() < deadline:
            await asyncio.sleep(min(1.0, max(0.0, deadline - loop.time())))
        if self.control.pending():
            log.warning("the learner did not read its stop command in %.0f s", self.cfg.learner_grace_s)


def _env_float(name: str) -> float | None:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return None


def _first(*values: float | None) -> Any:
    return next((v for v in values if v is not None), None)


def write_rows(inbox: Path, game_id: str, rows: Sequence[dict[str, Any]]) -> Path | None:
    """The learner's inbox protocol: write `.GAME.jsonl.tmp`, fsync, rename to the unique `GAME.jsonl`."""
    if not rows:
        return None
    inbox.mkdir(parents=True, exist_ok=True)
    tmp, final = inbox / f".{game_id}.jsonl.tmp", inbox / f"{game_id}.jsonl"
    with tmp.open("w") as f:
        f.write("".join(json.dumps(row) + "\n" for row in rows))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return final


def _power_off() -> None:
    subprocess.run(["shutdown", "-h", "now"], check=False)


# ---- the command line --------------------------------------------------------------------------------------------
def _fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}


def load_credit(cfg: RunConfig | None = None) -> Credit:
    """The learning stream's credit (`train.credit.Credit`), configured from the run config's `[credit]` table
    (`CreditConfig` fields: positive_control, margin_scale, on_error, ...) and, when it has a `[credit.critic]` table,
    with the critic (`train.critic.GBCritic`, `CriticConfig` fields). A clear error until credit is in this tree."""
    try:
        from agentenv_zeroad.train.credit import Credit as RealCredit
        from agentenv_zeroad.train.credit import CreditConfig
    except ImportError as e:
        raise SystemExit("agentenv_zeroad.train.credit is not installed: run with --play-only, or merge it") from e
    try:
        config = CreditConfig(**_fields(dict(cfg.credit))) if cfg and cfg.credit else None
        critic = None
        if cfg and cfg.critic is not None:
            from agentenv_zeroad.train.critic import CriticConfig, GBCritic

            critic = GBCritic(CriticConfig(**_fields(dict(cfg.critic))))
    except TypeError as e:
        raise SystemExit(f"the run config's [credit] table is not a CreditConfig/CriticConfig: {e}") from e
    return RealCredit(config, critic=critic)


def load_hard_states(cfg: RunConfig, tools: Sequence[Any], http: httpx.AsyncClient) -> HardStates | None:
    """The hard-state labeller of a run whose config enables it: the policy's prompt and order tools, a teacher client
    on the LiteLLM endpoint (its key from the environment, never printed) and the policy's tokenizer, loaded when the
    first labels are written."""
    hard = cfg.hard_states
    if not hard.enabled:
        return None
    from agentenv_zeroad import brains
    from agentenv_zeroad.spend import SpendGuard
    from agentenv_zeroad.train.distill import Policy
    from agentenv_zeroad.train.hardstate import HardStates

    iface = cfg.games.interface
    policies = {iface: Policy(brains.strategist_system(iface), tuple(brains.offered_tools(tools, iface)))}
    url, key = hard.base_url or os.environ[LITELLM_URL_ENV], os.environ[hard.api_key_env]

    def make_teacher(model: str, guard: SpendGuard) -> StrategistModel:
        return StrategistModel(model, url, key, guard, effort="low", max_tokens=1024, http=http)

    def tokenizer() -> Any:
        from transformers import AutoTokenizer  # the trainer image's, not the base install's

        return AutoTokenizer.from_pretrained(hard.tokenizer)

    return HardStates(
        hard, root=cfg.root, ledger=cfg.ledger_path, policies=policies, make_teacher=make_teacher,
        tokenizer=tokenizer, seed=cfg.seed,
    )  # fmt: skip


async def serve(args: argparse.Namespace) -> str:
    from agentenv_zeroad.pool import EnginePool
    from agentenv_zeroad.server import tool_specs
    from agentenv_zeroad.train.runner import run_game

    cfg = load_config(args.config)
    try:
        frontier_keys = check_run(cfg)
    except ValueError as e:
        raise SystemExit(f"the run config does not make a runnable run: {e}") from e
    missing = [k for k in [cfg.servers.api_key_env, *frontier_keys] if not os.environ.get(k)]
    needs_url = any(p.kind == "teacher_calibration" and not p.base_url for p in cfg.panels)
    needs_url = needs_url or (cfg.hard_states.enabled and not cfg.hard_states.base_url)
    if needs_url and not os.environ.get(LITELLM_URL_ENV):
        missing.append(LITELLM_URL_ENV)
    if missing:
        raise SystemExit(f"not set: {', '.join(missing)}")
    credit = PlayOnly() if args.play_only else load_credit(cfg)
    (cfg.root / "orchestrator").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.root / "orchestrator" / "config.toml")
    pool_options: dict[str, Any] = {
        "mode": args.mode,
        "image": args.image or cfg.engine_image,
        "require_patches": cfg.engine_patches,
    }
    limits = httpx.Limits(max_connections=512)
    async with (
        httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S, limits=limits) as http,
        EnginePool(cfg.engines, **pool_options) as pool,
    ):
        tools = await tool_specs()

        async def play(spec: GameSpec, out_dir: Path, model_for: ModelFactory) -> GameOutcome:
            return await run_game(spec, pool, out_dir, tools=tools, model_for=model_for)

        orchestrator = Orchestrator(
            cfg,
            fleet=Fleet(cfg.servers, os.environ[cfg.servers.api_key_env], http),
            play=play,
            credit=credit,
            syncer=Syncer(cfg.root, cfg.s3_uri),
            http=http,
            learner=not args.play_only,
            hard_states=None if args.play_only else load_hard_states(cfg, tools, http),
        )
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, orchestrator.request_stop, f"signal {sig.name}")
        return await orchestrator.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentenv_zeroad.train.orchestrate", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a training run from a config")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--play-only", action="store_true", help="credit nothing: play and keep games, train nothing")
    run.add_argument("--mode", choices=("docker", "local"), default="docker")
    run.add_argument("--image", default=None, help="the engine image (docker mode)")
    sub.add_parser("tools", help="print the order tools the policy is offered (interface v2), comma-separated")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cmd == "tools":
        from agentenv_zeroad.brains import offered_tools
        from agentenv_zeroad.server import tool_specs

        # the order tools the policy is offered on its interface (v2), not every tool the env has (get_brief, ...)
        print(",".join(t.name for t in offered_tools(asyncio.run(tool_specs()), "v2")))
        return 0
    reason = asyncio.run(serve(args))
    log.info("orchestrator done: %s", reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
