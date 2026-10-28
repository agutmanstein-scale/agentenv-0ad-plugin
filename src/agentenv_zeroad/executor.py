"""The per-seat executor: turns standing orders into real 0 A.D. commands every tick (spec §4.4).

Each tick (default 1 s) it reads the seat's `SeatView` (ids and positions, `view.py`) and, if given, the seat's U2
`PlayerSnapshot` (fog-limited enemies), then decides in a fixed priority order:
1. a house when population is about to cap (autopilot), and the phase advance if ordered;
2. research and buildings from the order book (placement through the engine's own check, `placement.py`);
3. builders for unfinished foundations, production from the order book, villagers up to the target (autopilot);
4. the army posture; idle and surplus gatherers to the economy split; stances; drop-sites and fields (autopilot).
A spend budget runs through the tick so a lower priority never spends what a higher one is saving for (the phase
advance reserves its cost once its requirements are met).

Commands go to the clock's per-player queue (`GameClock.queue`, sync, never blocks); the tick itself only awaits the
placement `/evaluate`. Idempotent across ticks: a unit, building or order that was just commanded is left alone until
the command shows in a later view or a cooldown passes, so a standing order never re-issues the same command every
tick. The engine answers commands with no result, so the executor checks effects instead: a `construct`, `train` or
`research` whose effect hasn't shown within `VERIFY_MS` is recorded in `rejections` (and logged) as rejected by the
engine, the order it served is put back and retried, and after `MAX_ATTEMPTS` it is recorded in `failures` (with the
reason) instead of disappearing. A tick is all or nothing: commands, effect checks and cooldowns are committed only
when the whole tick ran, so a tick that raises sends nothing, logs no false rejection and puts its orders back.

Command JSON follows the public mod's `simulation/helpers/Commands.js` (0.28).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agentenv_zeroad.army import ArmyDriver
from agentenv_zeroad.autopilot import Autopilot, worker_units
from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.commands import (
    cmd_construct,
    cmd_research,
    cmd_train,
)
from agentenv_zeroad.engine import GameEngine
from agentenv_zeroad.orders import (
    AdvancePhase,
    Army,
    Build,
    BuildRequest,
    OrderBook,
    OrderResult,
    Research,
    Say,
    SetEconomy,
    Train,
    TrainRequest,
    _Order,
    accepted,
    partial,
    rejected,
)
from agentenv_zeroad.placement import EnginePlacer, Placer, enemy_base_guess, search_for
from agentenv_zeroad.roles import RoleCatalog, RoleTable
from agentenv_zeroad.state import RESOURCES, PlayerSnapshot, Snapshot
from agentenv_zeroad.view import ExecutorView, Pos, SeatView, Structure, ViewQuery, distance

log = logging.getLogger(__name__)

TICK_S = 1.0
VERIFY_MS = 3_000
MAX_ATTEMPTS = 3
BUILDERS_PER_SITE = 3
MAX_QUEUE = 2
MAX_BATCH = 5

Queue = Callable[[int, dict[str, Any]], None]
PlaceOutcome = Literal["placed", "unaffordable", "waiting", "failed"]


@dataclass
class Issued:
    """A command whose effect should show in a later view; `check` says whether it has.

    `undo` puts back what the command took from its order (a build to place, units still to train); `rejected` runs
    when the engine refused it (undo, then count the attempt). `role`/`count` describe a train command.
    """

    what: str
    command: dict[str, Any]
    time_ms: int
    check: Callable[[SeatView], bool]
    request: BuildRequest | None = None
    role: str | None = None
    count: int = 0
    undo: Callable[[], None] | None = None
    rejected: Callable[[int], None] | None = None


@dataclass(frozen=True)
class Rejection:
    time_ms: int
    player: int
    what: str
    command: dict[str, Any]


@dataclass(frozen=True)
class Failure:
    """An accepted order the executor gave up on, and why (for the seat's results and brief)."""

    time_ms: int
    player: int
    key: str  # structure key, unit role or tech name
    what: str
    reason: str
    where: str = ""  # a build's anchor (`base`, `near_wood`, ...)


@dataclass
class Tick:
    """What one tick may still use (resources after reservations, units already given a job this tick) and what it
    has decided so far (commands, effect checks, cooldown marks), committed only when the whole tick ran."""

    seat: SeatView
    snapshot: PlayerSnapshot | None
    budget: dict[str, int]
    busy: set[int] = field(default_factory=set)
    commands: list[dict[str, Any]] = field(default_factory=list)
    issued: list[Issued] = field(default_factory=list)
    marks: dict[Any, int] = field(default_factory=dict)

    def affordable(self, cost: Mapping[str, int], count: int = 1) -> int:
        """How many of `cost` (up to `count`) the budget pays for now."""
        n = count
        for r, c in cost.items():
            if c > 0:
                n = min(n, self.budget.get(r, 0) // c)
        return max(n, 0)

    def spend(self, cost: Mapping[str, int], count: int = 1) -> None:
        for r, c in cost.items():
            self.budget[r] = self.budget.get(r, 0) - c * count

    def reserve(self, cost: Mapping[str, int]) -> None:
        """Hold back `cost` from everything decided after this point in the tick."""
        self.spend(cost)


class Executor:
    """Carries out one seat's order book and runs its autopilot."""

    def __init__(
        self,
        player: int,
        roles: RoleTable,
        *,
        queue: Queue,
        placer: Placer,
        players: Sequence[int] = (),
        book: OrderBook | None = None,
        autopilot: Autopilot | None = None,
    ):
        self.player = player
        self.roles = roles
        self.queue = queue
        self.placer = placer
        self.players = tuple(p for p in players if p != player)
        self.fallen: frozenset[int] = frozenset()  # opponents the engine has defeated (the match keeps it current)
        self.book = book or OrderBook(player)
        self.autopilot = autopilot or Autopilot()
        self.last_seat: SeatView | None = None
        self.last_snapshot: PlayerSnapshot | None = None
        self.pending: list[Issued] = []
        self.rejections: list[Rejection] = []
        self.issued: Counter[str] = Counter()
        self.placed: list[tuple[str, str, int]] = []  # (structure, where, time_ms) confirmed by a later view
        self.failures: list[Failure] = []
        self.enemy_structures: dict[int, dict[int, Pos]] = {}
        self._cooldown: dict[Any, int] = {}
        self._marks: dict[Any, int] = {}  # this tick's cooldown marks, committed with the tick
        self._research_attempts: Counter[str] = Counter()
        self.army = ArmyDriver()
        self._phase_at_order: str | None = None

    # ---- orders -------------------------------------------------------------------------------------------------
    def submit(self, order: _Order) -> OrderResult:
        """Validate `order` against the civ and the latest view and store it in the book."""
        handler = {
            SetEconomy: self._submit_economy,
            Train: self._submit_train,
            Build: self._submit_build,
            Research: self._submit_research,
            AdvancePhase: self._submit_phase,
            Army: self._submit_army,
            Say: self._submit_say,
        }[type(order)]
        return handler(order)  # type: ignore[operator]

    def _submit_economy(self, order: SetEconomy) -> OrderResult:
        self.book.set_split(order.split())
        text = ", ".join(f"{r} {v}%" for r, v in order.split().items())
        return accepted(f"Economy split set: {text}. Idle and surplus gatherers move over the next ticks.")

    def _submit_train(self, order: Train) -> OrderResult:
        if not self.roles.by_role.get(order.role):
            return rejected(
                "no_template", f"{self.roles.civ} has no {order.role} units.", self.roles.roles(), "train(role=..)"
            )
        at_options = self.roles.trained_at(order.role)
        if order.at is not None and order.at not in at_options:
            return rejected(
                "bad_building",
                f"A {order.at} can't train {order.role}; these can: {', '.join(at_options)}.",
                at_options,
                f'train(role="{order.role}", count={order.count}, at="{at_options[0]}")' if at_options else None,
            )
        self.book.train.append(TrainRequest(order.role, order.count, order.at))
        seat = self.last_seat
        if seat is not None and not self._trainers(seat, order.role, order.at):
            where = order.at or next(
                (b for b in at_options if b != "civil_centre"), at_options[0] if at_options else ""
            )
            return partial(
                f"Queued {order.count} {order.role}, but no building of yours can train them now; "
                f"they start once a {where} stands and its requirements are met.",
                suggest=f'build(structure="{where}", where="base")' if where else None,
            )
        return accepted(f"Queued {order.count} {order.role}; production starts as resources and houses allow.")

    def _submit_build(self, order: Build) -> OrderResult:
        tpl = self.roles.structure(order.structure)
        if tpl is None:
            names = sorted(self.roles.structures)
            return rejected("unknown_structure", f"{self.roles.civ} has no structure '{order.structure}'.", names)
        seat = self.last_seat
        if seat is not None and _needs_phase(tpl.requires, seat.phase):
            return rejected(
                "phase_required",
                f"A {order.structure} needs {', '.join(tpl.requires)}; you are in {seat.phase} phase.",
                suggest="advance_phase()",
            )
        if order.where.startswith("toward_"):
            target = int(order.where.removeprefix("toward_").removeprefix("player_"))
            if target not in self.players:
                options = [f"toward_player_{p}" for p in self.players] + ["base", "front"]
                return rejected("bad_target", f"Player {target} is not an opponent in this match.", options)
        self.book.build.append(BuildRequest(order.structure, order.where, tpl.name))
        cost = ", ".join(f"{v} {r}" for r, v in tpl.cost.items()) or "free"
        return accepted(f"Will place a {order.structure} at {order.where} ({cost}) when resources allow.")

    def _submit_research(self, order: Research) -> OrderResult:
        seat = self.last_seat
        offered = {} if seat is None else {t[0]: t for s in seat.finished() for t in s.techs}
        if seat is not None and order.tech not in offered:
            if order.tech in seat.researching:
                return rejected("in_progress", f"{order.tech} is already being researched.")
            return rejected("unknown_tech", f"No building of yours offers {order.tech}.", sorted(offered)[:30])
        self.book.research.append(order.tech)
        if seat is not None and not offered[order.tech][1]:
            return partial(f"Queued {order.tech}; its requirements are not met yet, so it waits until they are.")
        return accepted(f"Queued {order.tech}; it starts when resources allow.")

    def _submit_phase(self, order: AdvancePhase) -> OrderResult:
        seat = self.last_seat
        if seat is not None and seat.phase == "city":
            return rejected("already_max", "You are already in city phase.")
        self.book.advance_phase = True
        self._phase_at_order = seat.phase if seat is not None else None
        tech = self._phase_tech(seat) if seat is not None else None
        if tech is not None and not tech[1]:
            return partial(
                f"Advancing when requirements are met: {tech[0]} needs more structures "
                "(town: five village structures such as houses). The autopilot adds houses meanwhile."
            )
        return accepted("Advancing to the next phase as soon as resources allow; other spending waits for it.")

    def _submit_army(self, order: Army) -> OrderResult:
        target = order.target_player
        if target is not None and target not in self.players:
            options = [f"player_{p}" for p in self.players] + ["home", "center", "front"]
            return rejected("bad_target", f"Player {target} is not an opponent in this match.", options)
        if target is not None and target in self.fallen:
            options = [f"player_{p}" for p in self.standing] + ["home", "center", "front"]
            return rejected("defeated", f"Player {target} is already defeated.", options)
        time_ms = self.last_seat.time_ms if self.last_seat else 0
        self.book.army.action, self.book.army.target, self.book.army.since_ms = order.action, order.target, time_ms
        self.book.army.seq += 1
        self.book.army.focus = "any"
        if self.last_seat is not None and not self.last_seat.soldiers:
            return partial(f"Army set to {order.action} {order.target}, but you have no soldiers yet.")
        return accepted(f"Army set to {order.action} {order.target}.")

    def _submit_say(self, order: Say) -> OrderResult:
        self.book.said.append((self.last_seat.time_ms if self.last_seat else 0, order.text))
        return accepted("Said.")

    # ---- tick ---------------------------------------------------------------------------------------------------
    async def tick(self, seat: SeatView, snapshot: PlayerSnapshot | None = None) -> list[dict[str, Any]]:
        """Decide this tick's commands from the book and the view, queue them, and return them."""
        self.last_seat, self.last_snapshot = seat, snapshot
        self._verify(seat)
        self._learn(seat, snapshot)
        if seat.state != "active" or seat.base is None:
            return []
        t = Tick(seat, snapshot, dict(seat.resources))
        self._marks = t.marks
        try:
            self.autopilot.house(self, t)
            self._advance_phase(t)
            self._research(t)
            await self._build(t)
            self.autopilot.builders(self, t)
            self._train(t)
            self.autopilot.villagers(self, t)
            self.army.tick(self, t)
            self.autopilot.gatherers(self, t)
            self.autopilot.stances(self, t)
            self.autopilot.dropsites(self, t)
        except BaseException:
            for item in reversed(t.issued):
                if item.undo is not None:
                    item.undo()
            raise
        finally:
            self._marks = {}
        self._cooldown.update(t.marks)
        self.pending.extend(t.issued)
        for command in t.commands:
            self.queue(self.player, command)
            self.issued[command["type"]] += 1
        return t.commands

    def workers(self, t: Tick) -> list:
        """Units that gather: villagers, plus infantry while the army posture is `gather`."""
        return worker_units(list(t.seat.units), self.book.army.action == "gather")

    def desired_workers(self, t: Tick) -> dict[str, int]:
        return split_targets(self.book.split, len(self.workers(t)))

    def surplus(self, t: Tick) -> dict[str, int]:
        """Workers above (positive) or below (negative) each resource's share of the split."""
        target = self.desired_workers(t)
        current = Counter(u.gathering for u in self.workers(t) if u.doing == "gather")
        return {r: current.get(r, 0) - target[r] for r in RESOURCES}

    def ready(self, key: Any, now: int, cooldown: int) -> bool:
        last = self._marks.get(key, self._cooldown.get(key))
        return last is None or now - last >= cooldown

    def mark(self, key: Any, now: int) -> None:
        self._marks[key] = now

    def fail_attempt(
        self, req: BuildRequest | TrainRequest, now: int, what: str, reason: str, *, final: bool = False
    ) -> None:
        """Count a failed attempt at `req`; after `MAX_ATTEMPTS` (or a `final` one) the order fails and is recorded
        in `failures`."""
        req.attempts += 1
        if (final or req.attempts >= MAX_ATTEMPTS) and req.failure is None:
            req.failure = reason
            build = isinstance(req, BuildRequest)
            key, where = (req.structure, req.where) if build else (req.role, "")
            self.failures.append(Failure(now, self.player, key, what, reason, where))
            log.warning("player %d: gave up on %s: %s", self.player, what, reason)

    @property
    def standing(self) -> tuple[int, ...]:
        return tuple(p for p in self.players if p not in self.fallen)

    def enemy_bases(self, seat: SeatView) -> dict[int, Pos]:
        """Each standing opponent's base as this seat knows it: its seen structures' nearest, else a guess."""
        out: dict[int, Pos] = {}
        for q in self.standing:
            seen = list(self.enemy_structures.get(q, {}).values())
            if seen and seat.base is not None:
                out[q] = min(seen, key=lambda p: distance(p, seat.base))
            else:
                out[q] = enemy_base_guess(seat, q, (self.player, *self.players))
        return out

    def _learn(self, seat: SeatView, snapshot: PlayerSnapshot | None) -> None:
        if snapshot is None:
            return
        for q, view in snapshot.enemies.items():
            known = self.enemy_structures.setdefault(q, {})
            for b in view.buildings:
                if b.pos is not None:
                    known[b.id] = (float(b.pos[0]), float(b.pos[1]))

    def _verify(self, seat: SeatView) -> None:
        still: list[Issued] = []
        for item in self.pending:
            if item.check(seat):
                if item.request is not None and item.command["type"] == "construct":
                    self.placed.append((item.request.structure, item.request.where, seat.time_ms))
                continue
            if seat.time_ms - item.time_ms < VERIFY_MS:
                still.append(item)
                continue
            self.rejections.append(Rejection(seat.time_ms, self.player, item.what, item.command))
            log.warning("player %d: engine rejected %s: %s", self.player, item.what, item.command)
            if item.rejected is not None:
                item.rejected(seat.time_ms)
        self.pending = still

    def expect(self, t: Tick, item: Issued) -> None:
        t.commands.append(item.command)
        t.issued.append(item)

    # ---- research and phase -------------------------------------------------------------------------------------
    def _phase_tech(self, seat: SeatView) -> tuple[str, bool, dict[str, int]] | None:
        prefix = {"village": "phase_town", "town": "phase_city"}.get(seat.phase)
        if prefix is None:
            return None
        return next((t for s in seat.finished() for t in s.techs if t[0].startswith(prefix)), None)

    def _advance_phase(self, t: Tick) -> None:
        if not self.book.advance_phase:
            return
        seat = t.seat
        if self._phase_at_order is None:
            self._phase_at_order = seat.phase
        if seat.phase != self._phase_at_order:
            self.book.advance_phase = False
            return
        if any(r.startswith(("phase_town", "phase_city")) for r in seat.researching):
            return
        tech = self._phase_tech(seat)
        if tech is None:
            return
        name, can, cost = tech
        if not can:
            self.autopilot.village_structure(self, t)
            return
        self._start_research(t, name, cost, reserve=True)

    def _research(self, t: Tick) -> None:
        for tech in list(self.book.research):
            if tech in t.seat.researching:
                self.book.research.remove(tech)
                continue
            if not any(x[0] == tech for s in t.seat.finished() for x in s.techs):
                continue
            cost = next(x[2] for s in t.seat.finished() for x in s.techs if x[0] == tech)
            self._start_research(t, tech, cost, reserve=False)

    def _start_research(self, t: Tick, tech: str, cost: Mapping[str, int], *, reserve: bool) -> None:
        if not self.ready(("research", tech), t.seat.time_ms, VERIFY_MS):
            if reserve:
                t.reserve(cost)
            return
        building = next(
            (
                s
                for s in t.seat.finished()
                if (s.queue or 0) < MAX_QUEUE and any(x[0] == tech and x[1] for x in s.techs)
            ),
            None,
        )
        if building is None or not t.affordable(cost):
            if reserve:
                t.reserve(cost)
            return
        t.spend(cost)
        self.mark(("research", tech), t.seat.time_ms)
        command = cmd_research(building.id, tech)
        what = f"research {tech}"
        check = _researching_or_done(tech)
        self.expect(t, Issued(what, command, t.seat.time_ms, check, rejected=self._research_rejected(tech)))

    def _research_rejected(self, tech: str) -> Callable[[int], None]:
        def rejected_(now: int) -> None:
            self._research_attempts[tech] += 1
            if self._research_attempts[tech] < MAX_ATTEMPTS:
                return
            reason = "the engine refused the research command"
            if tech in self.book.research:
                self.book.research.remove(tech)
            elif tech.startswith(("phase_town", "phase_city")) and self.book.advance_phase:
                self.book.advance_phase = False
            else:
                return
            self.failures.append(Failure(now, self.player, tech, f"research {tech}", reason))
            log.warning("player %d: gave up on research %s: %s", self.player, tech, reason)

        return rejected_

    # ---- buildings ----------------------------------------------------------------------------------------------
    async def _build(self, t: Tick) -> None:
        for req in self.book.build:
            if req.placed_id is not None or req.failure is not None:
                continue
            if await self.place(t, req) == "unaffordable":
                break  # keep the order: the first unaffordable building blocks later ones, like a build queue
        self.book.build = [r for r in self.book.build if r.failure is None and not self._confirmed(t, r)]

    def _confirmed(self, t: Tick, req: BuildRequest) -> bool:
        return req.placed_id is not None and not any(p.request is req for p in (*self.pending, *t.issued))

    async def place(self, t: Tick, req: BuildRequest) -> PlaceOutcome:
        """Try to place `req` now. Only an `unaffordable` request holds back the ones after it."""
        now = t.seat.time_ms
        what = f"build {req.structure} at {req.where}"
        tpl = self.roles.structure(req.structure)
        if tpl is None:
            self.fail_attempt(req, now, what, f"unknown structure {req.structure}", final=True)
            return "failed"
        if any(p.request is req for p in self.pending):
            return "placed"
        if not t.affordable(tpl.cost):
            t.reserve(tpl.cost)
            return "unaffordable"
        search = search_for(t.seat, req.where, self.enemy_bases(t.seat))
        if search is None:
            return "waiting"
        builders = self.autopilot.pick_builders(self, t, search.anchors[0], BUILDERS_PER_SITE)
        if not builders:
            return "waiting"
        try:
            spot = await self.placer.find(self.player, tpl.name, search)
        except Exception as e:  # noqa: BLE001 - a failed search counts as a failed attempt, not a stuck seat
            log.warning("player %d: placement search for %s failed: %r", self.player, req.structure, e)
            self.fail_attempt(req, now, what, f"placement search failed: {e!r}")
            return "failed"
        if not spot.ok:
            log.info("player %d: no spot for %s at %s: %s", self.player, req.structure, req.where, spot.why())
            self.fail_attempt(req, now, what, f"no valid spot at {req.where}: {spot.why()}")
            return "failed"
        t.spend(tpl.cost)
        t.busy.update(builders)
        for b in builders:
            self.mark(("unit", b), now)
        req.placed_id = -1

        def undo() -> None:
            req.placed_id = None

        def rejected_(when: int) -> None:
            undo()
            self.fail_attempt(req, when, what, "the engine refused the construct command")

        command = cmd_construct(tpl.name, spot.x, spot.z, spot.angle, builders)
        check = _foundation_at(tpl.name, spot)
        label = f"construct {req.structure} at {req.where}"
        item = Issued(label, command, now, check, req, undo=undo, rejected=rejected_)
        self.expect(t, item)
        return "placed"

    # ---- production ---------------------------------------------------------------------------------------------
    def _trainers(self, seat: SeatView, role: str, at: str | None) -> list[tuple[Structure, str]]:
        """Finished buildings that can train some `role` unit now, with the template each would use."""
        names = self.roles.templates_for(role)
        out = []
        for s in seat.finished():
            if at is not None and s.key != at:
                continue
            tpl = next((n for n in names if n in s.trains), None)
            if tpl is not None:
                out.append((s, tpl))
        return out

    def train_at(
        self, t: Tick, role: str, count: int, at: str | None = None, request: TrainRequest | None = None
    ) -> int:
        """Queue up to `count` `role` units across free buildings; returns how many were queued.

        With a `request`, a batch the engine refuses goes back into it (and counts as a failed attempt)."""
        queued = 0
        room = t.seat.pop_cap - t.seat.pop - self._queued_pop(t.seat)
        for building, tpl in self._trainers(t.seat, role, at):
            if queued >= count or room <= 0:
                break
            if (building.queue or 0) >= MAX_QUEUE or not self.ready(("train", building.id), t.seat.time_ms, VERIFY_MS):
                continue
            unit = self.roles.units[tpl]
            n = t.affordable(unit.cost, min(count - queued, MAX_BATCH, room))
            if n <= 0:
                continue
            t.spend(unit.cost, n)
            room -= n * max(unit.population, 1)
            queued += n
            self.mark(("train", building.id), t.seat.time_ms)
            check = _queue_grew(t.seat, building, tpl, role, set(self.roles.templates_for(role)))
            undo, rejected_ = self._train_undo(request, n) if request is not None else (None, None)
            what = f"train {n} {tpl}"
            item = Issued(what, cmd_train(building.id, tpl, n), t.seat.time_ms, check, None, role, n, undo, rejected_)
            self.expect(t, item)
        return queued

    def _train_undo(self, req: TrainRequest, n: int) -> tuple[Callable[[], None], Callable[[int], None]]:
        def undo() -> None:
            req.remaining += n
            if not any(r is req for r in self.book.train):
                self.book.train.append(req)

        def rejected_(when: int) -> None:
            undo()
            self.fail_attempt(req, when, f"train {req.role}", "the engine refused the train command")

        return undo, rejected_

    def _queued_pop(self, seat: SeatView) -> int:
        """Population the queues will still claim: the units in batches waiting behind each building's first one.

        A batch reserves its population when it starts (`Trainer.Item.Start` in 0.28), so `seat.pop` already counts
        the batch in production."""
        total = 0
        for s in seat.finished():
            for tpl, n in s.queued[1:]:
                unit = self.roles.units.get(tpl)
                total += n * max(unit.population if unit else 1, 1)
        return total

    def _train(self, t: Tick) -> None:
        for req in list(self.book.train):
            if req.remaining > 0 and req.failure is None:
                req.remaining -= self.train_at(t, req.role, req.remaining, req.at, req)
        self.book.train = [r for r in self.book.train if r.remaining > 0 and r.failure is None]


def _needs_phase(requires: Sequence[str], phase: str) -> bool:
    rank = {"village": 0, "town": 1, "city": 2}
    need = max((rank[r.removeprefix("phase_").split("_")[0]] for r in requires if r.startswith("phase_")), default=0)
    return need > rank.get(phase, 0)


def _researching_or_done(tech: str) -> Callable[[SeatView], bool]:
    def check(seat: SeatView) -> bool:
        return tech in seat.researching or not any(x[0] == tech for s in seat.finished() for x in s.techs)

    return check


def _foundation_at(template: str, spot: Any) -> Callable[[SeatView], bool]:
    def check(seat: SeatView) -> bool:
        return any(
            s.template == template and s.pos is not None and distance(s.pos, (spot.x, spot.z)) < 4.0
            for s in seat.structures
        )

    return check


def _role_total(seat: SeatView, templates: set[str], role: str) -> int:
    """`role` units alive plus those waiting in any production queue."""
    queued = sum(n for s in seat.finished() for tpl, n in s.queued if tpl in templates)
    return len(seat.units_of(role)) + queued


def _queue_grew(
    seat: SeatView, building: Structure, template: str, role: str, templates: set[str]
) -> Callable[[SeatView], bool]:
    """The building holds more batches of `template`, or the seat has more `role` units alive or queued than when the
    command went out (the batch may already be out). Comparing alive-plus-queued keeps an earlier batch finishing
    (queued units walking out) from passing for ours, and another batch finishing from looking like a refusal."""
    batches = sum(1 for tpl, _ in building.queued if tpl == template)
    total = _role_total(seat, templates, role)

    def check(now: SeatView) -> bool:
        s = now.structure(building.id)
        if s is None or sum(1 for tpl, _ in s.queued if tpl == template) > batches:
            return True
        return _role_total(now, templates, role) > total

    return check


def split_targets(split: Mapping[str, int], workers: int) -> dict[str, int]:
    """Whole worker counts per resource for a percent split (largest remainder, so they add up to `workers`)."""
    raw = {r: workers * split.get(r, 0) / 100 for r in RESOURCES}
    out = {r: math.floor(v) for r, v in raw.items()}
    for r in sorted(RESOURCES, key=lambda r: raw[r] - out[r], reverse=True)[: workers - sum(out.values())]:
        out[r] += 1
    return out


class ExecutorLoop:
    """Runs every seat's executor once per `period` seconds beside the clock, with one view query per tick.

    It never blocks the clock: commands go through `GameClock.queue` and the loop's own awaits (view, snapshot,
    placement) only delay the executor. A failed tick is logged and the loop carries on at the next tick.
    """

    def __init__(
        self,
        clock: GameClock,
        executors: Mapping[int, Executor],
        *,
        view: Callable[[], Any],
        snapshot: Callable[[], Any] | None = None,
        period: float = TICK_S,
    ):
        self.clock = clock
        self.executors = dict(executors)
        self.view = view
        self.snapshot = snapshot
        self.period = period
        self.ticks = 0
        self.errors = 0
        self._stopping = False

    async def tick_once(self) -> ExecutorView:
        view: ExecutorView = await self.view()
        snap: Snapshot | None = await self.snapshot() if self.snapshot is not None else None
        for p, executor in self.executors.items():
            seat = view.seats.get(p)
            if seat is None:
                continue
            try:
                await executor.tick(seat, snap.players.get(p) if snap is not None else None)
            except Exception:  # noqa: BLE001 - one seat's bad tick must not stop the other seats
                self.errors += 1
                log.exception("player %d: executor tick failed", p)
        self.ticks += 1
        return view

    async def run(self) -> None:
        while not self._stopping:
            try:
                await self.tick_once()
            except Exception:  # noqa: BLE001 - one bad tick must not stop the empire
                self.errors += 1
                log.exception("executor tick failed")
            await asyncio.sleep(self.period)

    def stop(self) -> None:
        self._stopping = True


async def build_executors(
    engine: GameEngine,
    clock: GameClock,
    seats: Sequence[int],
    *,
    players: Sequence[int] | None = None,
    catalog: RoleCatalog | None = None,
    placer: Placer | None = None,
) -> tuple[dict[int, Executor], ViewQuery]:
    """One executor per driven seat (`seats`), each with its civ's role table, plus the shared view query.

    `players` lists every player in the match (opponents include Petra seats the executor doesn't drive); it defaults
    to `seats`.
    """
    query = ViewQuery(engine, seats)
    view = await query()
    catalog = catalog or RoleCatalog(engine)
    placer = placer or EnginePlacer(engine)
    everyone = tuple(players or seats)
    executors = {}
    for p in seats:
        roles = await catalog.table(view.seats[p].civ)
        executors[p] = Executor(p, roles, queue=clock.queue, placer=placer, players=everyone)
        executors[p].last_seat = view.seats[p]
    return executors, query
