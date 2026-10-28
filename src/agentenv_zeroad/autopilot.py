"""Autopilot defaults so a slow brain never leaves its empire frozen (spec §4.4). Deliberately mediocre.

- idle gatherers (and gatherers over their resource's share, two per tick) go to the resource furthest below the
  economy split, at the nearest site with a free slot; food falls back to fields once wild food is gone or far;
- a house goes up when free population room drops to `HOUSE_MARGIN`, ahead of everything else, and a house still
  waiting for a spot never holds back the buildings behind it;
- villagers keep training at the civic centre up to the order book's target, counting those queued or on the way;
- unfinished foundations without builders get some;
- soldiers use the engine's stances: `defensive` while they work or defend (they fight back and return), `aggressive`
  when they attack or raid; workers stay `passive` (they flee) as the engine sets them;
- a drop-site goes up next to wood, stone or metal when most of the workers on it stand far from any drop-site that
  takes it, and a field when wild food runs short.
Every building the autopilot asks for backs off after it fails at its spot (say, unexplored ground): `RETRY_MS`,
doubling with each further failure there, up to `RETRY_MAX_MS`. A standing need that can't be placed would otherwise
run a placement search every tick and add a failure every few seconds for the rest of the game.
Strategy, timing, composition, phase and attacks stay with the brains, with one exception the brain itself opts into:
the Luna player keeps a "not yet" answer to its phase and attack questions all match,
and when it still holds past the plan's deadline (`HELD_DEADLINES`, a share of the time cap) and reports that hold,
the autopilot carries out the plan's default (`held_default`: advance to town; attack the leading enemy; neither
while enemies are at the base). The match logs it as the autopilot's, "autopilot (Luna held)", never as Luna's choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentenv_zeroad.commands import cmd_gather, cmd_repair, cmd_stance
from agentenv_zeroad.orders import AdvancePhase, Army, BuildRequest, _Order
from agentenv_zeroad.placement import DROPSITE_SERVES
from agentenv_zeroad.state import RESOURCES
from agentenv_zeroad.view import Pos, ResourceSite, Unit, distance

if TYPE_CHECKING:
    from agentenv_zeroad.executor import Executor, Tick
    from agentenv_zeroad.roles import Template

HOUSE_MARGIN = 6
RETRY_MS = 30_000
RETRY_MAX_MS = 300_000
REBALANCE_PER_TICK = 2
FOOD_SITE_MAX_M = 60
WORKER_STANCE = "passive"
SOLDIER_STANCES = {"gather": "defensive", "defend": "defensive", "retreat": "defensive", "scout": "defensive"}
DROP_FOR = {"wood": "Storehouse", "stone": "Storehouse", "metal": "Storehouse", "food": "Farmstead"}
HELD_DEADLINES = {"phase": 0.25, "attack": 0.45}  # share of the time cap: the Luna player's plan (brains.py)
HELD_ATTACK_MIN = 4  # soldiers the default attack needs (the Luna player's REGROUP_ARMY)


def held_default(
    decision: str,
    *,
    time_ms: int,
    cap_ms: int,
    phase: str,
    advancing: bool,
    army_action: str,
    soldiers: int,
    enemies: dict[int, tuple[str, float]],
    threat: int = 0,
) -> tuple[_Order, str] | None:
    """The plan's default for a decision a brain held past its deadline, with the line the overlay shows; None when
    the deadline has not passed or the default would change nothing. `enemies`: player -> (name, score), the players
    still standing; the attack goes at the leader. `threat`: enemy soldiers at the base now. The brain held from a
    brief taken before its model call, so raiders may have arrived since; neither default runs while the base is
    under attack (the plan defends first): the attack would send the army away, and the advance would set its cost
    aside ahead of training and add buildings mid-raid."""
    deadline = HELD_DEADLINES.get(decision)
    if deadline is None or cap_ms <= 0 or time_ms < deadline * cap_ms:
        return None
    if threat > 0:
        return None
    if decision == "phase":
        if phase != "village" or advancing:
            return None
        return AdvancePhase(), "ordered the advance to town"  # an order: the executor may first save up for it
    if army_action in ("attack", "raid") or soldiers < HELD_ATTACK_MIN or not enemies:
        return None
    player, (name, _) = max(enemies.items(), key=lambda e: (e[1][1], -e[0]))
    return Army(action="attack", target=f"player_{player}"), f"sent the army to attack {name}"


class Autopilot:
    def __init__(self, *, house_margin: int = HOUSE_MARGIN):
        self.house_margin = house_margin

    # ---- buildings ----------------------------------------------------------------------------------------------
    def house(self, ex: Executor, t: Tick) -> None:
        seat = t.seat
        if seat.pop_cap >= seat.pop_max or seat.pop_cap - seat.pop > self.house_margin:
            return
        if any(s.kind == "House" for s in seat.foundations()) or self._requested(ex, "House"):
            return
        house = ex.roles.structure_of_kind("House")
        if house is not None:
            _request(ex, house, "base", seat.time_ms, front=True)

    def village_structure(self, ex: Executor, t: Tick) -> None:
        """The phase advance is waiting on structure requirements: add a house if none is on the way."""
        if any(s.kind == "House" for s in t.seat.foundations()) or self._requested(ex, "House"):
            return
        house = ex.roles.structure_of_kind("House")
        if house is not None:
            _request(ex, house, "base", t.seat.time_ms)

    def dropsites(self, ex: Executor, t: Tick) -> None:
        seat = t.seat
        for r in ("wood", "stone", "metal"):
            if self._requested(ex, DROP_FOR[r]) or any(s.kind == DROP_FOR[r] for s in seat.foundations()):
                continue
            workers = [u.pos for u in seat.units if u.doing == "gather" and u.gathering == r and u.pos is not None]
            drops = [s.pos for s in seat.dropsites(r)]
            far = sum(1 for p in workers if not drops or min(distance(p, d) for d in drops) > DROPSITE_SERVES)
            if len(workers) >= 3 and 2 * far > len(workers):
                tpl = ex.roles.structure_of_kind(DROP_FOR[r])
                if tpl is not None and _request(ex, tpl, f"near_{r}", seat.time_ms):
                    return
        self._fields(ex, t)

    def _fields(self, ex: Executor, t: Tick) -> None:
        seat = t.seat
        if self._requested(ex, "Field") or any(s.kind == "Field" for s in seat.foundations()):
            return
        wild = sum(max(s.max_gatherers - s.gatherers, 0) for s in seat.sites.get("food", ()) if s.distance <= 60)
        fields = sum(max(s.gatherers[1] - s.gatherers[0], 0) for s in seat.finished("Field") if s.gatherers)
        want = ex.desired_workers(t).get("food", 0)
        have = sum(1 for u in seat.units if u.gathering == "food" and u.doing == "gather")
        if wild + fields >= max(want - have, 1) + 2:
            return
        field_tpl = ex.roles.structure_of_kind("Field")
        if field_tpl is not None:
            _request(ex, field_tpl, "near_food", seat.time_ms)

    def _requested(self, ex: Executor, kind: str) -> bool:
        for req in ex.book.build:
            tpl = ex.roles.structure(req.structure)
            if tpl is not None and tpl.kind == kind and req.failure is None:
                return True
        return False

    def builders(self, ex: Executor, t: Tick) -> None:
        now = t.seat.time_ms
        for f in t.seat.foundations():
            if f.builders > 0 or f.pos is None or not ex.ready(("repair", f.id), now, 10_000):
                continue
            units = self.pick_builders(ex, t, f.pos, 2)
            if units:
                ex.mark(("repair", f.id), now)
                t.busy.update(units)
                for u in units:
                    ex.mark(("unit", u), now)
                t.commands.append(cmd_repair(units, f.id))

    def pick_builders(self, ex: Executor, t: Tick, near: Pos, count: int) -> list[int]:
        """Nearest free workers: idle ones first, then gatherers of the most over-staffed resource."""
        now = t.seat.time_ms
        free = [
            u
            for u in t.seat.villagers
            if u.id not in t.busy and u.pos is not None and ex.ready(("unit", u.id), now, 3_000)
        ]
        idle = sorted((u for u in free if u.doing == "idle"), key=lambda u: distance(u.pos, near))
        picked = idle[:count]
        if len(picked) < count:
            surplus = ex.surplus(t)
            busy = sorted(
                (u for u in free if u.doing == "gather" and u not in picked),
                key=lambda u: (-surplus.get(u.gathering or "", 0), distance(u.pos, near)),
            )
            picked += busy[: count - len(picked)]
        return [u.id for u in picked]

    # ---- production ---------------------------------------------------------------------------------------------
    def villagers(self, ex: Executor, t: Tick) -> None:
        """Train villagers up to the target, counting live ones, batches in queues and trains not yet seen."""
        templates = set(ex.roles.templates_for("villager"))
        have = len(t.seat.villagers)
        in_queues = sum(n for s in t.seat.finished() for tpl, n in s.queued if tpl in templates)
        on_the_way = sum(p.count for p in (*ex.pending, *t.issued) if p.role == "villager")
        need = ex.book.villager_target - have - in_queues - on_the_way
        if need > 0:
            ex.train_at(t, "villager", min(need, 2))

    # ---- workers ------------------------------------------------------------------------------------------------
    def gatherers(self, ex: Executor, t: Tick) -> None:
        seat = t.seat
        now = seat.time_ms
        workers = ex.workers(t)
        target = ex.desired_workers(t)
        current = {r: 0 for r in RESOURCES}
        for u in workers:
            if u.doing == "gather" and u.gathering in current:
                current[u.gathering] += 1
        free = [u for u in workers if u.id not in t.busy and u.pos is not None and ex.ready(("unit", u.id), now, 3_000)]
        idle = [u for u in free if u.doing == "idle"]
        movers = idle[:]
        over = sorted((r for r in RESOURCES if current[r] - target[r] >= 2), key=lambda r: target[r] - current[r])
        moved = 0
        for r in over:
            for u in (u for u in free if u.doing == "gather" and u.gathering == r):
                if moved >= REBALANCE_PER_TICK or current[r] - target[r] < 2:
                    break
                movers.append(u)
                current[r] -= 1
                moved += 1
        taken: dict[int, int] = {}
        for u in movers:
            r = max(RESOURCES, key=lambda r: (target[r] - current[r], -RESOURCES.index(r)))
            site = self._site(t, r, u.pos, taken) or self._any_site(t, u.pos, taken, target, current)
            if site is None:
                continue
            taken[site.id] = taken.get(site.id, 0) + 1
            current[site.type] += 1
            t.busy.add(u.id)
            ex.mark(("unit", u.id), now)
            t.commands.append(cmd_gather([u.id], site.id))

    def _site(self, t: Tick, resource: str, pos: Pos, taken: dict[int, int]) -> ResourceSite | None:
        sites = list(t.seat.sites.get(resource, ()))
        if resource == "food":
            sites = [s for s in sites if s.distance <= FOOD_SITE_MAX_M]
            for f in t.seat.finished("Field"):
                if f.gatherers and f.pos is not None:
                    sites.append(ResourceSite(f.id, "food", f.pos, 1, "grain", 0, f.gatherers[0], f.gatherers[1]))
        free = [s for s in sites if s.gatherers + taken.get(s.id, 0) < max(s.max_gatherers, 1)]
        if not free:
            return None
        return min(free, key=lambda s: s.distance + 0.25 * distance(s.pos, pos))

    def _any_site(self, t, pos, taken, target, current) -> ResourceSite | None:
        for r in sorted(RESOURCES, key=lambda r: current[r] - target[r]):
            if target[r] <= 0:
                continue
            site = self._site(t, r, pos, taken)
            if site is not None:
                return site
        return None

    # ---- stances ------------------------------------------------------------------------------------------------
    def stances(self, ex: Executor, t: Tick) -> None:
        want = SOLDIER_STANCES.get(ex.book.army.action, "aggressive")
        now = t.seat.time_ms
        wrong = [u.id for u in t.seat.soldiers if u.stance != want and ex.ready(("stance", u.id, want), now, 10_000)]
        if wrong:
            for uid in wrong:
                ex.mark(("stance", uid, want), now)
            t.commands.append(cmd_stance(wrong, want))


def _request(ex: Executor, tpl: Template, where: str, now: int, *, front: bool = False) -> bool:
    """Ask the book for `tpl` at `where` unless it is backing off there; returns whether it asked."""
    if _backing_off(ex, tpl.key, where, now):
        return False
    req = BuildRequest(tpl.key, where, tpl.name)
    if front:
        ex.book.build.insert(0, req)
    else:
        ex.book.build.append(req)
    return True


def _backing_off(ex: Executor, key: str, where: str, now: int) -> bool:
    """Whether `key` at `where` failed within its back-off: `RETRY_MS`, doubled for each failure there since that
    structure last went up there, at most `RETRY_MAX_MS`."""
    placed = max((at for s, w, at in ex.placed if s == key and w == where), default=-1)
    fails = [f.time_ms for f in ex.failures if f.key == key and f.where == where and f.time_ms > placed]
    if not fails:
        return False
    wait = min(RETRY_MS << min(len(fails) - 1, 10), RETRY_MAX_MS)
    return now - max(fails) < wait


def worker_units(units: list[Unit], soldiers_work: bool) -> list[Unit]:
    """Villagers, plus infantry when soldiers work (cavalry only hunt and champions and siege never gather)."""
    infantry = ("infantry_melee", "infantry_ranged")
    return [u for u in units if u.role == "villager" or (soldiers_work and u.role in infantry)]
