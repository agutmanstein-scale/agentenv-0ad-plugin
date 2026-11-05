"""The army posture as commands (spec §4.5 `army(action, target)`; postures are builder heuristics under §10).

- `gather` (default): soldiers work as gatherers (the autopilot moves them), nothing here;
- `attack` / `raid`: attack-walk to the known enemy structure nearest the group, else the guessed base; a raid sends
  cavalry, else ranged infantry, and targets workers. The posture's `focus` (set by the Luna reflex) redirects it:
  `army` walks at the enemy's visible soldiers, `workers` at its visible gatherers (attacking only workers),
  `buildings` at its nearest known structure attacking only structures (so it walks past their soldiers); with
  nothing of that kind known it falls back to the default;
- `defend`: gather at the point; every soldier to base when the snapshot reports enemies near it;
- `retreat`: walk home, then back to `gather`;
- `scout`: one cavalryman (else a soldier) walks to the target once.
An attack, raid or scout on a player the engine has defeated turns to the nearest opponent still standing.
Each unit is re-sent only when its target moved, its target classes changed (a new focus re-aims soldiers already
walking or fighting) or a cooldown passed, so a standing posture doesn't flood the queue.

Interface v2 sends attacks and raids in waves instead (`_waves`; every other posture is unchanged):
- soldiers not in a wave gather at the rally point, `point("front")`, and the group launches once `min_size` of them
  stand within `GATHER_M` of it, or `WAVE_WAIT_MS` after the order (or after the last launch, for the next wave),
  with whoever there is;
- a launched wave walks to the target as one attack-walk like v1's, and keeps its members: a wave whose survivors
  fall below `regroup_below` of its launch strength is dissolved and they walk back to the rally point, where they
  form the next wave with soldiers trained since;
- soldiers trained after a launch are never sent alone: they gather for the next wave;
- an identical army order is not a new order (the executor keeps `seq`), so the wave state and its timer survive the
  brain re-issuing the posture every turn; a changed order bumps `seq` and starts over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentenv_zeroad.commands import ATTACK_CLASSES, RAID_CLASSES, STRUCTURE_CLASSES, cmd_attack_walk, cmd_walk
from agentenv_zeroad.orders import Army, ArmyPosture
from agentenv_zeroad.placement import toward
from agentenv_zeroad.view import Pos, SeatView, Unit, distance

if TYPE_CHECKING:
    from agentenv_zeroad.executor import Executor, Tick

ARMY_COOLDOWN_MS = 8_000
ARRIVED_M = 25.0
WAVE_WAIT_MS = 90_000
GATHER_M = 40.0
HOME_M = 40.0


@dataclass
class Wave:
    """Soldiers launched together: who is still alive of them, and how many there were."""

    members: set[int]
    launched: int


@dataclass(frozen=True)
class ArmyReport:
    """Where a seat's army stands, for the v2 brief: soldiers by role, where they are (`none`, `home`, `rally`,
    `moving` or `engaged`), the waves out (alive, launched with) and the group gathering for the next one."""

    soldiers: dict[str, int]
    where: str
    waves: tuple[tuple[int, int], ...] = ()
    forming: int = 0
    gathered: int = 0
    min_size: int = 0
    goes_in_s: int | None = None


@dataclass
class WaveState:
    seq: int | None = None
    waves: list[Wave] = field(default_factory=list)
    forming_since: int | None = None


class ArmyDriver:
    """Turns one seat's army posture into attack-walk and walk commands, tick by tick."""

    def __init__(self) -> None:
        self._points: dict[int, tuple[Pos, Any]] = {}  # where each soldier was last sent, and to attack what
        self._wave = WaveState()

    def point(self, ex: Executor, seat: SeatView, target: str) -> Pos:
        bases = ex.enemy_bases(seat)
        if target == "home":
            return seat.base or seat.center
        if target == "center":
            return seat.center
        if target == "front":
            if not bases or seat.base is None:
                return seat.base or seat.center
            nearest = min(bases.values(), key=lambda p: distance(p, seat.base))
            return toward(seat.base, nearest, 80.0)
        q = int(target.removeprefix("player_"))
        return bases.get(q, seat.center)

    def tick(self, ex: Executor, t: Tick) -> None:
        posture = ex.book.army
        seat = t.seat
        soldiers = [u for u in seat.soldiers if u.pos is not None]
        if ex.interface == "v2" and posture.action in ("attack", "raid") and not soldiers:
            self._state(posture).forming_since = None  # nobody is gathering: the wait starts with the first soldier
        if posture.action == "gather" or not soldiers:
            return
        now = seat.time_ms
        fallen = Army(action="attack", target=posture.target).target_player in ex.fallen
        if fallen and posture.action in ("attack", "raid", "scout"):
            bases = ex.enemy_bases(seat)
            origin = seat.base or seat.center
            if bases:
                posture.target = f"player_{min(bases, key=lambda q: distance(bases[q], origin))}"
        point = self.point(ex, seat, posture.target)
        if posture.action in ("attack", "raid"):
            group, classes = _strike_group(soldiers, posture.action)
            if ex.interface == "v2":
                self._waves(ex, t, posture, group, classes)
                return
            point, classes = self._aim(ex, seat, posture, group, point, classes)
            self._send(ex, t, group, point, classes, fight=True)
        elif posture.action == "defend":
            if t.snapshot is not None and t.snapshot.enemies_near_base > 0:
                self._send(ex, t, soldiers, seat.base or point, ATTACK_CLASSES, fight=True, force=True)
            else:
                far = [u for u in soldiers if distance(u.pos, point) > 40 and u.doing in ("idle", "gather")]
                self._send(ex, t, far, point, ATTACK_CLASSES, fight=True)
        elif posture.action == "retreat":
            home = seat.base or point
            away = [u for u in soldiers if distance(u.pos, home) > 40]
            if not away:
                posture.action, posture.target = "gather", "home"
                return
            self._send(ex, t, away, home, None, fight=False)
        elif posture.action == "scout":
            scout = sorted(soldiers, key=lambda u: (u.role != "cavalry", u.id))[0]
            if ex.ready(("scout", scout.id, posture.since_ms), now, 10**9):
                ex.mark(("scout", scout.id, posture.since_ms), now)
                t.busy.add(scout.id)
                t.commands.append(cmd_walk([scout.id], point))
            else:
                t.busy.add(scout.id)

    def _aim(self, ex: Executor, seat: SeatView, posture: ArmyPosture, group: Sequence[Unit], point: Pos, classes):
        """Where an attack or raid by `group` goes, and what it hits: the known enemy structure nearest the group
        for a player target, then the posture's focus."""
        q = Army(action="attack", target=posture.target).target_player
        if q is None:
            return point, classes
        point = self._attack_target(ex, seat, q, group)
        return self._focus(ex, q, posture.focus, point, classes)

    def _waves(self, ex: Executor, t: Tick, posture: ArmyPosture, group: list[Unit], classes: Any) -> None:
        """One tick of an attack or raid in waves (interface v2): launch the gathered group when it is big enough or
        has waited long enough, keep each wave on its target, dissolve one that is spent, gather everyone else."""
        seat, now = t.seat, t.seat.time_ms
        state = self._state(posture)
        alive = {u.id: u for u in group}
        for wave in state.waves:
            wave.members &= alive.keys()
        spent = [w for w in state.waves if not w.members or len(w.members) < posture.regroup_below * w.launched]
        state.waves = [w for w in state.waves if w not in spent]
        rally = self.point(ex, seat, "front")
        fallen_back = [alive[i] for w in spent for i in w.members]
        if fallen_back:
            self._send(ex, t, fallen_back, rally, ATTACK_CLASSES, fight=True, force=True)
        out = {i for w in state.waves for i in w.members}
        forming = [u for u in group if u.id not in out]
        if forming and state.forming_since is None:
            state.forming_since = now
        waited = 0 if state.forming_since is None else now - state.forming_since
        if forming and (len(_gathered(forming, rally)) >= posture.min_size or waited >= WAVE_WAIT_MS):
            state.waves.append(Wave({u.id for u in forming}, len(forming)))
            state.forming_since, forming = None, []
        for wave in state.waves:
            members = [alive[i] for i in wave.members]
            point, wave_classes = self._aim(ex, seat, posture, members, self.point(ex, seat, posture.target), classes)
            self._send(ex, t, members, point, wave_classes, fight=True)
        if forming:
            self._send(ex, t, forming, rally, ATTACK_CLASSES, fight=True)

    def _state(self, posture: ArmyPosture) -> WaveState:
        """The waves of the standing order: a new order (its `seq` changed) starts over, its 90 s from the order."""
        if self._wave.seq != posture.seq:
            self._wave = WaveState(posture.seq, [], posture.since_ms)
        return self._wave

    def report(self, ex: Executor, seat: SeatView) -> ArmyReport:
        """The army's state for the brief; the wave numbers are the standing ones (interface v2)."""
        posture = ex.book.army
        soldiers = [u for u in seat.soldiers if u.pos is not None]
        counts: dict[str, int] = {}
        for u in soldiers:
            counts[u.role] = counts.get(u.role, 0) + 1
        rally, home = self.point(ex, seat, "front"), seat.base or seat.center
        where = _where(soldiers, rally, home)
        if posture.action not in ("attack", "raid") or self._wave.seq != posture.seq:
            return ArmyReport(counts, where, min_size=posture.min_size)
        group, _ = _strike_group(soldiers, posture.action)
        ids = {u.id for u in group}
        waves = tuple((len(w.members & ids), w.launched) for w in self._wave.waves)
        out = {i for w in self._wave.waves for i in w.members}
        forming = [u for u in group if u.id not in out]
        since = self._wave.forming_since
        goes_in = None if not forming or since is None else max(0, (WAVE_WAIT_MS - (seat.time_ms - since)) // 1000)
        return ArmyReport(counts, where, waves, len(forming), len(_gathered(forming, rally)), posture.min_size, goes_in)

    def _focus(self, ex: Executor, q: int, focus: str, point: Pos, classes) -> tuple[Pos, Any]:
        """Where the reflex's focus sends the attack, when the enemy's soldiers, workers or structures are known."""
        if focus == "buildings" and ex.enemy_structures.get(q):
            return point, STRUCTURE_CLASSES  # `point` is already the known structure nearest the group
        enemy = ex.last_snapshot.enemies.get(q) if ex.last_snapshot is not None else None
        if enemy is None:
            return point, classes
        if focus == "army" and enemy.soldiers_at is not None:
            return (float(enemy.soldiers_at[0]), float(enemy.soldiers_at[1])), ATTACK_CLASSES
        if focus == "workers" and enemy.workers_at is not None:
            return (float(enemy.workers_at[0]), float(enemy.workers_at[1])), RAID_CLASSES
        return point, classes

    def _attack_target(self, ex: Executor, seat: SeatView, q: int, group: Sequence[Any]) -> Pos:
        """The known enemy structure nearest the group (dropping ones the group reached and found gone)."""
        known = ex.enemy_structures.get(q, {})
        centre = _centroid([u.pos for u in group]) or seat.base or seat.center
        visible = set()
        if ex.last_snapshot is not None and q in ex.last_snapshot.enemies:
            visible = {b.id for b in ex.last_snapshot.enemies[q].buildings}
        for bid, pos in list(known.items()):
            if bid not in visible and any(u.pos and distance(u.pos, pos) < ARRIVED_M / 2 for u in group):
                del known[bid]
        if known:
            return min(known.values(), key=lambda p: distance(p, centre))
        return ex.enemy_bases(seat).get(q, seat.center)

    def _send(self, ex: Executor, t: Tick, units, point: Pos, classes, *, fight: bool, force: bool = False) -> None:
        now = t.seat.time_ms
        movers = []
        for u in units:
            t.busy.add(u.id)
            key = ("army", u.id)
            last = self._points.get(u.id)
            retarget = last is not None and last[1] != classes
            if u.doing == "fight" and not force and not retarget:
                continue
            moved_target = last is None or distance(last[0], point) > ARRIVED_M or retarget
            arrived = u.pos is not None and distance(u.pos, point) <= ARRIVED_M and u.doing == "idle"
            if arrived:
                continue
            if not moved_target and u.doing in ("move", "fight") and not force:
                continue
            if not ex.ready(key, now, ARMY_COOLDOWN_MS) and not moved_target:
                continue
            ex.mark(key, now)
            self._points[u.id] = (point, classes)
            movers.append(u.id)
        if movers:
            t.commands.append(cmd_attack_walk(movers, point, classes) if fight else cmd_walk(movers, point))


def _strike_group(soldiers: list[Unit], action: str) -> tuple[list[Unit], Any]:
    """Who an attack or raid sends, and the classes it hits: every soldier, or for a raid the cavalry (else ranged
    infantry)."""
    if action != "raid":
        return soldiers, ATTACK_CLASSES
    cavalry = [u for u in soldiers if u.role == "cavalry"]
    return cavalry or [u for u in soldiers if u.role == "infantry_ranged"], RAID_CLASSES


def _gathered(units: Sequence[Unit], rally: Pos) -> list[Unit]:
    return [u for u in units if u.pos is not None and distance(u.pos, rally) <= GATHER_M]


def _where(soldiers: list[Unit], rally: Pos, home: Pos) -> str:
    if not soldiers:
        return "none"
    if any(u.doing == "fight" for u in soldiers):
        return "engaged"
    centre = _centroid([u.pos for u in soldiers])
    if centre is not None and distance(centre, home) <= HOME_M:
        return "home"
    if centre is not None and distance(centre, rally) <= GATHER_M:
        return "rally"
    return "moving"


def _centroid(points: Sequence[Pos | None]) -> Pos | None:
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
