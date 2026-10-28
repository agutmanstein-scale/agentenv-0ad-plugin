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
Each unit is re-sent only when its target moved, its target classes changed (a new focus re-aims soldiers already
walking or fighting) or a cooldown passed, so a standing posture doesn't flood the queue.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agentenv_zeroad.commands import ATTACK_CLASSES, RAID_CLASSES, STRUCTURE_CLASSES, cmd_attack_walk, cmd_walk
from agentenv_zeroad.orders import Army
from agentenv_zeroad.placement import toward
from agentenv_zeroad.view import Pos, SeatView, distance

if TYPE_CHECKING:
    from agentenv_zeroad.executor import Executor, Tick

ARMY_COOLDOWN_MS = 8_000
ARRIVED_M = 25.0


class ArmyDriver:
    """Turns one seat's army posture into attack-walk and walk commands, tick by tick."""

    def __init__(self) -> None:
        self._points: dict[int, tuple[Pos, Any]] = {}  # where each soldier was last sent, and to attack what

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
        if posture.action == "gather" or not soldiers:
            return
        now = seat.time_ms
        point = self.point(ex, seat, posture.target)
        if posture.action in ("attack", "raid"):
            group = soldiers
            classes = ATTACK_CLASSES
            if posture.action == "raid":
                cavalry = [u for u in soldiers if u.role == "cavalry"]
                group = cavalry or [u for u in soldiers if u.role == "infantry_ranged"]
                classes = RAID_CLASSES
            q = Army(action="attack", target=posture.target).target_player
            if q is not None:
                point = self._attack_target(ex, seat, q, group)
                point, classes = self._focus(ex, q, posture.focus, point, classes)
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


def _centroid(points: Sequence[Pos | None]) -> Pos | None:
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
