"""A scripted bot (no model): boom, advance to town phase, train an army, attack the nearest enemy.

After town phase it keeps every army role in production (a new batch whenever the last one is queued), so an attack
that dies is followed by reinforcements walking the same attack order.

It plays through exactly the interface a brain uses: it reads the seat's fog-limited U2 snapshot and submits §4.5
orders to its executor (`Executor.submit`), so it exercises the order vocabulary end to end and serves as the $0
opponent and smoke test. Each step of its script fires once, when the snapshot shows its condition; the executor and
autopilot do all the unit-level work. Every order and its result is kept in `log` for tests and reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agentenv_zeroad.executor import Executor
from agentenv_zeroad.orders import (
    AdvancePhase,
    Army,
    Build,
    OrderResult,
    SetEconomy,
    Train,
    _Order,
)
from agentenv_zeroad.state import PlayerSnapshot
from agentenv_zeroad.view import distance

BOOM_SPLIT = SetEconomy(food=55, wood=45, stone=0, metal=0)
PHASE_SPLIT = SetEconomy(food=50, wood=50, stone=0, metal=0)
ARMY_SPLIT = SetEconomy(food=50, wood=35, stone=0, metal=15)
ADVANCE_AT_VILLAGERS = 24
ARMY_BATCHES = (("infantry_melee", 5), ("infantry_ranged", 5), ("cavalry", 2))
ATTACK_AT_ARMY = 16
ATTACK_LATEST_MS = 4 * 60_000  # attack this long after town phase even with a smaller army


@dataclass
class ScriptedBot:
    executor: Executor
    # How an order reaches the executor; the match passes one that also keeps it for the overlay's latest orders.
    submit: Callable[[_Order], OrderResult] | None = None
    log: list[tuple[int, _Order, OrderResult]] = field(default_factory=list)
    done: set[str] = field(default_factory=set)
    town_at_ms: int | None = None
    attack_at_ms: int | None = None
    target: int | None = None

    @property
    def player(self) -> int:
        return self.executor.player

    def order(self, time_ms: int, order: _Order) -> OrderResult:
        result = (self.submit or self.executor.submit)(order)
        self.log.append((time_ms, order, result))
        return result

    def once(self, key: str, time_ms: int, *orders: _Order) -> None:
        if key in self.done:
            return
        self.done.add(key)
        for order in orders:
            self.order(time_ms, order)

    def tick(self, me: PlayerSnapshot, time_ms: int) -> None:
        """Look at the seat's snapshot and submit whatever the script says is due."""
        villagers = me.units.get("villager", 0)
        self.once("boom", time_ms, BOOM_SPLIT, Build(structure="storehouse", where="near_wood"))
        if villagers >= 12:
            self.once("farm", time_ms, Build(structure="farmstead", where="near_food"))
        if villagers >= 16:
            self.once("barracks", time_ms, Build(structure="barracks", where="toward_" + str(self._target(me))))
        if villagers >= 20:
            self.once("mines", time_ms, Build(structure="storehouse", where="near_stone"))
            self.once("outpost", time_ms, Build(structure="sentry_tower", where="front"))
        if me.phase == "village" and (villagers >= ADVANCE_AT_VILLAGERS or time_ms >= 9 * 60_000):
            self.once("advance", time_ms, PHASE_SPLIT, AdvancePhase())
        if me.phase in ("town", "city"):
            if self.town_at_ms is None:
                self.town_at_ms = time_ms
            self.once("army", time_ms, ARMY_SPLIT)
            for role, batch in ARMY_BATCHES:
                if self.executor.book.pending_count(role) == 0:
                    self.order(time_ms, Train(role=role, count=batch))
            if me.army >= ATTACK_AT_ARMY or time_ms - self.town_at_ms >= ATTACK_LATEST_MS:
                if "attack" not in self.done:
                    self.attack_at_ms = time_ms
                self.once("attack", time_ms, Army(action="attack", target=f"player_{self._target(me)}"))

    def _target(self, me: PlayerSnapshot) -> int:
        """The nearest opponent, by seen structures or the executor's guess of its base."""
        if self.target is None:
            seat = self.executor.last_seat
            bases = self.executor.enemy_bases(seat) if seat is not None else {}
            origin = me.base or (seat.base if seat else None)
            if bases and origin is not None:
                self.target = min(bases, key=lambda q: distance(bases[q], origin))
            else:
                self.target = self.executor.players[0]
        return self.target
