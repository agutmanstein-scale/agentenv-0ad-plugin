"""An order-replay controller for the S1 tool check (docs/post-training-v2.md, stage S1): a seat with no model that
gives a logged order stream back through its executor, so the same moves can be played under either interface.

The stream comes from a finished game's results.json: each of a seat's logged `turns` has `began_ms` (the game time the
brain read its brief) and the `orders` it gave (`tool`, `arguments`). `ReplaySpec.from_results` turns them into a flat
list of `(time_ms, tool, arguments)`; the controller submits each one once the game clock reaches its time (plus an
optional `delay_ms`, for the fixed lag lockstep play adds), the way `ScriptedBot` is ticked by the match. Every order
goes to the executor as written, whatever status it got in the original game, because the point is what the executor
does with it now: under v2 an army(attack) with no wave parameters takes the default wave, and an invalid `at` is
ignored and auto-routed instead of rejected. A logged tool name the vocabulary doesn't have (an invented one) or
arguments that don't validate are not played; they are counted in `skipped`, by reason.

A replay spec is also the unit `scripts/s1_specs.py` writes: besides the orders it names the game it came from and
carries the settings needed to set the same match up again (`game`), which the match itself ignores.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentenv_zeroad.executor import Executor
from agentenv_zeroad.orders import ORDER_TYPES, OrderResult, _Order, parse_order
from agentenv_zeroad.state import PlayerSnapshot

TOOLS = frozenset(t.model_fields["kind"].default for t in ORDER_TYPES)


@dataclass(frozen=True)
class ReplayOrder:
    time_ms: int
    tool: str
    arguments: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"time_ms": self.time_ms, "tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ReplaySpec:
    """The orders to replay, in game-time order, and where they came from."""

    orders: tuple[ReplayOrder, ...]
    delay_ms: int = 0
    source: str = ""  # the game the orders were logged in
    slot: str = ""  # the seat they were logged for
    game: Mapping[str, Any] = field(default_factory=dict)  # how to set that game up again (not read by the match)

    @classmethod
    def from_results(
        cls,
        results: Mapping[str, Any],
        slot: str,
        *,
        source: str = "",
        delay_ms: int = 0,
        game: Mapping[str, Any] | None = None,
    ) -> ReplaySpec:
        """The order stream of seat `slot` in a finished game's results.json (its `turns`)."""
        turns = results["seats"][slot].get("turns") or []
        orders = [
            ReplayOrder(int(turn["began_ms"]), str(order["tool"]), dict(order.get("arguments") or {}))
            for turn in turns
            for order in turn.get("orders") or []
        ]
        orders.sort(key=lambda o: o.time_ms)
        return cls(tuple(orders), delay_ms, source, slot, game or {})

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ReplaySpec:
        orders = tuple(ReplayOrder(int(o["time_ms"]), str(o["tool"]), dict(o["arguments"])) for o in raw["orders"])
        return cls(
            orders,
            int(raw.get("delay_ms", 0)),
            str(raw.get("source", "")),
            str(raw.get("slot", "")),
            raw.get("game") or {},
        )

    @classmethod
    def load(cls, path: Path) -> ReplaySpec:
        return cls.from_json(json.loads(path.read_text()))

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "slot": self.slot,
            "delay_ms": self.delay_ms,
            "game": dict(self.game),
            "orders": [o.to_json() for o in self.orders],
        }


@dataclass
class ReplayController:
    """Submits a spec's orders to a seat's executor as the game clock reaches them (ticked like `ScriptedBot`)."""

    executor: Executor
    spec: ReplaySpec
    submit: Callable[[_Order], OrderResult] | None = None
    log: list[tuple[int, _Order, OrderResult]] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    _next: int = 0

    @property
    def player(self) -> int:
        return self.executor.player

    @property
    def finished(self) -> bool:
        return self._next >= len(self.spec.orders)

    def tick(self, me: PlayerSnapshot, time_ms: int) -> None:
        """Submit every order that has come due by `time_ms`, in the order it was logged."""
        orders = self.spec.orders
        while self._next < len(orders) and orders[self._next].time_ms + self.spec.delay_ms <= time_ms:
            logged = orders[self._next]
            self._next += 1
            if (order := self._parse(logged)) is None:
                continue
            self.log.append((time_ms, order, (self.submit or self.executor.submit)(order)))

    def _parse(self, logged: ReplayOrder) -> _Order | None:
        if logged.tool not in TOOLS:
            self.skipped[f"unknown tool {logged.tool}"] += 1
            return None
        try:
            return parse_order({"kind": logged.tool, **logged.arguments})
        except ValidationError:
            self.skipped[f"invalid arguments for {logged.tool}"] += 1
            return None
