"""Where a building goes: an anchor from the order's `where`, then the engine's own placement check (spec §4.5, §9).

`where` names an intent, not a spot:
- `base`: around the civic centre;
- `near_<resource>`: next to the nearest site of that resource that no drop-site for it already serves, else the
  nearest site at all (drop-sites, fields and farmsteads go there);
- `toward_<player>`: part-way from the base toward that player's base as this seat knows it;
- `front`: further toward the nearest enemy, as far forward as own territory allows.

Enemy bases are what the seat has seen (enemy structures in its fog-limited snapshot); until it has seen one, the
guess is the point opposite the own base across the map centre (exact for two players on a circle placement), for more
players the matching point of a circle through the own base.

`js/placement.js` then walks rings outward from the anchors and asks `BuildRestrictions.CheckPlacement()` at each
spot, so every accepted spot is one the engine's `construct` command also accepts at that moment. Anchors fall back
to the base when the first ring search fails (e.g. the front is outside own territory).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Protocol

from agentenv_zeroad.engine import GameEngine
from agentenv_zeroad.view import Pos, SeatView, distance

ANGLE = math.pi * 3 / 4
TOWARD_DISTANCE = 45.0
FRONT_DISTANCE = 80.0
DROPSITE_SERVES = 35.0


@dataclass(frozen=True)
class Placement:
    ok: bool
    x: float = 0.0
    z: float = 0.0
    angle: float = ANGLE
    tried: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Placement:
        return cls(
            ok=bool(raw.get("ok")),
            x=float(raw.get("x", 0.0)),
            z=float(raw.get("z", 0.0)),
            angle=float(raw.get("angle", ANGLE)),
            tried=int(raw.get("tried", 0)),
            reasons={_plain(k): int(v) for k, v in (raw.get("reasons") or {}).items()},
        )

    def why(self) -> str:
        if self.ok:
            return "placed"
        if not self.reasons:
            return "no candidate spot (all too close to other buildings)"
        return max(self.reasons.items(), key=lambda kv: kv[1])[0]


def _plain(message: str) -> str:
    """Turn the engine's translatable message ("%(name)s cannot be built ...") into plain text."""
    return (
        message.replace("%(name)s", "it")
        .replace("%(category)s", "building of that category")
        .replace("%(distance)s", "the minimum")
    )


@dataclass(frozen=True)
class Search:
    """One ring search: anchors in priority order and ring radii in metres."""

    anchors: tuple[Pos, ...]
    min_r: float = 12.0
    max_r: float = 90.0
    step: float = 6.0
    spacing: float = 3.0


class Placer(Protocol):
    async def find(self, player: int, template: str, search: Search) -> Placement: ...


def placement_js(player: int, template: str, search: Search, *, phase: float = 0.3) -> str:
    source = resources.files("agentenv_zeroad.js").joinpath("placement.js").read_text()
    config = {
        "player": int(player),
        "template": template,
        "anchors": [[round(a[0], 1), round(a[1], 1)] for a in search.anchors],
        "minR": search.min_r,
        "maxR": search.max_r,
        "step": search.step,
        "angle": ANGLE,
        "spacing": search.spacing,
        "phase": phase,
    }
    return source.replace("__CONFIG__", json.dumps(config))


class EnginePlacer:
    """Runs the ring search inside the engine (one `/evaluate`, a few ms)."""

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.calls = 0

    async def find(self, player: int, template: str, search: Search) -> Placement:
        self.calls += 1
        raw = await self.engine.evaluate(placement_js(player, template, search, phase=0.3 + 0.7 * self.calls))
        return Placement.from_json(raw if isinstance(raw, Mapping) else {"ok": False})


def enemy_base_guess(seat: SeatView, enemy: int, players: Sequence[int]) -> Pos:
    """Where `enemy`'s base probably is when the seat has not seen it: same circle as the own base, spaced evenly."""
    cx, cz = seat.center
    if seat.base is None:
        return (cx, cz)
    bx, bz = seat.base
    radius = max(distance(seat.base, seat.center), 1.0)
    angle = math.atan2(bz - cz, bx - cx)
    order = sorted(set(players) | {seat.player, enemy})
    n = len(order)
    step = (order.index(enemy) - order.index(seat.player)) % n
    a = angle + 2 * math.pi * step / max(n, 2)
    return (cx + radius * math.cos(a), cz + radius * math.sin(a))


def toward(origin: Pos, target: Pos, metres: float) -> Pos:
    d = distance(origin, target)
    if d < 1e-6:
        return origin
    f = min(1.0, metres / d)
    return (origin[0] + (target[0] - origin[0]) * f, origin[1] + (target[1] - origin[1]) * f)


def resource_anchor(seat: SeatView, resource: str) -> Pos | None:
    """The nearest site of `resource` that no drop-site for it serves yet, else the nearest site."""
    sites = seat.sites.get(resource, ())
    if not sites:
        return None
    unserved = [s for s in sites if s.distance > DROPSITE_SERVES]
    return (unserved or list(sites))[0].pos


def search_for(seat: SeatView, where: str, enemy_bases: Mapping[int, Pos]) -> Search | None:
    """The ring search for a `where` (see the module docstring); None when there is no base to anchor on."""
    if seat.base is None:
        return None
    base = seat.base
    fallback = Search((base,), min_r=15.0, max_r=100.0)
    if where == "base":
        return fallback
    if where.startswith("near_"):
        anchor = resource_anchor(seat, where.removeprefix("near_"))
        if anchor is None:
            return fallback
        return Search((anchor, base), min_r=8.0, max_r=40.0, step=4.0, spacing=2.0)
    target = _target_base(seat, where, enemy_bases)
    if target is None:
        return fallback
    metres = FRONT_DISTANCE if where == "front" else TOWARD_DISTANCE
    return Search((toward(base, target, metres), toward(base, target, metres / 2), base), min_r=0.0, max_r=40.0)


def _target_base(seat: SeatView, where: str, enemy_bases: Mapping[int, Pos]) -> Pos | None:
    if where == "front":
        if not enemy_bases or seat.base is None:
            return None
        return min(enemy_bases.values(), key=lambda p: distance(p, seat.base))
    digits = where.removeprefix("toward_").removeprefix("player_")
    return enemy_bases.get(int(digits)) if digits.isdigit() else None
