"""The executor's own per-tick view of each seat: ids and positions to act on (spec §4.4).

The U2 snapshot says *how much* (counts, totals, what the enemy shows); acting needs *which*: unit ids by role and
what each is doing, the buildings that can train or research something now, foundations waiting for builders, and
resource sites with ids. `js/executor_view.js` returns that for every seat in one `/evaluate` call.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from agentenv_zeroad.engine import GameEngine
from agentenv_zeroad.state import RESOURCES

RESOURCE_RADIUS = 220.0
SITES_PER_TYPE = 16
SOLDIER_ROLES = ("infantry_melee", "infantry_ranged", "cavalry", "siege", "champion")

Pos = tuple[float, float]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Unit:
    id: int
    role: str
    pos: Pos | None
    doing: str
    gathering: str | None
    stance: str

    @property
    def soldier(self) -> bool:
        return self.role in SOLDIER_ROLES


@dataclass(frozen=True)
class Structure:
    id: int
    template: str
    kind: str
    pos: Pos | None
    foundation: bool = False
    progress: int = 100
    builders: int = 0
    queue: int | None = None  # batches in the production queue
    queued: tuple[tuple[str, int], ...] = ()  # (unit template, count) per queued batch
    trains: tuple[str, ...] = ()
    techs: tuple[tuple[str, bool, dict[str, int]], ...] = ()
    drop: tuple[str, ...] = ()
    gatherers: tuple[int, int] | None = None  # (now, max) for fields and other owned resource supplies

    @property
    def key(self) -> str:
        return self.template.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ResourceSite:
    id: int
    type: str
    pos: Pos
    amount: int
    specific: str
    distance: int  # to the nearest finished drop-site that takes this resource
    gatherers: int
    max_gatherers: int


@dataclass(frozen=True)
class SeatView:
    player: int
    time_ms: int
    map_size: int
    civ: str
    state: str
    resources: dict[str, int]
    pop: int
    pop_cap: int
    pop_max: int
    phase: str
    researching: tuple[str, ...]
    units: tuple[Unit, ...]
    structures: tuple[Structure, ...]
    sites: dict[str, tuple[ResourceSite, ...]]
    base: Pos | None

    @classmethod
    def from_json(cls, player: int, time_ms: int, map_size: int, raw: Mapping[str, Any]) -> SeatView:
        units = tuple(
            Unit(int(u[0]), u[1], None if u[2] is None else (u[2], u[3]), u[4], u[5], u[6] or "")
            for u in raw.get("units", ())
        )
        structures = tuple(
            Structure(
                id=int(s["id"]),
                template=s["tpl"],
                kind=s["kind"],
                pos=None if s.get("pos") is None else tuple(s["pos"]),
                foundation=bool(s.get("foundation", False)),
                progress=int(s.get("progress", 100)),
                builders=int(s.get("builders", 0)),
                queue=s.get("queue"),
                queued=tuple((q[0], int(q[1])) for q in s.get("queued", ())),
                trains=tuple(s.get("trains", ())),
                techs=tuple((t[0], bool(t[1]), dict(t[2] or {})) for t in s.get("techs", ())),
                drop=tuple(s.get("drop", ())),
                gatherers=None if s.get("gatherers") is None else (int(s["gatherers"][0]), int(s["gatherers"][1])),
            )
            for s in raw.get("structures", ())
        )
        sites = {
            r: tuple(
                ResourceSite(int(x[0]), r, (x[1], x[2]), int(x[3]), x[4], int(x[5]), int(x[6]), int(x[7]))
                for x in raw.get("resource_sites", {}).get(r, ())
            )
            for r in RESOURCES
        }
        base = raw.get("base")
        return cls(
            player=player,
            time_ms=time_ms,
            map_size=map_size,
            civ=raw.get("civ", ""),
            state=raw.get("state", "active"),
            resources={r: int(raw.get("resources", {}).get(r, 0)) for r in RESOURCES},
            pop=int(raw.get("pop", 0)),
            pop_cap=int(raw.get("pop_cap", 0)),
            pop_max=int(raw.get("pop_max", 0)),
            phase=raw.get("phase", "village"),
            researching=tuple(raw.get("researching", ())),
            units=units,
            structures=structures,
            sites=sites,
            base=None if base is None else (base[0], base[1]),
        )

    def units_of(self, *roles: str) -> list[Unit]:
        return [u for u in self.units if u.role in roles]

    @property
    def villagers(self) -> list[Unit]:
        return self.units_of("villager")

    @property
    def soldiers(self) -> list[Unit]:
        return [u for u in self.units if u.soldier]

    def finished(self, kind: str | None = None) -> list[Structure]:
        return [s for s in self.structures if not s.foundation and (kind is None or s.kind == kind)]

    def foundations(self) -> list[Structure]:
        return [s for s in self.structures if s.foundation]

    def dropsites(self, resource: str) -> list[Structure]:
        return [s for s in self.finished() if resource in s.drop and s.pos is not None]

    def structure(self, sid: int) -> Structure | None:
        return next((s for s in self.structures if s.id == sid), None)

    @property
    def center(self) -> Pos:
        return (self.map_size / 2, self.map_size / 2)


@dataclass(frozen=True)
class ExecutorView:
    time_ms: int
    map_size: int
    seats: dict[int, SeatView] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ExecutorView:
        time_ms = int(raw.get("time_ms", 0))
        map_size = int(raw.get("map_size", 0))
        seats = {int(p): SeatView.from_json(int(p), time_ms, map_size, v) for p, v in raw.get("players", {}).items()}
        return cls(time_ms=time_ms, map_size=map_size, seats=seats)


def executor_view_js(players: Sequence[int], *, radius: float = RESOURCE_RADIUS, per_type: int = SITES_PER_TYPE) -> str:
    source = resources.files("agentenv_zeroad.js").joinpath("executor_view.js").read_text()
    config = {"players": [int(p) for p in players], "resourceRadius": radius, "perType": per_type}
    return source.replace("__CONFIG__", json.dumps(config))


class ViewQuery:
    """Reuses one JS string for the seats an executor loop drives."""

    def __init__(self, engine: GameEngine, players: Sequence[int]):
        self.engine = engine
        self.code = executor_view_js(players)

    async def __call__(self) -> ExecutorView:
        raw = await self.engine.evaluate(self.code)
        if not isinstance(raw, Mapping):
            raise ValueError(f"executor view query returned {type(raw).__name__}, not an object")
        return ExecutorView.from_json(raw)
