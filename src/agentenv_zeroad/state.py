"""Typed per-player snapshots of the game and the events between two of them (spec §4.3).

The `/step` state is the whole game with no fog and no economy totals, so every brain reads this view instead. One
`/evaluate` call (`js/snapshot.js`) returns all viewers at once; measured on the real engine at about 3 ms for two
players (see the engine test), well inside the 200 ms turn.

Units are counted in the six order roles of spec §4.5 (`villager | infantry_melee | infantry_ranged | cavalry |
siege | champion`) plus `other` for heroes, ships, traders and healers, which no order trains. Enemies are only
those the viewer can see this turn (the JS honours fog; see its header for the reveal-all quirk).

Events come from comparing two snapshots of the same player. Loss and training counts come from the engine's
`StatisticsTracker` totals, which are exact even when a unit is trained and another dies in the same interval;
roles and kinds come from count and id differences, which are best-effort labels.

Snapshots arrive five times a second but a strategist reads them every ~25 s (spec §4.6), so `EventTracker` keeps
the pending list short and meaningful: one under_attack per raid (a raid ends after `RAID_QUIET_MS` with no enemy
soldiers near base and no damage; a building first hit later in the raid is reported), per-tick unit_lost and
unit_trained events merged into one running event each, and urgent events never evicted by the length cap.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from agentenv_zeroad.engine import GameEngine

ROLES = ("villager", "infantry_melee", "infantry_ranged", "cavalry", "siege", "champion")
RESOURCES = ("food", "wood", "stone", "metal")
PHASES = ("village", "town", "city")
BASE_RADIUS = 80.0
URGENT_EVENTS = frozenset({"under_attack", "building_lost"})
MERGED_EVENTS = frozenset({"unit_lost", "unit_trained"})
RAID_QUIET_MS = 15_000

Pos = tuple[int, int]


def _pos(raw: Sequence[float] | None) -> Pos | None:
    return None if raw is None else (int(raw[0]), int(raw[1]))


@dataclass(frozen=True)
class Building:
    id: int
    kind: str
    pos: Pos | None
    hp: int
    foundation: bool = False
    queue: int | None = None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Building:
        return cls(
            id=int(raw["id"]),
            kind=str(raw["kind"]),
            pos=_pos(raw.get("pos")),
            hp=int(raw.get("hp", 100)),
            foundation=bool(raw.get("foundation", False)),
            queue=raw.get("queue"),
        )


@dataclass(frozen=True)
class EnemyView:
    """What one viewer can see of one enemy player right now."""

    player: int
    units: dict[str, int]
    buildings: tuple[Building, ...]
    nearest: int | None
    soldiers_at: tuple[int, int, int] | None = None  # (x, z, count): centroid of its visible soldiers
    workers_at: tuple[int, int, int] | None = None  # the same for its visible workers
    contact: int = 0  # its visible soldiers within reach of one of ours

    @classmethod
    def from_json(cls, player: int, raw: Mapping[str, Any]) -> EnemyView:
        buildings = tuple(Building.from_json({"hp": 100, **b}) for b in raw.get("buildings", ()))
        return cls(
            player=player,
            units=dict(raw.get("units", {})),
            buildings=buildings,
            nearest=raw.get("nearest"),
            soldiers_at=_group(raw.get("soldiers_at")),
            workers_at=_group(raw.get("workers_at")),
            contact=int(raw.get("contact") or 0),
        )

    @property
    def army(self) -> int:
        return sum(n for role, n in self.units.items() if role not in ("villager", "other"))


@dataclass(frozen=True)
class NearbyResource:
    amount: int
    sources: int


@dataclass(frozen=True)
class PlayerSnapshot:
    player: int
    name: str
    civ: str
    state: str
    resources: dict[str, int]
    gatherers: dict[str, int]
    pop: int
    pop_cap: int
    pop_max: int
    phase: str
    techs: tuple[str, ...]
    researching: tuple[str, ...]
    buildings: tuple[Building, ...]
    units: dict[str, int]
    idle_units: dict[str, int]
    idle_villagers: tuple[int, ...]
    idle_production: tuple[int, ...]
    nearby_resources: dict[str, NearbyResource]
    enemies: dict[int, EnemyView]
    enemies_near_base: int
    base: Pos | None
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, player: int, raw: Mapping[str, Any]) -> PlayerSnapshot:
        return cls(
            player=player,
            name=str(raw.get("name", "")),
            civ=str(raw.get("civ", "")),
            state=str(raw.get("state", "active")),
            resources={r: int(raw.get("resources", {}).get(r, 0)) for r in RESOURCES},
            gatherers={r: int(raw.get("gatherers", {}).get(r, 0)) for r in RESOURCES},
            pop=int(raw.get("pop", 0)),
            pop_cap=int(raw.get("pop_cap", 0)),
            pop_max=int(raw.get("pop_max", 0)),
            phase=str(raw.get("phase", "")),
            techs=tuple(raw.get("techs", ())),
            researching=tuple(raw.get("researching", ())),
            buildings=tuple(Building.from_json(b) for b in raw.get("buildings", ())),
            units={k: int(v) for k, v in raw.get("units", {}).items()},
            idle_units={k: int(v) for k, v in raw.get("idle_units", {}).items()},
            idle_villagers=tuple(int(i) for i in raw.get("idle_villagers", ())),
            idle_production=tuple(int(i) for i in raw.get("idle_production", ())),
            nearby_resources={
                r: NearbyResource(int(v.get("amount", 0)), int(v.get("sources", 0)))
                for r, v in raw.get("nearby_resources", {}).items()
            },
            enemies={int(q): EnemyView.from_json(int(q), v) for q, v in raw.get("enemies", {}).items()},
            enemies_near_base=int(raw.get("enemies_near_base", 0)),
            base=_pos(raw.get("base")),
            stats={k: int(v) for k, v in raw.get("stats", {}).items()},
        )

    @property
    def army(self) -> int:
        return sum(n for role, n in self.units.items() if role not in ("villager", "other"))

    def buildings_by_kind(self) -> dict[str, list[Building]]:
        out: dict[str, list[Building]] = {}
        for b in self.buildings:
            out.setdefault(b.kind, []).append(b)
        return out


@dataclass(frozen=True)
class Snapshot:
    time_ms: int
    players: dict[int, PlayerSnapshot]

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Snapshot:
        players = {int(p): PlayerSnapshot.from_json(int(p), v) for p, v in raw.get("players", {}).items()}
        return cls(time_ms=int(raw.get("time_ms", 0)), players=players)


def snapshot_js(players: Sequence[int], *, base_radius: float = BASE_RADIUS) -> str:
    """The JS for one `/evaluate` returning every viewer in `players`; enemies are looked up among all players."""
    source = resources.files("agentenv_zeroad.js").joinpath("snapshot.js").read_text()
    config = {"players": [int(p) for p in players], "baseRadius": base_radius}
    return source.replace("__CONFIG__", json.dumps(config))


async def query_snapshot(engine: GameEngine, players: Sequence[int], *, code: str | None = None) -> Snapshot:
    """Run the snapshot query for the viewers in `players`."""
    code = code or snapshot_js(players)
    raw = await engine.evaluate(code)
    if not isinstance(raw, Mapping):
        raise ValueError(f"snapshot query returned {type(raw).__name__}, not an object")
    return Snapshot.from_json(raw)


class SnapshotQuery:
    """Reuses one JS string and keeps the wall-clock cost of each query, for the < 30 ms p95 budget."""

    def __init__(self, engine: GameEngine, players: Sequence[int]):
        self.engine = engine
        self.players = tuple(players)
        self.code = snapshot_js(self.players)
        self.costs_s: list[float] = []

    async def __call__(self) -> Snapshot:
        started = time.perf_counter()
        snap = await query_snapshot(self.engine, self.players, code=self.code)
        self.costs_s.append(time.perf_counter() - started)
        return snap


@dataclass(frozen=True)
class Event:
    kind: str
    time_ms: int
    text: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def urgent(self) -> bool:
        return self.kind in URGENT_EVENTS

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "t": clock_text(self.time_ms), "text": self.text, **self.data}


def _group(raw: Any) -> tuple[int, int, int] | None:
    return (int(raw[0]), int(raw[1]), int(raw[2])) if isinstance(raw, list | tuple) and len(raw) == 3 else None


def clock_text(time_ms: int) -> str:
    seconds = max(0, time_ms) // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def diff_events(prev: PlayerSnapshot, cur: PlayerSnapshot, time_ms: int) -> list[Event]:
    """Events for one player between two of its snapshots, in a fixed order."""
    events: list[Event] = []
    events += _attack_events(prev, cur, time_ms)
    events += _building_events(prev, cur, time_ms)
    events += _unit_events(prev, cur, time_ms)
    if cur.phase != prev.phase and cur.phase in PHASES and _phase_rank(cur.phase) > _phase_rank(prev.phase):
        events.append(Event("phase_reached", time_ms, f"reached {cur.phase} phase", {"phase": cur.phase}))
    for tech in sorted(set(cur.techs) - set(prev.techs)):
        if not tech.startswith("phase_"):
            events.append(Event("tech_researched", time_ms, f"researched {tech}", {"tech": tech}))
    events += _resource_events(prev, cur, time_ms)
    return events


def _phase_rank(phase: str) -> int:
    return PHASES.index(phase) if phase in PHASES else -1


def _attack_events(prev: PlayerSnapshot, cur: PlayerSnapshot, time_ms: int) -> list[Event]:
    before = {b.id: b for b in prev.buildings}
    damaged = [b for b in cur.buildings if not b.foundation and b.id in before and b.hp < before[b.id].hp]
    if cur.enemies_near_base <= 0 and not damaged:
        return []
    if prev.enemies_near_base > 0 and not damaged:
        return []  # the same raid is still there; it was reported when it arrived
    attackers = sorted(q for q, view in cur.enemies.items() if view.army and view.nearest is not None)
    parts = []
    if cur.enemies_near_base:
        parts.append(f"{cur.enemies_near_base} enemy soldiers near base")
    if damaged:
        parts.append("damaged: " + ", ".join(sorted({b.kind for b in damaged})))
    data = {"near_base": cur.enemies_near_base, "damaged": [b.id for b in damaged], "by": attackers}
    return [Event("under_attack", time_ms, "under attack: " + "; ".join(parts), data)]


def _building_events(prev: PlayerSnapshot, cur: PlayerSnapshot, time_ms: int) -> list[Event]:
    now_ids = {b.id for b in cur.buildings}
    gone = [b for b in prev.buildings if b.id not in now_ids and not b.foundation]
    if "buildings_lost" in cur.stats:
        count = cur.stats["buildings_lost"] - prev.stats.get("buildings_lost", 0)
    else:
        count = len(gone)
    if count <= 0:
        return []
    kinds = sorted(b.kind for b in gone)
    text = f"lost {count} building(s)" + (f": {', '.join(kinds)}" if kinds else "")
    return [Event("building_lost", time_ms, text, {"count": count, "kinds": kinds})]


def _unit_events(prev: PlayerSnapshot, cur: PlayerSnapshot, time_ms: int) -> list[Event]:
    events: list[Event] = []
    delta = {r: cur.units.get(r, 0) - prev.units.get(r, 0) for r in set(cur.units) | set(prev.units)}
    gained = {r: d for r, d in sorted(delta.items()) if d > 0}
    dropped = {r: -d for r, d in sorted(delta.items()) if d < 0}
    lost = cur.stats.get("units_lost", 0) - prev.stats.get("units_lost", 0) if cur.stats else sum(dropped.values())
    trained = (
        cur.stats.get("units_trained", 0) - prev.stats.get("units_trained", 0) if cur.stats else sum(gained.values())
    )
    if lost > 0:
        events.append(_unit_event("unit_lost", time_ms, lost, dropped))
    if trained > 0:
        events.append(_unit_event("unit_trained", time_ms, trained, gained))
    return events


def _unit_event(kind: str, time_ms: int, count: int, roles: Mapping[str, int]) -> Event:
    verb = "lost" if kind == "unit_lost" else "trained"
    ordered = dict(sorted(roles.items()))
    return Event(kind, time_ms, f"{verb} {count} unit(s){_roles(ordered)}", {"count": count, **ordered})


def _roles(counts: Mapping[str, int]) -> str:
    return (": " + ", ".join(f"{n} {r}" for r, n in counts.items())) if counts else ""


def merge_events(events: Sequence[Event]) -> list[Event]:
    """Fold repeated unit_lost / unit_trained events into one each (summed counts, latest time), in time order."""
    out: list[Event] = []
    for e in events:
        if e.kind not in MERGED_EVENTS or "count" not in e.data:
            out.append(e)
            continue
        i = next((i for i, o in enumerate(out) if o.kind == e.kind and "count" in o.data), None)
        if i is None:
            out.append(e)
            continue
        old = out.pop(i)
        roles = {k: v for k, v in old.data.items() if k != "count"}
        for k, v in e.data.items():
            if k != "count":
                roles[k] = roles.get(k, 0) + v
        out.append(_unit_event(e.kind, e.time_ms, old.data["count"] + e.data["count"], roles))
    return out


def _base_ccs(player: PlayerSnapshot) -> set[int]:
    """The civic centres `nearby_resources` is summed around (the same rule as js/snapshot.js)."""
    return {b.id for b in player.buildings if b.kind == "CivCentre" and not b.foundation}


def _resource_events(prev: PlayerSnapshot, cur: PlayerSnapshot, time_ms: int) -> list[Event]:
    if _base_ccs(prev) != _base_ccs(cur):
        return []  # the area being summed changed (a CC was lost or finished), so the totals are not comparable
    events = []
    for r in RESOURCES:
        a, b = prev.nearby_resources.get(r), cur.nearby_resources.get(r)
        if a is None or b is None:
            continue
        ran_out = a.amount > 0 and b.amount == 0
        mine_gone = r in ("stone", "metal") and b.sources < a.sources
        if ran_out or mine_gone:
            left = "none left" if b.amount == 0 else f"{b.amount} left in {b.sources} source(s)"
            text = f"{r} exhausted near base ({left})"
            events.append(Event("resource_exhausted", time_ms, text, {"resource": r, "left": b.amount}))
    return events


class EventTracker:
    """Per-seat event history: feed every snapshot, take the events since the last brief with `drain`."""

    def __init__(self, player: int, *, keep: int = 50):
        self.player = player
        self.keep = keep
        self._last: PlayerSnapshot | None = None
        self._pending: list[Event] = []
        self._raid_until = -1
        self._raid_hit: set[int] = set()

    def update(self, snapshot: Snapshot) -> list[Event]:
        """Diff `snapshot` against the previous one and return the new events (after per-raid de-duplication)."""
        cur = snapshot.players.get(self.player)
        if cur is None:
            return []
        events = [] if self._last is None else diff_events(self._last, cur, snapshot.time_ms)
        self._last = cur
        events = self._one_alarm_per_raid(events, cur, snapshot.time_ms)
        self._pending = merge_events(self._pending + events)
        self._cap()
        return events

    def _one_alarm_per_raid(self, events: list[Event], cur: PlayerSnapshot, time_ms: int) -> list[Event]:
        attack = next((e for e in events if e.kind == "under_attack"), None)
        if time_ms > self._raid_until:
            self._raid_hit = set()
        ongoing = time_ms <= self._raid_until
        if cur.enemies_near_base > 0 or attack is not None:
            self._raid_until = time_ms + RAID_QUIET_MS
        if attack is None:
            return events
        hit = set(attack.data["damaged"])
        fresh = hit - self._raid_hit
        self._raid_hit |= hit
        if ongoing and not fresh:
            return [e for e in events if e is not attack]
        return events

    def _cap(self) -> None:
        """Drop the oldest non-urgent events beyond `keep`; urgent ones stay until drained."""
        excess = len(self._pending) - self.keep
        if excess <= 0:
            return
        drop = {id(e) for e in [e for e in self._pending if not e.urgent][:excess]}
        self._pending = [e for e in self._pending if id(e) not in drop]

    def has_urgent(self) -> bool:
        return any(e.urgent for e in self._pending)

    def peek(self) -> list[Event]:
        return list(self._pending)

    def drain(self) -> list[Event]:
        events, self._pending = self._pending, []
        return events
