"""Briefs: what each empire is told (spec §4.3), as text for strategists and compact JSON for Luna's questions.

Both carry the game clock and the events since the last brief, describe the army in the §4.5 order roles, and list
only enemies the seat can see. Briefs stay under a 3k-token budget (spec §4.3, for latency). Tokens are estimated
as characters / 4, which overestimates for this mostly-English, number-heavy text; long lists (buildings, enemy
structures, events) are capped so a late, sprawling game still fits. The event cap never drops an urgent event or a
phase change in favour of routine ones: a brief triggered by an attack must mention the attack.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from agentenv_zeroad.state import RESOURCES, ROLES, Event, PlayerSnapshot, Snapshot, clock_text, merge_events

TOKEN_BUDGET = 3000
MAX_EVENTS = 12
MAX_ENEMY_BUILDINGS = 6
MAX_TECHS = 20


def approx_tokens(text: str) -> int:
    """Token estimate used for the budget: one token per four characters, rounded up."""
    return math.ceil(len(text) / 4)


def _roles_text(counts: Mapping[str, int]) -> str:
    parts = [f"{counts[r]} {r}" for r in (*ROLES, "other") if counts.get(r)]
    return ", ".join(parts) if parts else "none"


def _buildings_text(me: PlayerSnapshot) -> list[str]:
    lines = []
    for kind, items in sorted(me.buildings_by_kind().items()):
        done = [b for b in items if not b.foundation]
        building = len(items) - len(done)
        ids = ",".join(str(b.id) for b in done[:4]) + ("…" if len(done) > 4 else "")
        extra = f", {building} under construction" if building else ""
        lines.append(f"  {kind}: {len(done)}{f' (ids {ids})' if ids else ''}{extra}")
    return lines


def _enemy_lines(me: PlayerSnapshot, names: Mapping[int, str]) -> list[str]:
    if not me.enemies:
        return ["Enemies in sight: none"]
    lines = ["Enemies in sight:"]
    for q, view in sorted(me.enemies.items()):
        who = names.get(q, f"player {q}")
        near = f", nearest {view.nearest} m from base" if view.nearest is not None else ""
        lines.append(f"  {who} (player {q}): units {_roles_text(view.units)}{near}")
        if view.buildings:
            shown = view.buildings[:MAX_ENEMY_BUILDINGS]
            more = len(view.buildings) - len(shown)
            text = ", ".join(f"{b.kind}@{b.pos[0]},{b.pos[1]}" if b.pos else b.kind for b in shown)
            lines.append(f"    structures: {text}" + (f" (+{more} more)" if more > 0 else ""))
    return lines


def text_brief(
    snapshot: Snapshot,
    player: int,
    events: Sequence[Event] = (),
    *,
    names: Mapping[int, str] | None = None,
) -> str:
    """The strategist's brief for `player`: plain text, at most `TOKEN_BUDGET` estimated tokens."""
    me = snapshot.players[player]
    names = names or {p: s.name for p, s in snapshot.players.items()}
    res = " ".join(f"{r} {me.resources[r]}" for r in RESOURCES)
    gat = " ".join(f"{r} {me.gatherers[r]}" for r in RESOURCES)
    near = " ".join(f"{r} {v.amount}" for r, v in me.nearby_resources.items())
    lines = [
        f"Time {clock_text(snapshot.time_ms)}. You are {me.name} ({me.civ}), player {player}, {me.state}.",
        f"Phase: {me.phase or 'none'}. Population {me.pop}/{me.pop_cap} (max {me.pop_max}).",
        f"Resources: {res}.",
        f"Gatherers: {gat}.",
        f"Resources left near base: {near}." if near else "Resources left near base: unknown.",
        f"Army by role: {_roles_text(me.units)} (fighting strength {me.army}).",
        f"Idle: {len(me.idle_villagers)} villagers; idle units {_roles_text(me.idle_units)}; "
        f"{len(me.idle_production)} production buildings with empty queues.",
        "Buildings:",
        *(_buildings_text(me) or ["  none"]),
    ]
    techs = [t for t in me.techs if not t.startswith(("phase_", "civbonuses/"))]
    shown = techs[-MAX_TECHS:]
    lines.append(
        f"Researched: {', '.join(shown) if shown else 'nothing yet'}"
        + ("" if len(shown) == len(techs) else f" (+{len(techs) - len(shown)} earlier)")
    )
    lines.append(f"Researching: {', '.join(me.researching) if me.researching else 'nothing'}")
    if me.stats:
        lines.append(
            f"Combat so far: killed {me.stats.get('enemy_units_killed', 0)} enemy units; lost "
            f"{me.stats.get('units_lost', 0)} units and {me.stats.get('buildings_lost', 0)} buildings."
        )
    if me.enemies_near_base:
        lines.append(f"WARNING: {me.enemies_near_base} enemy soldiers within reach of your base.")
    lines += _enemy_lines(me, names)
    lines.append(_events_text(events))
    return _fit("\n".join(lines))


def select_events(events: Sequence[Event], limit: int = MAX_EVENTS) -> list[Event]:
    """At most `limit` events in time order: repeated unit events merged, then urgent and phase events first, then
    the newest of the rest."""
    merged = merge_events(events)
    key = [(not (e.urgent or e.kind == "phase_reached"), -i) for i, e in enumerate(merged)]
    keep = sorted(sorted(range(len(merged)), key=lambda i: key[i])[:limit])
    return [merged[i] for i in keep]


def _events_text(events: Sequence[Event]) -> str:
    if not events:
        return "Since last brief: nothing notable."
    events = merge_events(events)
    shown = select_events(events)
    head = f"Since last brief ({len(events)} events" + (f", {len(shown)} shown" if len(shown) < len(events) else "")
    return head + "):\n" + "\n".join(f"  [{clock_text(e.time_ms)}] {e.text}" for e in shown)


def _fit(text: str, budget: int = TOKEN_BUDGET) -> str:
    """Hard cap: if every list cap still left the brief over budget, cut it at the budget and say so."""
    if approx_tokens(text) <= budget:
        return text
    marker = "\n[brief truncated]"
    return text[: budget * 4 - len(marker)] + marker


def json_brief(snapshot: Snapshot, player: int, events: Sequence[Event] = ()) -> dict[str, Any]:
    """Compact JSON brief for Luna decision questions: numbers and ids, no prose."""
    me = snapshot.players[player]
    return {
        "t": clock_text(snapshot.time_ms),
        "time_ms": snapshot.time_ms,
        "player": player,
        "civ": me.civ,
        "phase": me.phase,
        "pop": [me.pop, me.pop_cap],
        "res": me.resources,
        "gath": me.gatherers,
        "army": {r: n for r, n in me.units.items() if n},
        "idle": {
            "villagers": len(me.idle_villagers),
            "units": {r: n for r, n in me.idle_units.items() if n},
            "production": list(me.idle_production),
        },
        "researching": list(me.researching),
        "bld": {k: sum(1 for b in v if not b.foundation) for k, v in sorted(me.buildings_by_kind().items())},
        "killed": me.stats.get("enemy_units_killed", 0),
        "lost": me.stats.get("units_lost", 0),
        "near_base_left": {r: v.amount for r, v in me.nearby_resources.items()},
        "threat": me.enemies_near_base,
        "enemies": {
            str(q): {
                "units": {r: n for r, n in v.units.items() if n},
                "structures": len(v.buildings),
                "dist": v.nearest,
                "contact": v.contact,
            }
            for q, v in sorted(me.enemies.items())
        },
        "events": [e.to_json() for e in select_events(events)],
    }


def json_brief_text(snapshot: Snapshot, player: int, events: Sequence[Event] = ()) -> str:
    return json.dumps(json_brief(snapshot, player, events), separators=(",", ":"))
