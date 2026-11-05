"""The interface v2 brief and its per-civ reference (docs/post-training-v2.md, §1 "Brief" and "Anonymity").

Two texts replace v1's single brief:
- the **reference**: costs and requirements that never change in a match (units by role, buildings, the phase tech),
  one text per civ, sent once to the brain, which puts it in its system prompt. Every seat of a civ then shares one
  prompt prefix, which the model server caches, and the per-turn message stays small;
- the **brief**: what changed. It names players by number only (no policy label, model name or opponent level), and
  adds the seat's own score parts, its army (soldiers by role, where they are, losses since the last brief, the wave
  gathering against `min_size`) and what it can afford now. Exploration and the total score are left out: they are not
  what decides the match.

`net_worth` reads the seat's net worth from its snapshot (`js/snapshot.js`): the cost of its living units and finished
structures plus its banked resources, which decides a time-capped match under the `net_worth` score rule. A snapshot
without the number (an older recording) leaves the line out.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentenv_zeroad.army import ArmyReport
from agentenv_zeroad.brief import MAX_ENEMY_BUILDINGS, TOKEN_BUDGET, _events_text, _fit, _roles_text
from agentenv_zeroad.executor import Failure, _needs_phase
from agentenv_zeroad.roles import RoleTable
from agentenv_zeroad.state import RESOURCES, PlayerSnapshot, Snapshot, clock_text
from agentenv_zeroad.view import SeatView

NOTICES_HEADER = "Executor gave up since your last brief:"
NOTICE_LINES = 4
NOTICE_WIDTH = 120
MAX_TECHS_SHOWN = 3
RESOURCE_LETTER = {"food": "f", "wood": "w", "stone": "s", "metal": "m"}


def net_worth(snapshot: Snapshot, player: int) -> int | None:
    """The seat's net worth: the cost value of its living units and finished structures plus its banked resources,
    or None when the snapshot carries none."""
    return snapshot.players[player].net_worth


@dataclass(frozen=True)
class BriefContext:
    """What the match gathers for one seat's v2 brief, besides the snapshot and the events."""

    cap_ms: int  # game ms until the time cap; 0 when there is none
    scores: Mapping[str, float]
    army: ArmyReport
    orders: str  # the army posture as the brain gave it: "attack player_2, waves of 8, regrouping below 40%"
    killed: int
    lost: int
    villager_target: int
    affordable: str
    standing: str
    net_worth: int | None = None
    notices: Sequence[Failure] = field(default_factory=tuple)


def _cost(cost: Mapping[str, int]) -> str:
    return "".join(f"{v}{RESOURCE_LETTER[r]}" for r, v in cost.items() if r in RESOURCE_LETTER and v) or "free"


def _needs(requires: Sequence[str]) -> str:
    """`[town]` for a phase requirement, `[tech_name]` for any other; nothing for the village phase."""
    tags = [r.removeprefix("phase_") for r in requires if r != "phase_village"]
    return f"[{','.join(tags)}]" if tags else ""


def reference_text(roles: RoleTable, seat: SeatView | None = None) -> str:
    """The civ's static tables, as short as they can be: for each role the costs of its units with the phase they need
    and where they train (units alike in cost and need are one entry; ones gated by a special tech are left out unless
    the role has no other), then the buildings' costs and needs, and the phase advance's cost when the first view
    offers it."""
    trained_at: dict[str, set[str]] = {}
    for s in roles.structures.values():
        for name in s.trains:
            trained_at.setdefault(name, set()).add(s.key)
    lines = [f"Costs for {roles.civ} (f food, w wood, s stone, m metal; [..] what it needs; @ where it trains):"]
    for role in roles.roles():
        entries: dict[tuple[str, str], set[str]] = {}
        units = roles.by_role[role]
        plain = [u for u in units if all(r.startswith("phase_") for r in u.requires)]
        for u in plain or units[:1]:
            entries.setdefault((_cost(u.cost), _needs(u.requires)), set()).update(trained_at.get(u.name, ()))
        parts = [f"{cost}{need} @{','.join(sorted(where))}" for (cost, need), where in entries.items()]
        lines.append(f"{role}: " + "; ".join(parts))
    skipped = ("Wonder", "Wall", "Gate", "Palisade", "Dock", "Corral")
    builds = [
        f"{s.key} {_cost(s.cost)}{_needs(s.requires)}"
        for s in sorted(roles.structures.values(), key=lambda s: s.key)
        if s.kind not in skipped
    ]
    lines.append("Buildings: " + "; ".join(builds))
    tech = next((t for s in seat.finished() for t in s.techs if t[0].startswith("phase_")), None) if seat else None
    if tech is not None:
        lines.append(f"Phase advance: {tech[0]} {_cost(tech[2])}.")
    return "\n".join(lines)


def affordable_text(roles: RoleTable, seat: SeatView, resources: Mapping[str, int], phase_tech) -> str:
    """What the seat's resources pay for now: units its buildings can train (up to the population room), buildings
    its phase allows, and the next phase advance."""
    room = max(seat.pop_cap - seat.pop, 0)
    units = []
    for role in roles.roles():
        names = set(roles.templates_for(role))
        trainable = [roles.units[n] for s in seat.finished() for n in s.trains if n in names]
        if not trainable:
            continue
        cheapest = min(trainable, key=lambda u: sum(u.cost.values()))
        n = min([room, *(resources.get(r, 0) // c for r, c in cheapest.cost.items() if c > 0)])
        if n > 0:
            units.append(f"{n} {role}")
    builds = [
        s.key
        for s in sorted(roles.structures.values(), key=lambda s: s.key)
        if not _needs_phase(s.requires, seat.phase)
        and s.kind not in ("CivCentre", "Wall", "Gate", "Wonder")
        and all(resources.get(r, 0) >= c for r, c in s.cost.items())
    ]
    text = f"Can afford now: {', '.join(units) or 'no units'}; build {', '.join(builds) or 'nothing'}"
    if phase_tech is not None:
        name, ready, cost = phase_tech
        short = {r: c - resources.get(r, 0) for r, c in cost.items() if resources.get(r, 0) < c}
        gap = f"short by {_cost(short)}" if short else "affordable" + ("" if ready else " but needs more structures")
        text += f"; {name} {gap}"
    return text + "."


def notices_lines(failures: Sequence[Failure]) -> list[str]:
    """The executor's give-ups, merged by order and reason, longest messages cut: the section a brief ends with."""
    merged = Counter((f.what, f.reason) for f in failures)
    lines = []
    for (what, reason), n in list(merged.items())[:NOTICE_LINES]:
        text = f"  {what}{f' (x{n})' if n > 1 else ''}: {reason}"
        lines.append(text if len(text) <= NOTICE_WIDTH else text[: NOTICE_WIDTH - 1] + "…")
    if len(merged) > NOTICE_LINES:
        lines.append(f"  (+{len(merged) - NOTICE_LINES} more)")
    return [NOTICES_HEADER, *lines] if lines else []


def notices_in(brief: str) -> list[str]:
    """The give-up lines of a v2 brief (without the header), for the brain's memory."""
    lines = brief.splitlines()
    if NOTICES_HEADER not in lines:
        return []
    out = []
    for line in lines[lines.index(NOTICES_HEADER) + 1 :]:
        if not line.startswith("  "):
            break
        out.append(line.strip())
    return out


def _buildings(me: PlayerSnapshot) -> str:
    parts = []
    for kind, items in sorted(me.buildings_by_kind().items()):
        done = sum(1 for b in items if not b.foundation)
        parts.append(f"{kind} {done}" + (f"+{len(items) - done}" if len(items) > done else ""))
    return ", ".join(parts) or "none"


def _army_lines(me: PlayerSnapshot, ctx: BriefContext) -> list[str]:
    a = ctx.army
    soldiers = sum(a.soldiers.values())
    roles = _roles_text(a.soldiers)
    head = f"Army: {roles} ({soldiers} soldiers), {a.where}. Orders: {ctx.orders}."
    lines = [head]
    if a.forming:
        wait = f", goes in {a.goes_in_s} s whatever its size" if a.goes_in_s is not None else ""
        lines.append(
            f"Next wave: {a.gathered} of {a.min_size} gathered at the rally point ({a.forming} soldiers){wait}."
        )
    for alive, launched in a.waves:
        lines.append(f"Wave out: {alive} of {launched} alive.")
    lines.append(f"Since your last brief: killed {ctx.killed} enemy units, lost {ctx.lost} units.")
    return lines


def _enemy_lines(me: PlayerSnapshot) -> list[str]:
    if not me.enemies:
        return ["Enemies in sight: none"]
    lines = ["Enemies in sight:"]
    for q, view in sorted(me.enemies.items()):
        near = f", nearest {view.nearest} m from your base" if view.nearest is not None else ""
        lines.append(f"  player {q}: units {_roles_text(view.units)}{near}")
        if view.buildings:
            shown = view.buildings[:MAX_ENEMY_BUILDINGS]
            more = len(view.buildings) - len(shown)
            text = ", ".join(f"{b.kind}@{b.pos[0]},{b.pos[1]}" if b.pos else b.kind for b in shown)
            lines.append(f"    structures: {text}" + (f" (+{more} more)" if more > 0 else ""))
    return lines


def _opponents(snapshot: Snapshot, player: int) -> str:
    rows = [
        f"player {q} {'defeated' if p.state == 'defeated' else 'standing'}"
        for q, p in sorted(snapshot.players.items())
        if q != player
    ]
    return f"Opponents: {', '.join(rows) or 'none'}."


def text_brief_v2(snapshot: Snapshot, player: int, events: Sequence, ctx: BriefContext) -> str:
    """The seat's v2 brief: numbers for players, own score and army, what it can afford, then what happened."""
    me = snapshot.players[player]
    clock = clock_text(snapshot.time_ms)
    timing = f"{clock}, {clock_text(max(0, ctx.cap_ms - snapshot.time_ms))} left" if ctx.cap_ms > 0 else (
        f"{clock}, no time limit (the game is won by conquest)"
    )
    res = " ".join(f"{r} {me.resources[r]}" for r in RESOURCES)
    gat = " ".join(f"{r} {me.gatherers[r]}" for r in RESOURCES)
    score = f"Score: economy {ctx.scores.get('economy', 0):.0f}, military {ctx.scores.get('military', 0):.0f}."
    if ctx.net_worth is not None:
        score += f" Net worth {ctx.net_worth}."
    villagers = me.units.get("villager", 0)
    lines = [
        f"Time {timing}. You are player {player} ({me.civ}), "
        f"{me.state}.",
        f"Phase: {me.phase or 'none'}. Population {me.pop}/{me.pop_cap} (max {me.pop_max}).",
        f"Resources: {res}. Gatherers: {gat}.",
        f"Villagers {villagers} (target {ctx.villager_target}); idle: {len(me.idle_villagers)} villagers, "
        f"{len(me.idle_production)} production buildings with empty queues.",
        score,
        *_army_lines(me, ctx),
        ctx.affordable,
        f"Buildings: {_buildings(me)}.",
    ]
    techs = [t for t in me.techs if not t.startswith(("phase_", "civbonuses/"))]
    shown = techs[-MAX_TECHS_SHOWN:]
    lines.append(
        f"Researched: {', '.join(shown) if shown else 'nothing yet'}"
        + ("" if len(shown) == len(techs) else f" (+{len(techs) - len(shown)} earlier)")
        + f". Researching: {', '.join(me.researching) if me.researching else 'nothing'}."
    )
    if me.enemies_near_base:
        lines.append(f"WARNING: {me.enemies_near_base} enemy soldiers within reach of your base.")
    lines += [_opponents(snapshot, player), *_enemy_lines(me), *notices_lines(ctx.notices)]
    lines.append(_events_text(events))
    lines.append(f"Standing orders: {ctx.standing}")
    return _fit("\n".join(lines), TOKEN_BUDGET)
