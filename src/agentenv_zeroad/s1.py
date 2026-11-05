"""The S1 tool check's data side (docs/post-training-v2.md, stage S1): which pilot games to replay, and the metrics of a
folder of finished games per interface.

`scripts/s1_specs.py` and `scripts/s1_summary.py` are thin command lines over this module.

A game here is a results.json (`LiveMatch.results`) and the path it was found at. Its *focus seat* is the one under
test: a replay seat, else a scripted bot, else the pilot's policy (`policy v<N>`), else the first agent. Its opponent is
labelled `petra-<level>` for Petra, else the other seat's name in lower case. The pilot's results do not record Petra's
level, so for those the label comes from the round's `specs.json` or the run's `games.jsonl` (found above the results
file), else from the game id (`r00-p-petra-2`).

The metrics are the S1 gate's: field placement failures per game, villagers at 15 minutes, idle production buildings
(the mean count over the game), the minute the town phase is reached, the median army at 10 and 15 minutes, and the win
rate by opponent. They are read from the results the match writes: the focus seat's `failures` and `metrics`, and the
timeline rows (`villagers` where the match recorded them, else population minus army); the idle production buildings
are the seat's `idle_production_s` over the game's length, which is their mean count.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_zeroad.replay import ReplaySpec

FOCUS_KINDS = ("replay", "bot")
POLICY_NAME = re.compile(r"^policy v(\d+)$")
PETRA_IN_ID = re.compile(r"petra-(\d)")
TOWN_PHASES = ("town", "city")
MINUTE_MS = 60_000


@dataclass(frozen=True)
class Game:
    path: Path
    results: Mapping[str, Any]

    @property
    def game_id(self) -> str:
        """The pilot's game id (the folder under `games/`), else the results file's stem."""
        parts = self.path.parts
        return parts[parts.index("games") + 1] if "games" in parts else self.path.stem

    @property
    def unique_id(self) -> str:
        """`<box>-<run>-<game id>`: the two folders above `rounds/` and the game id, since the pilot reuses game ids
        across its boxes and runs. A path without that layout is its own game id."""
        parts = self.path.parts
        if "games" not in parts:
            return self.game_id
        above = parts[: parts.index("games")]
        if "rounds" not in above:
            return self.game_id
        runs = above[: len(above) - 1 - above[::-1].index("rounds")]
        return "-".join((*runs[-2:], self.game_id))

    @property
    def focus(self) -> str | None:
        seats = self.results["seats"]
        for kind in FOCUS_KINDS:
            slot = next((s for s, v in seats.items() if v.get("kind") == kind), None)
            if slot is not None:
                return slot
        policy = [s for s, v in seats.items() if POLICY_NAME.match(str(v.get("name", "")))]
        agents = [s for s, v in seats.items() if v.get("kind") == "agent"]
        return (policy or agents or [None])[0]

    @property
    def group(self) -> str:
        """`<seat kind>/<interface>` of the focus seat: what a summary column is."""
        seat = self.results["seats"][self.focus]
        return f"{seat.get('kind')}/{seat.get('interface', 'v1')}"

    @property
    def policy_version(self) -> int | None:
        match = POLICY_NAME.match(str(self.results["seats"][self.focus].get("name", ""))) if self.focus else None
        return int(match.group(1)) if match else None

    @property
    def opponent(self) -> str:
        seats = self.results["seats"]
        others = [s for s in seats if s != self.focus]
        if not others:
            return "none"
        seat = seats[others[0]]
        if seat.get("kind") != "petra":
            return str(seat.get("name", "unknown")).lower()
        level = seat.get("ai_difficulty")
        if level is not None:
            return f"petra-{level}"
        return self._label_from_logs() or (
            f"petra-{found[1]}" if (found := PETRA_IN_ID.search(self.game_id)) else "petra"
        )

    def _label_from_logs(self) -> str | None:
        """The pilot's round and run logs name each game's opponent: `specs.json` (round) and `games.jsonl` (run)."""
        for folder in self.path.parents:
            specs, games = folder / "specs.json", folder / "games.jsonl"
            if specs.is_file():
                row = next((s for s in json.loads(specs.read_text()) if s.get("game_id") == self.game_id), None)
                if row is not None and row.get("opponent", {}).get("kind") == "petra":
                    return f"petra-{row['opponent']['level']}"
            if games.is_file():
                for line in games.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("game_id") == self.game_id and row.get("opponent"):
                        return str(row["opponent"])
        return None


def load_games(root: Path) -> Iterator[Game]:
    for path in sorted(root.rglob("*results.json")):
        yield Game(path, json.loads(path.read_text()))


def pick_games(
    games: Sequence[Game], n: int, *, version: int | None = 0, min_turns: int = 15, seed: int = 0
) -> list[Game]:
    """`n` games whose policy seat logged at least `min_turns` turns (those of policy `version`, if any), spread
    over the opponents round-robin; the order within an opponent is shuffled by `seed`."""
    pool = [
        g
        for g in games
        if g.focus is not None
        and len(g.results["seats"][g.focus].get("turns") or []) >= min_turns
        and (version is None or g.policy_version == version)
    ]
    by_opponent: dict[str, list[Game]] = defaultdict(list)
    for g in pool:
        by_opponent[g.opponent].append(g)
    rng = random.Random(seed)
    for group in by_opponent.values():
        rng.shuffle(group)
    picked: list[Game] = []
    queues = [by_opponent[k] for k in sorted(by_opponent)]
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def replay_spec(game: Game, *, delay_ms: int = 0) -> ReplaySpec:
    """The focus seat's logged orders, with what is needed to set the same match up again."""
    results = game.results
    seats = {
        slot: {"kind": seat.get("kind"), "civ": seat.get("civ"), "name": seat.get("name")}
        for slot, seat in results["seats"].items()
    }
    setup = {
        "map": results.get("map"),
        "time_cap_s": results.get("time_cap_s"),
        "ended_by": results.get("ended_by"),
        "opponent": game.opponent,
        "seats": seats,
    }
    return ReplaySpec.from_results(results, game.focus, source=game.unique_id, delay_ms=delay_ms, game=setup)


@dataclass(frozen=True)
class GameMetrics:
    group: str
    opponent: str
    points: float | None  # 1 for a win, 0.5 for a draw, 0 for a loss; None while undecided
    field_failures: int
    villagers_15: int | None
    idle_production: float | None
    town_minute: float | None
    army_10: int | None
    army_15: int | None


def _row_at(timeline: Sequence[Mapping[str, Any]], minute: int) -> Mapping[str, Any] | None:
    """The last row at or before `minute`, None if the game ended before it."""
    at = minute * MINUTE_MS
    if not timeline or timeline[-1]["t_ms"] <= at:
        return None
    return [r for r in timeline if r["t_ms"] <= at][-1]


def _is_field_placement_failure(failure: Mapping[str, Any]) -> bool:
    return failure.get("key") == "field" and str(failure.get("reason", "")).startswith("no valid spot")


def game_metrics(game: Game) -> GameMetrics | None:
    slot = game.focus
    if slot is None:
        return None
    results = game.results
    seat = results["seats"][slot]
    timeline = [
        {**row["players"][slot], "t_ms": row["t_ms"]} for row in results.get("timeline", ()) if slot in row["players"]
    ]
    at15, at10 = _row_at(timeline, 15), _row_at(timeline, 10)
    town = next((r["t_ms"] / MINUTE_MS for r in timeline if r.get("phase") in TOWN_PHASES), None)
    match_s = (seat.get("metrics") or {}).get("match_s") or results.get("game_time_s") or 0
    idle_s = (seat.get("metrics") or {}).get("idle_production_s")
    outcome = seat.get("outcome")
    return GameMetrics(
        group=game.group,
        opponent=game.opponent,
        points={"won": 1.0, "drawn": 0.5, "lost": 0.0}.get(outcome),
        field_failures=sum(1 for f in seat.get("failures", ()) if _is_field_placement_failure(f)),
        villagers_15=None if at15 is None else int(at15.get("villagers", at15.get("pop", 0) - at15.get("army", 0))),
        idle_production=None if idle_s is None or not match_s else idle_s / match_s,
        town_minute=town,
        army_10=None if at10 is None else int(at10.get("army", 0)),
        army_15=None if at15 is None else int(at15.get("army", 0)),
    )


def _median(values: Sequence[float | int | None]) -> str:
    kept = [v for v in values if v is not None]
    return f"{statistics.median(kept):.1f} (n={len(kept)})" if kept else "n/a"


def _mean(values: Sequence[float | int | None]) -> str:
    kept = [v for v in values if v is not None]
    return f"{statistics.fmean(kept):.2f} (n={len(kept)})" if kept else "n/a"


def format_summary(metrics: Sequence[GameMetrics]) -> str:
    """One table: a column per `<seat kind>/<interface>` group, a row per S1 metric, then the win rate by opponent."""
    groups = sorted({m.group for m in metrics})
    by_group = {g: [m for m in metrics if m.group == g] for g in groups}
    width = 24
    lines = [f"{'':<38}" + "".join(f"{g + f' ({len(by_group[g])} games)':>{width}}" for g in groups)]

    def row(label: str, cell) -> None:
        lines.append(f"{label:<38}" + "".join(f"{cell(by_group[g]):>{width}}" for g in groups))

    row("field placement failures per game", lambda ms: _mean([m.field_failures for m in ms]))
    row("villagers at 15 min (median)", lambda ms: _median([m.villagers_15 for m in ms]))
    row("idle production buildings (mean)", lambda ms: _mean([m.idle_production for m in ms]))
    row("games reaching town phase", lambda ms: f"{sum(m.town_minute is not None for m in ms)}/{len(ms)}")
    row("minute of town phase (median)", lambda ms: _median([m.town_minute for m in ms]))
    row("army at 10 min (median)", lambda ms: _median([m.army_10 for m in ms]))
    row("army at 15 min (median)", lambda ms: _median([m.army_15 for m in ms]))
    lines += ["", "win rate by opponent (a draw counts half):"]
    for opponent in sorted({m.opponent for m in metrics}):
        row(f"  {opponent}", lambda ms, o=opponent: _wins([m for m in ms if m.opponent == o]))
    return "\n".join(lines)


def _wins(ms: Sequence[GameMetrics]) -> str:
    decided = [m.points for m in ms if m.points is not None]
    return f"{sum(decided) / len(decided):.0%} ({sum(decided):g}/{len(decided)})" if decided else "-"
