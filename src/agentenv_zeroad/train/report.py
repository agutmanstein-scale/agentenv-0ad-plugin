"""report.json for a v2 run: every series the v2 dashboard draws, from the runner's and the learner's folders.

    python -m agentenv_zeroad.train.report --games DIR --learner ROOT --out report.json [--ledger spend.jsonl]
        [--panels DIR --panel BASE:OTHER] [--learner-slot 1,2]

`--games` is the orchestrator's `ROOT/games` (one `<game_id>/results.json` per training game, `games.jsonl`),
`--panels` its `ROOT/panels` (one folder per panel, `NAME/<game_id>/results.json`, every game of a panel pinned to one
version), `--learner` the learner's `--root` (health.jsonl, adapters/v*/manifest.json).
The file is rewritten whole each time, so the dashboard fetches one
small file. Sections:

- `elo`: the Bradley-Terry board (train/stats.py) with bootstrap intervals, the policy's hourly bins as `curve`.
- `trend`, `milestones`, `panel`: the trend test, each milestone with its checks and when it passed (and
  `milestone_rule`, the schedule they were judged on), and the panel test when `--panel` names two panels (folders
  under `--panels`, played at different times, even by different runs). Panel games are kept off the Elo board.
- `behaviour`: per training hour and overall, what the policy does with its army and economy: army at minutes 5, 10
  and 15, the time of the first attack order and the number of attack and raid orders (a proxy: the env's wave
  launches are not in results.json), kills at home and away (from the engine's combat
  counters when a game has them), villagers at minute 15, conquest share (games that ended by conquest) and conquest
  wins, hold share, the tool mix and the rejected share. These move before Elo does.
- `health`: health.jsonl as a series, with the trips and alarms each update recorded.
- `throughput` and `spend`: games and trained decisions per hour; opponent spend from the games' own estimates and, if
  given, the budget ledger.

A game counts when it finished by conquest or a cap. Its policy seat is the one marked `"trainable": true` in the spec,
or `--learner-slot`, or else the seat with the highest policy version (the learner against a frozen v0, a base policy
against a Petra or a frontier model); two seats of one version are a mirror and count seat 1. A version's training hour
is when its adapter was published, counted from the run's first update (the first health record; with none, the
first published adapter, and with neither the first finished game).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentenv_zeroad import spend as spend_module
from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train import stats

RESULT = {"won": 1.0, "drawn": 0.5, "lost": 0.0}
ARMY_MINUTES = (5, 10, 15)
HEALTH_KEYS = (
    "update", "version", "trained", "grad_norm", "kl_after_mb", "kl_to_base", "entropy", "clipped_share",
    "tis_truncation_share", "hold_share", "tokens_p50", "calls_per_decision", "exact_share", "distinct_fraction",
    "rejected_share", "invented_tool_share", "adv_mean", "adv_sd", "lr", "seconds", "stale_dropped",
)  # fmt: skip


@dataclass
class Played:
    """One finished game from its policy seat's side."""

    game_id: str
    version: int
    opponent: str
    result: float
    margin: float | None
    ended_by: str
    finished_at: float
    spend_usd: float
    metrics: dict[str, Any] = field(default_factory=dict)
    tools: Counter[str] = field(default_factory=Counter)
    turns: int = 0
    holds: int = 0
    orders: int = 0
    rejected: int = 0


def build(
    games: Path,
    learner: Path | None = None,
    *,
    learners: Sequence[str] | None = None,
    milestones: Sequence[stats.Milestone] = stats.DEFAULT_MILESTONES,
    panels: Path | None = None,
    panel: tuple[str, str] | None = None,
    ledger: Path | None = None,
    bootstrap: int = 200,
    now: float | None = None,
) -> dict[str, Any]:
    """The report for a run's folders, as of `now` (default: the clock)."""
    now = time.time() if now is None else now
    health = read_health(learner)
    published = published_at(learner)
    played, unfinished = read_games(games, learners)
    if health:
        start = float(health[0]["time"]) - float(health[0].get("seconds") or 0)
    else:
        start = min([*published.values(), *(p.finished_at for p in played)], default=now)
    hours_of = hours_function(published, start)
    rows = [stats.Game(p.version, hours_of(p.version), p.opponent, p.result, p.margin) for p in played]
    horizon = max((now - start) / 3600, max((r.hours for r in rows), default=0.0))
    report: dict[str, Any] = {
        "schema": "v2",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run": {
            "hours": round(horizon, 2),
            "games": len(played),
            "unfinished": unfinished,
            "versions": max((p.version for p in played), default=0),
            "policy_games": sum(p.version > 0 for p in played),
        },
        "elo": stats.elo_by_hour(rows, bootstrap=bootstrap),
        "trend": stats.trend_test(rows),
        "milestone_rule": {
            "check_every_hours": stats.CHECK_EVERY_HOURS,
            "pool_hours": stats.POOL_HOURS,
            "alpha": stats.ALPHA,
        },
        "milestones": stats.milestones(rows, milestones, now_hours=horizon),
        "panel": _panel(panels, panel, learners, hours_of),
        "behaviour": _behaviour(played, hours_of),
        "health": _health(health, start),
        "throughput": _throughput(played, health, start, horizon),
        "spend": _spend(played, ledger),
    }
    report["milestones_passed"] = [
        {"name": m["name"], "at": m["passed_at"]} for m in report["milestones"] if m["passed_at"] is not None
    ]
    return report


def write(path: Path, report: Mapping[str, Any]) -> Path:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, separators=(",", ":")))
    tmp.replace(path)
    return path


# ---- reading the folders ----------------------------------------------------------------------------------------
def read_health(learner: Path | None) -> list[dict[str, Any]]:
    """The update records of health.jsonl (rollback events are kept out of the series)."""
    path = learner / "health.jsonl" if learner else None
    if path is None or not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in records if "update" in r]


def published_at(learner: Path | None) -> dict[int, float]:
    """When each adapter version was published (its manifest's time)."""
    manifests = sorted((learner / "adapters").glob("v*/manifest.json")) if learner else []
    out = {}
    for path in manifests:
        manifest = json.loads(path.read_text())
        out[int(manifest["version"])] = float(manifest["time"])
    return out


def hours_function(published: Mapping[int, float], start: float) -> Callable[[int], float]:
    """hours_of(version): the training hours into the run at which the version was published. v0 is hour 0, and a
    version with no manifest takes the hours of the newest earlier one."""
    hours = {v: max(0.0, (t - start) / 3600) for v, t in published.items()}

    def hours_of(version: int) -> float:
        earlier = [v for v in hours if v <= version]
        return 0.0 if version == 0 or not earlier else hours[max(earlier)]

    return hours_of


def read_games(games: Path, learners: Sequence[str] | None) -> tuple[list[Played], int]:
    played, unfinished = [], 0
    for path in sorted(games.glob("*/results.json")):
        results = json.loads(path.read_text())
        game = _played(results, path, learners)
        if game is None:
            unfinished += 1
        else:
            played.append(game)
    return played, unfinished


def _played(results: Mapping[str, Any], path: Path, learners: Sequence[str] | None) -> Played | None:
    if results.get("status") != "finished" or results.get("ended_by") not in ("conquest", *rs.CAP_ENDS):
        return None
    slot = rs.policy_slot(results, learners)
    opp = rs.other_slot(results, slot) if slot else None
    if slot is None or opp is None:
        return None
    version = rs.policy_version(results, slot)
    outcome = results["seats"][slot].get("outcome")
    if version is None or outcome not in RESULT:
        return None
    game = Played(
        str((results.get("spec") or {}).get("game_id") or path.parent.name),
        version,
        rs.player_id(results, opp),
        RESULT[outcome],
        rs.final_margin(results, slot, opp)[0],
        str(results["ended_by"]),
        path.stat().st_mtime,
        float(results.get("spend_usd_est") or 0.0),
    )
    _tally(game, results, slot)
    return game


# ---- behaviour --------------------------------------------------------------------------------------------------
def _tally(game: Played, results: Mapping[str, Any], slot: str) -> None:
    seat = results["seats"][slot]
    m: dict[str, Any] = {
        "conquest": float(game.ended_by == "conquest"),
        "conquest_win": float(game.ended_by == "conquest" and game.result == 1.0),
    }
    for minute in ARMY_MINUTES:
        row = rs.timeline_at(results, minute * 60_000)
        reached = bool(results.get("timeline")) and results["timeline"][-1]["t_ms"] >= minute * 60_000
        mine = ((row or {}).get("players") or {}).get(slot) or {}
        m[f"army_{minute}"] = float(mine["army"]) if reached and mine.get("army") is not None else None
        if minute == 15:
            workers = mine.get("villagers", mine.get("gatherers"))
            m["villagers_15"] = float(workers) if reached and workers is not None else None
    attacks = _attack_orders(seat.get("turns") or [])
    m["first_attack_s"] = attacks[0] / 1000 if attacks else None
    m["attack_orders"] = float(len(attacks))
    combat = seat.get("combat") or {}
    m["kills"] = _number(combat.get("enemy_units_killed"))
    m["lost"] = _number(combat.get("units_lost"))
    m["kills_home"], m["kills_away"] = _number(combat.get("kills_home")), _number(combat.get("kills_away"))
    m["field_failures"] = float(sum(rs.is_field_placement_failure(f) for f in seat.get("failures") or []))
    game.metrics = m
    for turn in seat.get("turns") or []:
        game.turns += 1
        game.holds += rs.is_hold(turn)
        for order in turn.get("orders") or []:
            game.orders += 1
            game.rejected += order.get("status") == "rejected"
            game.tools[str(order.get("tool"))] += 1


def _attack_orders(turns: Sequence[Mapping[str, Any]]) -> list[float]:
    """When the policy gave a new attack or raid order: an accepted army order that is not a re-issue of the previous
    army order (any other army order, such as gather, makes the same attack count again)."""
    given, last = [], None
    for turn in turns:
        for order in turn.get("orders") or []:
            if order.get("tool") != "army" or order.get("status") == "rejected":
                continue
            args = order.get("arguments") or {}
            if args.get("action") not in ("attack", "raid"):
                last = None
                continue
            key = (args["action"], json.dumps(args.get("target"), sort_keys=True), args.get("min_size"))
            if key != last:
                given.append(float(turn["began_ms"]))
            last = key
    return given


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _summary(group: Sequence[Played]) -> dict[str, Any]:
    keys = sorted({k for g in group for k in g.metrics})
    out: dict[str, Any] = {"games": len(group)}
    for key in keys:
        out[key] = _mean([g.metrics.get(key) for g in group])
    turns, orders = sum(g.turns for g in group), sum(g.orders for g in group)
    tools = sum((g.tools for g in group), Counter())
    out["hold_share"] = sum(g.holds for g in group) / turns if turns else None
    out["rejected_share"] = sum(g.rejected for g in group) / orders if orders else None
    out["tool_mix"] = {t: n / orders for t, n in sorted(tools.items())} if orders else {}
    out["kill_loss"] = out["kills"] / out["lost"] if out.get("kills") is not None and out.get("lost") else None
    return out


def _behaviour(played: Sequence[Played], hours_of: Callable[[int], float]) -> dict[str, Any]:
    by_hour: dict[int, list[Played]] = defaultdict(list)
    for p in played:
        by_hour[int(hours_of(p.version))].append(p)
    return {
        "overall": _summary(played),
        "by_hour": [{"hour": h, **_summary(g)} for h, g in sorted(by_hour.items())],
    }


# ---- the rest ---------------------------------------------------------------------------------------------------
def _panel(
    panels: Path | None,
    panel: tuple[str, str] | None,
    learners: Sequence[str] | None,
    hours_of: Callable[[int], float],
) -> dict[str, Any] | None:
    """The panel test between two panel folders; each folder's games are the policy version they were pinned to."""
    if panel is None or panels is None:
        return None
    arms, versions = [], []
    for name in panel:
        played, _ = read_games(panels / name, learners)
        arms.append([stats.Game(p.version, hours_of(p.version), p.opponent, p.result, p.margin) for p in played])
        versions.append(Counter(p.version for p in played).most_common(1)[0][0] if played else None)
    return {
        "base": versions[0],
        "other": versions[1],
        "base_name": panel[0],
        "other_name": panel[1],
        **stats.panel_test(*arms),
    }


def _health(health: Sequence[Mapping[str, Any]], start: float) -> dict[str, Any]:
    series = []
    for r in health:
        row = {k: r.get(k) for k in HEALTH_KEYS}
        tripwires = r.get("tripwires") or {}
        row["hours"] = round((float(r["time"]) - start) / 3600, 3)
        row["trips"] = [t["name"] for t in tripwires.get("trips", [])]
        row["alarms"] = [t["name"] for t in tripwires.get("alarms", [])]
        series.append(row)
    latest = health[-1].get("tripwires") if health else None
    return {"updates": series, "latest": latest, "tripped": sum(bool(r["trips"]) for r in series)}


def _throughput(
    played: Sequence[Played], health: Sequence[Mapping[str, Any]], start: float, horizon: float
) -> dict[str, Any]:
    games: Counter[int] = Counter(int((p.finished_at - start) // 3600) for p in played)
    trained: Counter[int] = Counter()
    for r in health:
        trained[int((float(r["time"]) - start) // 3600)] += int(r.get("trained") or 0)
    seconds = [float(r["seconds"]) for r in health if r.get("seconds") is not None]
    hours = sorted(set(games) | set(trained))
    return {
        "by_hour": [{"hour": h, "games": games[h], "decisions_trained": trained[h]} for h in hours],
        "games_per_hour": len(played) / horizon if horizon > 0 else None,
        "decisions_trained": sum(trained.values()),
        "updates": len(health),
        "seconds_per_update": statistics.fmean(seconds) if seconds else None,
    }


def _spend(played: Sequence[Played], ledger: Path | None) -> dict[str, Any]:
    by_opponent: Counter[str] = Counter()
    for p in played:
        by_opponent[p.opponent] += p.spend_usd
    return {
        "games_usd": round(sum(by_opponent.values()), 4),
        "by_opponent": {k: round(v, 4) for k, v in sorted(by_opponent.items()) if v},
        "ledger_usd": round(spend_module.ledger_total(ledger), 4) if ledger else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", type=Path, required=True)
    p.add_argument("--learner", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--panels", type=Path, help="the orchestrator's ROOT/panels")
    p.add_argument("--panel", help="BASE:OTHER, two panel folder names under --panels")
    p.add_argument("--learner-slot", help="comma-separated learner slots, when the specs do not mark them")
    p.add_argument("--bootstrap", type=int, default=200)
    a = p.parse_args(argv)
    panel = None
    if a.panel:
        base, _, other = a.panel.partition(":")
        if not (base and other and a.panels):
            raise SystemExit("--panel BASE:OTHER needs two panel names and --panels")
        panel = (base, other)
    learners = [s for s in a.learner_slot.split(",") if s] if a.learner_slot else None
    report = build(
        a.games, a.learner, learners=learners, panels=a.panels, panel=panel, ledger=a.ledger, bootstrap=a.bootstrap
    )
    write(a.out, report)


if __name__ == "__main__":
    main()
