"""report.json: every series the progress dashboard draws, from the pilot's folder. The driver rewrites it after each
round and each training step, so fetching that one small file is enough to redraw the dashboard.

- `ratings`: after each round, a fit over every game up to it: the policy version that played the round, each Petra
  level and the frontier model (Petra medium = 1000), so the policy's line climbs past fixed reference lines.
- `probes`: each round's probe games by opponent: outcome and score margin.
- `training`: each round's training games by opponent (the ladder's mix), with wins and the mean margin.
- `health`: per round, the share of the policy's orders rejected, of its calls missed, its median call time, clock
  overruns and finished games.
- `trainer`: per policy version, what its training step reported (decisions, mean KL, mean loss, seconds).
- `games`: the score over game time of the first and the latest round's probe against Petra medium.
- `results`: every finished game as `[round, policy version, opponent, result]`, for a fit across runs (`combine`).
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentenv_zeroad.pilot import ratings, scoring

FEATURED = "petra-3"


def build(home: Path, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    records = _read_jsonl(home / "games.jsonl")
    rounds = sorted({r["round"] for r in records})
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": dict(state or {}),
        "rounds": rounds,
        "ratings": _ratings(records, rounds),
        "probes": [_row(r) for r in records if r["kind"] == "probe"],
        "training": _training(records),
        "health": [_health(r, [g for g in records if g["round"] == r]) for r in rounds],
        "trainer": _trainer(home),
        "games": _featured(home, records, rounds),
        "results": [
            [r["round"], r["policy_version"], r["opponent"], scoring.RESULT[r["outcome"]]]
            for r in records
            if r.get("outcome") in scoring.RESULT
        ],
    }


def write(home: Path, state: Mapping[str, Any] | None = None) -> Path:
    path = home / "report.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(build(home, state), separators=(",", ":")))
    tmp.replace(path)
    return path


def _ratings(records: Sequence[dict[str, Any]], rounds: Sequence[int]) -> list[dict[str, Any]]:
    out = []
    for r in rounds:
        upto = [g for g in records if g["round"] <= r and g.get("outcome") in scoring.RESULT]
        board = ratings.fit([(f"v{g['policy_version']}", g["opponent"], scoring.RESULT[g["outcome"]]) for g in upto])
        version = next(g["policy_version"] for g in records if g["round"] == r)
        out.append({"round": r, "version": version, "policy": board.get(f"v{version}"), "board": board})
    return out


def _row(r: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("round", "policy_version", "opponent", "outcome", "margin", "policy_score", "opponent_score", "status")
    return {k: r.get(k) for k in keep}


def _training(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["kind"] == "train":
            groups[(r["round"], r["opponent"])].append(r)
    out = []
    for (round_, opp), games in sorted(groups.items()):
        margins = [g["margin"] for g in games if g.get("margin") is not None]
        out.append(
            {
                "round": round_,
                "opponent": opp,
                "games": len(games),
                "wins": sum(1 for g in games if g.get("outcome") == "won"),
                "draws": sum(1 for g in games if g.get("outcome") == "drawn"),
                "mean_margin": round(statistics.fmean(margins), 1) if margins else None,
            }
        )
    return out


def _health(round_: int, games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    orders = sum(g.get("orders") or 0 for g in games)
    calls = sum(g.get("calls") or 0 for g in games)
    latencies = [g["latency_p50_s"] for g in games if g.get("latency_p50_s") is not None]
    return {
        "round": round_,
        "rejected_share": round(sum(g.get("rejected") or 0 for g in games) / orders, 4) if orders else None,
        "missed_share": round(sum(g.get("missed") or 0 for g in games) / calls, 4) if calls else None,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "overruns": sum(g.get("overruns") or 0 for g in games),
        "finished": sum(1 for g in games if g.get("status") == "finished"),
        "games": len(games),
        "spend_usd": round(sum(g.get("spend_usd") or 0 for g in games), 4),
    }


def _trainer(home: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((home / "adapters").glob("v*/train.json"), key=lambda p: int(p.parent.name[1:])):
        out.append({"version": int(path.parent.name[1:]), **json.loads(path.read_text())})
    return out


def _featured(home: Path, records: Sequence[dict[str, Any]], rounds: Sequence[int]) -> list[dict[str, Any]]:
    """The probe against Petra medium in the first and the latest round: both players' totals over game time."""
    out = []
    for r in sorted({rounds[0], rounds[-1]}) if rounds else []:
        probes = [g for g in records if g["round"] == r and g["kind"] == "probe"]
        game = next((g for g in probes if g["opponent"] == FEATURED), None)
        if game is None:
            continue
        found = sorted((home / "rounds" / f"r{r:02d}" / "games" / game["game_id"]).glob("*/results.json"))
        if not found:
            continue
        results = json.loads(found[-1].read_text())
        task = json.loads((home / "rounds" / f"r{r:02d}" / "bundle" / "tasks" / f"{game['game_id']}.json").read_text())
        me = next(s["player_id"] for s in task if s["type"] == "add_player_slot" and s.get("player_name") == "policy")
        points = [
            {
                "t_ms": row["t_ms"],
                **{
                    ("policy" if slot == me else "opponent"): (p or {}).get("total")
                    for slot, p in row["players"].items()
                },
            }
            for row in results.get("timeline") or []
        ]
        out.append({"round": r, "version": game["policy_version"], "outcome": game.get("outcome"), "points": points})
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
