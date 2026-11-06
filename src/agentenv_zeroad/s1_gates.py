"""The S1 gates as numbers: PASS or FAIL for each, from the games the batch played (docs/post-training-v2.md, S1).

The gates, on the groups `s1.py` makes (`<seat kind>/<interface>` of the seat under test):
1. field placement failures per game, scripted bot v2 against v1: down at least 80%;
2. villagers at 15 minutes (median), bot v2 against v1: up at least 30%;
3. the median army at 10 minutes of the replayed Qwen orders under v2 (`replay/v2`): at least 8;
4. the bot's win rate in v2 not lower than in v1, for each Petra level where the v1 bot wins at least 10% of its
   games: Fisher's exact test, one-sided, at 5% (a draw is not a win).

A gate with no games, or whose v1 baseline is zero (a drop from nothing), is NO DATA, and the batch passes only when
every gate passes. The replays' own field failures and villagers are in the report beside the bot's but gate nothing:
the bot is the clean pair (same script, same seeds and civs under both interfaces). The calibration numbers
(`s1_calibrate.py`) are copied into the report as they are.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import comb
from typing import Any

from agentenv_zeroad.s1 import GameMetrics, format_summary

FIELD_DROP = 0.80
VILLAGER_GAIN = 0.30
ARMY_10_MIN = 8
WIN_RATE_FLOOR = 0.10
ALPHA = 0.05
PASS, FAIL, NO_DATA = "PASS", "FAIL", "NO DATA"


def fisher_lower(wins_a: int, n_a: int, wins_b: int, n_b: int) -> float:
    """One-sided Fisher exact p-value that arm B's win rate is lower than arm A's: the chance, with the total wins
    fixed, of B having this few wins or fewer."""
    total, n = wins_a + wins_b, n_a + n_b
    if n == 0:
        return 1.0
    low = max(0, total - n_a)
    tail = sum(comb(n_b, k) * comb(n_a, total - k) for k in range(low, wins_b + 1))
    return tail / comb(n, total)


def _by_group(metrics: Sequence[GameMetrics]) -> dict[str, list[GameMetrics]]:
    groups: dict[str, list[GameMetrics]] = defaultdict(list)
    for m in metrics:
        groups[m.group].append(m)
    return groups


def _mean(values: Sequence[float | int | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _median(values: Sequence[float | int | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.median(kept) if kept else None


def _gate(status: str, rule: str, detail: str, **numbers: Any) -> dict[str, Any]:
    return {"status": status, "rule": rule, "detail": detail, **numbers}


def _field_failures(v1: Sequence[GameMetrics], v2: Sequence[GameMetrics], who: str) -> dict[str, Any]:
    rule = f"field placement failures per game, {who} v2 against v1: down at least {FIELD_DROP:.0%}"
    a, b = _mean([m.field_failures for m in v1]), _mean([m.field_failures for m in v2])
    if a is None or b is None:
        return _gate(NO_DATA, rule, f"no {who} games under both interfaces", v1=a, v2=b, change=None)
    if a == 0:
        return _gate(NO_DATA, rule, f"v1 had no field failures to remove ({a:.2f} -> {b:.2f})", v1=a, v2=b, change=None)
    change = b / a - 1
    return _gate(
        PASS if change <= -FIELD_DROP else FAIL,
        rule,
        f"{a:.2f} -> {b:.2f} ({change:+.0%}), n={len(v1)}/{len(v2)}",
        v1=a,
        v2=b,
        change=change,
    )


def _villagers(v1: Sequence[GameMetrics], v2: Sequence[GameMetrics], who: str) -> dict[str, Any]:
    rule = f"villagers at 15 min (median), {who} v2 against v1: up at least {VILLAGER_GAIN:.0%}"
    a, b = _median([m.villagers_15 for m in v1]), _median([m.villagers_15 for m in v2])
    if not a or b is None:
        return _gate(NO_DATA, rule, f"no {who} games under both interfaces reached minute 15", v1=a, v2=b, change=None)
    change = b / a - 1
    return _gate(
        PASS if change >= VILLAGER_GAIN else FAIL,
        rule,
        f"{a:g} -> {b:g} ({change:+.0%}), n={len(v1)}/{len(v2)}",
        v1=a,
        v2=b,
        change=change,
    )


def _army(replay_v2: Sequence[GameMetrics]) -> dict[str, Any]:
    rule = f"median army at 10 min of the replays under v2: at least {ARMY_10_MIN}"
    value = _median([m.army_10 for m in replay_v2])
    if value is None:
        return _gate(NO_DATA, rule, "no replay/v2 game reached minute 10", value=None)
    return _gate(PASS if value >= ARMY_10_MIN else FAIL, rule, f"{value:g}, n={len(replay_v2)}", value=value)


def _wins(ms: Sequence[GameMetrics]) -> tuple[int, int]:
    decided = [m.points for m in ms if m.points is not None]
    return sum(1 for p in decided if p == 1.0), len(decided)


def _win_rates(v1: Sequence[GameMetrics], v2: Sequence[GameMetrics]) -> dict[str, Any]:
    rule = (
        f"bot win rate not lower in v2 (Fisher one-sided, p < {ALPHA:g} fails) at each Petra level "
        f"where the v1 bot wins at least {WIN_RATE_FLOOR:.0%}"
    )
    levels: dict[str, dict[str, Any]] = {}
    for opponent in sorted({m.opponent for m in (*v1, *v2)}):
        w1, n1 = _wins([m for m in v1 if m.opponent == opponent])
        w2, n2 = _wins([m for m in v2 if m.opponent == opponent])
        rate1 = w1 / n1 if n1 else None
        tested = bool(n1 and n2 and rate1 is not None and rate1 >= WIN_RATE_FLOOR)
        p = fisher_lower(w1, n1, w2, n2) if tested else None
        levels[opponent] = {"v1": [w1, n1], "v2": [w2, n2], "tested": tested, "p": p}
    tested = {k: v for k, v in levels.items() if v["tested"]}
    lines = [
        f"{k}: {v['v1'][0]}/{v['v1'][1]} -> {v['v2'][0]}/{v['v2'][1]}"
        + (f" (p={v['p']:.3f})" if v["tested"] else " (v1 under the floor: not tested)")
        for k, v in levels.items()
    ]
    if not levels:
        return _gate(NO_DATA, rule, "no bot games", levels=levels)
    failed = [k for k, v in tested.items() if v["p"] < ALPHA]
    detail = "; ".join(lines) + ("" if tested else "; no level to test")
    return _gate(FAIL if failed else PASS, rule, detail, levels=levels)


def evaluate(metrics: Sequence[GameMetrics]) -> dict[str, Any]:
    """Every gate's result, the replay-side figures that gate nothing, and the overall verdict."""
    groups = _by_group(metrics)
    bot1, bot2 = groups.get("bot/v1", []), groups.get("bot/v2", [])
    rep1, rep2 = groups.get("replay/v1", []), groups.get("replay/v2", [])
    gates = {
        "field_failures": _field_failures(bot1, bot2, "bot"),
        "villagers_15": _villagers(bot1, bot2, "bot"),
        "army_10_replay_v2": _army(rep2),
        "bot_win_rate": _win_rates(bot1, bot2),
    }
    statuses = {g["status"] for g in gates.values()}
    overall = FAIL if FAIL in statuses else PASS if statuses == {PASS} else NO_DATA
    return {
        "overall": overall,
        "gates": gates,
        "replay_figures": {
            "field_failures": _field_failures(rep1, rep2, "replay"),
            "villagers_15": _villagers(rep1, rep2, "replay"),
        },
        "games": {g: len(ms) for g, ms in sorted(groups.items())},
    }


def format_gates(report: Mapping[str, Any]) -> str:
    lines = ["S1 gates (docs/post-training-v2.md):"]
    for gate in report["gates"].values():
        lines.append(f"  {gate['status']:<8}{gate['rule']}")
        lines.append(f"          {gate['detail']}")
    lines.append("not gating, the replays' own figures:")
    for gate in report["replay_figures"].values():
        lines.append(f"  {gate['status']:<8}{gate['rule']}: {gate['detail']}")
    lines.append(f"overall: {report['overall']}")
    return "\n".join(lines)


def format_calibration(calibration: Mapping[str, Any]) -> str:
    """The calibration numbers as lines: per level of engines in flight, the cost of a game and the engines' memory."""
    host = calibration.get("host", {})
    lines = [f"E1 CPU calibration ({host.get('vcpus', '?')} vCPUs, {host.get('mem_total_mb', '?')} MB):"]
    for level, row in sorted(calibration.get("levels", {}).items(), key=lambda kv: int(kv[0])):
        rss = row.get("rss_mb") or {}
        lines.append(
            f"  {int(level):>2} engines: {row.get('games_finished', 0)}/{row.get('games', 0)} games finished, "
            f"wall {row.get('wall_s', 0):.0f} s, {_fmt(row.get('core_s_per_game'))} core-s per game, "
            f"{_fmt(row.get('mean_cores'))} cores busy, engine RSS median {_fmt(rss.get('median'))} MB "
            f"max {_fmt(rss.get('max'))} MB, all engines {_fmt(rss.get('sum_max'))} MB"
        )
        by_minute = row.get("step_ms_by_minute") or {}
        if by_minute:
            picks = [m for m in ("0", "5", "10", "15", "20", "25", "29") if m in by_minute]
            cells = ", ".join(f"min {m}: {by_minute[m]['median']:.0f}/{by_minute[m]['p95']:.0f}" for m in picks)
            lines.append(f"      step ms (median/p95) {cells}")
    if not calibration.get("levels"):
        lines.append("  (no calibration run)")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def summary(
    metrics: Sequence[GameMetrics], calibration: Mapping[str, Any] | None, excluded: int = 0
) -> tuple[str, dict]:
    """The text and the JSON of the whole S1 report."""
    report = evaluate(metrics)
    parts = [format_summary(metrics)]
    if excluded:
        parts.append(f"({excluded} games left out: they did not finish)")
    parts += ["", format_gates(report)]
    if calibration is not None:
        parts += ["", format_calibration(calibration)]
    report = {**report, "excluded_unfinished": excluded, "calibration": calibration}
    return "\n".join(parts), report
