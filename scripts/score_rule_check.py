"""Check the net-worth score rule on the pilot's games (docs/post-training-v2.md, section 2).

    uv run python scripts/score_rule_check.py [ROOT] [--out PATH] [--minutes 10 15 20]

Reads every `*results.json` under ROOT (the pilot's fetched games, 809 of them) and, for the games that ended by
conquest, asks how well each measure's leader at minutes 10, 15 and 20 predicts the conquest winner: the engine's
official score total, economy plus military (the total without exploration), and a proxy net worth. It prints one table
and saves it to PATH (by default beside the design's research, outside git).

The pilot's timelines hold score parts, population, army, gatherers, the sum of the banked resources and the building
count, but no cost-valued net worth, so the proxy is built from those counts:

    proxy = banked + 50 * (pop - army) + 100 * army + 1500 + 150 * (buildings - 1)

- a villager costs 50 (food), a soldier 100 (an infantryman's 50 food and 50 wood; cavalry and champions cost more);
- the first building is the civic centre at 1,500 (500 wood, 500 stone, 500 metal); each other building is taken at
  150, between a house or storehouse (100-150 wood) and a barracks (300).

Its limits are listed in LIMITS and printed under the table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "live/fetched/pilot-final"
DEFAULT_OUT = "live/fetched/design-2026-11-05/score_rule_check.txt"
MINUTES = (10, 15, 20)
CLOSE_RATIO = 0.8  # "close": the trailing player's official total is at least this share of the leader's
VILLAGER_COST, SOLDIER_COST, CIVIC_CENTRE_COST, BUILDING_COST = 50, 100, 1500, 150

LIMITS = """Limits of the proxy and of this check:
- unit and building prices are averages: champions, siege, cavalry, temples, fortresses and towers are undervalued,
  and the building count includes unfinished foundations and fields (the real measure counts finished structures);
- researched techs and the phase advance cost resources that net worth never counts, in the proxy and the real rule;
- pop minus army counts heroes, traders and ships as villagers;
- only games that ended by conquest and were still running at the minute are used, so lopsided games dominate: the
  winner usually led on everything from minute 10, which caps what any measure can show;
- the official score's military part adds kills, so it already carries who is winning fights, while net worth is
  built to decide a cap, not to predict a conquest; a measure that predicts the winner well is not thereby the right
  thing to reward;
- the score parts and the proxy share inputs (gathering feeds both), so they are not independent evidence."""

Row = Mapping[str, Any]


def proxy_net_worth(row: Row) -> float:
    """The documented proxy (see the module docstring) for one player's timeline row."""
    pop, army = row.get("pop", 0), row.get("army", 0)
    buildings = row.get("buildings", 0)
    built = CIVIC_CENTRE_COST + BUILDING_COST * (buildings - 1) if buildings > 0 else 0
    return row.get("resources", 0) + VILLAGER_COST * max(pop - army, 0) + SOLDIER_COST * army + built


MEASURES: dict[str, Callable[[Row], float]] = {
    "official total": lambda r: r.get("total", 0.0),
    "economy + military": lambda r: r.get("economy", 0.0) + r.get("military", 0.0),
    "proxy net worth": proxy_net_worth,
}


def all_results(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(root.rglob("*results.json")):
        yield json.loads(path.read_text())


def winner_of(results: Mapping[str, Any]) -> str | None:
    won = [slot for slot, seat in results["seats"].items() if seat.get("outcome") == "won"]
    return won[0] if len(won) == 1 else None


def row_at(timeline: list[Mapping[str, Any]], minute: int) -> Mapping[str, Row] | None:
    """The players of the last timeline row at or before `minute`, None if the game ended before it."""
    at = minute * 60_000
    if timeline[-1]["t_ms"] <= at:
        return None
    before = [r for r in timeline if r["t_ms"] <= at]
    return before[-1]["players"] if before else None


def leader(players: Mapping[str, Row], measure: Callable[[Row], float]) -> str | None:
    """The slot with the highest value, None on a tie for the top."""
    values = {slot: measure(row) for slot, row in players.items()}
    top = max(values.values())
    leaders = [slot for slot, v in values.items() if v == top]
    return leaders[0] if len(leaders) == 1 else None


def is_close(players: Mapping[str, Row]) -> bool:
    totals = sorted(MEASURES["official total"](row) for row in players.values())
    return len(totals) >= 2 and totals[-1] > 0 and totals[-2] / totals[-1] >= CLOSE_RATIO


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def sign_test(b: int, c: int) -> float:
    """Two-sided exact binomial p for `b` wins against `c` losses among discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def evaluate(root: Path, minutes: tuple[int, ...]) -> dict[str, Any]:
    scanned = conquest = 0
    hits: dict[tuple[str, int], list[bool]] = {(m, k): [] for m in minutes for k in MEASURES}
    close: dict[tuple[str, int], list[bool]] = {(m, k): [] for m in minutes for k in MEASURES}
    paired: dict[int, list[tuple[bool, bool]]] = {m: [] for m in minutes}
    agent_won: dict[int, list[bool]] = {m: [] for m in minutes}
    for results in all_results(root):
        scanned += 1
        if results.get("ended_by") != "conquest" or not results.get("timeline"):
            continue
        conquest += 1
        winner = winner_of(results)
        if winner is None:
            continue
        for m in minutes:
            players = row_at(results["timeline"], m)
            if players is None or winner not in players:
                continue
            picks = {name: leader(players, fn) for name, fn in MEASURES.items()}
            if any(p is None for p in picks.values()):
                continue
            for name, pick in picks.items():
                hits[(m, name)].append(pick == winner)
                if is_close(players):
                    close[(m, name)].append(pick == winner)
            paired[m].append((picks["proxy net worth"] == winner, picks["official total"] == winner))
            agent_won[m].append(results["seats"][winner].get("kind") == "agent")
    return {
        "scanned": scanned,
        "conquest": conquest,
        "hits": hits,
        "close": close,
        "paired": paired,
        "agent_won": agent_won,
    }


def format_table(data: Mapping[str, Any], minutes: tuple[int, ...]) -> str:
    lines = [
        "Score rule check on the pilot's games: how well the leader at minute M predicts the conquest winner.",
        f"{data['scanned']} results.json files scanned; {data['conquest']} ended by conquest.",
        "Games count at minute M if they ended by conquest after M and no measure tied at the top.",
        "",
        "proxy net worth = banked + 50 * (pop - army) + 100 * army + 1500 + 150 * (buildings - 1)",
        "",
        f"{'measure':<22}" + "".join(f"{'min ' + str(m):>26}" for m in minutes),
    ]
    lines += _rows(data["hits"], minutes)
    lines += [
        "",
        f"close games only (the trailing player's official total is at least {CLOSE_RATIO:.0%} of the leader's):",
    ]
    lines += _rows(data["close"], minutes)
    lines += ["", "agent seat won the game (base rate)  " + "  ".join(_rate(data["agent_won"][m], m) for m in minutes)]
    lines += ["", "proxy net worth against official total, games where exactly one of them picked the winner:"]
    for m in minutes:
        pairs = data["paired"][m]
        b = sum(1 for proxy, official in pairs if proxy and not official)
        c = sum(1 for proxy, official in pairs if official and not proxy)
        lines.append(
            f"  minute {m}: proxy right only {b}, official right only {c}, sign test p = {sign_test(b, c):.3f}"
        )
    return "\n".join([*lines, "", LIMITS])


def _rows(hits: Mapping[tuple[int, str], list[bool]], minutes: tuple[int, ...]) -> list[str]:
    lines = []
    for name in MEASURES:
        cells = []
        for m in minutes:
            outcome = hits[(m, name)]
            lo, hi = wilson(sum(outcome), len(outcome))
            share = sum(outcome) / len(outcome) if outcome else float("nan")
            cells.append(f"{share:6.1%} [{lo:5.1%},{hi:5.1%}] n={len(outcome):<4}")
        lines.append(f"{name:<22}" + "".join(f"{c:>26}" for c in cells))
    return lines


def _rate(flags: list[bool], minute: int) -> str:
    return f"min {minute}: {sum(flags) / len(flags):.1%}" if flags else f"min {minute}: n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="folder holding the pilot's results.json files")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to save the table")
    parser.add_argument("--minutes", type=int, nargs="+", default=list(MINUTES))
    args = parser.parse_args(argv)
    minutes = tuple(args.minutes)
    text = format_table(evaluate(Path(args.root), minutes), minutes)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nsaved to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
