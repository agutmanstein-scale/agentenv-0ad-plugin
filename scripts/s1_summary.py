"""Print the S1 metrics of a folder of finished games, per seat kind and interface (docs/post-training-v2.md, stage S1).

    uv run python scripts/s1_summary.py ROOT [ROOT ...]

Reads every `*results.json` under each ROOT and prints one table with a column per `<seat kind>/<interface>` of the seat
under test (replay, scripted bot, or the policy): field placement failures per game, villagers at 15 minutes, idle
production buildings, how soon the town phase comes, the median army at 10 and 15 minutes, and the win rate by
opponent. Run the same replay specs under v1 and v2 and read the two columns side by side. The pilot's own results work
too, as the `agent/v1` column. The gate: field failures down 80%, villagers at 15 minutes up 30%, less idle
production, and a median army of at least 8 at 10 minutes for Qwen's replayed orders under v2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentenv_zeroad.s1 import format_summary, game_metrics, load_games


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="+", help="folders holding results.json files")
    args = parser.parse_args(argv)
    metrics = [m for root in args.roots for game in load_games(Path(root)) if (m := game_metrics(game)) is not None]
    if not metrics:
        print("no results.json files found")
        return 1
    print(format_summary(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
