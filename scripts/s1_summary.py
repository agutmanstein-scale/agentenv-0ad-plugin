"""Print the S1 metrics of a folder of finished games, per seat kind and interface (docs/post-training-v2.md, stage S1).

    uv run python scripts/s1_summary.py ROOT [ROOT ...] [--calibration FILE] [--out DIR]

Reads every `*results.json` under each ROOT and prints one table with a column per `<seat kind>/<interface>` of the seat
under test (replay, scripted bot, or the policy): field placement failures per game, villagers at 15 minutes, idle
production buildings, how soon the town phase comes, the median army at 10 and 15 minutes, and the win rate by
opponent. Run the same replay specs under v1 and v2 and read the two columns side by side. The pilot's own results work
too, as the `agent/v1` column. A game that did not finish is left out.
With `--calibration` (the batch's calibration.json) or `--out`, the report also has the S1 gates as PASS or FAIL
(`agentenv_zeroad.s1_gates`: field failures down 80% and villagers at 15 minutes up 30% for the bot under v2, a median
army of at least 8 at 10 minutes for Qwen's replayed orders under v2, the bot's win rate not lower in v2) and the
calibration numbers. `--out DIR` writes the report to `DIR/s1-summary.txt` and `DIR/s1-summary.json` as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentenv_zeroad.s1 import format_summary, load_metrics
from agentenv_zeroad.s1_gates import summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="+", help="folders holding results.json files")
    parser.add_argument("--calibration", type=Path, help="calibration.json of the batch's E1 CPU calibration")
    parser.add_argument("--out", type=Path, help="folder for s1-summary.txt and s1-summary.json (adds the gates)")
    args = parser.parse_args(argv)
    metrics, excluded = load_metrics([Path(r) for r in args.roots])
    if not metrics:
        print("no results.json files found")
        return 1
    if args.calibration is None and args.out is None:
        print(format_summary(metrics))
        return 0
    calibration = json.loads(args.calibration.read_text()) if args.calibration and args.calibration.is_file() else None
    text, report = summary(metrics, calibration, excluded)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "s1-summary.txt").write_text(text + "\n")
        (args.out / "s1-summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
