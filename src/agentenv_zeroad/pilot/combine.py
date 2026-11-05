"""The dashboard's data for several pilot runs (one per box): each run's report.json, and one rating fit over every
run's games, so the runs share a scale. A run's policy version is its own player (`a:v3`); Petra's levels and the
frontier models are the same players in every run, which is what ties the runs together. A snapshot opponent is the
same run's earlier version. Petra medium is 1000.

  python -m agentenv_zeroad.pilot.combine --run a=live/fetched/pilot-a/report.json --run b=... --out dashboard.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentenv_zeroad.pilot import ratings

VERSION = re.compile(r"^v\d+$")


def combine(runs: Mapping[str, Mapping[str, Any]], *, shared: bool = False) -> dict[str, Any]:
    """With `shared` the runs are one pooled policy: version n is the same player in every run, one line."""
    games = []
    for run_id, rep in runs.items():
        owner = "pool" if shared else run_id
        for _, version, opponent, result in rep.get("results") or []:
            other = f"{owner}:{opponent}" if VERSION.match(opponent) else opponent
            games.append((f"{owner}:v{version}", other, float(result)))
    board = ratings.fit(games)
    lines = []
    groups = {"pool": [r for rep in runs.values() for r in rep.get("results") or []]} if shared else {
        run_id: rep.get("results") or [] for run_id, rep in runs.items()}  # fmt: skip
    for run_id, results in groups.items():
        versions = sorted({int(v) for _, v, _, _ in results})
        points = [[v, board[f"{run_id}:v{v}"]] for v in versions if f"{run_id}:v{v}" in board]
        lines.append({"run": run_id, "points": points})
    shared = {p: r for p, r in board.items() if ":" not in p}
    return {"runs": [{"id": k, **v} for k, v in runs.items()], "pooled": {"lines": lines, "board": shared}}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True, help="ID=path/to/report.json")
    p.add_argument("--out", required=True)
    p.add_argument("--example", action="store_true", help="mark the data as example data")
    p.add_argument("--shared-versions", action="store_true", help="the runs pooled one policy (pilot --pool)")
    a = p.parse_args(argv)
    runs = {}
    for item in a.run:
        run_id, _, path = item.partition("=")
        runs[run_id] = json.loads(Path(path).read_text())
    out = combine(runs, shared=a.shared_versions)
    out["example"] = a.example or any(r.get("example") for r in runs.values())
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
