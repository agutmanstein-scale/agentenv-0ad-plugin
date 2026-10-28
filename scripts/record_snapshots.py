"""Record real snapshot-query results from the engine into tests/fixtures/ for the unit tests.

Runs a 2-player Petra game (Rome vs Persia, Mainland 192, seed 3) in the zeroad-engine:0.28 image and saves the raw
`/evaluate` reply of `js/snapshot.js` at the chosen game times. The replies are already compact (a few KB); this
script only rounds nothing and drops nothing, so the fixtures are exactly what the engine said.

Usage: uv run python scripts/record_snapshots.py [--out tests/fixtures]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentenv_zeroad.engine import EngineOptions, free_port, launch_engine
from agentenv_zeroad.scenario import Seat, build_scenario
from agentenv_zeroad.state import snapshot_js

SEATS = [Seat("Rome", "rome", controller="petra"), Seat("Persia", "pers", controller="petra")]
RECORD_AT_S = (10, 60, 120)


async def record(out: Path) -> None:
    code = snapshot_js([1, 2])
    async with launch_engine(EngineOptions(port=free_port())) as engine:
        await engine.reset(build_scenario(SEATS, size=192, seed=3))
        steps = 0
        for seconds in RECORD_AT_S:
            while steps < seconds * 5:
                await engine.step()
                steps += 1
            raw = await engine.evaluate(code)
            path = out / f"snapshot_2p_{seconds:03d}s.json"
            path.write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")
            print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "tests" / "fixtures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    asyncio.run(record(args.out))


if __name__ == "__main__":
    main()
