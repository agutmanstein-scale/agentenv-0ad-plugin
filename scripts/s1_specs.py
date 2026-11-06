"""Pick pilot games and write the replay specs the S1 tool check plays back (docs/post-training-v2.md, stage S1).

    uv run python scripts/s1_specs.py ROOT --out DIR [--n 12] [--version 0] [--min-turns 15] [--delay-ms 0] [--seed 0]
                                      [--petra-only]

Reads the pilot's results.json files under ROOT and picks N games of policy `--version` (the base model is 0; `any`
takes every version) whose policy seat logged at least `--min-turns` turns, spread over the opponents (`--petra-only`:
only games against Petra at a known level, the ones a replay can set up again with no model seat). For each it
writes `DIR/<id>.json`, a replay spec (`agentenv_zeroad.replay.ReplaySpec`): the seat's logged orders with their
game times, and the settings of the game they came from. The id is `<box>-<run>-<game id>` (the pilot reuses game ids
across boxes and runs), and it is the spec's `source`. An `index.json` beside them lists what was picked. A spec that
already exists, or an id picked twice, stops the script before it writes anything.

To run S1, play each spec as a `replay` seat (lobby slot `controller: "replay"`, `replay_spec: <path>`) once per
interface (`interface: "v1"` and `"v2"`) in the game it came from, then compare the results with `s1_summary.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentenv_zeroad.s1 import load_games, pick_games, replay_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="folder holding the pilot's results.json files")
    parser.add_argument("--out", required=True, help="folder to write the specs to")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--version", default="0", help="policy version to take, or `any`")
    parser.add_argument("--min-turns", type=int, default=15)
    parser.add_argument("--delay-ms", type=int, default=0, help="shift every order later (lockstep adds 5000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--petra-only", action="store_true", help="only games against Petra at a known level")
    args = parser.parse_args(argv)
    version = None if args.version == "any" else int(args.version)
    games = pick_games(
        list(load_games(Path(args.root))),
        args.n,
        version=version,
        min_turns=args.min_turns,
        seed=args.seed,
        petra_only=args.petra_only,
    )
    out = Path(args.out)
    ids = [g.unique_id for g in games]
    clashes = sorted({i for i in ids if ids.count(i) > 1 or (out / f"{i}.json").exists()})
    if clashes:
        print(f"refusing to overwrite or repeat {', '.join(clashes)} in {out}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for game in games:
        spec = replay_spec(game, delay_ms=args.delay_ms)
        (out / f"{game.unique_id}.json").write_text(json.dumps(spec.to_json(), indent=1))
        index.append(
            {
                "id": game.unique_id,
                "game_id": game.game_id,
                "opponent": game.opponent,
                "slot": spec.slot,
                "orders": len(spec.orders),
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} replay specs to {out}")
    for row in index:
        print(f"  {row['id']:<40} vs {row['opponent']:<10} seat {row['slot']} {row['orders']} orders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
