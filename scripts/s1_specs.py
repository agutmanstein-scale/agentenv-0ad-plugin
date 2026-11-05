"""Pick pilot games and write the replay specs the S1 tool check plays back (docs/post-training-v2.md, stage S1).

    uv run python scripts/s1_specs.py ROOT --out DIR [--n 12] [--version 0] [--min-turns 15] [--delay-ms 0] [--seed 0]

Reads the pilot's results.json files under ROOT and picks N games of policy `--version` (the base model is 0; `any`
takes every version) whose policy seat logged at least `--min-turns` turns, spread over the opponents. For each it
writes `DIR/<game_id>.json`, a replay spec (`agentenv_zeroad.replay.ReplaySpec`): the seat's logged orders with their
game times, and the settings of the game they came from. An `index.json` beside them lists what was picked.

To run S1, play each spec as a `replay` seat (lobby slot `controller: "replay"`, `replay_spec: <path>`) once per
interface (`interface: "v1"` and `"v2"`) in the game it came from, then compare the results with `s1_summary.py`.
"""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args(argv)
    version = None if args.version == "any" else int(args.version)
    games = pick_games(
        list(load_games(Path(args.root))), args.n, version=version, min_turns=args.min_turns, seed=args.seed
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for game in games:
        spec = replay_spec(game, delay_ms=args.delay_ms)
        (out / f"{game.game_id}.json").write_text(json.dumps(spec.to_json(), indent=1))
        index.append(
            {"game_id": game.game_id, "opponent": game.opponent, "slot": spec.slot, "orders": len(spec.orders)}
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} replay specs to {out}")
    for row in index:
        print(f"  {row['game_id']:<20} vs {row['opponent']:<10} seat {row['slot']} {row['orders']} orders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
