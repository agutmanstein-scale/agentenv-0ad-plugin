"""The S1 tool check's games (docs/post-training-v2.md, stage S1), as the game specs the lockstep runner plays.

    python -m agentenv_zeroad.s1_batch package ROOT SPECS_DIR --out DIR     # laptop: the inputs to upload
    python -m agentenv_zeroad.s1_batch build --inputs DIR --out DIR [--n 100]   # box: bot.jsonl and replay.jsonl

No game has a model seat: they are played by `python -m agentenv_zeroad.train.runner`, with the engines from its pool.
- Bot games: the scripted bot against Petra at levels 1, 2 and 3, the bot's seat on interface v1 and on v2 (Petra
  seats are always v1), `n` games per arm and level on a 30-minute mainland map. Game `i` has the same seed, AI seed
  and civs (drawn from the executor's `roles.CIVS`) under both interfaces and at every level, and the bot takes the
  first start on even `i` and the second on odd, so the two arms differ only in the interface. The specs are
  ordered by `i`, then level, then arm: a batch cut short by its wall-clock limit still has balanced cells.
- Replay games: the pilot games `scripts/s1_specs.py` picked, each replayed under v1 and under v2 in the setup it
  came from (map, seed, both civs, Petra at the level it played, the replay seat on the start the policy had).
  The orders go in the spec file with the game-time lag lockstep play adds (`--delay-ms 5000` when it was written).
  `package` copies the specs and only the pilot results.json files they name into one folder and a tarball: the
  originals are the `agent/v1` column of the summary, and the opponent labels the pilot did not record go in a
  small `specs.json` beside each, where `s1.Game` looks for them.
- Calibration games (`calibration_games`): the E1 CPU calibration's, played by `s1_calibrate`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentenv_zeroad.interface import Interface
from agentenv_zeroad.replay import ReplaySpec
from agentenv_zeroad.roles import CIVS
from agentenv_zeroad.s1 import PETRA_LABEL, load_games
from agentenv_zeroad.train.spec import GameSpec, SeatSpec, load_specs

BOT_LEVELS = (1, 2, 3)
ARMS: tuple[Interface, ...] = ("v1", "v2")
BOT_SEED_BASE = 41_000  # game i of the bot batch has seed BOT_SEED_BASE + i; the pilot's seeds are 9000s
CALIBRATION_SEED_BASE = 47_000
TIME_CAP_MINUTES = 30.0
CALIBRATION_LEVEL = 3  # the Petra level the calibration games play
SPECS_DIR, PILOT_DIR = "specs", "pilot"  # inside the inputs folder
INPUTS_TARBALL = "s1-inputs.tgz"


def _civs(key: str) -> tuple[str, str]:
    """Two civs of the executor's, drawn from `key` alone (a string seed is stable across runs and machines)."""
    rng = random.Random(key)
    return rng.choice(CIVS), rng.choice(CIVS)


def _bot_vs_petra(game_id: str, *, i: int, level: int, interface: Interface, seed: int, civ_key: str) -> GameSpec:
    bot_civ, petra_civ = _civs(civ_key)
    bot = SeatSpec(kind="bot", civ=bot_civ, interface=interface)
    petra = SeatSpec(kind="petra", civ=petra_civ, difficulty=level, interface="v1")
    return GameSpec(
        game_id=game_id,
        seats=(bot, petra) if i % 2 == 0 else (petra, bot),
        map="mainland",
        seed=seed,
        ai_seed=seed,
        time_cap_minutes=TIME_CAP_MINUTES,
    )


def bot_games(n: int, levels: Sequence[int] = BOT_LEVELS, *, seed_base: int = BOT_SEED_BASE) -> list[GameSpec]:
    """The bot batch: `n` games per interface and Petra level, ordered by game, then level, then interface."""
    return [
        _bot_vs_petra(
            f"bot-{arm}-petra{level}-{i:03d}",
            i=i,
            level=level,
            interface=arm,
            seed=seed_base + i,
            civ_key=f"s1-bot-{level}-{i}",
        )
        for i in range(n)
        for level in levels
        for arm in ARMS
    ]


def calibration_games(count: int, *, seed_base: int = CALIBRATION_SEED_BASE) -> list[GameSpec]:
    """`count` games of the v2 bot against Petra at `CALIBRATION_LEVEL`, one per engine of a calibration level."""
    return [
        _bot_vs_petra(
            f"calib-{count:02d}-{i:02d}",
            i=i,
            level=CALIBRATION_LEVEL,
            interface="v2",
            seed=seed_base + i,
            civ_key=f"s1-calibration-{i}",
        )
        for i in range(count)
    ]


def spec_files(specs_dir: Path) -> list[Path]:
    return sorted(p for p in specs_dir.glob("*.json") if p.name != "index.json")


def _resolve_sources(specs: dict[Path, ReplaySpec], pilot_root: Path) -> None:
    known = {g.unique_id for g in load_games(pilot_root)}
    missing = sorted(s.source for s in specs.values() if s.source not in known)
    if missing:
        raise ValueError(f"no pilot results.json under {pilot_root} for the replay sources: {', '.join(missing)}")


def replay_games(specs_dir: Path, pilot_root: Path) -> list[GameSpec]:
    """Each replay spec in `specs_dir` as two games, v1 then v2, in the setup its source game had.

    Every spec's source must have its results.json under `pilot_root`, its opponent must be Petra at a known level,
    and its seat must be one of the source's two starts; anything else is an error, not a skipped game."""
    specs = {path: ReplaySpec.load(path) for path in spec_files(specs_dir)}
    _resolve_sources(specs, pilot_root)
    games: list[GameSpec] = []
    for path, spec in specs.items():
        setup = spec.game
        opponent = str(setup.get("opponent", ""))
        if not PETRA_LABEL.fullmatch(opponent):
            raise ValueError(f"{spec.source}: the opponent is {opponent!r}, not Petra at a known level")
        level = int(opponent.removeprefix("petra-"))
        seats = setup.get("seats") or {}
        if spec.slot not in seats or set(seats) != {"1", "2"}:
            raise ValueError(
                f"{spec.source}: replay seat {spec.slot!r} is not one of the game's starts {sorted(seats)}"
            )
        civs = {slot: seats[slot].get("civ") for slot in seats}
        if not all(civ in CIVS for civ in civs.values()):
            raise ValueError(f"{spec.source}: the game's civs are {civs}, not all among the executor's")
        game_map = setup.get("map") or {}
        for arm in ARMS:
            replay_seat = SeatSpec(kind="replay", civ=civs[spec.slot], replay=str(path.resolve()), interface=arm)
            seat_for = {
                slot: replay_seat
                if slot == spec.slot
                else SeatSpec(kind="petra", civ=civs[slot], difficulty=level, interface="v1")
                for slot in ("1", "2")
            }
            games.append(
                GameSpec(
                    game_id=f"replay-{arm}-{spec.source}",
                    seats=(seat_for["1"], seat_for["2"]),
                    map=game_map.get("name") or "mainland",
                    seed=int(game_map["seed"]),
                    ai_seed=int(game_map["seed"]),
                    size=int(game_map.get("size") or 128),
                    time_cap_minutes=float(setup.get("time_cap_s") or TIME_CAP_MINUTES * 60) / 60,
                )
            )
    return games


def write_specs(specs: Sequence[GameSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(spec.to_json()) + "\n" for spec in specs))
    load_specs(path)  # what the runner will read must load: valid seats, unique game ids


def package_inputs(root: Path, specs_dir: Path, out: Path) -> Path:
    """Copy the replay specs and the pilot results.json each names (only those) into `out/inputs`, and tar it as
    `out/s1-inputs.tgz`. Returns the tarball's path."""
    index = json.loads((specs_dir / "index.json").read_text())
    games = {g.unique_id: g for g in load_games(root)}
    missing = sorted(row["id"] for row in index if row["id"] not in games)
    if missing:
        raise ValueError(f"no pilot results.json under {root} for: {', '.join(missing)}")
    inputs = out / "inputs"
    shutil.rmtree(inputs, ignore_errors=True)
    (inputs / SPECS_DIR).mkdir(parents=True)
    for path in spec_files(specs_dir):
        shutil.copy2(path, inputs / SPECS_DIR / path.name)
    for row in index:
        game = games[row["id"]]
        copy = inputs / PILOT_DIR / game.path.relative_to(root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(game.path, copy)
        _label_opponent(copy.parent / "specs.json", game.game_id, row["opponent"])
    tarball = out / INPUTS_TARBALL
    with tarfile.open(tarball, "w:gz") as tar:
        for child in sorted(inputs.iterdir()):
            tar.add(child, arcname=child.name)
    return tarball


def _label_opponent(path: Path, game_id: str, opponent: str) -> None:
    """The opponent's label for `s1.Game` (`specs.json`, as the pilot's rounds write it), for a result the pilot
    did not record Petra's level in."""
    level = re.fullmatch(r"petra-(\d)", opponent)
    if level is None:
        return
    rows: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    rows.append({"game_id": game_id, "opponent": {"kind": "petra", "level": int(level.group(1))}})
    path.write_text(json.dumps(rows))


def build(inputs: Path, out: Path, n: int, levels: Sequence[int] = BOT_LEVELS) -> dict[str, int]:
    """Write `bot.jsonl` and `replay.jsonl` into `out` from the unpacked inputs; returns the game counts."""
    bot = bot_games(n, levels)
    replay = replay_games(inputs / SPECS_DIR, inputs / PILOT_DIR)
    write_specs(bot, out / "bot.jsonl")
    write_specs(replay, out / "replay.jsonl")
    return {"bot": len(bot), "replay": len(replay), "replay_specs": len(replay) // len(ARMS)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_zeroad.s1_batch", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package", help="the replay specs and the pilot results they name, as a tarball")
    package.add_argument("root", type=Path, help="the pilot's results folder (live/fetched/pilot-final)")
    package.add_argument("specs_dir", type=Path, help="the folder scripts/s1_specs.py wrote")
    package.add_argument("--out", type=Path, required=True)
    make = sub.add_parser("build", help="bot.jsonl and replay.jsonl for the runner")
    make.add_argument("--inputs", type=Path, required=True, help="the unpacked inputs folder")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--n", type=int, default=100, help="bot games per interface and Petra level")
    make.add_argument("--levels", default=",".join(map(str, BOT_LEVELS)))
    args = parser.parse_args(argv)
    if args.command == "package":
        tarball = package_inputs(args.root, args.specs_dir, args.out)
        print(f"{tarball} ({tarball.stat().st_size / 1e6:.1f} MB)")
        return 0
    if args.n < 1:
        parser.error("--n must be at least 1")
    counts = build(args.inputs, args.out, args.n, [int(x) for x in args.levels.split(",")])
    print(
        f"bot: {counts['bot']} games ({args.n} per interface and level); "
        f"replay: {counts['replay']} games from {counts['replay_specs']} specs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
