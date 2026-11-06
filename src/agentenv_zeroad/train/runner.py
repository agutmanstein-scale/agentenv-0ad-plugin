"""The training runner: play a list of game specs on an engine pool, `concurrency` games at a time.

    python -m agentenv_zeroad.train.runner --specs specs.jsonl --out DIR --engines N [--concurrency K] [--workers W]
        [--engines-per-worker 4] [--image zeroad-engine:0.28-rt] [--replays]

Each game's files land in `DIR/<game_id>/` (results.json; commands.txt with `--replays`), and one line per finished
game is appended to `DIR/games.jsonl`. A game that fails is recorded as failed and its engine is restarted before it
is leased again; the other games go on.

Workers are separate processes, each with its own pool of its share of the engines and up to `--concurrency` games in
flight on one asyncio loop, started with PYTHONHASHSEED=0 so that two runs of one spec make the same set-ordered
choices. There are `--workers` of them, or by default one per `--engines-per-worker` engines (4): a loop that
serialises the HTTP, JSON and bot work of a dozen engines is a queue the engines wait in. The parent only splits the
specs round-robin, gives each worker its own port range (`PORT_STRIDE` ports, all below Linux's ephemeral range, or
it refuses to start) and waits.

The engine image defaults to the patched training engine (`pool.TRAIN_IMAGE`, built by `scripts/build-engine.sh rt`):
its idle engines sleep instead of spinning a core and `step_raw` skips the state JSON. The pool refuses an image whose
`zeroad.patches` label lacks `pool.TRAIN_PATCHES`; `--allow-unpatched` runs on any image (the PPA's, to compare).

`run_games` and `run_game` are the async API for callers that already have a pool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from agentenv_zeroad.engine import PORT_STOP
from agentenv_zeroad.grading import BRAIN_TIMEOUT_S
from agentenv_zeroad.llm import DEFAULT_TIMEOUT_S, ToolSpec
from agentenv_zeroad.pool import BASE_PORT, TRAIN_IMAGE, TRAIN_PATCHES, EnginePool
from agentenv_zeroad.train.game import ModelFactory, build_model, play_game
from agentenv_zeroad.train.spec import GameSpec, load_specs

log = logging.getLogger(__name__)

PORT_STRIDE = 100  # ports between two workers' ranges: a worker has at most this many engines
ENGINES_PER_WORKER = 4  # one worker's loop keeps up with this many engines stepping flat out (re-measure: s1_calibrate)


@dataclass(frozen=True)
class GameOutcome:
    """The line `games.jsonl` keeps for a game."""

    game_id: str
    status: str  # finished, failed, cancelled
    ended_by: str | None = None
    outcomes: Mapping[str, str | None] | None = None
    game_time_s: float | None = None
    wall_s: float | None = None
    detail: str | None = None


async def run_game(
    spec: GameSpec,
    pool: EnginePool,
    out_dir: Path,
    *,
    tools: Sequence[ToolSpec],
    model_for: ModelFactory,
    brain_timeout_s: float = BRAIN_TIMEOUT_S,
    match_options: Mapping[str, Any] | None = None,
) -> GameOutcome:
    """Lease an engine, play `spec` on it, and return its outcome; a failure is an outcome, not an exception.

    An engine whose game failed (it stopped stepping, or the game raised) is marked broken and restarted before its
    next lease. A lease that can't be had, because the restart failed, is that game's failed outcome too, and the
    engine stays broken for the next try.
    """
    game_dir = out_dir / spec.game_id
    try:
        async with pool.lease() as handle:
            try:
                results = await play_game(
                    spec,
                    handle,
                    game_dir,
                    tools=tools,
                    model_for=model_for,
                    brain_timeout_s=brain_timeout_s,
                    match_options=match_options,
                )
            except Exception as e:  # noqa: BLE001 - one game's failure must not stop the others
                log.exception("game %s failed", spec.game_id)
                handle.mark_broken()
                return GameOutcome(spec.game_id, "failed", detail=f"{type(e).__name__}: {str(e)[:300]}")
            if results["status"] == "failed":
                handle.mark_broken()
    except Exception as e:  # noqa: BLE001 - an engine that won't restart costs this game, not the worker
        log.exception("game %s got no engine", spec.game_id)
        return GameOutcome(spec.game_id, "failed", detail=f"no engine: {type(e).__name__}: {str(e)[:300]}")
    return GameOutcome(
        spec.game_id,
        results["status"],
        results["ended_by"],
        {slot: seat["outcome"] for slot, seat in results["seats"].items()},
        results["game_time_s"],
        results["wall_s"],
        results["detail"],
    )


async def run_games(
    specs: Sequence[GameSpec],
    pool: EnginePool,
    out_dir: Path,
    *,
    concurrency: int,
    tools: Sequence[ToolSpec] | None = None,
    model_for: ModelFactory | None = None,
    brain_timeout_s: float = BRAIN_TIMEOUT_S,
    match_options: Mapping[str, Any] | None = None,
) -> list[GameOutcome]:
    """Play every spec, `concurrency` at a time, and return the outcomes in spec order."""
    if tools is None:
        from agentenv_zeroad.server import tool_specs

        tools = await tool_specs()
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(max(1, concurrency))
    ledger = out_dir / "games.jsonl"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S, limits=httpx.Limits(max_connections=512)) as http:
        factory = model_for or (lambda seat, slot: build_model(seat, slot, http=http))

        async def one(spec: GameSpec) -> GameOutcome:
            async with gate:
                outcome = await run_game(
                    spec,
                    pool,
                    out_dir,
                    tools=tools,
                    model_for=factory,
                    brain_timeout_s=brain_timeout_s,
                    match_options=match_options,
                )
            with ledger.open("a") as f:
                f.write(json.dumps(asdict(outcome)) + "\n")
            return outcome

        return list(await asyncio.gather(*(one(spec) for spec in specs)))


# ---- the command line --------------------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.train.runner", description=__doc__.split("\n\n")[0])
    p.add_argument("--specs", type=Path, required=True, help="JSON lines, one game spec each")
    p.add_argument("--out", type=Path, required=True, help="folder for the games' files")
    p.add_argument("--engines", type=int, default=4, help="engines in all workers together")
    p.add_argument("--concurrency", type=int, default=None, help="games in flight per worker (default: its engines)")
    p.add_argument("--workers", type=int, default=None, help="worker processes (default: engines / engines-per-worker)")
    p.add_argument(
        "--engines-per-worker", type=int, default=ENGINES_PER_WORKER, help="engines one worker process drives"
    )
    p.add_argument("--mode", choices=("docker", "local"), default="docker")
    p.add_argument("--image", default=TRAIN_IMAGE, help="the engine image (docker mode)")
    p.add_argument("--allow-unpatched", action="store_true", help="accept an image without the training engine patches")
    p.add_argument("--base-port", type=int, default=BASE_PORT)
    p.add_argument("--replays", action="store_true", help="also copy each game's engine replay (commands.txt)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def split(items: Sequence[Any], parts: int) -> list[list[Any]]:
    """`items` dealt round-robin into `parts` lists."""
    return [list(items[i::parts]) for i in range(parts)]


def engines_per_worker(engines: int, workers: int) -> list[int]:
    return [engines // workers + (1 if i < engines % workers else 0) for i in range(workers)]


def check_ports(base_port: int, workers: int, engines_in_largest: int) -> None:
    """Every worker's range must hold its engines and end below `PORT_STOP`, where Linux hands out ephemeral ports
    (an engine port in that range can be taken by any outgoing connection); otherwise refuse, instead of running
    engines on ports the OS may give away."""
    if engines_in_largest > PORT_STRIDE:
        raise SystemExit(f"a worker would run {engines_in_largest} engines; its port range holds {PORT_STRIDE}")
    end = base_port + workers * PORT_STRIDE
    if end > PORT_STOP:
        raise SystemExit(
            f"{workers} workers from port {base_port} need ports up to {end}, past {PORT_STOP} where the OS's "
            "ephemeral ports begin: lower --base-port or use fewer workers"
        )


def worker_count(engines: int, *, workers: int | None = None, per_worker: int = ENGINES_PER_WORKER) -> int:
    """`workers` if given, else enough for `per_worker` engines each; never more workers than engines."""
    if per_worker < 1 or (workers is not None and workers < 1):
        raise ValueError("workers and engines-per-worker must be at least 1")
    wanted = workers if workers is not None else -(-engines // per_worker)
    return max(1, min(wanted, engines))


async def work(args: argparse.Namespace) -> list[GameOutcome]:
    specs = load_specs(args.specs)
    pool_options: dict[str, Any] = {
        "mode": args.mode,
        "base_port": args.base_port,
        "port_stop": args.base_port + PORT_STRIDE,
        "keep_replays": args.replays,
        "image": args.image,
        "require_patches": () if args.allow_unpatched else TRAIN_PATCHES,
    }
    async with EnginePool(args.engines, **pool_options) as pool:
        return await run_games(specs, pool, args.out, concurrency=min(args.concurrency or args.engines, args.engines))


async def supervise(args: argparse.Namespace) -> int:
    """Split the specs among the workers, run each as a child process and return the worst exit code."""
    specs = args.specs.read_text().splitlines()
    specs = [line for line in specs if line.strip()]
    load_specs(args.specs)  # a bad spec fails here, before any engine starts
    workers = min(worker_count(args.engines, workers=args.workers, per_worker=args.engines_per_worker), len(specs) or 1)
    check_ports(args.base_port, workers, max(engines_per_worker(args.engines, workers)))
    shards = args.out / ".shards"
    shards.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    procs = []
    for i, (lines, engines) in enumerate(
        zip(split(specs, workers), engines_per_worker(args.engines, workers), strict=True)
    ):
        shard = shards / f"shard-{i}.jsonl"
        shard.write_text("\n".join(lines) + "\n")
        cmd = [
            sys.executable, "-m", "agentenv_zeroad.train.runner", "--worker",
            "--specs", str(shard), "--out", str(args.out), "--engines", str(engines),
            "--mode", args.mode, "--base-port", str(args.base_port + i * PORT_STRIDE),
        ]  # fmt: skip
        if args.concurrency:
            cmd += ["--concurrency", str(args.concurrency)]
        cmd += ["--image", args.image]
        if args.allow_unpatched:
            cmd.append("--allow-unpatched")
        if args.replays:
            cmd.append("--replays")
        procs.append(await asyncio.create_subprocess_exec(*cmd, env=env))
    codes = await asyncio.gather(*(proc.wait() for proc in procs))
    return max(codes, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _parser().parse_args(argv)
    if args.worker:
        outcomes = asyncio.run(work(args))
        failed = sum(1 for o in outcomes if o.status != "finished")
        log.info("worker done: %d games, %d not finished", len(outcomes), failed)
        return 0
    return asyncio.run(supervise(args))


if __name__ == "__main__":
    raise SystemExit(main())
