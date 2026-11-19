"""The showcase's command line.

    python -m agentenv_zeroad.showcase run --out DIR --config live/showcase/showcase.toml [--hours H] [--max-games N]
        [--rotation a,b,c] [--mode live|replay]
    python -m agentenv_zeroad.showcase verify DIR

`run` plays a session on this box (`session.run_session`) with the box's real engine, budget guard, power-off deadline
and broadcast; `verify` re-checks every game of a finished session against the 13-3's lockstep rules and exits 1 if any
game has a problem. Nothing here prints a key, an endpoint or a prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from agentenv_zeroad import live
from agentenv_zeroad.engine import EngineOptions, RunningEngine, free_port, start_engine
from agentenv_zeroad.llm import DEFAULT_TIMEOUT_S, base_of
from agentenv_zeroad.showcase.results import GameRow, verify_lockstep
from agentenv_zeroad.showcase.session import SessionDeps, SessionOptions, game_spec, run_session
from agentenv_zeroad.showcase.specs import ShowcaseConfig
from agentenv_zeroad.train.calibration import LITELLM_URL_ENV
from agentenv_zeroad.train.game import build_model
from agentenv_zeroad.train.panels import UNSET_URL
from agentenv_zeroad.train.spec import SeatSpec

DISPLAY = ":0"
LIVE_HOME_ENV = "ZEROAD_LIVE_HOME"
DEFAULT_LIVE_HOME = "/opt/zeroad-live"
DEADLINE_FILE_ENV = "ZEROAD_DEADLINE_FILE"
DEFAULT_DEADLINE_FILE = "/etc/zeroad-live/deadline"
BUDGET_ENV = "LIVE_BUDGET_USD"
VLLM_URL_ENV = "VLLM_BASE_URL"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
READY_TIMEOUT_S = 180.0

log = logging.getLogger("agentenv_zeroad.showcase")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.showcase", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play a session")
    run.add_argument("--out", type=Path, required=True, help="folder for session.json and the games")
    run.add_argument("--config", type=Path, default=Path("live/showcase/showcase.toml"))
    run.add_argument("--hours", type=float, default=6.0)
    run.add_argument("--max-games", type=int, default=None)
    run.add_argument("--rotation", default=None, help="opponents in turn, comma separated (default: the config's)")
    run.add_argument("--mode", choices=("live", "replay"), default="live")
    verify = sub.add_parser("verify", help="check a finished session's games against the lockstep rules")
    verify.add_argument("dir", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        return verify_dir(args.dir)
    if args.mode == "replay":
        print("replay mode is not built yet", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return asyncio.run(run_command(args))


def verify_dir(directory: Path) -> int:
    """Print each game's problems (the row's, and a fresh check of its results.json); exit 1 if there are any."""
    data = json.loads((directory / "session.json").read_text())
    found = 0
    for raw in data.get("rows", []):
        row = GameRow.from_json(raw)
        problems = list(row.problems)
        results = directory / "games" / row.game_id / "results.json"
        if results.is_file() and row.blame != "engine":
            problems += [p for p in verify_lockstep(json.loads(results.read_text())) if p not in problems]
        for problem in problems:
            print(f"{row.game_id}: {problem}")
        found += len(problems)
    return 1 if found else 0


async def run_command(args: argparse.Namespace) -> int:
    cfg = ShowcaseConfig.load(args.config)
    rotation = tuple(args.rotation.split(",")) if args.rotation else cfg.rotation
    opts = SessionOptions(
        out=args.out,
        rotation=rotation,
        hours=args.hours,
        max_games=args.max_games,
        mode=args.mode,
        policy_url=os.environ.get(VLLM_URL_ENV, DEFAULT_VLLM_URL),
        teacher_url=os.environ.get(LITELLM_URL_ENV, UNSET_URL),
    )
    broadcast = live.Broadcast(
        live.BroadcastSettings(
            display=DISPLAY,
            encoder="nvenc",
            low_graphics=False,
            rtmp_url=os.environ.get("ZEROAD_RTMP_URL"),
        ),
        args.out / "broadcast",
    )
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S, limits=httpx.Limits(max_connections=64)) as http:
        deps = SessionDeps(
            start_engine=start_paced_engine,
            budget_ok=budget_ok,
            poweroff_left_s=poweroff_left_s,
            model_for=lambda seat, slot: build_model(seat, slot, http=http, cap_usd=opts.seat_cap_usd),
            preflight=lambda: preflight_problems(cfg, opts, http),
            now=time.monotonic,
            sleep=asyncio.sleep,
        )
        session = await run_session(opts, cfg, deps, broadcast)
    print(f"session ended: {session.ended} ({session.ended_why}); {len(session.rows)} games in {opts.out}")
    return 0


async def start_paced_engine() -> RunningEngine:
    options = EngineOptions(
        port=free_port(), visual=True, display=DISPLAY, paced=True, extra_args=live.engine_args(low_graphics=False)
    )
    return await start_engine(options, mode="local", ready_timeout=READY_TIMEOUT_S)


async def budget_ok(planned_usd: float, planned_hours: float) -> bool:
    """`budget.sh check` for the game about to start: its exit status, never its output's key-bearing parts (it has
    none: it prints dollars and hours)."""
    budget = os.environ.get(BUDGET_ENV)
    if not budget:
        log.error("%s is not set: the budget guard cannot check", BUDGET_ENV)
        return False
    script = Path(os.environ.get(LIVE_HOME_ENV, DEFAULT_LIVE_HOME)) / "box" / "budget.sh"
    proc = await asyncio.create_subprocess_exec(
        str(script),
        "check",
        "--planned-usd",
        f"{planned_usd:g}",
        "--planned-hours",
        f"{planned_hours:g}",
        "--budget",
        budget,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    log.info("budget.sh check: %s", out.decode(errors="replace").strip()[-300:])
    return proc.returncode == 0


def poweroff_left_s() -> float:
    """Seconds to the box's power-off deadline; 0 (so no game starts) when the deadline can't be read."""
    path = Path(os.environ.get(DEADLINE_FILE_ENV, DEFAULT_DEADLINE_FILE))
    try:
        return float(path.read_text().strip()) - time.time()
    except (OSError, ValueError):
        log.error("no readable power-off deadline in %s: no game will start", path.name)
        return 0.0


async def preflight_problems(cfg: ShowcaseConfig, opts: SessionOptions, http: httpx.AsyncClient) -> list[str]:
    """One check per model in the rotation (and the policy's): its server answers `GET /v1/models` with the seat's key.
    A model call would spend, and the first game's own calls find anything this misses."""
    seats: dict[str, SeatSpec] = {}
    run_cfg = cfg.run_config()
    panel = next(p for p in run_cfg.panels if p.name == cfg.panel)
    for opponent in opts.rotation:
        for seat in game_spec(cfg, run_cfg, panel, opts, cfg.first_index, opponent).seats:
            if seat.kind == "llm" and seat.model:
                seats[seat.model] = seat
    return [p for p in [await _check(seat, os.environ, http) for seat in seats.values()] if p]


async def _check(seat: SeatSpec, environ: Mapping[str, str], http: httpx.AsyncClient) -> str | None:
    assert seat.base_url and seat.api_key_env
    key = environ.get(seat.api_key_env)
    if not key:
        return f"{seat.model}: the key variable {seat.api_key_env} is not set"
    try:
        response = await http.get(base_of(seat.base_url) + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as e:
        return f"{seat.model}: no answer ({type(e).__name__})"
    return None if response.status_code == 200 else f"{seat.model}: HTTP {response.status_code}"


if __name__ == "__main__":
    sys.exit(main())
