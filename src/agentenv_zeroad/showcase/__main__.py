"""The showcase's command line.

    python -m agentenv_zeroad.showcase run --out DIR --config live/showcase/showcase.toml [--hours H] [--max-games N]
        [--rotation a,b,c] [--mode live|replay] [--continue EARLIER_SESSION_DIR]
    python -m agentenv_zeroad.showcase verify DIR

`run` plays a session on this box (`session.run_session`) with the box's real engine, budget guard, power-off deadline
and broadcast; `verify` re-checks every game of a finished session against the 13-3's lockstep rules and exits 1 if any
game has a problem. `run` exits 0 only when the session ran and ended normally (hours, games, a stop by hand, or
a refusal after at least one game) and non-zero on a refusal before the first game or a crash: the script that
runs it powers the box off on 0 alone. Nothing here prints a key, an endpoint or a prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from pathlib import Path

import httpx

from agentenv_zeroad import live
from agentenv_zeroad.engine import EngineOptions, RunningEngine, free_port, start_engine
from agentenv_zeroad.llm import DEFAULT_TIMEOUT_S, base_of
from agentenv_zeroad.showcase.replay import start_headless_engine, start_replay_process
from agentenv_zeroad.showcase.results import NO_DECISION, GameRow, Session, verify_lockstep
from agentenv_zeroad.showcase.session import SessionDeps, SessionOptions, game_spec, run_session
from agentenv_zeroad.showcase.specs import ShowcaseConfig
from agentenv_zeroad.spend import append_ledger
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
VLLM_URL_ENV = "ZEROAD_VLLM_URL"
LEDGER_ENV = "ZEROAD_SPEND_LEDGER"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
READY_TIMEOUT_S = 180.0
PREFLIGHT_TRIES = 3
PREFLIGHT_RETRY_S = 10.0
GAME_PREFLIGHT_TRIES = 5  # a later game's two seats: more patience, since a game is at stake
GAME_PREFLIGHT_RETRY_S = 15.0
NORMAL_ENDS = frozenset({"hours", "max_games", "stopped"})
PING_BODY = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}

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
    run.add_argument(
        "--continue",
        dest="continue_from",
        type=Path,
        default=None,
        help="an earlier session's folder: its rows start this session's table (default: start at 0-0)",
    )
    verify = sub.add_parser("verify", help="check a finished session's games against the lockstep rules")
    verify.add_argument("dir", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        return verify_dir(args.dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every request URL at INFO
    return asyncio.run(run_command(args))


def unwritable(ledger: Path, out: Path) -> str | None:
    """Why the session can't write what it must (the spend ledger, its own folder), or None. The container runs as
    another user against a read-only state folder, so this is checked before a game costs anything."""
    try:
        with ledger.open("a"):
            pass
    except OSError as e:
        return f"cannot append to the spend ledger {ledger.name} ({type(e).__name__}): its spend would be lost"
    probe = out / ".write-check"
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink()
    except OSError as e:
        return f"cannot write to the output folder {out.name} ({type(e).__name__})"
    return None


def verify_dir(directory: Path) -> int:
    """Print each game's problems (the row's, and a fresh check of its results.json); exit 1 if there are any."""
    data = json.loads((directory / "session.json").read_text())
    found = 0
    for raw in data.get("rows", []):
        row = GameRow.from_json(raw)
        problems = [p for p in row.problems if not (row.blame == "engine" and p == NO_DECISION)]
        results = directory / "games" / row.game_id / "results.json"
        if row.result == "pending":
            problems.append("the game never finished (pending)")
        elif results.is_file() and row.blame != "engine":
            problems += [p for p in verify_lockstep(json.loads(results.read_text())) if p not in problems]
        for problem in problems:
            print(f"{row.game_id}: {problem}")
        found += len(problems)
    return 1 if found else 0


async def run_command(args: argparse.Namespace) -> int:
    ledger = os.environ.get(LEDGER_ENV)
    if not ledger:
        print(f"{LEDGER_ENV} is not set: the session's spend would never reach the box ledger", file=sys.stderr)
        return 2
    if problem := unwritable(Path(ledger), args.out):
        print(problem, file=sys.stderr)
        return 2
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
        continue_from=args.continue_from,
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
            preflight=lambda skip: preflight_problems(cfg, opts, http, skip),
            check_game=lambda opponent: game_problems(cfg, opts, http, opponent),
            record_spend=lambda what, usd: append_ledger(Path(ledger), what, usd),
            now=time.monotonic,
            sleep=asyncio.sleep,
            start_headless=lambda: start_headless_engine(args.out / "headless-home"),
            start_replay=lambda commands: start_replay_process(commands, DISPLAY),
        )
        session = await run_session(opts, cfg, deps, broadcast)
    print(f"session ended: {session.ended} ({session.ended_why}); {len(session.rows)} games in {opts.out}")
    return exit_code(session)


def exit_code(session: Session) -> int:
    """0 if the session ran and ended normally, else 1 (a refusal before its first game of its own). A crash never gets
    here: it raises."""
    own = [r for r in session.rows if r.carried_from is None]
    return 0 if session.ended in NORMAL_ENDS or own else 1


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
        "--no-record",  # the container's state folder is read-only: skip boxtime.json
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


async def preflight_problems(
    cfg: ShowcaseConfig,
    opts: SessionOptions,
    http: httpx.AsyncClient,
    skip: Collection[str] = (),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list[str]:
    """One check per model in the rotation (and the policy's), before the first game: `check_with_retries` of each."""
    seats: dict[str, SeatSpec] = {}
    for opponent in (o for o in opts.rotation if o not in skip):
        for _, seat in _llm_seats(cfg, opts, opponent):
            seats[seat.model or ""] = seat
    found = [await check_with_retries(seat, os.environ, http, sleep) for seat in seats.values()]
    return [p for p in found if p]


def _llm_seats(cfg: ShowcaseConfig, opts: SessionOptions, opponent: str) -> list[tuple[str, SeatSpec]]:
    """The model seats of the game against `opponent`, each with its role (`policy` or `opponent`)."""
    run_cfg = cfg.run_config()
    panel = next(p for p in run_cfg.panels if p.name == cfg.panel)
    spec = game_spec(cfg, run_cfg, panel, opts, cfg.first_index, opponent)
    return [("policy" if s.trainable else "opponent", s) for s in spec.seats if s.kind == "llm" and s.model]


async def game_problems(
    cfg: ShowcaseConfig,
    opts: SessionOptions,
    http: httpx.AsyncClient,
    opponent: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, str]:
    """The problems of one later game's two seats by role, each seat tried GAME_PREFLIGHT_TRIES times
    GAME_PREFLIGHT_RETRY_S apart (a Petra seat has no model to check)."""
    found: dict[str, str] = {}
    for role, seat in _llm_seats(cfg, opts, opponent):
        problem = await check_with_retries(
            seat, os.environ, http, sleep, tries=GAME_PREFLIGHT_TRIES, retry_s=GAME_PREFLIGHT_RETRY_S
        )
        if problem:
            found[role] = problem
    return found


async def check_with_retries(
    seat: SeatSpec,
    environ: Mapping[str, str],
    http: httpx.AsyncClient,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    *,
    tries: int = PREFLIGHT_TRIES,
    retry_s: float = PREFLIGHT_RETRY_S,
) -> str | None:
    """`_check`, up to `tries` times `retry_s` apart: a model is reported only if every try fails."""
    problem = await _check(seat, environ, http)
    for _ in range(tries - 1):
        if problem is None:
            break
        await sleep(retry_s)
        problem = await _check(seat, environ, http)
    return problem


async def _check(seat: SeatSpec, environ: Mapping[str, str], http: httpx.AsyncClient) -> str | None:
    """None if the server answers with the seat's key and serves the seat's model, else the problem (never the key or
    the endpoint). The model is served if `/v1/models` lists it; a LiteLLM wildcard route may not list its ids, so a
    model not listed is tried with one chat completion of one token, which must answer 200."""
    assert seat.base_url and seat.api_key_env
    key = environ.get(seat.api_key_env)
    if not key:
        return f"{seat.model}: the key variable {seat.api_key_env} is not set"
    headers = {"Authorization": f"Bearer {key}"}
    base = base_of(seat.base_url)
    try:
        response = await http.get(base + "/v1/models", headers=headers)
        if response.status_code == 200 and seat.model in _listed(response):
            return None
        reply = await http.post(base + "/v1/chat/completions", headers=headers, json={"model": seat.model, **PING_BODY})
    except httpx.HTTPError as e:
        return f"{seat.model}: no answer ({type(e).__name__})"
    return None if reply.status_code == 200 else f"{seat.model}: HTTP {reply.status_code}"


def _listed(response: httpx.Response) -> set[str]:
    try:
        return {str(m["id"]) for m in response.json()["data"]}
    except (ValueError, KeyError, TypeError):
        return set()


if __name__ == "__main__":
    sys.exit(main())
