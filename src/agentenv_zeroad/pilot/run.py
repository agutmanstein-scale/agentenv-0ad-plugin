"""The pilot's round loop, run on the box (box/pilot.sh starts it as the zeroad-pilot unit).

Each round: pick the training opponents (`ladder.allot`) and the probe panel; write each game as a task into the
round's copy of the bundle; play them all at once, four tasks per `agent-env run`; read every game's results.json;
append the records to games.jsonl, rate every player, and write progress.md; then (unless --baseline) train the next
policy version on the round's training decisions in the trainer container and load its adapter into vLLM. Before a
round it asks budget.sh whether the box hours so far, the round's and its frontier spend still fit the budget, and it
stops after the round in play when the file STOP is in its home. State is in state.json, saved once a round is
recorded and again once its training is done, so a restart neither replays a round nor skips its training; a
--baseline round's decisions are what the next start trains first. With --share-gpu (a one-GPU box) vLLM's container
is stopped while the trainer has the GPU and started again after, before the new adapter is loaded. --objective best
trains only on the round's decisions that did better than their position predicted (advantage over --best-threshold)
with no rejected order, each weighted 1: the policy imitates its own best play, a lower-noise signal than the
policy gradient over every decision (--objective pg).

Pooled (--pool URI --pool-self NAME --pool-members a,b,c): every box plays the same version each round. After it,
each trains its own update from that version on its own games, puts it in the pool (URI/rNN/NAME.tgz, then
NAME.json, so a marker means a whole upload), waits for the other members' (up to --pool-wait-s), and averages every
update it has that started from the same version, weighted by decisions (trainer/average.py). Every member computes
the same average from the same updates, so all play the same next version with no coordinator. A member that is late
or out of step is left out of that round's average. URI is s3://... (the box's own role reads and writes it) or a
folder (tests).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import httpx

from agentenv_zeroad.pilot import ladder, ratings, report, scoring
from agentenv_zeroad.pilot.games import CIVS, GameSpec, Opponent, PilotSettings, task, version_model
from agentenv_zeroad.spend import append_ledger

log = logging.getLogger("pilot")
TASKS_PER_RUN = 4  # agent-env runs at most four tasks of one run at a time
FRONTIER = (
    "sonnet=anthropic/claude-sonnet-5-5,opus=anthropic/claude-opus-5-5,astra=openai/gpt-6-astra,sol=openai/gpt-6-sol"
)
SNAPSHOT_EVERY = 2  # a policy version joins the training pool every this many versions
SNAPSHOTS_KEPT = 3


@dataclasses.dataclass
class State:
    round: int = 0  # the next round to play
    version: int = 0  # the policy version the next round plays
    snapshots: list[int] = dataclasses.field(default_factory=list)
    started_at: float = dataclasses.field(default_factory=time.time)
    pending_train: bool = False  # the last round played is recorded but not trained on yet

    @classmethod
    def load(cls, path: Path) -> State:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=1))
        os.replace(tmp, path)


def plan_round(
    state: State, played: Sequence[ladder.Played], args: argparse.Namespace, rng: random.Random
) -> list[GameSpec]:
    pool = ladder.ladder(state.snapshots)
    rates = ladder.win_rates(pool, played, state.round)
    specs = []
    for i, opp in enumerate(ladder.allot(args.train_games, pool, rates), start=1):
        specs.append(_spec(f"r{state.round:02d}-t{i:02d}", state, "train", opp, rng.randrange(1, 9000), rng))
    frontier = ladder.frontier_for(state.round, frontier_list(args.frontier), args.frontier_per_round,
                                   args.frontier_offset)  # fmt: skip
    previous = state.version - 1 if state.version > 0 else None
    for opp in ladder.probe_panel(previous, frontier) if args.probes else []:
        seed = ladder.probe_seed(opp)
        probe_rng = random.Random(seed)  # the same starts and civs every round
        specs.append(_spec(f"r{state.round:02d}-p-{opp.id}", state, "probe", opp, seed, probe_rng))
    return specs


def frontier_list(spec: str) -> list[Opponent]:
    """`id=model,...` (e.g. `opus=anthropic/claude-opus-5-5`) as probe-only frontier opponents, in that order."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, model = item.partition("=")
        if not name or not model:
            raise ValueError(f"--frontier: {item!r} is not id=model")
        out.append(Opponent(name, "frontier", model=model, trainable=False))
    return out


def _spec(game_id: str, state: State, kind: str, opp: Opponent, seed: int, rng: random.Random) -> GameSpec:
    return GameSpec(
        game_id,
        state.round,
        kind,  # type: ignore[arg-type]
        seed,
        opp,
        state.version,
        policy_slot=rng.choice(("1", "2")),
        policy_civ=rng.choice(CIVS),
        opponent_civ=rng.choice(CIVS),
    )


def write_bundle(specs: Sequence[GameSpec], settings: PilotSettings, round_dir: Path) -> Path:
    bundle = round_dir / "bundle"
    src = resources.files("agentenv_zeroad") / "bundles" / "zeroad"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(str(src), bundle)
    for spec in specs:
        out = round_dir / "games" / spec.game_id
        out.mkdir(parents=True, exist_ok=True)
        (bundle / "tasks" / f"{spec.game_id}.json").write_text(json.dumps(task(spec, settings, str(out)), indent=1))
    os.chmod(bundle, 0o700)  # the task files hold the box's vLLM key
    return bundle


async def play(specs: Sequence[GameSpec], bundle: Path, round_dir: Path, timeout_s: float) -> None:
    """Every game at once, `TASKS_PER_RUN` per `agent-env run`; a run still going at `timeout_s` is stopped."""
    logs = round_dir / "logs"
    logs.mkdir(exist_ok=True)
    groups = [specs[i : i + TASKS_PER_RUN] for i in range(0, len(specs), TASKS_PER_RUN)]
    procs = []
    for n, group in enumerate(groups):
        cmd = [str(Path(sys.executable).parent / "agent-env"), "run", str(bundle)]
        for spec in group:
            cmd += ["--task", spec.game_id]
        out = open(logs / f"run-{n}.log", "wb")  # noqa: SIM115 - closed below, after the process ends
        procs.append((await asyncio.create_subprocess_exec(*cmd, stdout=out, stderr=subprocess.STDOUT), out))
        await asyncio.sleep(5)  # stagger the env starts
    try:
        await asyncio.wait_for(asyncio.gather(*(p.wait() for p, _ in procs)), timeout_s)
    except TimeoutError:
        log.error("round past its %.0f s: stopping its runs", timeout_s)
        for p, _ in procs:
            if p.returncode is None:
                p.terminate()
        await asyncio.gather(*(p.wait() for p, _ in procs))
    finally:
        for _, out in procs:
            out.close()


def collect(specs: Sequence[GameSpec], round_dir: Path) -> tuple[list[scoring.GameRecord], list[scoring.Decision]]:
    """Each game's record and its training decisions. save_zeroad_results keeps a match's files under its folder at
    the run instance's path (e.g. `@local/var/.../r00-t01-4decfz7k/zeroad-lb-<lobby>-results.json`)."""
    records, decisions = [], []
    for spec in specs:
        found = sorted((round_dir / "games" / spec.game_id).rglob("*results.json"), key=lambda p: p.stat().st_mtime)
        results = json.loads(found[-1].read_text()) if found else None
        records.append(scoring.record(spec, results, None if found else "no results.json"))
        if results is not None and spec.kind == "train":
            decisions += scoring.decisions(spec, results)
    return records, decisions


def rate(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    games = [
        (f"v{r['policy_version']}", r["opponent"], scoring.RESULT[r["outcome"]])
        for r in records
        if r.get("outcome") in scoring.RESULT
    ]
    return ratings.fit(games)


def progress(records: Sequence[dict[str, Any]], board: dict[str, float]) -> str:
    """progress.md: the ratings, then each round's probes and training games."""
    lines = ["# Post-training pilot", "", "## Ratings (Petra medium = 1000)", ""]
    lines += [f"- {p}: {r:.0f}" for p, r in sorted(board.items(), key=lambda kv: -kv[1])]
    for round_ in sorted({r["round"] for r in records}):
        rows = [r for r in records if r["round"] == round_]
        lines += ["", f"## Round {round_} (policy v{rows[0]['policy_version']})", ""]
        lines += ["| Game | Opponent | Outcome | Margin | Calls (missed) | Orders rejected | p50 call | Overruns |"]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["kind"] != "probe", r["game_id"])):
            margin = f"{r['margin']:+.0f}" if r.get("margin") is not None else "-"
            p50 = f"{r['latency_p50_s']:.1f} s" if r.get("latency_p50_s") is not None else "-"
            lines.append(
                f"| {r['game_id']} | {r['opponent']} | {r.get('outcome') or r['status']} | {margin} | "
                f"{r['calls']} ({r['missed']}) | {r['rejected']}/{r['orders']} | {p50} | {r.get('overruns')} |"
            )
    return "\n".join(lines) + "\n"


def best_decisions(decisions_path: Path, threshold: float) -> Path:
    """The round's decisions worth imitating, each with advantage 1, in train-best.jsonl beside them."""
    kept = []
    for row in _read(decisions_path):
        if row["advantage"] > threshold and row["rejected_share"] == 0:
            kept.append({**row, "advantage": 1.0})
    out = decisions_path.with_name("train-best.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    log.info("objective best: %d of the round's decisions kept", len(kept))
    return out


def train(
    args: argparse.Namespace, home: Path, round_: int, version: int, decisions_path: Path, out_name: str = ""
) -> Path:
    """Train policy version `version + 1` from `version` on the round's decisions, in the trainer container (into
    adapters/`out_name`, by default v`version + 1`)."""
    if args.objective == "best":
        decisions_path = best_decisions(decisions_path, args.best_threshold)
    out_name = out_name or f"v{version + 1}"
    out = home / "adapters" / out_name
    init = f"/pilot/adapters/v{version}" if version > 0 else ""
    gpus = args.train_gpus.split(",")
    cmd = [
        "docker", "run", "--rm", "--gpus", f'"device={args.train_gpus}"', "--ipc=host",
        "-v", f"{home}:/pilot", "-v", f"{args.hf_cache}:/root/.cache/huggingface",
        args.trainer_image,
        "torchrun", "--nproc_per_node", str(len(gpus)), "/app/train.py",
        "--decisions", f"/pilot/{decisions_path.relative_to(home)}",
        "--base", args.base_model, "--init", init, "--out", f"/pilot/adapters/{out_name}",
        "--samples", str(args.train_samples), "--batch", str(args.train_batch), "--seed", str(round_),
        "--lr", str(args.train_lr),
    ]  # fmt: skip
    with open(home / "rounds" / f"r{round_:02d}" / "train.log", "wb") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    if rc != 0 or not (out / "adapter_config.json").exists():
        raise RuntimeError(f"training v{version + 1} failed (exit {rc}): see rounds/r{round_:02d}/train.log")
    return out


def pool_round(args: argparse.Namespace, home: Path, round_: int, version: int, local: Path) -> list[str]:
    """Share this box's update of `version` (trained on round `round_`), gather the members', and average them into
    adapters/v`version + 1`. The members averaged, this box first."""
    key = f"r{round_:02d}"
    marker = {"member": args.pool_self, "from_version": version, **_train_stats(local)}
    _pool_put(args.pool, f"{key}/{args.pool_self}", local, marker)
    members = [m for m in args.pool_members.split(",") if m]
    deadline = time.monotonic() + args.pool_wait_s
    while True:
        markers = _pool_markers(args.pool, key)
        ready = [m for m in members if markers.get(m, {}).get("from_version") == version]
        if set(ready) >= set(members) or time.monotonic() > deadline:
            break
        log.info("pool %s: have %s, waiting for %s", key, ready, sorted(set(members) - set(ready)))
        time.sleep(30)
    gathered = home / "adapters" / "pool" / key
    dirs = []
    for m in [args.pool_self] + [m for m in ready if m != args.pool_self]:
        dest = gathered / m
        if m == args.pool_self:
            dest = local
        else:
            _pool_get(args.pool, f"{key}/{m}", dest)
        dirs.append(dest)
    left_out = sorted(set(members) - set(ready))
    if left_out:
        log.warning("pool %s: averaging without %s (late, or not on v%d)", key, left_out, version)
    average(args, home, dirs, home / "adapters" / f"v{version + 1}")
    return [d.name if d != local else args.pool_self for d in dirs]


def average(args: argparse.Namespace, home: Path, members: Sequence[Path], out: Path) -> None:
    """adapters/v<next> as the decision-weighted mean of the members' adapters (trainer/average.py, no GPU)."""
    cmd = ["docker", "run", "--rm", "-v", f"{home}:/pilot", args.trainer_image, "python3", "/app/average.py",
           "--out", f"/pilot/{out.relative_to(home)}"]  # fmt: skip
    for m in members:
        cmd += ["--member", f"/pilot/{m.relative_to(home)}"]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0 or not (out / "adapter_config.json").exists():
        raise RuntimeError(f"averaging into {out.name} failed: {run.stderr[-500:]}")
    log.info("pool: %s = mean of %s: %s", out.name, [m.name for m in members], run.stdout.strip()[-300:])


def _train_stats(adapter: Path) -> dict[str, Any]:
    path = adapter / "train.json"
    stats = json.loads(path.read_text()) if path.exists() else {}
    return {"decisions": stats.get("decisions"), "mean_kl": stats.get("mean_kl")}


def _pool_put(pool: str, name: str, adapter: Path, marker: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tar, meta = Path(tmp) / "a.tgz", Path(tmp) / "a.json"
        subprocess.run(["tar", "-C", str(adapter), "-czf", str(tar), "."], check=True)
        meta.write_text(json.dumps(marker))
        if pool.startswith("s3://"):
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(tar), f"{pool}/{name}.tgz"], check=True)
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(meta), f"{pool}/{name}.json"], check=True)
        else:
            target = Path(pool) / f"{name}.tgz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tar, target)
            shutil.copy(meta, Path(pool) / f"{name}.json")


def _pool_markers(pool: str, key: str) -> dict[str, dict[str, Any]]:
    """Each member's marker for round `key` (member -> its JSON)."""
    out: dict[str, dict[str, Any]] = {}
    if pool.startswith("s3://"):
        listing = subprocess.run(["aws", "s3", "ls", f"{pool}/{key}/"], capture_output=True, text=True, check=False)
        for line in listing.stdout.splitlines():
            name = line.split()[-1] if line.strip() else ""
            if name.endswith(".json"):
                got = subprocess.run(["aws", "s3", "cp", f"{pool}/{key}/{name}", "-"], capture_output=True, text=True)
                if got.returncode == 0:
                    out[name[:-5]] = json.loads(got.stdout)
    else:
        for path in (Path(pool) / key).glob("*.json"):
            out[path.stem] = json.loads(path.read_text())
    return out


def _pool_get(pool: str, name: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "a.tgz"
        if pool.startswith("s3://"):
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"{pool}/{name}.tgz", str(tar)], check=True)
        else:
            shutil.copy(Path(pool) / f"{name}.tgz", tar)
        subprocess.run(["tar", "-C", str(dest), "-xzf", str(tar)], check=True)


@contextlib.contextmanager
def gpu_for_training(args: argparse.Namespace) -> Iterator[None]:
    """With --share-gpu: vLLM's container stops for the trainer and is back up (healthy) after, even if it failed."""
    if not args.share_gpu:
        yield
        return
    subprocess.run(["docker", "stop", "-t", "60", args.vllm_container], check=True, capture_output=True)
    try:
        yield
    finally:
        subprocess.run(["docker", "start", args.vllm_container], check=True, capture_output=True)
        wait_healthy(args, args.vllm_wait_s)


def wait_healthy(args: argparse.Namespace, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if httpx.get(f"{args.vllm_host}/health", timeout=10).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"vLLM not healthy {timeout_s:.0f} s after its restart")
        time.sleep(10)


def load_adapter(args: argparse.Namespace, version: int) -> None:
    body = {"lora_name": version_model(version), "lora_path": f"/adapters/v{version}"}
    r = httpx.post(f"{args.vllm_host}/v1/load_lora_adapter", json=body, headers=_auth(args), timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"vLLM refused adapter v{version}: {r.status_code} {r.text[:300]}")


def ensure_adapters(args: argparse.Namespace, versions: Sequence[int]) -> None:
    """Load every version a round plays that vLLM doesn't serve (a restarted vLLM keeps no runtime adapter)."""
    r = httpx.get(f"{args.vllm_host}/v1/models", headers=_auth(args), timeout=60)
    r.raise_for_status()
    served = {m["id"] for m in r.json().get("data", [])}
    for v in sorted(set(versions)):
        if v > 0 and version_model(v) not in served:
            log.info("loading adapter v%d into vLLM", v)
            load_adapter(args, v)


def _auth(args: argparse.Namespace) -> dict[str, str]:
    return {"Authorization": f"Bearer {args.vllm_key}"}


def budget_ok(args: argparse.Namespace, round_hours: float, planned_usd: float) -> bool:
    if not args.budget_script:
        return True
    cmd = [args.budget_script, "check", "--planned-usd", f"{planned_usd:.2f}", "--planned-hours", f"{round_hours:.2f}"]
    run = subprocess.run([*cmd, "--budget", str(args.budget)], capture_output=True, text=True, check=False)
    log.info("budget: %s", run.stdout.strip())
    return run.returncode == 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    home = Path(args.home)
    if args.recollect is not None:
        records = recollect(home, args.recollect)
        for r in records:
            log.info("%s vs %s: %s %s %s", r.game_id, r.opponent, r.status, r.outcome, r.margin)
        return 0
    (home / "rounds").mkdir(parents=True, exist_ok=True)
    (home / "adapters").mkdir(exist_ok=True)
    state = State.load(home / "state.json")
    games_path = home / "games.jsonl"
    settings = PilotSettings(args.vllm_url, args.vllm_key, args.cap_minutes, args.speed)
    rng = random.Random(args.seed + state.round)
    train_s = 0 if args.baseline else (35 if args.share_gpu else 20) * 60
    round_hours = (args.cap_minutes * 60 / args.speed + 15 * 60 + train_s) / 3600
    last = args.rounds if not args.baseline else state.round + 1
    while state.round < last:
        if state.pending_train and not args.baseline:
            trained = home / "rounds" / f"r{state.round - 1:02d}" / "decisions.jsonl"
            if len(_read(trained)) < args.min_decisions:
                log.error("round %d left %d trainable decisions (< %d): stopping", state.round - 1,
                          len(_read(trained)), args.min_decisions)  # fmt: skip
                return 4
            with gpu_for_training(args):
                local = train(args, home, state.round - 1, state.version, trained,
                              f"local-v{state.version + 1}" if args.pool else "")  # fmt: skip
            if args.pool:
                pool_round(args, home, state.round - 1, state.version, local)
            load_adapter(args, state.version + 1)
            state.version += 1
            if state.version % SNAPSHOT_EVERY == 0:
                state.snapshots = (state.snapshots + [state.version])[-SNAPSHOTS_KEPT:]
            state.pending_train = False
            state.save(home / "state.json")
            report.write(home, dataclasses.asdict(state))
        if (home / "STOP").exists():
            log.info("STOP found: stopping before round %d", state.round)
            break
        probes = ladder.frontier_for(state.round, frontier_list(args.frontier), args.frontier_per_round,
                                     args.frontier_offset) if args.probes else []  # fmt: skip
        if not budget_ok(args, round_hours, args.frontier_usd * len(probes)):
            log.error("the budget would be passed: stopping before round %d", state.round)
            return 3
        played = [
            ladder.Played(r["opponent"], r["round"], scoring.RESULT[r["outcome"]])
            for r in _read(games_path)
            if r.get("outcome") in scoring.RESULT
        ]
        specs = plan_round(state, played, args, rng)
        ensure_adapters(args, [s.policy_version for s in specs] + [_version_of(s.opponent) for s in specs])
        round_dir = home / "rounds" / f"r{state.round:02d}"
        log.info("round %d: policy v%d, %d games: %s", state.round, state.version, len(specs),
                 ", ".join(f"{s.game_id}:{s.opponent.id}" for s in specs))  # fmt: skip
        bundle = write_bundle(specs, settings, round_dir)
        (round_dir / "specs.json").write_text(json.dumps([dataclasses.asdict(s) for s in specs]))
        asyncio.run(play(specs, bundle, round_dir, timeout_s=round_hours * 3600 + 600))
        records, decisions = collect(specs, round_dir)
        with games_path.open("a") as f:
            for rec in records:
                f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
        for rec in records:
            if rec.spend_usd > 0 and args.ledger:
                append_ledger(Path(args.ledger), f"pilot {rec.game_id}", rec.spend_usd)
        history = _history(home, state.round)
        scoring.advantages(decisions, history)
        dec_path = round_dir / "decisions.jsonl"
        with dec_path.open("w") as f:
            for d in decisions:
                f.write(json.dumps(dataclasses.asdict(d)) + "\n")
        all_records = _read(games_path)
        board = rate(all_records)
        (home / "progress.md").write_text(progress(all_records, board))
        summary = {
            "round": state.round,
            "version": state.version,
            "games": len(records),
            "finished": sum(1 for r in records if r.status == "finished"),
            "decisions": len(decisions),
            "ratings": board,
        }
        (round_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info("round %d done: %s", state.round, json.dumps(summary))
        state.round += 1
        state.pending_train = True
        state.save(home / "state.json")
        report.write(home, dataclasses.asdict(state))
    log.info("pilot stopped at round %d (policy v%d)", state.round, state.version)
    return 0


def recollect(home: Path, round_: int) -> list[scoring.GameRecord]:
    """Read round `round_`'s games again (its results, as they are now) and replace its records, decisions,
    progress page and report; the round loop's state is left as it is."""
    round_dir = home / "rounds" / f"r{round_:02d}"
    specs = load_specs(round_dir)
    records, decisions = collect(specs, round_dir)
    kept = [r for r in _read(home / "games.jsonl") if r["round"] != round_]
    rows = kept + [dataclasses.asdict(r) for r in records]
    (home / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["round"])))
    scoring.advantages(decisions, _history(home, round_))
    with (round_dir / "decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(dataclasses.asdict(d)) + "\n")
    all_records = _read(home / "games.jsonl")
    (home / "progress.md").write_text(progress(all_records, rate(all_records)))
    report.write(home, dataclasses.asdict(State.load(home / "state.json")))
    return records


def load_specs(round_dir: Path) -> list[GameSpec]:
    """The round's game specs: specs.json, or rebuilt from its task files (rounds played before specs.json)."""
    path = round_dir / "specs.json"
    if path.exists():
        return [GameSpec(**{**s, "opponent": Opponent(**s["opponent"])}) for s in json.loads(path.read_text())]
    round_ = int(round_dir.name[1:])
    specs = []
    for task_path in sorted((round_dir / "bundle" / "tasks").glob(f"r{round_:02d}-*.json")):
        steps = {s["id"]: s for s in json.loads(task_path.read_text())}
        slots = [s for s in steps.values() if s["type"] == "add_player_slot"]
        mine = next(s for s in slots if s.get("player_name") == "policy")
        theirs = next(s for s in slots if s is not mine)
        model = steps["play-policy"]["model"]
        version = 0 if model == version_model(0) else int(model.rsplit("-v", 1)[1])
        game_id = task_path.stem
        if theirs["player_kind"] == "ai":
            opp = ladder.petra(int(theirs["game_settings"]["ai_difficulty"]))
        elif steps["play-opponent"]["model"].startswith("local/"):
            opp = Opponent(theirs["game_settings"]["label"], "snapshot", model=steps["play-opponent"]["model"])
        else:
            frontier_model = steps["play-opponent"]["model"]
            opp = Opponent(game_id.split("-p-", 1)[1], "frontier", model=frontier_model, trainable=False)
        specs.append(
            GameSpec(
                game_id,
                round_,
                "probe" if "-p-" in game_id else "train",
                steps["lobby"]["game_settings"]["seed"],
                opp,
                version,
                mine["player_id"],
                mine["game_settings"]["civ"],
                theirs["game_settings"]["civ"],
            )  # fmt: skip
        )
    return specs


def _version_of(opponent: Opponent) -> int:
    return int(opponent.id[1:]) if opponent.kind == "snapshot" else 0


def _history(home: Path, round_: int) -> list[scoring.Decision]:
    """The decisions of the last rounds before `round_`, for the value estimates' fit."""
    out = []
    for r in range(max(0, round_ - ladder.RECENT_ROUNDS), round_):
        path = home / "rounds" / f"r{r:02d}" / "decisions.jsonl"
        for row in _read(path):
            row["prompt_token_ids"], row["token_ids"] = [], []  # the fit needs only features and targets
            out.append(scoring.Decision(**row))
    return out


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.pilot.run", description=__doc__.splitlines()[0])
    p.add_argument("--home", required=True, help="the pilot's folder (rounds, adapters, games.jsonl, state.json)")
    p.add_argument("--rounds", type=int, default=14)
    p.add_argument("--baseline", action="store_true", help="play one round and train nothing")
    p.add_argument("--recollect", type=int, help="read this round's games again, replace its records, and stop")
    p.add_argument("--train-games", type=int, default=16)
    p.add_argument("--probes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cap-minutes", type=float, default=30.0)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--vllm-url", default="http://172.17.0.1:8000", help="vLLM as the agents' containers reach it")
    p.add_argument("--vllm-host", default="http://172.17.0.1:8000", help="vLLM as this process reaches it")
    p.add_argument("--vllm-key-env", default="VLLM_API_KEY", help="the variable holding vLLM's key")
    p.add_argument("--frontier", default=FRONTIER, help="probe-only frontier models, id=model,...; '' none")
    p.add_argument("--frontier-per-round", type=int, default=0, help="probe this many of them a round, in turn; 0 all")
    p.add_argument("--frontier-offset", type=int, default=0, help="where in the list the turns start")
    p.add_argument("--frontier-usd", type=float, default=3.0, help="the most one frontier probe game spends")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--trainer-image", default="zeroad-trainer")
    p.add_argument("--train-gpus", default="2,3")
    p.add_argument("--share-gpu", action="store_true", help="one GPU: stop vLLM while the trainer runs")
    p.add_argument("--vllm-container", default="zeroad-vllm")
    p.add_argument("--vllm-wait-s", type=float, default=900.0)
    p.add_argument("--train-samples", type=int, default=384)
    p.add_argument("--train-batch", type=int, default=64, help="decisions per optimizer step")
    p.add_argument("--train-lr", type=float, default=2e-5)
    p.add_argument("--objective", choices=("pg", "best"), default="pg")
    p.add_argument("--best-threshold", type=float, default=0.5, help="--objective best keeps advantage above this")
    p.add_argument("--pool", default="", help="pool every box's update each round: s3://bucket/prefix/pool/ID")
    p.add_argument("--pool-self", default="", help="this box's name in the pool")
    p.add_argument("--pool-members", default="", help="every member's name, comma-separated")
    p.add_argument("--pool-wait-s", type=float, default=3600.0, help="how long to wait for the others' updates")
    p.add_argument("--min-decisions", type=int, default=100)
    p.add_argument("--hf-cache", default="/var/lib/zeroad-live/hf")
    p.add_argument("--budget-script", default="")
    p.add_argument("--budget", type=float, default=0.0)
    p.add_argument("--ledger", default="")
    args = p.parse_args(argv)
    args.vllm_key = os.environ.get(args.vllm_key_env, "")
    if not args.vllm_key:
        p.error(f"no vLLM key in ${args.vllm_key_env}")
    if args.budget_script and args.budget <= 0:
        p.error("--budget-script needs --budget")
    if args.pool and (not args.pool_self or args.pool_self not in args.pool_members.split(",")):
        p.error("--pool needs --pool-self, one of --pool-members")
    return args


if __name__ == "__main__":
    sys.exit(main())
