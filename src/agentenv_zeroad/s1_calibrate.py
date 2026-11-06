"""The S1 batch's E1 CPU calibration (docs/post-training-v2.md, stage S1): what a 30-minute game costs on a CPU box.

    python -m agentenv_zeroad.s1_calibrate --out DIR [--levels 12,24,44] [--per-worker 4,24] [--idle-engines 1,20]
        [--idle-s 10] [--sample-s 10] [--level-timeout-s 2400] [--image zeroad-engine:0.28-rt]
        [--baseline-image zeroad-engine:0.28] [--parity-minutes 3] [--allow-unpatched]

For each level, a pool of that many engines plays that many full-length games at once (the v2 bot against Petra 3,
`s1_batch.calibration_games`), so every engine is busy for the whole game. With `--per-worker` a level is run once per
value P (capped at the level's engines): the engines are split among ceil(engines / P) worker processes, each with its
own asyncio loop and pool, so the run separates an engine that is slow because the box is out of cores from one that is
slow because a Python loop is serialising too many engines. Without it a level is one loop, as before. A worker that
dies before its engines are up fails that cell (an `error` in its row), not the calibration. The run records:
- step time by game minute: the wall time of each `/step` (one 200 ms turn), from a timer on every engine, with the
  count, median, 95th percentile and maximum per game minute over all the level's games;
- core-seconds: each engine container's CPU time (its cgroup's `usage_usec`), summed over the games' run from after
  the engines had started, per game and as the mean number of cores busy;
- Python cost: the CPU each worker process used in its game phase (its own `process_time`, from the go signal to its
  last game: core-s per game in all, and the busiest worker's cores, which near 1.0 is a loop that is full) and the
  event-loop lag of each worker (a 50 ms timer's overshoot: median, 95th percentile, max; the p95 per worker is in the
  row), which is the time every engine of that worker waits on Python before its next request is even sent;
- engine RSS: each container's `anon` memory (its cgroup's `memory.stat`; the engine's own pages, not the shared
  page cache of the image) sampled every `--sample-s`: the median and maximum of one engine, and the largest sum.
A worker finishes its games, writes `done-<i>` and waits for the parent's `sampled-<i>` before closing its engines, so
the last CPU sample still sees them and no engine's core-seconds are cut short. `DIR/calibration.json` is
rewritten after every level, so a run cut short keeps the levels it finished; `DIR/level-<N>[x<P>]/<game>/results.json`
are the games.

`--idle-engines N,...` first measures idle engines: N engines each with a game reset and one step done, then nothing
sent for `--idle-s` seconds, and the cores the containers burn in that time. An engine that waits for its next `/step`
without spinning burns about 0; the PPA's burns about 1 core (this is what `patches/rl-idle-wait.patch` fixes). It runs
on both `--baseline-image` (the PPA's, `idle_baseline` in the report) and `--image` (`idle`), so before and after are in
one report. The image the levels run on must carry the training patches (its `zeroad.patches` label) unless
`--allow-unpatched` is given, and `step_raw` asks it for no state (`/step?state=0`).

Parity: one short game (`--parity-minutes` of game time, the calibration game of seed 47000) is played on the baseline
and on the patched image, and the report says whether the bot's commands (their sha256), the outcomes, the scores and
the game's length are identical: rl-realtime changes how `/reset` parses `playerID`, and a patch that changed what the
simulation does must not feed the training.

Docker engines are found with `docker ps`, and their cgroups through `/proc/<pid>/cgroup` (cgroup v2, what Ubuntu
24.04 uses); where a cgroup can't be read the CPU and memory fields are null, and the step times still stand.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from agentenv_zeroad.engine import (
    DEFAULT_IMAGE,
    IDLE_WAIT_PATCH,
    TURN_MS,
    Engine,
    EngineError,
    EngineOptions,
    RunningEngine,
    image_patches,
    start_engine,
)
from agentenv_zeroad.pool import BASE_PORT, TRAIN_IMAGE, TRAIN_PATCHES, EnginePool, StartOne
from agentenv_zeroad.s1_batch import calibration_games
from agentenv_zeroad.train.runner import PORT_STRIDE, GameOutcome, check_ports, engines_per_worker, run_games

log = logging.getLogger(__name__)

LEVELS = (12, 24, 44)
PER_WORKER = (4, 24)  # four engines per worker, and one loop for 24 (a level of 44 is then two loops of 22)
PARITY_SEED_GAME = 0
READY_TIMEOUT_S = 900.0  # a worker's engines start within this, or the cell fails
SAMPLED_WAIT_S = 120.0  # a finished worker waits this long for the parent's last CPU sample before closing its engines
MIB = 2**20
Discover = Callable[[], Awaitable[dict[str, int]]]


class StepTimer:
    """Wall milliseconds of every `/step`, by the game minute the step belongs to."""

    def __init__(self) -> None:
        self.by_minute: dict[int, list[float]] = defaultdict(list)

    def record(self, step: int, seconds: float) -> None:
        self.by_minute[step * TURN_MS // 60_000].append(seconds * 1000)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for minute, values in sorted(self.by_minute.items()):
            ordered = sorted(values)
            out[str(minute)] = {
                "steps": len(ordered),
                "mean": round(statistics.fmean(ordered), 2),
                "median": round(statistics.median(ordered), 2),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
                "max": round(ordered[-1], 2),
            }
        return out


class TimedEngine:
    """An engine client whose `step` and `step_raw` are timed; everything else is the engine's own."""

    def __init__(self, inner: Engine, timer: StepTimer):
        self._inner = inner
        self._timer = timer
        self._step = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._step = 0
        return await self._inner.reset(*args, **kwargs)

    async def step(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await self._inner.step(*args, **kwargs)
        finally:
            self._timer.record(self._step, time.perf_counter() - started)
            self._step += 1

    async def step_raw(self, *args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        try:
            await self._inner.step_raw(*args, **kwargs)
        finally:
            self._timer.record(self._step, time.perf_counter() - started)
            self._step += 1


@dataclass(frozen=True)
class Reading:
    cpu_us: int
    anon: int
    current: int


class CgroupSampler:
    """Reads each engine container's CPU time and memory from its cgroup; `discover` names the containers and their
    main pids. Paths are injectable so tests can point them at a fake tree."""

    def __init__(
        self, discover: Discover, *, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
    ):
        self.discover = discover
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root

    def _cgroup(self, pid: int) -> Path | None:
        try:
            lines = (self.proc_root / str(pid) / "cgroup").read_text().splitlines()
        except OSError:
            return None
        path = next((line[3:] for line in lines if line.startswith("0::")), None)
        return None if path is None else self.cgroup_root / path.lstrip("/")

    @staticmethod
    def _field(path: Path, key: str) -> int | None:
        try:
            for line in path.read_text().splitlines():
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
        except (OSError, ValueError):
            pass
        return None

    async def sample(self) -> dict[str, Reading]:
        readings: dict[str, Reading] = {}
        for name, pid in (await self.discover()).items():
            group = self._cgroup(pid)
            if group is None:
                continue
            cpu = self._field(group / "cpu.stat", "usage_usec")
            anon = self._field(group / "memory.stat", "anon")
            try:
                current = int((group / "memory.current").read_text())
            except (OSError, ValueError):
                current = None
            if cpu is not None and anon is not None:
                readings[name] = Reading(cpu, anon, current or 0)
        return readings


async def docker_engines() -> dict[str, int]:
    """Name -> main pid of every running `zeroad-engine-*` container."""

    async def run(*cmd: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL
        )
        out, _ = await proc.communicate()
        return out.decode()

    ids = (await run("docker", "ps", "-q", "--filter", "name=zeroad-engine-")).split()
    if not ids:
        return {}
    found: dict[str, int] = {}
    for line in (await run("docker", "inspect", "-f", "{{.Name}} {{.State.Pid}}", *ids)).splitlines():
        name, _, pid = line.partition(" ")
        if pid.strip().isdigit() and int(pid) > 0:
            found[name.lstrip("/")] = int(pid)
    return found


class Meter:
    """Core-seconds and memory over a phase: the first sample is the baseline, the last the end."""

    def __init__(self) -> None:
        self.base: dict[str, int] = {}
        self.last: dict[str, int] = {}
        self.anon: list[int] = []
        self.sums: list[int] = []
        self.series: list[dict[str, float]] = []
        self._first = True
        self._prev: tuple[float, float] | None = None

    def add(self, at: float, readings: dict[str, Reading]) -> None:
        for name, r in readings.items():
            self.base.setdefault(name, r.cpu_us if self._first else 0)  # a container born mid-phase starts from 0
            self.last[name] = max(r.cpu_us, self.last.get(name, 0))
        cpu_s = sum(self.last[n] - self.base[n] for n in self.last) / 1e6
        if not self._first:
            self.anon += [r.anon for r in readings.values()]
            self.sums.append(sum(r.anon for r in readings.values()))
            cores = (cpu_s - self._prev[1]) / (at - self._prev[0]) if self._prev and at > self._prev[0] else 0.0
            self.series.append(
                {
                    "t_s": round(at, 1),
                    "engines": len(readings),
                    "cores": round(cores, 2),
                    "rss_mb": round(self.sums[-1] / MIB),
                }
            )
        self._first = False
        self._prev = (at, cpu_s)

    @property
    def core_seconds(self) -> float | None:
        return sum(self.last[n] - self.base[n] for n in self.last) / 1e6 if self.last else None

    def rss_mb(self) -> dict[str, float] | None:
        if not self.anon:
            return None
        return {
            "median": round(statistics.median(self.anon) / MIB, 1),
            "max": round(max(self.anon) / MIB, 1),
            "sum_max": round(max(self.sums) / MIB, 1),
        }


class LoopLag:
    """How late a 50 ms timer fires on this event loop: the time every request of every engine on it waits for the loop
    before it is even sent. A loop that is full (its worker near one core) shows it here, not in the engine."""

    PERIOD_S = 0.05

    def __init__(self) -> None:
        self.samples_ms: list[float] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            due = time.perf_counter() + self.PERIOD_S
            await asyncio.sleep(self.PERIOD_S)
            self.samples_ms.append(max(0.0, (time.perf_counter() - due) * 1000))


def lag_summary(samples_ms: Sequence[float]) -> dict[str, float] | None:
    if not samples_ms:
        return None
    ordered = sorted(samples_ms)
    return {
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "max": round(ordered[-1], 2),
    }


def _cpu_s() -> float:
    """This process's own CPU time (user + system), seconds."""
    return time.process_time()


async def check_image(image: str, *, mode: Literal["docker", "local"], require: Sequence[str]) -> bool:
    """Refuse an image docker lacks or that lacks a patch the run requires (the pool does this for its own starts; the
    calibration's timed starts bypass it); whether `step_raw` may ask this image for no state."""
    if mode != "docker":
        return False
    patches = await image_patches(image)
    if patches is None:
        raise EngineError(f"engine image {image} is not built here: scripts/build-engine.sh rt")
    missing = [p for p in require if p not in patches]
    if missing:
        raise EngineError(
            f"engine image {image} lacks the patches {', '.join(missing)} (its zeroad.patches label is "
            f"{' '.join(patches) or 'empty'}): REBUILD=1 scripts/build-engine.sh rt, or --allow-unpatched"
        )
    return IDLE_WAIT_PATCH in patches


def _timed_start(
    timer: StepTimer, mode: Literal["docker", "local"], image: str, no_state_step: bool = False
) -> StartOne:
    async def start_one(options: EngineOptions, home: Path | None) -> RunningEngine:
        running = await start_engine(
            options,
            mode=mode,
            image=image,
            ready_timeout=300.0,
            docker={"reap": False},
            home=home,
            no_state_step=no_state_step,
        )
        return RunningEngine(TimedEngine(running.engine, timer), running.launcher)  # type: ignore[arg-type]

    return start_one


def level_key(count: int, per_worker: int | None) -> str:
    return str(count) if per_worker is None else f"{count}x{per_worker}"


async def run_level(
    count: int,
    out: Path,
    *,
    per_worker: int | None = None,
    discover: Discover = docker_engines,
    sampler: CgroupSampler | None = None,
    mode: Literal["docker", "local"] = "docker",
    image: str = TRAIN_IMAGE,
    require_patches: Sequence[str] = TRAIN_PATCHES,
    base_port: int = BASE_PORT,
    sample_s: float = 10.0,
    timeout_s: float = 2400.0,
    ready_timeout_s: float = READY_TIMEOUT_S,
    match_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Play `count` calibration games on `count` engines at once and measure them: in this process on one loop, or,
    with `per_worker` below `count`, in ceil(count / per_worker) worker processes."""
    sampler = sampler or CgroupSampler(discover)
    no_state = await check_image(image, mode=mode, require=require_patches)
    if per_worker is not None and per_worker < count:
        return await _run_level_in_workers(
            count, out, per_worker, sampler=sampler, mode=mode, image=image, no_state_step=no_state,
            base_port=base_port, sample_s=sample_s, timeout_s=timeout_s, ready_timeout_s=ready_timeout_s,
        )  # fmt: skip
    timer, meter, lag = StepTimer(), Meter(), LoopLag()
    specs = calibration_games(count)
    async with EnginePool(
        count, mode=mode, image=image, base_port=base_port, start_one=_timed_start(timer, mode, image, no_state)
    ) as pool:
        t0, cpu0 = time.monotonic(), _cpu_s()
        meter.add(0.0, await sampler.sample())
        lag.start()
        games = asyncio.create_task(
            run_games(specs, pool, out, concurrency=count, tools=[], match_options=match_options)
        )
        timed_out = False
        while not games.done():
            await asyncio.wait({games}, timeout=sample_s)
            meter.add(time.monotonic() - t0, await sampler.sample())
            if not games.done() and time.monotonic() - t0 > timeout_s:
                timed_out = True
                games.cancel()
                await asyncio.gather(games, return_exceptions=True)
        wall = time.monotonic() - t0
        python_s = _cpu_s() - cpu0
        await lag.stop()
    outcomes = [] if timed_out else games.result()
    detail = [_worker_row(0, count, python_s, wall, lag.samples_ms)]
    return _report(
        count, 1, count if per_worker is None else per_worker, outcomes, timed_out, wall, meter, timer.summary(),
        lag.samples_ms, detail,
    )  # fmt: skip


def _worker_row(index: int, engines: int, python_s: float, wall: float, lag_ms: Sequence[float]) -> dict[str, Any]:
    lag = lag_summary(lag_ms)
    return {
        "index": index,
        "engines": engines,
        "python_cpu_s": round(python_s, 2),
        "wall_s": round(wall, 1),
        "python_cores_busy": round(python_s / wall, 3) if wall > 0 else None,
        "loop_lag_p95_ms": None if lag is None else lag["p95"],
    }


def _report(
    count: int,
    workers: int,
    per_worker: int,
    outcomes: Sequence[GameOutcome],
    timed_out: bool,
    wall: float,
    meter: Meter,
    step_ms: dict[str, dict[str, float]],
    lag_ms: Sequence[float],
    worker_rows: Sequence[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    finished = [o for o in outcomes if o.status == "finished"]
    core_s = meter.core_seconds
    python_s = sum(r["python_cpu_s"] for r in worker_rows) if worker_rows else None
    busy = [r["python_cores_busy"] for r in worker_rows if r["python_cores_busy"] is not None]
    p95 = [r["loop_lag_p95_ms"] for r in worker_rows if r["loop_lag_p95_ms"] is not None]
    return {
        "engines": count,
        "workers": workers,
        "engines_per_worker": per_worker,
        "games": count,
        "games_finished": len(finished),
        "timed_out": timed_out,
        "error": error,
        "wall_s": round(wall, 1),
        "game_time_s": [o.game_time_s for o in outcomes],
        "core_s_total": None if core_s is None else round(core_s, 1),
        "core_s_per_game": None if core_s is None else round(core_s / count, 1),
        "mean_cores": None if core_s is None or wall <= 0 else round(core_s / wall, 2),
        "python_core_s_per_game": None if python_s is None else round(python_s / count, 1),
        "python_cores_busy": max(busy) if busy else None,  # the busiest worker: near 1.0, its loop is full
        "python_cores_busy_total": None if python_s is None or wall <= 0 else round(python_s / wall, 2),
        "loop_lag_ms": lag_summary(lag_ms),
        "loop_lag_p95_ms_worst_worker": max(p95) if p95 else None,
        "workers_detail": list(worker_rows),
        "rss_mb": meter.rss_mb(),
        "step_ms_by_minute": step_ms,
        "series": meter.series,
    }


def _failed_cell(count: int, workers: int, per_worker: int, error: str) -> dict[str, Any]:
    return _report(count, workers, per_worker, [], False, 0.0, Meter(), {}, [], [], error)


# ---- worker processes: one loop per few engines -------------------------------------------------------------------
async def _wait_for(path: Path, *, poll_s: float = 0.2, timeout: float | None = None) -> bool:
    """Wait for `path` to exist; False if `timeout` seconds pass first."""

    async def poll() -> None:
        while not path.exists():
            await asyncio.sleep(poll_s)

    try:
        await asyncio.wait_for(poll(), timeout)
    except TimeoutError:
        return False
    return True


async def cell_worker(
    first: int,
    engines: int,
    total: int,
    out: Path,
    sync: Path,
    index: int,
    *,
    mode: Literal["docker", "local"],
    image: str,
    base_port: int,
    timeout_s: float,
    no_state_step: bool = False,
    sampled_wait_s: float = SAMPLED_WAIT_S,
    match_options: Mapping[str, Any] | None = None,
) -> None:
    """One worker process of a level: start `engines` engines, say so (`ready-<index>`), wait for the parent's `go` (it
    takes the CPU baseline once every worker's engines are up), play games `first..first+engines` of the level's
    `total` with the same timer and loop-lag meter as the in-process level, write `result-<index>.json` (its own
    game-phase Python CPU and wall time among it), then say `done-<index>` and keep its engines until the parent has
    taken its last CPU sample (`sampled-<index>`), so the engines' core-seconds are not cut short."""
    timer, lag = StepTimer(), LoopLag()
    specs = calibration_games(total)[first : first + engines]
    async with EnginePool(
        engines, mode=mode, image=image, base_port=base_port, port_stop=base_port + PORT_STRIDE,
        start_one=_timed_start(timer, mode, image, no_state_step),
    ) as pool:  # fmt: skip
        (sync / f"ready-{index}").write_text("")
        await _wait_for(sync / "go")
        started, cpu0 = time.monotonic(), _cpu_s()
        lag.start()
        games = asyncio.create_task(
            run_games(specs, pool, out, concurrency=engines, tools=[], match_options=match_options)
        )
        await asyncio.wait({games}, timeout=timeout_s)
        timed_out = not games.done()
        if timed_out:
            games.cancel()
            await asyncio.gather(games, return_exceptions=True)
        python_s, wall = _cpu_s() - cpu0, time.monotonic() - started
        await lag.stop()
        outcomes = [] if timed_out else games.result()
        (sync / f"result-{index}.json.tmp").write_text(
            json.dumps(
                {
                    "index": index,
                    "engines": engines,
                    "timed_out": timed_out,
                    "wall_s": wall,
                    "python_cpu_s": python_s,
                    "outcomes": [asdict(o) for o in outcomes],
                    "steps_ms": {str(m): v for m, v in timer.by_minute.items()},
                    "lag_ms": lag.samples_ms,
                }
            )
        )
        (sync / f"result-{index}.json.tmp").replace(sync / f"result-{index}.json")
        (sync / f"done-{index}").write_text("")
        await _wait_for(sync / f"sampled-{index}", timeout=sampled_wait_s)


def merge_results(
    results: Sequence[Mapping[str, Any]],
) -> tuple[StepTimer, list[GameOutcome], bool, float, list[float]]:
    """The workers' result files as one level: a step timer over all their samples, every outcome, whether any worker
    ran out of time, the Python CPU seconds of all of them (each its own game phase), and every loop-lag sample."""
    timer = StepTimer()
    outcomes: list[GameOutcome] = []
    lag: list[float] = []
    for result in results:
        for minute, values in result["steps_ms"].items():
            timer.by_minute[int(minute)].extend(values)
        outcomes += [GameOutcome(**o) for o in result["outcomes"]]
        lag += result["lag_ms"]
    return (
        timer,
        outcomes,
        any(r["timed_out"] for r in results),
        sum(r["python_cpu_s"] for r in results),
        lag,
    )


async def _run_level_in_workers(
    count: int,
    out: Path,
    per_worker: int,
    *,
    sampler: CgroupSampler,
    mode: Literal["docker", "local"],
    image: str,
    no_state_step: bool,
    base_port: int,
    sample_s: float,
    timeout_s: float,
    ready_timeout_s: float,
) -> dict[str, Any]:
    workers = -(-count // per_worker)
    shares = engines_per_worker(count, workers)
    check_ports(base_port, workers, max(shares))
    sync = out / ".sync"
    shutil.rmtree(sync, ignore_errors=True)  # a rerun of this cell must not see the last run's go, ready or done files
    sync.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    procs, first = [], 0
    for i, engines in enumerate(shares):
        cmd = [
            sys.executable, "-m", "agentenv_zeroad.s1_calibrate", "--cell-worker", "--out", str(out),
            "--sync-dir", str(sync), "--worker-index", str(i), "--first-game", str(first),
            "--engines", str(engines), "--games-total", str(count), "--mode", mode, "--image", image,
            "--base-port", str(base_port + i * PORT_STRIDE), "--level-timeout-s", str(timeout_s),
        ]  # fmt: skip
        if no_state_step:
            cmd.append("--no-state-step")
        procs.append(await asyncio.create_subprocess_exec(*cmd, env=env))
        first += engines
    waits = asyncio.gather(*(proc.wait() for proc in procs))
    meter = Meter()
    done = lambda: all((sync / f"done-{i}").exists() for i in range(workers))  # noqa: E731
    try:
        ready_by = time.monotonic() + ready_timeout_s
        while not all((sync / f"ready-{i}").exists() for i in range(workers)):  # every worker's engines are up
            if waits.done():
                return _failed_cell(
                    count, workers, per_worker, f"a worker exited before its engines were ready: {waits.result()}"
                )
            if time.monotonic() > ready_by:
                return _failed_cell(
                    count, workers, per_worker, f"the workers' engines were not ready in {ready_timeout_s:.0f} s"
                )
            await asyncio.sleep(0.5)
        t0 = time.monotonic()
        meter.add(0.0, await sampler.sample())
        (sync / "go").write_text("")
        next_sample = t0 + sample_s
        while not waits.done() and not done():
            await asyncio.wait({waits}, timeout=min(0.2, sample_s))
            if time.monotonic() >= next_sample:
                meter.add(time.monotonic() - t0, await sampler.sample())
                next_sample += sample_s
            if time.monotonic() - t0 > timeout_s + 180:  # a worker stops its games at timeout_s by itself
                break
        wall = time.monotonic() - t0
        if done():  # the engines are still up: take the last reading, then let the workers close them
            meter.add(wall, await sampler.sample())
        for i in range(workers):
            (sync / f"sampled-{i}").write_text("")
        await asyncio.wait({waits}, timeout=SAMPLED_WAIT_S + 30)
    finally:
        for proc in procs:
            if proc.returncode is None:
                proc.kill()
        await asyncio.gather(waits, return_exceptions=True)
    results = [
        json.loads((sync / f"result-{i}.json").read_text())
        for i in range(workers)
        if (sync / f"result-{i}.json").is_file()
    ]
    timer, outcomes, timed_out, _, lag = merge_results(results)
    error = None
    if len(results) < workers:
        timed_out = True
        error = f"{workers - len(results)} of {workers} workers wrote no result"
    if timed_out:
        outcomes = []
    rows = [_worker_row(r["index"], r["engines"], r["python_cpu_s"], r["wall_s"], r["lag_ms"]) for r in results]
    return _report(count, workers, per_worker, outcomes, timed_out, wall, meter, timer.summary(), lag, rows, error)


# ---- idle engines: what does an engine burn while nobody sends /step ----------------------------------------------
async def run_idle(
    count: int,
    *,
    idle_s: float = 10.0,
    discover: Discover = docker_engines,
    sampler: CgroupSampler | None = None,
    mode: Literal["docker", "local"] = "docker",
    image: str = TRAIN_IMAGE,
    require_patches: Sequence[str] = (),
    base_port: int = BASE_PORT,
    settle_s: float = 3.0,
) -> dict[str, Any]:
    """Start `count` engines, reset a calibration game on each and step it once, then send nothing for `idle_s`
    seconds and report the CPU the containers burn: cores in all and per engine. An engine that blocks while it waits
    burns about 0; one that spins its main loop burns about 1."""
    sampler = sampler or CgroupSampler(discover)
    scenario = calibration_games(1)[0].match_config().scenario()
    async with EnginePool(count, mode=mode, image=image, base_port=base_port, require_patches=require_patches) as pool:
        engines = [pool._slots[i].running.engine for i in range(count)]  # noqa: SLF001 - the probe talks to each engine
        await asyncio.gather(*(e.reset(scenario) for e in engines))
        await asyncio.gather(*(e.step_raw() for e in engines))
        await asyncio.sleep(settle_s)
        before = await sampler.sample()
        t0 = time.monotonic()
        await asyncio.sleep(idle_s)
        after = await sampler.sample()
        elapsed = time.monotonic() - t0
    burned = sum(after[n].cpu_us - before[n].cpu_us for n in after if n in before) / 1e6
    cores = burned / elapsed if elapsed > 0 and before else None
    return {
        "image": image,
        "engines": count,
        "idle_s": round(elapsed, 1),
        "cores": None if cores is None else round(cores, 2),
        "cores_per_engine": None if cores is None else round(cores / count, 3),
    }


# ---- parity: the patched engine must play the same game -----------------------------------------------------------
PARITY_FIELDS = ("status", "ended_by", "game_time_s", "seats", "commands", "commands_sha256")


def _parity_row(image: str, outcome: GameOutcome, results: Mapping[str, Any]) -> dict[str, Any]:
    commands = json.dumps(results.get("commands", {}), sort_keys=True, separators=(",", ":"))
    return {
        "image": image,
        "status": outcome.status,
        "ended_by": outcome.ended_by,
        "game_time_s": outcome.game_time_s,
        "seats": {slot: {"outcome": s.get("outcome"), "score": s.get("score")} for slot, s in results["seats"].items()},
        "commands": sum(len(v) for v in results.get("commands", {}).values()),
        "commands_sha256": hashlib.sha256(commands.encode()).hexdigest(),
    }


async def run_parity(
    out: Path,
    images: Sequence[str],
    *,
    minutes: float = 3.0,
    mode: Literal["docker", "local"] = "docker",
    base_port: int = BASE_PORT,
    require_patches: Mapping[str, Sequence[str]] | None = None,
    timeout_s: float = 1200.0,
    match_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One short game with identical specs (same seed, civs and bot) on each of `images`, and whether the games match:
    the same commands, outcomes, scores and length. A game that fails on an image is a row with its error, not an
    exception."""
    spec = replace(calibration_games(1)[PARITY_SEED_GAME], game_id="parity", time_cap_minutes=minutes)
    rows: list[dict[str, Any]] = []
    for n, image in enumerate(images):
        folder = out / f"parity-{n}"
        try:
            async with EnginePool(
                1, mode=mode, image=image, base_port=base_port, require_patches=(require_patches or {}).get(image, ())
            ) as pool:
                [outcome] = await asyncio.wait_for(
                    run_games([spec], pool, folder, concurrency=1, tools=[], match_options=match_options), timeout_s
                )
            results = json.loads((folder / spec.game_id / "results.json").read_text())
            rows.append(_parity_row(image, outcome, results))
        except Exception as e:  # noqa: BLE001 - a broken image is the finding, the calibration goes on
            rows.append({"image": image, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return {"minutes": minutes, **compare_parity(rows)}


def compare_parity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the games in `rows` (one per image, the first the baseline) are the same game."""
    first = rows[0] if rows else {}
    differing = sorted({f for r in rows[1:] for f in PARITY_FIELDS if r.get(f) != first.get(f)})
    errors = [r["image"] for r in rows if "error" in r]
    return {
        "games": list(rows),
        "same": len(rows) > 1 and not errors and not differing,
        "differences": differing,
        "errors": errors,
    }


def host_facts() -> dict[str, Any]:
    mem = None
    try:
        mem = next(
            int(line.split()[1]) // 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal")
        )
    except (OSError, StopIteration, ValueError):
        pass
    return {"vcpus": os.cpu_count(), "mem_total_mb": mem}


def plan_cells(levels: Sequence[int], per_worker: Sequence[int]) -> list[tuple[int, int | None]]:
    """The (engines, engines per worker) cells to run: every level once per distinct share, a share of at least the
    level's engines being one loop, and levels alone (one loop, the old way) when no share is given."""
    if not per_worker:
        return [(count, None) for count in levels]
    cells: list[tuple[int, int | None]] = []
    for count in levels:
        for share in dict.fromkeys(min(p, count) for p in per_worker):
            cells.append((count, share))
    return cells


async def calibrate(
    levels: Sequence[int],
    out: Path,
    *,
    per_worker: Sequence[int] = (),
    idle_engines: Sequence[int] = (),
    idle_s: float = 10.0,
    baseline_image: str | None = DEFAULT_IMAGE,
    require_patches: Sequence[str] = TRAIN_PATCHES,
    parity_minutes: float = 0.0,
    **options: Any,
) -> dict[str, Any]:
    """The whole calibration: idle engines on the baseline and the run's image, the parity game, then every cell.
    `require_patches` is what the run's image must carry (`()` accepts any); the baseline is never required to."""
    image = options.get("image", TRAIN_IMAGE)
    baseline = baseline_image if baseline_image and baseline_image != image else None
    report: dict[str, Any] = {
        "host": host_facts(),
        "image": image,
        "baseline_image": baseline,
        "levels": {},
        "idle": {},
        "idle_baseline": {},
    }
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        (out / "calibration.json").write_text(json.dumps(report, indent=1) + "\n")

    mode, base_port = options.get("mode", "docker"), options.get("base_port", BASE_PORT)
    await check_image(image, mode=mode, require=require_patches)  # before any measurement: a wrong image fails at once
    idle_options = {k: v for k, v in options.items() if k in ("mode", "base_port", "discover", "sampler")}
    for count in idle_engines:
        for key, which, required in (("idle_baseline", baseline, ()), ("idle", image, require_patches)):
            if which is None:
                continue
            log.info("calibration: %d idle engines on %s for %.0f s", count, which, idle_s)
            report[key][str(count)] = await _guarded(
                run_idle(count, idle_s=idle_s, image=which, require_patches=required, **idle_options)
            )
            save()
    if parity_minutes > 0 and baseline:
        log.info("calibration: parity game (%.1f game minutes) on %s and %s", parity_minutes, baseline, image)
        report["parity"] = await run_parity(
            out, [baseline, image], minutes=parity_minutes, mode=mode, base_port=base_port,
            require_patches={image: require_patches},
        )  # fmt: skip
        save()
    for count, share in plan_cells(levels, per_worker):
        key = level_key(count, share)
        log.info("calibration: %d engines, %d games, %s engines per worker", count, count, share or "all")
        report["levels"][key] = await _guarded(
            run_level(count, out / f"level-{key}", per_worker=share, require_patches=require_patches, **options),
            failed=lambda error, count=count, share=share: _failed_cell(count, 0, share or count, error),
        )
        save()
    return report


async def _guarded(
    work: Awaitable[dict[str, Any]], failed: Callable[[str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """One measurement's result; if it raises (an engine that won't start, a port plan that doesn't fit) the cell
    records the error and the calibration goes on with the next."""
    try:
        return await work
    except Exception as e:  # noqa: BLE001 - one cell's failure must not cost the cells after it
        log.exception("calibration cell failed")
        error = f"{type(e).__name__}: {str(e)[:300]}"
        return failed(error) if failed else {"error": error}


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.s1_calibrate", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--levels", default=",".join(map(str, LEVELS)), help="engines in flight, one level after another")
    p.add_argument(
        "--per-worker",
        default=",".join(map(str, PER_WORKER)),
        help="engines per worker process, one run of each level per value ('' for one loop per level)",
    )
    p.add_argument("--idle-engines", default="", help="measure this many idle engines first (e.g. 1,20)")
    p.add_argument("--idle-s", type=float, default=10.0, help="seconds the idle engines get no /step")
    p.add_argument("--mode", choices=("docker", "local"), default="docker")
    p.add_argument("--image", default=TRAIN_IMAGE, help="the engine image the levels run on")
    p.add_argument(
        "--baseline-image", default=DEFAULT_IMAGE, help="the PPA image the idle engines and parity compare to"
    )
    p.add_argument(
        "--parity-minutes", type=float, default=3.0, help="game minutes of the same-seed parity game; 0: none"
    )
    p.add_argument("--allow-unpatched", action="store_true", help="accept an --image without the training patches")
    p.add_argument("--base-port", type=int, default=BASE_PORT)
    p.add_argument("--sample-s", type=float, default=10.0)
    p.add_argument("--level-timeout-s", type=float, default=2400.0)
    for name in ("--cell-worker", "--no-state-step"):
        p.add_argument(name, action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--sync-dir", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--first-game", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--engines", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--games-total", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.cell_worker:
        asyncio.run(
            cell_worker(
                args.first_game, args.engines, args.games_total, args.out, args.sync_dir, args.worker_index,
                mode=args.mode, image=args.image, base_port=args.base_port, timeout_s=args.level_timeout_s,
                no_state_step=args.no_state_step,
            )
        )  # fmt: skip
        return 0
    report = asyncio.run(
        calibrate(
            _ints(args.levels),
            args.out,
            per_worker=_ints(args.per_worker),
            idle_engines=_ints(args.idle_engines),
            idle_s=args.idle_s,
            baseline_image=args.baseline_image,
            require_patches=() if args.allow_unpatched else TRAIN_PATCHES,
            parity_minutes=args.parity_minutes,
            mode=args.mode,
            image=args.image,
            base_port=args.base_port,
            sample_s=args.sample_s,
            timeout_s=args.level_timeout_s,
        )
    )
    print(
        json.dumps(
            {
                "idle": report["idle"],
                "idle_baseline": report["idle_baseline"],
                "parity": report.get("parity"),
                "levels": {
                    k: {m: v for m, v in row.items() if m not in ("series", "step_ms_by_minute", "workers_detail")}
                    for k, row in report["levels"].items()
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
