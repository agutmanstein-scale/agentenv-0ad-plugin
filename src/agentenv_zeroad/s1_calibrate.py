"""The S1 batch's E1 CPU calibration (docs/post-training-v2.md, stage S1): what a 30-minute game costs on a CPU box.

    python -m agentenv_zeroad.s1_calibrate --out DIR [--levels 1,20,40,60] [--sample-s 10] [--level-timeout-s 2400]

For each level, a pool of that many engines plays that many full-length games at once (the v2 bot against Petra 3,
`s1_batch.calibration_games`), so every engine is busy for the whole game, and the run records:
- step time by game minute: the wall time of each `/step` (one 200 ms turn), from a timer on every engine, with the
  count, median, 95th percentile and maximum per game minute over all the level's games;
- core-seconds: each engine container's CPU time (its cgroup's `usage_usec`), summed over the games' run from after
  the engines had started, per game and as the mean number of cores busy;
- engine RSS: each container's `anon` memory (its cgroup's `memory.stat`; the engine's own pages, not the shared
  page cache of the image) sampled every `--sample-s`: the median and maximum of one engine, and the largest sum.
The first level should be 1: one game alone, the step times with nothing else running. `DIR/calibration.json` is
rewritten after every level, so a run cut short keeps the levels it finished; `DIR/level-<N>/<game>/results.json` are
the games. Docker engines are found with `docker ps`, and their cgroups through `/proc/<pid>/cgroup` (cgroup v2, what
Ubuntu 24.04 uses); where a cgroup can't be read the CPU and memory fields are null, and the step times still stand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentenv_zeroad.engine import DEFAULT_IMAGE, TURN_MS, Engine, EngineOptions, RunningEngine, start_engine
from agentenv_zeroad.pool import BASE_PORT, EnginePool
from agentenv_zeroad.s1_batch import calibration_games
from agentenv_zeroad.train.runner import run_games

log = logging.getLogger(__name__)

LEVELS = (1, 20, 40, 60)
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


async def run_level(
    count: int,
    out: Path,
    *,
    discover: Discover = docker_engines,
    sampler: CgroupSampler | None = None,
    mode: Literal["docker", "local"] = "docker",
    image: str = DEFAULT_IMAGE,
    base_port: int = BASE_PORT,
    sample_s: float = 10.0,
    timeout_s: float = 2400.0,
    match_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Play `count` calibration games on `count` engines at once and measure them."""
    timer, meter = StepTimer(), Meter()
    sampler = sampler or CgroupSampler(discover)

    async def start_one(options: EngineOptions, home: Path | None) -> RunningEngine:
        running = await start_engine(
            options, mode=mode, image=image, ready_timeout=300.0, docker={"reap": False}, home=home
        )
        return RunningEngine(TimedEngine(running.engine, timer), running.launcher)  # type: ignore[arg-type]

    specs = calibration_games(count)
    async with EnginePool(count, mode=mode, image=image, base_port=base_port, start_one=start_one) as pool:
        t0 = time.monotonic()
        meter.add(0.0, await sampler.sample())
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
    outcomes = [] if timed_out else games.result()
    finished = [o for o in outcomes if o.status == "finished"]
    core_s = meter.core_seconds
    return {
        "engines": count,
        "games": count,
        "games_finished": len(finished),
        "timed_out": timed_out,
        "wall_s": round(wall, 1),
        "game_time_s": [o.game_time_s for o in outcomes],
        "core_s_total": None if core_s is None else round(core_s, 1),
        "core_s_per_game": None if core_s is None else round(core_s / count, 1),
        "mean_cores": None if core_s is None or wall <= 0 else round(core_s / wall, 2),
        "rss_mb": meter.rss_mb(),
        "step_ms_by_minute": timer.summary(),
        "series": meter.series,
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


async def calibrate(levels: Sequence[int], out: Path, **options: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"host": host_facts(), "levels": {}}
    out.mkdir(parents=True, exist_ok=True)
    for count in levels:
        log.info("calibration: %d engines, %d games", count, count)
        report["levels"][str(count)] = await run_level(count, out / f"level-{count}", **options)
        (out / "calibration.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.s1_calibrate", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--levels", default=",".join(map(str, LEVELS)), help="engines in flight, one level after another")
    p.add_argument("--mode", choices=("docker", "local"), default="docker")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--base-port", type=int, default=BASE_PORT)
    p.add_argument("--sample-s", type=float, default=10.0)
    p.add_argument("--level-timeout-s", type=float, default=2400.0)
    args = p.parse_args(argv)
    levels = [int(x) for x in args.levels.split(",")]
    report = asyncio.run(
        calibrate(
            levels,
            args.out,
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
                k: {m: v for m, v in row.items() if m not in ("series", "step_ms_by_minute")}
                for k, row in report["levels"].items()
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
