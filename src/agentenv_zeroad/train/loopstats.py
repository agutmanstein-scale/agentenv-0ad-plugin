"""What one asyncio process spends and how late its event loop runs: its own CPU and the loop's lag.

A training process drives many games from one loop (the orchestrator all of them, a runner worker its engines'); when
the loop is saturated every game's HTTP, JSON and bot work queues behind the others, and the engines idle. The lag is
how much later than asked a `sleep` wakes: near 0 on a loop with room, tens of ms when it is the bottleneck. CPU is
the process's own user+system time, so a loop that is saturated shows about 1.0 cores.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROC_STAT = Path("/proc/self/stat")


def process_cpu_s(stat: Path = PROC_STAT) -> float:
    """This process's user + system CPU seconds, from /proc/self/stat (all its threads); `time.process_time()` where
    there is no /proc."""
    try:
        fields = stat.read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")  # utime, stime: fields 14 and 15
    except (OSError, IndexError, ValueError):
        return time.process_time()


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


class LoopMonitor:
    """Samples the running loop's lag every `interval` seconds and reports it with the process CPU on `snapshot()`."""

    def __init__(
        self,
        interval: float = 0.05,
        *,
        clock: Callable[[], float] = time.monotonic,
        cpu: Callable[[], float] = process_cpu_s,
    ):
        self.interval = interval
        self._clock = clock
        self._cpu = cpu
        self._lags: list[float] = []
        self._task: asyncio.Task[None] | None = None
        self._mark = (clock(), cpu())
        self.cpu_start = self._mark[1]

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            before = self._clock()
            await asyncio.sleep(self.interval)
            self._lags.append(max(0.0, self._clock() - before - self.interval))

    def snapshot(self) -> dict[str, Any]:
        """CPU seconds since the process started, the CPU used per second of wall time since the last snapshot (1.0
        is one core), and the loop lag in ms (p50, p95, max over the samples since the last snapshot; null with
        none)."""
        now, cpu = self._clock(), self._cpu()
        before_now, before_cpu = self._mark
        self._mark = (now, cpu)
        lags, self._lags = sorted(self._lags), []
        out: dict[str, Any] = {
            "cpu_s": round(cpu - self.cpu_start, 2),
            "cpu_cores": round((cpu - before_cpu) / (now - before_now), 3) if now > before_now else None,
            "lag_samples": len(lags),
            "lag_ms": None,
        }
        if lags:
            out["lag_ms"] = {
                "p50": round(_percentile(lags, 0.5) * 1000, 2),
                "p95": round(_percentile(lags, 0.95) * 1000, 2),
                "max": round(lags[-1] * 1000, 2),
            }
        return out
