"""The process CPU and event-loop lag the orchestrator (and the calibration's workers) report."""

import asyncio
import time

import pytest

from agentenv_zeroad.train.loopstats import LoopMonitor, process_cpu_s

pytestmark = pytest.mark.anyio


def test_process_cpu_reads_utime_and_stime_from_proc_stat(tmp_path, monkeypatch):
    stat = tmp_path / "stat"
    # pid (comm with a space and a paren) state ppid ... utime=300 stime=150 (fields 14, 15)
    stat.write_text(
        "42 (py (x)) S 1 42 42 0 -1 4194304 100 0 0 0 300 150 0 0 20 0 1 0 5 1000 100 18446744073709551615\n"
    )
    monkeypatch.setattr("os.sysconf", lambda name: 100)
    assert process_cpu_s(stat) == 4.5


def test_process_cpu_falls_back_to_process_time_without_proc(tmp_path):
    assert process_cpu_s(tmp_path / "missing") >= 0.0


async def test_the_monitor_reports_how_late_a_blocked_loop_wakes():
    monitor = LoopMonitor(interval=0.01)
    monitor.start()
    await asyncio.sleep(0.05)
    end = time.perf_counter() + 0.15
    while time.perf_counter() < end:  # a blocked loop (busy): the sleeper wakes ~150 ms late
        pass
    await asyncio.sleep(0.05)
    snap = monitor.snapshot()
    await monitor.stop()
    assert snap["lag_samples"] >= 3 and snap["lag_ms"]["max"] >= 100
    assert snap["lag_ms"]["p50"] <= snap["lag_ms"]["p95"] <= snap["lag_ms"]["max"]
    assert snap["cpu_s"] >= 0.1  # the busy-wait above


async def test_a_snapshot_covers_the_time_since_the_last_one():
    times = iter([0.0, 10.0, 20.0])
    cpu = iter([5.0, 7.0, 7.5])
    monitor = LoopMonitor(clock=lambda: next(times), cpu=lambda: next(cpu))
    monitor._lags = [0.001, 0.002, 0.003, 0.004]
    first = monitor.snapshot()
    assert first["cpu_s"] == 2.0 and first["cpu_cores"] == 0.2 and first["lag_samples"] == 4
    assert first["lag_ms"] == {"p50": 3.0, "p95": 4.0, "max": 4.0}
    second = monitor.snapshot()
    assert second["cpu_s"] == 2.5 and second["cpu_cores"] == 0.05
    assert second["lag_ms"] is None and second["lag_samples"] == 0  # nothing sampled since the first
