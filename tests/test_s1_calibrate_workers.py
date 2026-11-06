"""The calibration's levels of engines x engines per worker, its loop-lag and Python-cost meters, and idle engines."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentenv_zeroad import pool, s1_calibrate
from agentenv_zeroad.s1_batch import calibration_games
from agentenv_zeroad.s1_gates import format_calibration
from agentenv_zeroad.train.runner import GameOutcome
from tests.test_s1_batch import fake_cgroups
from tests.zeroad_fakes import FakeStarter, ScriptedEngine, placer

pytestmark = pytest.mark.anyio


def test_a_level_is_run_once_per_distinct_share_and_a_share_over_its_engines_is_one_loop():
    cells = s1_calibrate.plan_cells([20, 40], [2, 4, 11])
    assert cells == [(20, 2), (20, 4), (20, 11), (40, 2), (40, 4), (40, 11)]
    assert s1_calibrate.plan_cells([1, 20], [4, 11]) == [(1, 1), (20, 4), (20, 11)]
    assert s1_calibrate.plan_cells([5], [8, 9, 5]) == [(5, 5)]  # three shares all mean one loop of five
    assert s1_calibrate.plan_cells([1, 20], []) == [(1, None), (20, None)]  # the old, single-loop levels
    assert s1_calibrate.level_key(20, 4) == "20x4" and s1_calibrate.level_key(20, None) == "20"


async def test_the_loop_lag_meter_sees_a_blocked_loop_and_nothing_on_an_idle_one():
    lag = s1_calibrate.LoopLag()
    lag.start()
    await asyncio.sleep(0.2)
    time.sleep(0.25)  # a loop doing synchronous work, as a worker serialising JSON for its engines does
    await asyncio.sleep(0.1)
    await lag.stop()
    summary = s1_calibrate.lag_summary(lag.samples_ms)
    assert summary is not None and summary["max"] > 150 and summary["median"] < 50
    assert s1_calibrate.lag_summary([]) is None


def test_worker_results_merge_into_one_timer_one_outcome_list_and_a_summed_python_cost():
    def result(minute, values, game_id, *, timed_out=False):
        outcome = {"game_id": game_id, "status": "finished", "game_time_s": 60.0}
        return {
            "timed_out": timed_out, "wall_s": 5.0, "python_cpu_s": 2.5, "outcomes": [outcome],
            "steps_ms": {str(minute): values}, "lag_ms": [1.0, 2.0],
        }  # fmt: skip

    timer, outcomes, timed_out, python_s, lag = s1_calibrate.merge_results(
        [result(0, [10.0, 20.0], "a"), result(0, [30.0], "b", timed_out=True), result(1, [40.0], "c")]
    )
    assert sorted(timer.by_minute) == [0, 1] and timer.by_minute[0] == [10.0, 20.0, 30.0]
    assert [o.game_id for o in outcomes] == ["a", "b", "c"] and all(isinstance(o, GameOutcome) for o in outcomes)
    assert timed_out is True and python_s == 7.5 and lag == [1.0, 2.0] * 3


def short_games(monkeypatch, count):
    games = [g.__class__(**{**g.__dict__, "time_cap_minutes": 0.5}) for g in calibration_games(count)]
    monkeypatch.setattr(s1_calibrate, "calibration_games", lambda total: games[:total])


class FakeProc:
    """A worker process that runs the calibration's own worker function on this loop."""

    returncode = None

    def __init__(self, cmd, match_options):
        flags = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        self.flags = flags
        self.task = asyncio.create_task(
            s1_calibrate.cell_worker(
                int(flags["--first-game"]), int(flags["--engines"]), int(flags["--games-total"]), Path(flags["--out"]),
                Path(flags["--sync-dir"]), int(flags["--worker-index"]), mode="local", image=flags["--image"],
                base_port=int(flags["--base-port"]), timeout_s=float(flags["--level-timeout-s"]),
                no_state_step="--no-state-step" in cmd, sampled_wait_s=10.0, match_options=match_options,
            )
        )  # fmt: skip

    async def wait(self):
        await self.task
        self.returncode = 0
        return 0

    def kill(self):
        self.task.cancel()


async def test_a_level_in_workers_splits_games_and_engines_and_merges_the_workers_results(tmp_path, monkeypatch):
    short_games(monkeypatch, 5)
    started = []

    events = []

    class Launcher:
        async def stop(self) -> None:
            events.append("stop")

        async def clear_replays(self) -> None: ...
        async def copy_replay(self, dest):
            return None

    async def fake_start_engine(options, **kwargs):
        return SimpleNamespace(engine=ScriptedEngine(), launcher=Launcher())

    async def fake_exec(*cmd, env):
        assert env["PYTHONHASHSEED"] == "0"
        proc = FakeProc(list(cmd), {"placer": placer()})
        started.append(proc.flags)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(s1_calibrate, "start_engine", fake_start_engine)
    inner = fake_cgroups(tmp_path, {"e0": (1_000_000, 600 * 2**20), "e1": (1_000_000, 650 * 2**20)})

    class Sampler:
        async def sample(self):
            events.append("sample")
            return await inner.sample()

    stale = tmp_path / "level-5x2" / ".sync"
    stale.mkdir(parents=True)
    for name in ("go", "done-0", "sampled-1", "result-2.json", "stale"):  # a rerun of the cell must not see these
        (stale / name).write_text("{}")
    result = await s1_calibrate.run_level(
        5, tmp_path / "level-5x2", per_worker=2, sampler=Sampler(), mode="local", base_port=24_000, sample_s=0.05
    )
    assert not (stale / "stale").exists()
    assert "sample" in events and "stop" in events
    assert max(i for i, e in enumerate(events) if e == "sample") < min(i for i, e in enumerate(events) if e == "stop")
    # the last CPU sample is taken while every worker's engines are still up
    assert [(f["--first-game"], f["--engines"], f["--base-port"]) for f in started] == [
        ("0", "2", "24000"), ("2", "2", "24100"), ("4", "1", "24200"),
    ]  # fmt: skip
    assert (result["engines"], result["workers"], result["engines_per_worker"]) == (5, 3, 2)
    assert result["games"] == result["games_finished"] == 5 and result["timed_out"] is False
    assert result["step_ms_by_minute"]["0"]["steps"] > 0 and result["loop_lag_ms"] is not None
    assert result["python_core_s_per_game"] is not None and result["rss_mb"]["max"] == 650.0
    rows = result["workers_detail"]
    assert [(r["index"], r["engines"]) for r in rows] == [(0, 2), (1, 2), (2, 1)] and result["error"] is None
    assert result["python_cores_busy"] == max(r["python_cores_busy"] for r in rows)
    assert result["python_core_s_per_game"] == pytest.approx(sum(r["python_cpu_s"] for r in rows) / 5, abs=0.1)
    assert result["loop_lag_p95_ms_worst_worker"] == max(r["loop_lag_p95_ms"] for r in rows)
    for i in range(5):
        assert (tmp_path / "level-5x2" / f"calib-05-{i:02d}" / "results.json").is_file()


class Dead:
    returncode = 1

    async def wait(self):
        return 1

    def kill(self):
        return None


async def test_a_worker_that_dies_before_its_engines_are_ready_fails_that_cell_with_its_error(tmp_path, monkeypatch):
    async def fake_exec(*cmd, env):
        return Dead()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    sampler = fake_cgroups(tmp_path, {"e0": (1, 1)})
    cell = await s1_calibrate.run_level(4, tmp_path / "l", per_worker=2, sampler=sampler, mode="local", sample_s=0.01)
    assert "exited before its engines were ready" in cell["error"]
    assert (cell["games_finished"], cell["workers"], cell["engines_per_worker"]) == (0, 2, 2)


async def test_workers_that_never_get_their_engines_up_fail_the_cell_after_the_ready_timeout(tmp_path, monkeypatch):
    class Hung(Dead):
        returncode = None

        def __init__(self):
            self.dead = asyncio.Event()

        async def wait(self):
            await self.dead.wait()
            return -9

        def kill(self):
            self.returncode = -9
            self.dead.set()

    async def fake_exec(*cmd, env):
        return Hung()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    cell = await s1_calibrate.run_level(
        4, tmp_path / "l", per_worker=2, sampler=fake_cgroups(tmp_path, {"e0": (1, 1)}), mode="local",
        sample_s=0.01, ready_timeout_s=0.3,
    )  # fmt: skip
    assert "not ready in 0 s" in cell["error"]


async def test_a_cell_that_raises_is_an_error_row_and_the_calibration_goes_on_to_the_next(tmp_path, monkeypatch):
    calls = []

    async def fake_level(count, out, *, per_worker=None, **kw):
        calls.append((count, per_worker))
        if count == 12:
            raise RuntimeError("no free ports")
        return {"engines": count, "games": count, "games_finished": count, "error": None}

    monkeypatch.setattr(s1_calibrate, "run_level", fake_level)
    report = await s1_calibrate.calibrate(
        [12, 24], tmp_path, per_worker=[4, 24], mode="local", baseline_image=None, require_patches=()
    )
    assert calls == [(12, 4), (12, 12), (24, 4), (24, 24)]
    assert "RuntimeError: no free ports" in report["levels"]["12x4"]["error"]
    assert report["levels"]["24x24"]["games_finished"] == 24
    assert json.loads((tmp_path / "calibration.json").read_text())["levels"].keys() == report["levels"].keys()


async def test_idle_engines_are_measured_on_the_ppa_image_and_the_patched_one_in_one_calibration(tmp_path, monkeypatch):
    seen = []

    async def fake_idle(count, *, image, require_patches=(), **kw):
        seen.append((count, image, tuple(require_patches)))
        return {"image": image, "engines": count, "cores": 0.0 if "rt" in image else float(count)}

    monkeypatch.setattr(s1_calibrate, "run_idle", fake_idle)
    report = await s1_calibrate.calibrate(
        [],
        tmp_path,
        idle_engines=[1, 20],
        mode="local",
        image="zeroad-engine:0.28-rt",
        require_patches=("rl-idle-wait",),
    )
    assert seen == [
        (1, "zeroad-engine:0.28", ()), (1, "zeroad-engine:0.28-rt", ("rl-idle-wait",)),
        (20, "zeroad-engine:0.28", ()), (20, "zeroad-engine:0.28-rt", ("rl-idle-wait",)),
    ]  # fmt: skip
    assert report["idle_baseline"]["20"]["cores"] == 20.0 and report["idle"]["20"]["cores"] == 0.0
    assert report["baseline_image"] == "zeroad-engine:0.28"
    only = await s1_calibrate.calibrate(
        [], tmp_path, idle_engines=[1], mode="local", image="zeroad-engine:0.28-rt", baseline_image=""
    )
    assert only["idle_baseline"] == {} and only["baseline_image"] is None


async def test_the_calibration_refuses_an_image_without_the_patches_before_measuring_anything(tmp_path, monkeypatch):
    async def patches(image):
        return () if image == "zeroad-engine:0.28-rt" else None

    monkeypatch.setattr(s1_calibrate, "image_patches", patches)
    with pytest.raises(s1_calibrate.EngineError, match="lacks the patches rl-realtime, rl-idle-wait"):
        await s1_calibrate.calibrate([12], tmp_path, mode="docker", image="zeroad-engine:0.28-rt")
    with pytest.raises(s1_calibrate.EngineError, match="not built here"):
        await s1_calibrate.calibrate([12], tmp_path, mode="docker", image="nope:1")
    assert await s1_calibrate.check_image("zeroad-engine:0.28-rt", mode="docker", require=()) is False
    monkeypatch.setattr(s1_calibrate, "image_patches", lambda image: _async(("rl-realtime", "rl-idle-wait")))
    assert await s1_calibrate.check_image("x", mode="docker", require=("rl-idle-wait",)) is True


async def _async(value):
    return value


async def test_the_parity_game_is_the_same_spec_on_each_image_and_reports_matching_games(tmp_path, monkeypatch):
    short_games(monkeypatch, 1)
    starter = FakeStarter()

    async def scripted(options, **kw):
        running = await starter(options, kw.get("home"))
        running.engine = ScriptedEngine()
        return running

    monkeypatch.setattr(pool, "start_engine", scripted)
    parity = await s1_calibrate.run_parity(
        tmp_path, ["zeroad-engine:0.28", "zeroad-engine:0.28-rt"], minutes=0.5, mode="local", base_port=24_600,
        match_options={"placer": placer()},
    )  # fmt: skip
    assert parity["errors"] == [] and parity["same"] is True and parity["differences"] == []
    first, second = parity["games"]
    assert first["image"] == "zeroad-engine:0.28" and second["image"] == "zeroad-engine:0.28-rt"
    assert first["commands_sha256"] == second["commands_sha256"]
    assert set(first["seats"]) == {"1", "2"} and first["status"] == "finished"


def test_parity_differs_on_any_command_outcome_score_or_length_and_a_failed_image_is_not_a_match():
    row = {
        "image": "a", "status": "finished", "ended_by": "time_cap", "game_time_s": 30.0,
        "seats": {"1": {"outcome": "won", "score": 5}}, "commands": 10, "commands_sha256": "x",
    }  # fmt: skip
    assert s1_calibrate.compare_parity([row, {**row, "image": "b"}])["same"] is True
    for field, value in (
        ("commands_sha256", "y"), ("game_time_s", 31.0), ("seats", {"1": {"outcome": "won", "score": 6}}),
        ("ended_by", "conquest"),
    ):  # fmt: skip
        verdict = s1_calibrate.compare_parity([row, {**row, "image": "b", field: value}])
        assert verdict["same"] is False and verdict["differences"] == [field]
    failed = s1_calibrate.compare_parity([row, {"image": "b", "error": "EngineError: boom"}])
    assert failed["same"] is False and failed["errors"] == ["b"]
    assert s1_calibrate.compare_parity([row])["same"] is False  # nothing to compare with


class Burning:
    """A sampler whose engines have each used `cpu_per_call` more CPU seconds by every sample."""

    def __init__(self, names, cpu_per_call):
        self.names, self.cpu_per_call, self.calls = names, cpu_per_call, 0

    async def sample(self):
        self.calls += 1
        us = int(self.calls * self.cpu_per_call * 1e6)
        return {n: s1_calibrate.Reading(us, 0, 0) for n in self.names}


async def test_idle_engines_get_a_game_and_one_step_then_their_cpu_is_read_over_the_silent_window(monkeypatch):
    starter = FakeStarter()
    monkeypatch.setattr(pool, "start_engine", lambda options, **kw: starter(options, kw.get("home")))
    sampler = Burning(["a", "b"], cpu_per_call=0.5)  # 0.5 core-s each over the window of 0.2 s: 5 cores for 2 engines
    result = await s1_calibrate.run_idle(2, idle_s=0.2, sampler=sampler, mode="local", settle_s=0.0, base_port=24_300)
    assert [r.engine.resets for r in starter.started] == [1, 1] and [r.engine.raw_steps for r in starter.started] == [
        1,
        1,
    ]
    assert (
        result["engines"] == 2
        and 4 < result["cores"] < 6.5
        and result["cores_per_engine"] == pytest.approx(result["cores"] / 2, abs=0.01)
    )
    quiet = await s1_calibrate.run_idle(
        2, idle_s=0.1, sampler=Burning(["a"], 0.0), mode="local", settle_s=0.0, base_port=24_400
    )
    assert quiet["cores"] == 0.0


def idle(image, engines, cores):
    return {"image": image, "engines": engines, "idle_s": 10.0, "cores": cores, "cores_per_engine": cores / engines}


def parity_game(image, commands, digest):
    return {
        "image": image, "commands": commands, "commands_sha256": digest, "ended_by": "time_cap",
        "game_time_s": 180.0, "seats": {"1": {"outcome": "won", "score": 9}},
    }  # fmt: skip


def test_the_report_lists_idle_engines_on_both_images_then_parity_then_levels_with_the_python_cost():
    calibration = {
        "host": {"vcpus": 48, "mem_total_mb": 96000},
        "image": "zeroad-engine:0.28-rt",
        "idle_baseline": {"1": idle("zeroad-engine:0.28", 1, 1.01), "20": idle("zeroad-engine:0.28", 20, 20.2)},
        "idle": {
            "20": idle("zeroad-engine:0.28-rt", 20, 0.4),
            "1": {"error": "EngineError: boom", "image": "zeroad-engine:0.28-rt"},
        },
        "parity": {
            "minutes": 3.0, "same": False, "differences": ["commands_sha256"], "errors": [],
            "games": [
                parity_game("zeroad-engine:0.28", 40, "aaaaaaaaaaaa1"),
                parity_game("zeroad-engine:0.28-rt", 41, "bbbbbbbbbbbb2"),
            ],
        },
        "levels": {
            "20": {"games": 20, "games_finished": 20, "wall_s": 400.0, "core_s_per_game": 100.0, "mean_cores": 5.0},
            "20x4": {"workers": 5, "games": 20, "games_finished": 20, "wall_s": 90.0, "core_s_per_game": 60.0,
                     "mean_cores": 13.3, "python_core_s_per_game": 4.5, "python_cores_busy": 0.93,
                     "python_cores_busy_total": 4.6, "loop_lag_p95_ms_worst_worker": 7.0,
                     "loop_lag_ms": {"median": 0.5, "p95": 3.0, "max": 40.0},
                     "step_ms_by_minute": {"0": {"median": 5.0, "p95": 9.0}}},
            "20x2": {"workers": 10, "games": 20, "games_finished": 0, "wall_s": 0.0, "error": "a worker exited"},
            "1": {"games": 1, "games_finished": 1, "wall_s": 76.0, "core_s_per_game": 40.0, "mean_cores": 0.5},
        },
    }  # fmt: skip
    text = format_calibration(calibration)
    assert "engine zeroad-engine:0.28-rt" in text
    spin, fix = (
        text.index("20 idle engines on zeroad-engine:0.28,"),
        text.index("20 idle engines on zeroad-engine:0.28-rt,"),
    )
    assert spin < fix < text.index("parity game") < text.index(" 1 engines:")
    assert "1.010 per engine" in text and "0.020 per engine" in text and "1.010 per engine" in text
    assert " 1 idle engines on zeroad-engine:0.28-rt: FAILED: EngineError: boom" in text
    assert "DIFFERS in commands_sha256" in text and "40 commands (aaaaaaaaaaaa)" in text
    order = [
        text.index(x)
        for x in (" 1 engines:", "20 engines in 10 workers of 2:", "20 engines in 5 workers of 4:", "20 engines: 20/20")
    ]
    assert order == sorted(order)
    assert "FAILED: a worker exited" in text
    assert (
        "python 4.5 core-s per game, busiest worker 0.93 cores (4.60 in all), "
        "loop lag ms median 0.5 p95 3.0 max 40.0 (worst worker p95 7.0)"
    ) in text
    assert "step ms (median/p95) min 0: 5/9" in text
    assert "identical commands" in format_calibration(
        {"parity": {"minutes": 3, "same": True, "games": [{"image": "a"}, {"image": "b"}]}, "levels": {}}
    )
