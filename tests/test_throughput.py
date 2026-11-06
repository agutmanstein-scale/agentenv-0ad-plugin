"""CPU throughput of a training box: how many worker processes drive the engines."""

import asyncio
import json

import pytest

from agentenv_zeroad.engine import PORT_STOP
from agentenv_zeroad.pool import TRAIN_IMAGE, TRAIN_PATCHES
from agentenv_zeroad.train import runner
from tests.test_train_runner import spec_json


# ---- worker processes ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "engines, workers, per_worker, expected",
    [
        (44, None, 4, 11),
        (40, None, 2, 20),
        (40, None, 11, 4),
        (5, None, 4, 2),
        (1, None, 4, 1),
        (3, 10, 4, 3),  # never more workers than engines
        (44, 4, 4, 4),  # an explicit --workers wins over --engines-per-worker
    ],
)
def test_workers_are_one_per_engines_per_worker_unless_given_and_never_more_than_engines(
    engines, workers, per_worker, expected
):
    assert runner.worker_count(engines, workers=workers, per_worker=per_worker) == expected


@pytest.mark.parametrize("workers, per_worker", [(0, 4), (None, 0), (-1, 4)])
def test_a_worker_count_of_zero_is_refused(workers, per_worker):
    with pytest.raises(ValueError, match="at least 1"):
        runner.worker_count(8, workers=workers, per_worker=per_worker)


def test_the_default_gives_each_worker_four_engines():
    assert runner.ENGINES_PER_WORKER == 4
    args = runner._parser().parse_args(["--specs", "s", "--out", "o", "--engines", "44"])
    assert (args.workers, args.concurrency, args.engines_per_worker) == (None, None, 4)


async def supervise(tmp_path, monkeypatch, argv, specs=12):
    path = tmp_path / "specs.jsonl"
    path.write_text("\n".join(json.dumps(spec_json(f"g{i}")) for i in range(specs)) + "\n")
    started = []

    class Child:
        async def wait(self):
            return 0

    async def fake_exec(*cmd, env):
        started.append((cmd, env))
        return Child()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    args = runner._parser().parse_args(["--specs", str(path), "--out", str(tmp_path / "out"), *argv])
    assert await runner.supervise(args) == 0
    return started


@pytest.mark.anyio
async def test_the_supervisor_starts_a_worker_per_four_engines_with_each_workers_engines_as_its_concurrency(
    tmp_path, monkeypatch
):
    started = await supervise(tmp_path, monkeypatch, ["--engines", "10"])
    engines = [c[c.index("--engines") + 1] for c, _ in started]
    assert engines == ["4", "3", "3"]
    for cmd, env in started:
        assert env["PYTHONHASHSEED"] == "0" and "--concurrency" not in cmd  # the worker plays on all of its engines


@pytest.mark.anyio
async def test_engines_per_worker_sets_the_worker_count_and_an_explicit_concurrency_is_passed_on(tmp_path, monkeypatch):
    started = await supervise(
        tmp_path, monkeypatch, ["--engines", "8", "--engines-per-worker", "2", "--concurrency", "1"]
    )
    assert len(started) == 4
    assert {c[c.index("--concurrency") + 1] for c, _ in started} == {"1"}
    ports = [int(c[c.index("--base-port") + 1]) for c, _ in started]
    assert ports == [16_000, 16_100, 16_200, 16_300]


@pytest.mark.anyio
async def test_a_worker_plays_on_all_its_engines_when_no_concurrency_is_given(tmp_path, monkeypatch):
    seen = {}

    class FakePool:
        def __init__(self, size, **options):
            seen["size"] = size

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    async def no_games(specs, pool, out_dir, *, concurrency, **kwargs):
        seen["concurrency"] = concurrency
        return []

    monkeypatch.setattr(runner, "EnginePool", FakePool)
    monkeypatch.setattr(runner, "run_games", no_games)
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(spec_json()) + "\n")
    args = runner._parser().parse_args(["--specs", str(path), "--out", str(tmp_path), "--engines", "3", "--worker"])
    await runner.work(args)
    assert seen == {"size": 3, "concurrency": 3}


# ---- ports: every worker's range stays below the OS's ephemeral ports ----------------------------------------------
def test_the_port_stride_keeps_the_biggest_calibration_below_the_ephemeral_range():
    assert runner.PORT_STRIDE == 100
    workers = runner.worker_count(400, per_worker=4)  # a 400-engine box is 100 workers, far past any real one
    runner.check_ports(runner.BASE_PORT, workers, 4)
    assert runner.BASE_PORT + workers * runner.PORT_STRIDE <= PORT_STOP


def test_too_many_workers_or_engines_for_the_port_ranges_fail_loudly():
    with pytest.raises(SystemExit, match="ephemeral"):
        runner.check_ports(30_000, 40, 4)  # 30,000 + 40 * 100 > 32,768
    with pytest.raises(SystemExit, match="its port range holds 100"):
        runner.check_ports(16_000, 1, 101)


@pytest.mark.anyio
async def test_a_supervisor_that_would_pass_the_ephemeral_range_starts_nothing(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="ephemeral"):
        await supervise(tmp_path, monkeypatch, ["--engines", "40", "--base-port", "32000"])


# ---- the image ----------------------------------------------------------------------------------------------------
def test_the_runner_defaults_to_the_patched_training_image():
    args = runner._parser().parse_args(["--specs", "s", "--out", "o"])
    assert (args.image, args.allow_unpatched) == (TRAIN_IMAGE, False)


@pytest.mark.anyio
async def test_the_supervisor_passes_the_image_and_the_unpatched_opt_in_to_every_worker(tmp_path, monkeypatch):
    started = await supervise(tmp_path, monkeypatch, ["--engines", "8"])
    assert {c[c.index("--image") + 1] for c, _ in started} == {TRAIN_IMAGE}
    assert all("--allow-unpatched" not in c for c, _ in started)
    started = await supervise(
        tmp_path, monkeypatch, ["--engines", "8", "--image", "zeroad-engine:0.28", "--allow-unpatched"]
    )
    assert all("--allow-unpatched" in c and "zeroad-engine:0.28" in c for c, _ in started)


@pytest.mark.anyio
async def test_a_worker_makes_its_pool_require_the_patches_unless_told_not_to(tmp_path, monkeypatch):
    seen = []

    class FakePool:
        def __init__(self, size, **options):
            seen.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    async def no_games(*args, **kwargs):
        return []

    monkeypatch.setattr(runner, "EnginePool", FakePool)
    monkeypatch.setattr(runner, "run_games", no_games)
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(spec_json()) + "\n")
    base = ["--specs", str(path), "--out", str(tmp_path), "--worker"]
    await runner.work(runner._parser().parse_args(base))
    await runner.work(runner._parser().parse_args([*base, "--allow-unpatched"]))
    assert [o["require_patches"] for o in seen] == [TRAIN_PATCHES, ()]
    assert all(o["image"] == TRAIN_IMAGE and o["port_stop"] == o["base_port"] + runner.PORT_STRIDE for o in seen)
