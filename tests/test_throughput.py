"""CPU throughput of a training box: how many worker processes drive the engines."""

import asyncio
import json

import pytest

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
    assert ports == [16_000, 17_000, 18_000, 19_000]


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
