"""Default pool sizing: a box's physical cores, and the run config's engines, concurrency and engine image."""

from pathlib import Path

import pytest

from agentenv_zeroad import pool
from agentenv_zeroad.pool import TRAIN_IMAGE, physical_cores
from agentenv_zeroad.train import config as train_config
from agentenv_zeroad.train.config import RunConfig
from tests.test_train_orchestrate import raw_config


@pytest.fixture(autouse=True)
def all_cpus_allowed(monkeypatch):
    """The affinity mask of the machine running the tests must not decide these tests' counts."""
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)


def fake_cpus(root: Path, topology: dict[int, str]) -> Path:
    """cpu N -> the `core_cpus_list` of its core (the CPUs that share it)."""
    for cpu, siblings in topology.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "core_cpus_list").write_text(f"{siblings}\n")
    (root / "cpufreq").mkdir()  # not a cpu
    return root


def test_hyperthreads_count_once_by_their_shared_core_list(tmp_path):
    ht = fake_cpus(tmp_path / "ht", {0: "0,2", 1: "1,3", 2: "0,2", 3: "1,3"})
    assert physical_cores(ht) == 2
    two_sockets = fake_cpus(tmp_path / "sockets", {0: "0", 1: "1", 2: "2", 3: "3"})  # core ids repeat, lists don't
    assert physical_cores(two_sockets) == 4


def test_an_older_topology_without_a_core_list_falls_back_to_package_and_core_id(tmp_path):
    for cpu, (package, core) in {0: (0, 0), 1: (1, 0), 2: (0, 0), 3: (1, 0)}.items():
        d = tmp_path / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "physical_package_id").write_text(f"{package}\n")
        (d / "core_id").write_text(f"{core}\n")
    assert physical_cores(tmp_path) == 2


def test_only_the_cpus_this_process_may_run_on_are_counted(tmp_path, monkeypatch):
    topology = {i: f"{i % 4},{i % 4 + 4}" for i in range(8)}  # 4 cores, 2 threads each
    root = fake_cpus(tmp_path, topology)
    assert physical_cores(root) == 4
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: {0, 1, 4, 5}, raising=False)  # a 2-core cpuset
    assert physical_cores(root) == 2
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert physical_cores(root) == 1


def test_without_a_topology_the_cores_are_half_the_allowed_cpus_and_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    assert physical_cores(tmp_path / "missing") == 24
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert physical_cores(tmp_path / "missing") == 1
    monkeypatch.delattr(pool.os, "sched_getaffinity", raising=False)  # macOS has none: fall back to the cpu count
    monkeypatch.setattr(pool.os, "cpu_count", lambda: 8)
    assert physical_cores(tmp_path / "missing") == 4


def test_a_run_config_without_engines_gets_the_physical_cores_and_runs_every_engine_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(train_config, "physical_cores", lambda: 24)
    raw = raw_config(tmp_path)
    del raw["engines"], raw["concurrency"]
    cfg = RunConfig.from_dict(raw)
    assert (cfg.engines, cfg.concurrency, cfg.engine_image) == (24, 24, TRAIN_IMAGE)
    cfg = RunConfig.from_dict({**raw, "concurrency": 6})
    assert (cfg.engines, cfg.concurrency) == (24, 6)
    cfg = RunConfig.from_dict({**raw, "engines": 40, "engine_image": "zeroad-engine:0.28"})
    assert (cfg.engines, cfg.concurrency, cfg.engine_image) == (40, 40, "zeroad-engine:0.28")


def test_engines_and_concurrency_cannot_be_negative_and_concurrency_cannot_pass_engines(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        RunConfig.from_dict(raw_config(tmp_path, engines=-1))
    with pytest.raises(ValueError, match="concurrency must be from 1"):
        RunConfig.from_dict(raw_config(tmp_path, engines=2, concurrency=3))


def test_the_config_check_prints_the_engines_and_image_the_box_script_reports(tmp_path, capsys, monkeypatch):
    cfg = RunConfig.from_dict(raw_config(tmp_path, engines=7, concurrency=5))
    monkeypatch.setattr(train_config, "load_config", lambda path: cfg)
    monkeypatch.setattr(train_config, "check_run", lambda cfg: [])
    assert train_config.main(["run.toml"]) == 0
    out = capsys.readouterr().out
    assert "TRAIN_ENGINES=7" in out and "TRAIN_CONCURRENCY=5" in out and f"TRAIN_ENGINE_IMAGE={TRAIN_IMAGE}" in out
    assert "TRAIN_ENGINE_PATCHES='rl-realtime rl-idle-wait'" in out
