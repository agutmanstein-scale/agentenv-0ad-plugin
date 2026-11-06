"""Default pool sizing: a box's physical cores, and the run config's engines, concurrency and engine image."""

from pathlib import Path

import pytest

from agentenv_zeroad import pool
from agentenv_zeroad.pool import TRAIN_IMAGE, physical_cores
from agentenv_zeroad.train import config as train_config
from agentenv_zeroad.train.config import RunConfig
from tests.test_train_orchestrate import raw_config


def fake_cpus(root: Path, topology: dict[int, tuple[int, int]]) -> Path:
    for cpu, (package, core) in topology.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "physical_package_id").write_text(f"{package}\n")
        (d / "core_id").write_text(f"{core}\n")
    (root / "cpufreq").mkdir()  # not a cpu
    return root


def test_hyperthreads_count_once_and_so_do_cores_of_one_id_in_two_sockets_count_twice(tmp_path):
    ht = fake_cpus(tmp_path / "ht", {0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (0, 1)})
    assert physical_cores(ht) == 2
    sockets = fake_cpus(tmp_path / "sockets", {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)})
    assert physical_cores(sockets) == 4


def test_without_a_topology_the_cores_are_half_the_logical_cpus_and_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.os, "cpu_count", lambda: 48)
    assert physical_cores(tmp_path / "missing") == 24
    monkeypatch.setattr(pool.os, "cpu_count", lambda: 1)
    assert physical_cores(tmp_path / "missing") == 1


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
