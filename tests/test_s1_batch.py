"""The S1 batch's games (bot and replay specs, the inputs package), its gates and its CPU calibration's measuring."""

import json
import tarfile
from math import comb
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentenv_zeroad import s1_calibrate
from agentenv_zeroad.engine import TURN_MS, FakeEngine
from agentenv_zeroad.replay import ReplaySpec
from agentenv_zeroad.roles import CIVS
from agentenv_zeroad.s1 import GameMetrics, game_metrics, load_games, load_metrics
from agentenv_zeroad.s1_batch import (
    ARMS,
    BOT_SEED_BASE,
    bot_games,
    build,
    calibration_games,
    package_inputs,
    replay_games,
    write_specs,
)
from agentenv_zeroad.s1_batch import main as batch_main
from agentenv_zeroad.s1_gates import evaluate, fisher_lower, format_calibration, summary
from agentenv_zeroad.train.spec import load_specs
from tests.script_loader import load_script
from tests.test_s1 import game, results, write_pilot_game
from tests.zeroad_fakes import ScriptedEngine, placer

pytestmark = pytest.mark.anyio


# ---- the bot games -------------------------------------------------------------------------------------------------
def test_the_bot_batch_is_n_games_per_interface_and_petra_level_with_petra_always_on_v1():
    games = bot_games(5)
    assert len(games) == 5 * 3 * 2
    cells = {(g.game_id.split("-")[1], g.game_id.split("-")[2]) for g in games}
    assert cells == {(arm, f"petra{level}") for arm in ARMS for level in (1, 2, 3)}
    for g in games:
        bot = next(s for s in g.seats if s.kind == "bot")
        petra = next(s for s in g.seats if s.kind == "petra")
        assert petra.interface == "v1" and bot.interface == g.game_id.split("-")[1]
        assert petra.difficulty == int(g.game_id.split("-")[2].removeprefix("petra"))
        assert {bot.civ, petra.civ} <= set(CIVS)
        assert (g.map, g.time_cap_minutes, g.size) == ("mainland", 30.0, 128)
    assert len({g.game_id for g in games}) == len(games)


def test_the_two_interfaces_play_the_same_seed_civs_and_start_so_only_the_interface_differs():
    by_id = {g.game_id: g for g in bot_games(6)}
    for i in range(6):
        for level in (1, 2, 3):
            v1, v2 = by_id[f"bot-v1-petra{level}-{i:03d}"], by_id[f"bot-v2-petra{level}-{i:03d}"]
            assert (v1.seed, v1.ai_seed) == (v2.seed, v2.ai_seed) == (BOT_SEED_BASE + i,) * 2
            assert [(s.kind, s.civ, s.difficulty) for s in v1.seats] == [
                (s.kind, s.civ, s.difficulty) for s in v2.seats
            ]
            assert v1.seats[0].kind == ("bot" if i % 2 == 0 else "petra")  # the bot alternates its start
        assert by_id[f"bot-v1-petra1-{i:03d}"].seats[0].civ == by_id[f"bot-v1-petra1-{i:03d}"].seats[0].civ
    assert [g.game_id for g in bot_games(6)] == [g.game_id for g in bot_games(6)]  # deterministic


def test_the_bot_specs_are_ordered_by_game_so_a_batch_cut_short_keeps_balanced_cells():
    ids = [g.game_id for g in bot_games(4)]
    assert ids[:6] == [f"bot-{arm}-petra{lv}-000" for lv in (1, 2, 3) for arm in ARMS]
    assert [int(i.rsplit("-", 1)[1]) for i in ids] == sorted(int(i.rsplit("-", 1)[1]) for i in ids)


def test_bot_specs_are_written_as_json_lines_the_runner_loads(tmp_path):
    write_specs(bot_games(2), tmp_path / "bot.jsonl")
    loaded = load_specs(tmp_path / "bot.jsonl")
    assert loaded == bot_games(2)
    assert all(s.kind in ("bot", "petra") for g in loaded for s in g.seats)  # no model seat anywhere


def test_the_calibration_games_are_one_v2_bot_game_per_engine_with_their_own_seeds():
    games = calibration_games(20)
    assert len(games) == 20 and len({g.seed for g in games}) == 20 and len({g.game_id for g in games}) == 20
    assert all(g.time_cap_minutes == 30.0 and {s.kind for s in g.seats} == {"bot", "petra"} for g in games)
    assert all(next(s for s in g.seats if s.kind == "bot").interface == "v2" for g in games)


# ---- the replay games and the inputs package ---------------------------------------------------------------------
def write_game(root: Path, box: str, run: str, game_id: str) -> None:
    write_pilot_game(root, box, run, game_id, level=None, turns=20)
    path = next((root / box / run).rglob(f"games/{game_id}/**/*results.json"))
    raw = json.loads(path.read_text())
    raw["seats"]["1"]["civ"], raw["seats"]["2"]["civ"] = "rome", "pers"
    path.write_text(json.dumps(raw))


def pilot_with_games(root: Path, count: int = 3) -> Path:
    """A pilot results folder: games r00-t0N with Petra's level known only from the round's specs.json."""
    rows = []
    for n in range(count):
        write_game(root, "a", "pilot-1", f"r00-t0{n}")
        rows.append({"game_id": f"r00-t0{n}", "opponent": {"kind": "petra", "level": 1 + n % 3}})
    (root / "a" / "pilot-1" / "rounds" / "r00" / "specs.json").write_text(json.dumps(rows))
    return root


def picked_specs(root: Path, out: Path, n: int = 3) -> Path:
    assert (
        load_script("s1_specs").main(
            [str(root), "--out", str(out), "--n", str(n), "--petra-only", "--delay-ms", "5000"]
        )
        == 0
    )
    return out


def test_petra_only_picks_games_against_petra_at_a_known_level(tmp_path):
    root = pilot_with_games(tmp_path / "pilot", 2)
    write_game(root, "b", "pilot-1", "r00-t09")  # no label anywhere: just "petra"
    games = list(load_games(root))
    from agentenv_zeroad.s1 import pick_games

    assert len(pick_games(games, 10)) == 3
    assert sorted(g.game_id for g in pick_games(games, 10, petra_only=True)) == ["r00-t00", "r00-t01"]
    sonnet = game(tmp_path, "s", level=1)
    sonnet.results["seats"]["2"] = {"kind": "agent", "name": "Sonnet", "outcome": "lost"}
    assert pick_games([sonnet], 5, petra_only=True) == []


def test_each_replay_spec_is_two_games_in_the_source_setup_with_petra_on_v1(tmp_path):
    root = pilot_with_games(tmp_path / "pilot")
    specs = picked_specs(root, tmp_path / "specs")
    inputs = tmp_path / "out"
    package_inputs(root, specs, inputs)
    games = replay_games(inputs / "inputs" / "specs", inputs / "inputs" / "pilot")
    assert len(games) == 6
    for g in games:
        arm = g.game_id.split("-")[1]
        replay = next(s for s in g.seats if s.kind == "replay")
        petra = next(s for s in g.seats if s.kind == "petra")
        assert replay.interface == arm and petra.interface == "v1"
        spec = ReplaySpec.load(Path(replay.replay))
        assert spec.source in g.game_id and spec.delay_ms == 5000 and g.seats[int(spec.slot) - 1] is replay
        assert (g.map, g.seed, g.size, g.time_cap_minutes) == ("mainland", 9, 128, 30.0)
        assert (replay.civ, petra.civ) == (("rome", "pers") if spec.slot == "1" else ("pers", "rome"))
    levels = sorted({next(s for s in g.seats if s.kind == "petra").difficulty for g in games})
    assert levels == [1, 2, 3]
    write_specs(games, tmp_path / "replay.jsonl")
    assert len(load_specs(tmp_path / "replay.jsonl")) == 6


def test_a_replay_spec_whose_pilot_result_is_missing_is_an_error_not_a_skipped_game(tmp_path):
    root = pilot_with_games(tmp_path / "pilot")
    specs = picked_specs(root, tmp_path / "specs")
    empty = tmp_path / "no-pilot"
    empty.mkdir()
    with pytest.raises(ValueError, match="no pilot results.json"):
        replay_games(specs, empty)


def test_a_replay_of_a_game_against_a_model_is_refused(tmp_path):
    root = pilot_with_games(tmp_path / "pilot", 1)
    specs = picked_specs(root, tmp_path / "specs", 1)
    [path] = [p for p in specs.glob("*.json") if p.name != "index.json"]
    raw = json.loads(path.read_text())
    raw["game"]["opponent"] = "sonnet"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="not Petra at a known level"):
        replay_games(specs, root)


def test_the_package_has_the_specs_and_only_the_pilot_results_they_name(tmp_path):
    root = pilot_with_games(tmp_path / "pilot", 4)
    specs = picked_specs(root, tmp_path / "specs", 2)
    out = tmp_path / "out"
    tarball = package_inputs(root, specs, out)
    picked = {row["id"] for row in json.loads((specs / "index.json").read_text())}
    assert len(picked) == 2
    copied = sorted((out / "inputs" / "pilot").rglob("*results.json"))
    assert len(copied) == 2 and {g.unique_id for g in load_games(out / "inputs" / "pilot")} == picked
    with tarfile.open(tarball) as tar:
        names = tar.getnames()
    assert sum(n.endswith("results.json") for n in names) == 2
    assert {n for n in names if n.startswith("specs/")} == {f"specs/{i}.json" for i in picked}


def test_the_copied_results_keep_petras_level_so_the_summary_labels_the_originals(tmp_path):
    root = pilot_with_games(tmp_path / "pilot", 3)
    specs = picked_specs(root, tmp_path / "specs", 3)
    out = tmp_path / "out"
    package_inputs(root, specs, out)
    labels = sorted(g.opponent for g in load_games(out / "inputs" / "pilot"))
    assert labels == ["petra-1", "petra-2", "petra-3"]  # the bare results say only "petra"
    assert {g.opponent for g in load_games(root)} == {"petra-1", "petra-2", "petra-3"}


def test_build_writes_both_spec_files_and_the_cli_reports_the_counts(tmp_path, capsys):
    root = pilot_with_games(tmp_path / "pilot")
    specs = picked_specs(root, tmp_path / "specs")
    out = tmp_path / "out"
    package_inputs(root, specs, out)
    assert build(out / "inputs", tmp_path / "batch", 4) == {"bot": 24, "replay": 6, "replay_specs": 3}
    assert len(load_specs(tmp_path / "batch" / "bot.jsonl")) == 24
    assert batch_main(["build", "--inputs", str(out / "inputs"), "--out", str(tmp_path / "again"), "--n", "2"]) == 0
    assert "bot: 12 games (2 per interface and level); replay: 6 games from 3 specs" in capsys.readouterr().out
    assert batch_main(["package", str(root), str(specs), "--out", str(tmp_path / "pack")]) == 0
    assert (tmp_path / "pack" / "s1-inputs.tgz").is_file()


# ---- the gates ---------------------------------------------------------------------------------------------------
def metric(group, opponent="petra-1", points=1.0, failures=0, villagers=40, army10=10):
    return GameMetrics(group, opponent, points, failures, villagers, 1.0, 8.0, army10, 12)


def good_games():
    games = []
    for i in range(10):
        games += [
            metric("bot/v1", failures=5, villagers=30, points=1.0 if i < 5 else 0.0),
            metric("bot/v2", failures=0, villagers=45, points=1.0 if i < 5 else 0.0),
            metric("replay/v1", failures=4, villagers=20),
            metric("replay/v2", failures=0, villagers=30, army10=9),
        ]
    return games


def test_every_gate_passes_when_the_numbers_meet_them():
    report = evaluate(good_games())
    assert report["overall"] == "PASS"
    assert {k: g["status"] for k, g in report["gates"].items()} == dict.fromkeys(report["gates"], "PASS")
    assert report["gates"]["field_failures"]["change"] == -1.0 and report["gates"]["villagers_15"]["change"] == 0.5


@pytest.mark.parametrize(
    ("group", "field", "value", "gate"),
    [
        ("bot/v2", "failures", 2, "field_failures"),  # 5 -> 2 is down 60%, not 80%
        ("bot/v2", "villagers", 38, "villagers_15"),  # 30 -> 38 is up 27%
        ("replay/v2", "army10", 7, "army_10_replay_v2"),
    ],
)
def test_a_gate_whose_number_falls_short_fails_and_so_does_the_batch(group, field, value, gate):
    field_of = {"failures": "field_failures", "villagers": "villagers_15", "army10": "army_10"}[field]
    games = [GameMetrics(**{**g.__dict__, field_of: value}) if g.group == group else g for g in good_games()]
    report = evaluate(games)
    assert report["gates"][gate]["status"] == "FAIL" and report["overall"] == "FAIL"


def test_a_v1_with_nothing_to_remove_or_no_games_is_no_data_not_a_pass():
    games = [
        GameMetrics(**{**g.__dict__, "field_failures": 0}) if g.group.startswith("bot") else g for g in good_games()
    ]
    report = evaluate(games)
    assert report["gates"]["field_failures"]["status"] == "NO DATA" and report["overall"] == "NO DATA"
    assert evaluate([])["overall"] == "NO DATA"


def fisher_by_hand(wins_a, n_a, wins_b, n_b):
    total, n = wins_a + wins_b, n_a + n_b
    return sum(comb(n_b, k) * comb(n_a, total - k) for k in range(wins_b + 1) if 0 <= total - k <= n_a) / comb(n, total)


def test_fishers_one_sided_p_is_the_hypergeometric_tail():
    assert fisher_lower(40, 100, 20, 100) == pytest.approx(fisher_by_hand(40, 100, 20, 100))
    assert fisher_lower(5, 10, 1, 10) == pytest.approx(fisher_by_hand(5, 10, 1, 10))
    assert fisher_lower(5, 10, 5, 10) > 0.5 and fisher_lower(0, 10, 0, 10) == 1.0
    assert fisher_lower(2, 10, 8, 10) > 0.99  # v2 higher: nothing to reject


def win_rate_gate(v1_wins, v2_wins, n=100, level="petra-1"):
    games = [metric("bot/v1", level, 1.0 if i < v1_wins else 0.0) for i in range(n)]
    games += [metric("bot/v2", level, 1.0 if i < v2_wins else 0.0) for i in range(n)]
    return evaluate(games)["gates"]["bot_win_rate"]


def test_a_significantly_lower_v2_win_rate_fails_where_v1_wins_at_least_a_tenth():
    assert win_rate_gate(40, 20)["status"] == "FAIL"
    assert win_rate_gate(40, 35)["status"] == "PASS"  # lower, but not significantly
    assert win_rate_gate(40, 60)["status"] == "PASS"


def test_a_level_where_v1_wins_under_a_tenth_is_not_tested_and_draws_are_not_wins():
    gate = win_rate_gate(5, 0)  # v1 wins 5%: under the floor, so v2's 0 is not tested
    assert (
        gate["status"] == "PASS"
        and gate["levels"]["petra-1"]["tested"] is False
        and "no level to test" in gate["detail"]
    )
    games = [metric("bot/v1", "petra-2", 0.5) for _ in range(20)] + [
        metric("bot/v2", "petra-2", 1.0) for _ in range(20)
    ]
    assert evaluate(games)["gates"]["bot_win_rate"]["levels"]["petra-2"]["v1"] == [0, 20]


def test_the_summary_text_and_json_have_the_gates_and_the_calibration_numbers():
    calibration = {
        "host": {"vcpus": 48, "mem_total_mb": 96000},
        "levels": {
            "20": {"games": 20, "games_finished": 20, "wall_s": 600.0, "core_s_per_game": 410.5, "mean_cores": 13.7,
                   "rss_mb": {"median": 700.0, "max": 910.0, "sum_max": 15000.0},
                   "step_ms_by_minute": {"0": {"median": 5.0, "p95": 9.0}, "29": {"median": 60.0, "p95": 120.0}}},
            "1": {"games": 1, "games_finished": 1, "wall_s": 300.0, "core_s_per_game": 250.0, "mean_cores": 0.8,
                  "rss_mb": None},
        },
    }  # fmt: skip
    text, report = summary(good_games(), calibration, excluded=2)
    assert "PASS" in text and "overall: PASS" in text and "2 games left out" in text
    assert "20 engines: 20/20 games finished" in text and "410.5 core-s per game" in text and "min 29: 60/120" in text
    assert text.index(" 1 engines") < text.index("20 engines")
    assert json.loads(json.dumps(report))["gates"]["field_failures"]["status"] == "PASS"
    assert "(no calibration run)" in format_calibration({})


def test_the_summary_script_writes_the_text_and_the_json_beside_the_games(tmp_path, capsys):
    pilot = tmp_path / "games"
    for i, (kind, iface) in enumerate([("bot", "v1"), ("bot", "v2")]):
        folder = pilot / f"g{i}"
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(results(kind=kind, name="Bot 1", interface=iface, level=1)))
    failed = results(kind="bot", name="Bot 1", interface="v2", level=1)
    failed["status"] = "failed"
    (pilot / "bad").mkdir()
    (pilot / "bad" / "results.json").write_text(json.dumps(failed))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"host": {}, "levels": {}}))
    out = tmp_path / "out"
    summary_script = load_script("s1_summary")
    assert summary_script.main([str(pilot), "--calibration", str(calibration), "--out", str(out)]) == 0
    text = (out / "s1-summary.txt").read_text()
    assert "S1 gates" in text and "1 games left out" in text
    report = json.loads((out / "s1-summary.json").read_text())
    assert report["games"] == {"bot/v1": 1, "bot/v2": 1} and report["excluded_unfinished"] == 1
    assert report["overall"] in ("PASS", "FAIL", "NO DATA") and "calibration" in report
    capsys.readouterr()
    assert summary_script.main([str(pilot)]) == 0 and "S1 gates" not in capsys.readouterr().out  # as before
    metrics, excluded = load_metrics([pilot])
    assert (len(metrics), excluded) == (2, 1) and game_metrics(game(tmp_path, "x")) is not None


# ---- the calibration's measuring ---------------------------------------------------------------------------------
def test_step_times_are_bucketed_by_game_minute_and_summarised():
    timer = s1_calibrate.StepTimer()
    per_minute = 60_000 // TURN_MS
    for step in range(2 * per_minute):
        timer.record(step, 0.010 if step < per_minute else 0.020)
    timer.record(per_minute, 0.100)  # the slowest step of minute 1
    summary_ = timer.summary()
    assert set(summary_) == {"0", "1"} and summary_["0"]["steps"] == per_minute
    assert summary_["0"]["median"] == 10.0 and summary_["1"]["max"] == 100.0 and summary_["1"]["p95"] == 20.0


async def test_a_timed_engine_times_both_step_calls_and_starts_again_at_minute_zero_on_a_reset():
    timer = s1_calibrate.StepTimer()
    engine = s1_calibrate.TimedEngine(FakeEngine(), timer)
    for _ in range(3):
        await engine.step()
    await engine.step_raw()
    assert sum(len(v) for v in timer.by_minute.values()) == 4 and engine._step == 4
    await engine.reset({"settings": {}})
    assert engine._step == 0 and engine.time_ms == 0  # attributes come from the engine itself


def fake_cgroups(tmp_path: Path, usage: dict[str, tuple[int, int]]) -> s1_calibrate.CgroupSampler:
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    pids = {}
    for pid, (name, (cpu_us, anon)) in enumerate(usage.items(), 100):
        (proc / str(pid)).mkdir(parents=True, exist_ok=True)
        (proc / str(pid) / "cgroup").write_text(f"0::/system.slice/docker-{name}.scope\n")
        group = cg / "system.slice" / f"docker-{name}.scope"
        group.mkdir(parents=True, exist_ok=True)
        (group / "cpu.stat").write_text(f"usage_usec {cpu_us}\nuser_usec 1\nsystem_usec 2\n")
        (group / "memory.stat").write_text(f"anon {anon}\nfile 5\n")
        (group / "memory.current").write_text(f"{anon + 5}\n")
        pids[name] = pid

    async def discover():
        return pids

    return s1_calibrate.CgroupSampler(discover, proc_root=proc, cgroup_root=cg)


async def test_the_sampler_reads_each_containers_cpu_and_anon_memory_through_its_pids_cgroup(tmp_path):
    sampler = fake_cgroups(tmp_path, {"a": (3_000_000, 700 * 2**20), "b": (1_000_000, 500 * 2**20)})
    readings = await sampler.sample()
    assert readings["a"] == s1_calibrate.Reading(3_000_000, 700 * 2**20, 700 * 2**20 + 5)
    assert set(readings) == {"a", "b"}
    broken = s1_calibrate.CgroupSampler(
        lambda: _pids({"gone": 999}), proc_root=tmp_path / "proc", cgroup_root=tmp_path / "cg"
    )
    assert await broken.sample() == {}  # a container whose cgroup can't be read is left out, not an error


async def _pids(pids):
    return pids


def test_the_meter_counts_core_seconds_from_the_baseline_and_the_memory_after_it():
    R = s1_calibrate.Reading
    meter = s1_calibrate.Meter()
    meter.add(0.0, {"a": R(5_000_000, 100 * 2**20, 0), "b": R(2_000_000, 100 * 2**20, 0)})  # the baseline: startup CPU
    meter.add(10.0, {"a": R(15_000_000, 600 * 2**20, 0), "b": R(12_000_000, 800 * 2**20, 0)})
    meter.add(
        20.0, {"a": R(25_000_000, 700 * 2**20, 0), "b": R(22_000_000, 900 * 2**20, 0), "c": R(4_000_000, 50 * 2**20, 0)}
    )
    assert meter.core_seconds == pytest.approx(44.0)  # a 20 + b 20 since the baseline, c 4 from its own start
    assert meter.rss_mb() == {"median": 700.0, "max": 900.0, "sum_max": 1650.0}
    assert [row["cores"] for row in meter.series] == [2.0, 2.4]


async def test_a_calibration_level_plays_its_games_and_reports_cost_memory_and_step_times(tmp_path, monkeypatch):
    class Launcher:
        async def stop(self) -> None: ...
        async def clear_replays(self) -> None: ...
        async def copy_replay(self, dest):
            return None

    async def fake_start_engine(options, **kwargs):
        return SimpleNamespace(engine=ScriptedEngine(), launcher=Launcher())

    short = [g.__class__(**{**g.__dict__, "time_cap_minutes": 0.5}) for g in calibration_games(2)]
    monkeypatch.setattr(s1_calibrate, "start_engine", fake_start_engine)
    monkeypatch.setattr(s1_calibrate, "calibration_games", lambda count: short[:count])
    sampler = fake_cgroups(tmp_path, {"e0": (1_000_000, 600 * 2**20), "e1": (1_000_000, 650 * 2**20)})
    result = await s1_calibrate.run_level(
        2, tmp_path / "level-2", sampler=sampler, mode="local", base_port=23_900, sample_s=0.01,
        match_options={"placer": placer()},
    )  # fmt: skip
    assert result["games"] == result["games_finished"] == 2 and result["timed_out"] is False
    assert result["step_ms_by_minute"]["0"]["steps"] > 0 and set(result["step_ms_by_minute"]) == {"0"}
    assert result["rss_mb"]["max"] == 650.0
    assert (tmp_path / "level-2" / "calib-02-00" / "results.json").is_file()
    assert (tmp_path / "level-2" / "games.jsonl").read_text().count("\n") == 2
