"""The teacher calibration panel: its specs, its config, its place in the orchestrator, its report and its pick."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.train import report as R
from agentenv_zeroad.train import stats
from agentenv_zeroad.train.calibration import LITELLM_URL_ENV, UNSET_URL, calibration_specs, panel_specs
from agentenv_zeroad.train.config import RunConfig, check_run, load_config
from agentenv_zeroad.train.game import build_model
from agentenv_zeroad.train.spec import SeatSpec
from tests.test_train_orchestrate import config as run_config
from tests.test_train_orchestrate import drive, make
from tests.train_fakes import results

pytestmark = pytest.mark.anyio

TEACHERS = [
    {"name": "astra", "model": "openai/gpt-6-astra"},
    {"name": "opus", "model": "anthropic/claude-opus-5-5", "effort": "high"},
]
PANEL = {"name": "teacher-calibration", "kind": "teacher_calibration", "at_hours": 0, "teachers": TEACHERS}


def cfg_with(tmp_path: Path, panel: dict | None = None, **over) -> RunConfig:
    return run_config(tmp_path / "run", panels=[{**PANEL, **(panel or {})}], **over)


def teacher_seat(spec):
    (seat,) = [s for s in spec.seats if s.kind == "llm"]
    return seat


def petra_seat(spec):
    (seat,) = [s for s in spec.seats if s.kind == "petra"]
    return seat


# ---- the specs --------------------------------------------------------------------------------------------------
def test_ten_games_are_dealt_over_each_teacher_and_rung_with_the_extras_on_petra_3(tmp_path):
    cfg = cfg_with(tmp_path)
    specs = calibration_specs(cfg, cfg.panels[0], {})
    assert len(specs) == 10
    cells = {}
    for spec in specs:
        key = (teacher_seat(spec).name, petra_seat(spec).difficulty)
        cells[key] = cells.get(key, 0) + 1
    assert cells == {("astra", 3): 3, ("opus", 3): 3, ("astra", 4): 2, ("opus", 4): 2}
    assert len({s.game_id for s in specs}) == 10


def test_teachers_play_llm_v2_seats_and_petra_plays_its_v1_controller_at_the_policys_latency(tmp_path):
    cfg = cfg_with(tmp_path, games={"latency_ms": 7000})
    specs = calibration_specs(cfg, cfg.panels[0], {LITELLM_URL_ENV: "https://litellm.example/v1"})
    assert {s.latency_ms for s in specs} == {7000} == {cfg.games.latency_ms}
    for spec in specs:
        teacher, petra = teacher_seat(spec), petra_seat(spec)
        assert teacher.interface == "v2" and petra.interface == "v1" and not teacher.trainable
        assert teacher.base_url == "https://litellm.example/v1" and teacher.api_key_env == "LITELLM_API_KEY"
        assert spec.time_cap_minutes == cfg.games.time_cap_minutes and spec.score_rule == cfg.games.score_rule
    models = {teacher_seat(s).name: (teacher_seat(s).model, teacher_seat(s).effort) for s in specs}
    assert models == {"astra": ("openai/gpt-6-astra", "low"), "opus": ("anthropic/claude-opus-5-5", "high")}


def test_a_panel_can_set_its_own_latency_and_endpoint_and_a_missing_endpoint_is_a_placeholder(tmp_path):
    cfg = cfg_with(tmp_path, {"latency_ms": 3000, "base_url": "http://x/v1", "api_key_env": "OTHER_KEY"})
    spec = calibration_specs(cfg, cfg.panels[0], {LITELLM_URL_ENV: "http://ignored"})[0]
    assert spec.latency_ms == 3000 and teacher_seat(spec).base_url == "http://x/v1"
    assert teacher_seat(spec).api_key_env == "OTHER_KEY"
    bare = cfg_with(tmp_path)
    assert teacher_seat(calibration_specs(bare, bare.panels[0], {})[0]).base_url == UNSET_URL


def test_every_teacher_faces_the_same_positions_and_takes_both_sides(tmp_path):
    cfg = cfg_with(
        tmp_path, held_out={"civ_pairs": [["rome", "pers"]], "maps": ["mainland_held_out"], "seeds": [900001, 900002]}
    )
    specs = calibration_specs(cfg, cfg.panels[0], {})
    by_cell = {}
    for spec in specs:
        by_cell.setdefault((teacher_seat(spec).name, petra_seat(spec).difficulty), []).append(spec)
    for j in range(2):
        positions = {
            (s[j].map, s[j].seed, s[j].ai_seed, teacher_seat(s[j]).civ, petra_seat(s[j]).civ) for s in by_cell.values()
        }
        assert len(positions) == 1  # the j-th game of every cell is the same position
    assert {s.map for s in specs} == {"mainland_held_out"} and {s.seed for s in specs} <= {900001, 900002}
    assert {(teacher_seat(s).civ, petra_seat(s).civ) for s in specs} == {("rome", "pers")}
    assert {s.seats.index(teacher_seat(s)) for s in specs} == {0, 1}
    assert [s.to_json() for s in specs] == [s.to_json() for s in calibration_specs(cfg, cfg.panels[0], {})]


def test_panel_specs_reads_a_spec_file_panel_as_before_and_makes_the_calibration_ones(tmp_path):
    spec_file = tmp_path / "panel.jsonl"
    spec_file.write_text(
        json.dumps({"game_id": "p0", "seats": [{"kind": "petra", "civ": "pers"}, {"kind": "petra", "civ": "rome"}]})
    )
    cfg = run_config(tmp_path / "run", panels=[{"name": "mid", "at_hours": 1, "specs": str(spec_file)}, PANEL])
    assert [s.game_id for s in panel_specs(cfg, cfg.panels[0])] == ["p0"]
    assert len(panel_specs(cfg, cfg.panels[1], {})) == 10


# ---- the config -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"teachers": []}, "needs teachers"),
        ({"games": 3}, "below one per teacher and rung"),
        ({"rungs": [3, 9]}, "0 to 5"),
        ({"teachers": [TEACHERS[0], TEACHERS[0]]}, "unique"),
        ({"kind": "mystery"}, "panel kind"),
        ({"latency_ms": 0}, "latency_ms"),
        ({"teachers": [{"name": "a", "model": "m", "surprise": 1}]}, "unknown teacher keys"),
    ],
)
def test_a_calibration_panel_that_could_not_run_is_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        cfg_with(tmp_path, change)


def test_a_spec_panel_still_needs_its_file(tmp_path):
    with pytest.raises(ValueError, match="specs is required"):
        run_config(tmp_path / "run", panels=[{"name": "mid", "at_hours": 1}])


def test_the_toml_example_loads_and_the_check_names_the_frontier_key(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        f"""run_id = "r1"
root = "{tmp_path}/run1"
budget_usd = 100
box_usd_per_hour = 10
duration_hours = 10
[servers]
urls = ["http://a:8000"]
adapter_dir = "/train/run1/adapters"
[[panels]]
name = "teacher-calibration"
kind = "teacher_calibration"
at_hours = 0
games = 12
rungs = [3, 4]
teachers = [{{ name = "astra", model = "openai/gpt-6-astra" }}, {{ name = "opus", model = "anthropic/claude-opus-5" }}]
"""
    )
    cfg = load_config(path)
    panel = cfg.panels[0]
    assert panel.kind == "teacher_calibration" and panel.games == 12 and panel.rungs == (3, 4)
    assert [t.name for t in panel.teachers] == ["astra", "opus"] and panel.specs is None
    assert check_run(cfg) == ["LITELLM_API_KEY"]


def test_a_frontier_seat_sends_no_sampler_fields_and_a_local_seat_still_does():
    teacher = SeatSpec(
        kind="llm", civ="rome", name="astra", model="openai/gpt-6-astra", base_url="http://x", api_key_env="K"
    )
    local = SeatSpec(kind="llm", civ="rome", model="local/policy-v1", base_url="http://x", api_key_env="K")
    frontier = build_model(teacher, "1", environ={"K": "k"})
    served = build_model(local, "1", environ={"K": "k"})
    assert isinstance(frontier, StrategistModel) and "return_token_ids" not in frontier.extra_body
    assert served.extra_body["return_token_ids"] is True and served.extra_body["logprobs"] is True


# ---- in the orchestrator ----------------------------------------------------------------------------------------
async def test_the_calibration_panel_plays_its_games_on_the_base_model_apart_from_training(tmp_path, monkeypatch):
    monkeypatch.setenv(LITELLM_URL_ENV, "https://litellm.example/v1")
    t = make(tmp_path, monkeypatch, concurrency=4, panels=[{**PANEL, "games": 4}])
    t.orch.start_due_panels()
    assert t.orch.panels["teacher-calibration"].games == 4 and t.orch.panels["teacher-calibration"].version == 0
    assert len(t.orch.panel_queue) == 4
    folder = t.root / "panels" / "teacher-calibration"
    saved = [json.loads(line) for line in (folder / "specs.jsonl").read_text().splitlines()]
    assert len(saved) == 4 and {s["latency_ms"] for s in saved} == {t.cfg.games.latency_ms}

    await drive(t.orch, lambda: t.orch.counts["panel_finished"] == 4)
    summary = json.loads((folder / "summary.json").read_text())
    assert summary == {"name": "teacher-calibration", "version": 0, "games": 4, "finished": 4, "voided": 0}
    assert not t.credit.games  # a panel's games are never credited
    assert all(p.version is None or p.version == 0 for p in t.orch.panel_queue)
    assert all(spec.game_id.startswith("teacher-calibration-v0-") for spec in t.play.specs[:4])


# ---- the report -------------------------------------------------------------------------------------------------
def calibration_game(folder: Path, game_id: str, teacher: str, rung: int, outcome: str, model: str = "m/x") -> None:
    outcomes = (outcome, {"won": "lost", "lost": "won", "drawn": "drawn"}[outcome])
    r = results(
        game_id,
        outcomes=outcomes,
        trainable=(False, False),
        model_1=model,
        opponent={"kind": "petra", "civ": "pers", "difficulty": rung, "behaviour": "balanced"},
    )
    r["spec"]["seats"][0]["name"] = teacher
    r["seats"]["1"]["name"] = teacher
    (folder / game_id).mkdir(parents=True)
    (folder / game_id / "results.json").write_text(json.dumps(r))


def calibration_folder(tmp_path: Path) -> Path:
    panels = tmp_path / "panels"
    folder = panels / "teacher-calibration"
    plan = [
        ("astra", 3, "won", "openai/gpt-6-astra"), ("astra", 3, "won", "openai/gpt-6-astra"),
        ("astra", 3, "lost", "openai/gpt-6-astra"), ("astra", 4, "lost", "openai/gpt-6-astra"),
        ("opus", 3, "won", "anthropic/claude-opus-5-5"), ("opus", 3, "lost", "anthropic/claude-opus-5-5"),
        ("opus", 3, "lost", "anthropic/claude-opus-5-5"), ("opus", 4, "won", "anthropic/claude-opus-5-5"),
    ]  # fmt: skip
    for i, (teacher, rung, outcome, model) in enumerate(plan):
        calibration_game(folder, f"g{i}", teacher, rung, outcome, model)
    return panels


def test_the_report_summarises_win_rate_per_teacher_and_rung_with_wilson_intervals(tmp_path):
    panels = calibration_folder(tmp_path)
    summary = R.teacher_calibration(panels / "teacher-calibration")
    assert summary is not None and summary["games"] == 8
    astra, opus = summary["teachers"]["astra"], summary["teachers"]["opus"]
    assert astra["model"] == "openai/gpt-6-astra" and astra["rungs"]["petra-3"]["games"] == 3
    assert astra["rungs"]["petra-3"]["win_rate"] == pytest.approx(2 / 3)
    lo, hi = stats.wilson_interval(2, 3)
    assert astra["rungs"]["petra-3"]["lo"] == pytest.approx(lo) and astra["rungs"]["petra-3"]["hi"] == pytest.approx(hi)
    assert 0 < lo < 2 / 3 < hi < 1 and hi - lo > 0.5  # three games say little, and the interval says so
    assert opus["rungs"]["petra-4"]["win_rate"] == 1.0 and opus["rungs"]["petra-3"]["win_rate"] == pytest.approx(1 / 3)
    assert summary["best"]["name"] == "astra" and summary["best"]["model"] == "openai/gpt-6-astra"


def test_report_json_carries_the_calibration_and_none_without_the_panel(tmp_path):
    panels = calibration_folder(tmp_path)
    games = tmp_path / "games"
    games.mkdir()
    built = R.build(games, panels=panels, now=1.0, bootstrap=0)
    assert built["teacher_calibration"]["best"]["name"] == "astra" and built["sft_selection"] is None
    json.dumps(built)
    assert R.build(games, now=1.0, bootstrap=0)["teacher_calibration"] is None
    assert R.build(games, panels=tmp_path / "nothing", now=1.0, bootstrap=0)["teacher_calibration"] is None


def test_unfinished_games_and_games_without_a_teacher_and_a_petra_seat_are_not_counted(tmp_path):
    folder = tmp_path / "cal"
    calibration_game(folder, "ok", "astra", 3, "won")
    bad = json.loads((folder / "ok" / "results.json").read_text())
    for name, change in (("failed", {"status": "failed"}), ("stopped", {"ended_by": "stopped"})):
        (folder / name).mkdir()
        (folder / name / "results.json").write_text(json.dumps({**bad, **change}))
    mirror = results("mirror", opponent={"kind": "llm"}, trainable=(False, False))
    (folder / "mirror").mkdir()
    (folder / "mirror" / "results.json").write_text(json.dumps(mirror))
    assert [(g.teacher, g.opponent) for g in R.read_teacher_games(folder)] == [("astra", "petra-3")]


# ---- the pick ---------------------------------------------------------------------------------------------------
def summary_of(rows: dict[str, dict[str, list[float]]]) -> dict:
    games = [
        stats.TeacherGame(teacher, rung, result)
        for teacher, by in rows.items()
        for rung, rs in by.items()
        for result in rs
    ]
    return stats.teacher_calibration(games)


def test_the_teacher_is_the_best_against_petra_3_not_the_best_overall():
    summary = summary_of(
        {"astra": {"petra-3": [1, 1, 0], "petra-4": [0, 0]}, "opus": {"petra-3": [1, 0, 0], "petra-4": [1, 1]}}
    )
    assert stats.pick_teacher(summary) == "astra"


def test_a_tie_on_petra_3_goes_to_the_better_petra_4_record_then_to_the_name():
    tie = {"astra": {"petra-3": [1, 0], "petra-4": [0, 0]}, "opus": {"petra-3": [0, 1], "petra-4": [1, 0]}}
    assert stats.pick_teacher(summary_of(tie)) == "opus"
    assert stats.pick_teacher(summary_of({"b": {"petra-3": [1]}, "a": {"petra-3": [1]}})) == "a"


def test_a_teacher_that_never_played_petra_3_cannot_be_picked_and_nobody_gives_none():
    assert stats.pick_teacher(summary_of({"astra": {"petra-4": [1, 1]}, "opus": {"petra-3": [0]}})) == "opus"
    assert summary_of({"astra": {"petra-4": [1]}})["best"] is None
    assert stats.teacher_calibration([])["best"] is None


def test_a_drawn_game_counts_half_and_the_interval_of_no_games_is_everything():
    summary = summary_of({"astra": {"petra-3": [0.5, 0.5]}})
    assert summary["teachers"]["astra"]["rungs"]["petra-3"]["win_rate"] == 0.5
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.7
