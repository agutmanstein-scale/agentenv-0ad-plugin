"""The v2 report: reading a run's folders into report.json."""

import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from agentenv_zeroad.train import report as R  # noqa: E402
from agentenv_zeroad.train import stats  # noqa: E402
from tests.train_fakes import results, timeline, turn  # noqa: E402

T0 = 1_700_000_000.0


def wave(began_ms, target="enemy_cc", min_size=12, status="accepted"):
    t = turn(began_ms, tools=("army",), status=status, model="local/policy-v1")
    t["orders"][0]["arguments"] = {"action": "attack", "target": target, "min_size": min_size}
    return t


def save(games_dir, game_id, finished_at, **kwargs):
    kwargs.setdefault("opponent", {"kind": "petra", "civ": "pers", "difficulty": 1, "behaviour": "balanced"})
    kwargs.setdefault("rows", timeline(minutes=20, army=(30, 10)))
    r = results(game_id, **kwargs)
    folder = games_dir / game_id
    folder.mkdir(parents=True)
    path = folder / "results.json"
    path.write_text(json.dumps(r))
    os.utime(path, (finished_at, finished_at))
    return r


def run_folders(tmp_path):
    games, learner = tmp_path / "games", tmp_path / "learner"
    for v, at in ((1, 1.2), (2, 2.5)):
        manifest = learner / "adapters" / f"v{v}" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"version": v, "time": T0 + at * 3600}))
    health = [
        {"update": 1, "version": 1, "time": T0 + 60, "seconds": 60.0, "trained": 256, "grad_norm": 0.5,
         "hold_share": 0.2, "tripwires": {"trips": [], "alarms": []}},
        {"update": 2, "version": 2, "time": T0 + 2.5 * 3600, "seconds": 90.0, "trained": 256, "grad_norm": 0.7,
         "hold_share": 0.9, "tripwires": {"trips": [{"name": "hold_share"}], "alarms": [{"name": "kl_to_base"}]}},
        {"event": "rollback", "restored": 1, "version": 3, "time": T0 + 2.6 * 3600},
    ]  # fmt: skip
    learner.mkdir(exist_ok=True)
    (learner / "health.jsonl").write_text("\n".join(json.dumps(h) for h in health))
    games.mkdir()
    # base policy before training: lost to petra-1
    base = "local/qwen-base"
    save(games, "a0", T0 + 60, model_1=base, outcomes=("lost", "won"), net_worth=(1000.0, 3000.0),
         turns={"1": [turn(0, tools=("hold",), model=base), turn(30_000, model=base)]})  # fmt: skip
    # v1: won by conquest, with a wave and its re-issue, one rejected order, one hold turn
    save(games, "a1", T0 + 1.5 * 3600, model_1="local/policy-v1", ended_by="conquest", outcomes=("won", "lost"),
         turns={"1": [wave(100_000, min_size=12), wave(125_000, min_size=12), wave(400_000, min_size=20),
                      turn(450_000, tools=("hold",), model="local/policy-v1"),
                      turn(500_000, tools=("economy",), status="rejected", model="local/policy-v1")]})  # fmt: skip
    # v2: a cap win; frozen-v0 opponent, so the opponent is "v0"
    save(games, "a2", T0 + 2.8 * 3600, model_1="local/policy-v2", opponent={"kind": "llm"}, trainable=(False, False),
         outcomes=("won", "lost"), net_worth=(6000.0, 4000.0), model_2="local/policy-v0",
         turns={"1": [turn(0, model="local/policy-v2")]})  # fmt: skip
    # a game that never finished, and one the clock stopped
    save(games, "u1", T0 + 3 * 3600, status="failed")
    save(games, "u2", T0 + 3 * 3600, ended_by="stopped")
    return games, learner


def test_elo_by_training_hour_has_the_base_policy_and_each_hours_versions_as_players(tmp_path):
    games, learner = run_folders(tmp_path)
    report = R.build(games, learner, bootstrap=20, now=T0 + 3.5 * 3600)
    assert [c["label"] for c in report["elo"]["curve"]] == ["v0", "h01", "h02"]
    assert report["run"] == {
        "hours": 3.5,
        "games": 3,
        "unfinished": 2,
        "versions": 2,
        "policy_games": 2,
    }  # from the first update
    assert report["schema"] == "v2" and json.loads(json.dumps(report)) == report


def test_the_seat_a_game_is_reported_from_is_the_trainable_one_else_the_highest_version(tmp_path):
    games, learner = run_folders(tmp_path)
    played, _ = R.read_games(games, None)
    by_id = {p.game_id: p for p in played}
    assert by_id["a2"].opponent == "v0" and by_id["a2"].version == 2  # v2 against the frozen v0, not the other way
    assert by_id["a0"].opponent == "petra-1" and by_id["a0"].version == 0 and by_id["a0"].result == 0.0
    assert by_id["a1"].result == 1.0 and by_id["a1"].ended_by == "conquest"
    assert by_id["a2"].margin == 2000.0  # the net-worth margin
    forced, _ = R.read_games(games, ["2"])
    assert {p.game_id: p.version for p in forced}["a2"] == 0  # the caller can name the learner seat


def test_milestones_use_the_hours_versions_were_published_not_the_wall_clock_of_the_games(tmp_path):
    games, learner = run_folders(tmp_path)
    report = R.build(games, learner, bootstrap=5, now=T0 + 6.5 * 3600, milestones=[stats.Milestone("petra-1", 0.5)])
    assert [c["at"] for c in report["milestones"][0]["checks"]] == [3.0, 6.0]
    assert report["milestones"][0]["checks"][0]["games"] == 1  # v1's win, in the two bins before hour 3
    assert report["milestones_passed"] == []


def test_behaviour_by_hour_reads_army_waves_holds_tools_and_rejections(tmp_path):
    games, learner = run_folders(tmp_path)
    report = R.build(games, learner, bootstrap=5, now=T0 + 3.5 * 3600)
    by_hour = {b["hour"]: b for b in report["behaviour"]["by_hour"]}
    assert sorted(by_hour) == [0, 1, 2]
    h1 = by_hour[1]
    assert (
        h1["conquest"] == 1.0 and h1["attack_orders"] == 2.0 and h1["first_attack_s"] == 100.0
    )  # the re-issue is not new
    assert h1["conquest_win"] == 1.0
    assert h1["hold_share"] == pytest.approx(1 / 5) and h1["rejected_share"] == pytest.approx(1 / 5)
    assert h1["tool_mix"] == {
        "army": pytest.approx(3 / 5),
        "economy": pytest.approx(1 / 5),
        "hold": pytest.approx(1 / 5),
    }
    assert h1["army_5"] == pytest.approx(15.0) and h1["army_10"] == pytest.approx(30.0)
    assert h1["army_15"] == pytest.approx(45.0) and h1["villagers_15"] == pytest.approx(30.0)  # gatherers stand in
    assert by_hour[2]["attack_orders"] == 0.0 and by_hour[2]["first_attack_s"] is None
    assert report["behaviour"]["overall"]["games"] == 3


def test_behaviour_counts_kills_at_home_and_away_and_failed_field_placements_per_game(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    r = save(games, "k1", T0 + 60, model_1="local/policy-v1", turns={"1": [turn(0, model="local/policy-v1")]})
    seat = r["seats"]["1"]
    seat["combat"] = {"enemy_units_killed": 9, "units_lost": 3, "kills_home": 4, "kills_away": 5}
    seat["failures"] = [
        {"key": "field", "reason": "no valid spot near the farmstead"},
        {"key": "field", "reason": "no valid spot near the town centre"},
        {"key": "field", "reason": "not enough wood"},
        {"key": "house", "reason": "no valid spot"},
    ]
    (games / "k1" / "results.json").write_text(json.dumps(r))
    played, _ = R.read_games(games, None)
    m = played[0].metrics
    assert (m["kills_home"], m["kills_away"], m["field_failures"]) == (4.0, 5.0, 2.0)


def test_a_game_that_ended_before_a_minute_has_no_army_for_it(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    save(games, "short", T0, ended_by="conquest", rows=timeline(minutes=8, army=(30, 10)))
    played, _ = R.read_games(games, None)
    assert played[0].metrics["army_5"] is not None and played[0].metrics["army_10"] is None


def test_health_series_carry_each_updates_trips_and_alarms_and_skip_rollback_events(tmp_path):
    games, learner = run_folders(tmp_path)
    health = R.build(games, learner, bootstrap=5, now=T0 + 3.5 * 3600)["health"]
    assert [u["update"] for u in health["updates"]] == [1, 2]
    assert health["updates"][1]["trips"] == ["hold_share"] and health["updates"][1]["alarms"] == ["kl_to_base"]
    assert health["updates"][0]["hours"] == pytest.approx(60 / 3600, abs=0.01)  # counted from the first update
    assert health["tripped"] == 1 and health["latest"]["trips"][0]["name"] == "hold_share"


def test_throughput_and_spend(tmp_path):
    games, learner = run_folders(tmp_path)
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(json.dumps({"what": "panel", "usd_est": 12.5}) + "\n")
    path = games / "a0" / "results.json"
    r = json.loads(path.read_text())
    r["spend_usd_est"] = 1.25
    path.write_text(json.dumps(r))
    report = R.build(games, learner, bootstrap=5, ledger=ledger, now=T0 + 4 * 3600)
    assert report["throughput"]["decisions_trained"] == 512 and report["throughput"]["updates"] == 2
    assert report["throughput"]["seconds_per_update"] == pytest.approx(75.0)
    assert sum(h["games"] for h in report["throughput"]["by_hour"]) == 3
    assert report["spend"] == {"games_usd": 1.25, "by_opponent": {"petra-1": 1.25}, "ledger_usd": 12.5}


def save_panel(root, name, version, base_margin, count=6):
    panel = root / "panels" / name
    panel.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        save(panel, f"{name}-{i}", T0 + i, model_1=version, net_worth=(base_margin + 50 * i, 3000.0))


def test_the_panel_test_reads_two_panel_folders_in_the_orchestrators_layout_and_keeps_them_off_the_board(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    save(games, "train-v5", T0, model_1="local/policy-v5", net_worth=(-9e6, 3000.0), outcomes=("lost", "won"))
    save_panel(tmp_path, "s2", "local/qwen-base", 1000.0)
    save_panel(tmp_path, "final", "local/policy-v5", 4000.0)
    report = R.build(games, panels=tmp_path / "panels", panel=("s2", "final"), bootstrap=5, now=T0 + 100)
    panel = report["panel"]
    assert panel["n_base"] == 6 and panel["n_other"] == 6 and panel["difference"] == pytest.approx(3000.0)
    assert panel["p"] < 0.001 and (panel["base"], panel["other"]) == (0, 5)
    assert (panel["base_name"], panel["other_name"]) == ("s2", "final")
    assert report["run"]["games"] == 1  # panel games are not training games
    assert R.build(games, bootstrap=5, now=T0 + 100)["panel"] is None


def test_a_conquest_loss_is_not_a_conquest_win(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    save(games, "lost", T0, model_1="local/policy-v1", ended_by="conquest", outcomes=("lost", "won"))
    overall = R.build(games, bootstrap=5, now=T0 + 100)["behaviour"]["overall"]
    assert overall["conquest"] == 1.0 and overall["conquest_win"] == 0.0


def test_an_attack_given_again_after_another_army_order_counts_and_raids_are_included():
    def order(began_ms, **args):
        t = turn(began_ms, tools=("army",), model="m")
        t["orders"][0]["arguments"] = args
        return t

    turns = [
        order(1000, action="attack", target="a"), order(2000, action="gather"),
        order(3000, action="attack", target="a"),
        order(4000, action="raid", target="b"),
    ]  # fmt: skip
    assert R._attack_orders(turns) == [1000.0, 3000.0, 4000.0]


def test_the_runs_clock_starts_at_the_first_update_not_at_an_earlier_game(tmp_path):
    games, learner = run_folders(tmp_path)
    early = R.build(games, learner, bootstrap=5, now=T0 + 3.5 * 3600)
    older = save(games, "ancient", T0 - 10 * 3600, model_1="local/qwen-base", outcomes=("lost", "won"))
    assert older
    later = R.build(games, learner, bootstrap=5, now=T0 + 3.5 * 3600)
    assert later["run"]["hours"] == early["run"]["hours"]


def test_an_empty_run_still_reports(tmp_path):
    (tmp_path / "games").mkdir()
    report = R.build(tmp_path / "games", None, bootstrap=5, now=T0)
    assert report["run"]["games"] == 0 and report["elo"] == {"board": {}, "curve": []}
    assert report["behaviour"]["by_hour"] == [] and report["health"]["updates"] == []


def test_the_command_line_writes_report_json_atomically(tmp_path):
    games, learner = run_folders(tmp_path)
    save_panel(tmp_path, "s2", "local/qwen-base", 1000.0, 2)
    save_panel(tmp_path, "final", "local/policy-v2", 2000.0, 2)
    out = tmp_path / "report.json"
    R.main(["--games", str(games), "--learner", str(learner), "--out", str(out), "--bootstrap", "5",
            "--panels", str(tmp_path / "panels"), "--panel", "s2:final"])  # fmt: skip
    report = json.loads(out.read_text())
    assert report["schema"] == "v2" and report["panel"]["other_name"] == "final"
    assert not list(tmp_path.glob("*.tmp"))


def test_every_field_the_dashboard_reads_from_a_v2_report_exists_in_one(tmp_path):
    page = (Path(__file__).resolve().parents[1] / "docs" / "pilot-dashboard.html").read_text()
    games, learner = run_folders(tmp_path)
    report = R.build(games, learner, bootstrap=5, now=T0 + 3.5 * 3600)
    for top in set(re.findall(r"\bV\.(\w+)", page)):
        assert top in report, top
    for top, inner in set(re.findall(r"\bV\.(\w+)\.(\w+)", page)):
        if isinstance(report[top], dict):
            assert inner in report[top], f"{top}.{inner}"
    for key in re.findall(r'\["(\w+)", "[^"]+", "--s\d"\]', page):
        rows = report["behaviour"]["by_hour"] + report["health"]["updates"]
        assert any(key in row for row in rows), key
    assert re.search(r'data\.schema === "v2"', page) and report["schema"] == "v2"


def test_report_json_shows_the_label_spend_beside_the_aux_alpha(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"what": "hard-state labels h0", "usd_est": 1.25},
                {"what": "hard-state labels h1", "usd_est": 0.75},
                {"what": "panel mid", "usd_est": 9.0},
            ]
        )
        + "\n"
    )
    health = [
        {"aux_alpha": 0.2, "aux_rows": 0},
        {"aux_alpha": 0.2, "aux_rows": 8},
        {"aux_alpha": 0.199, "aux_rows": 5},
        {"aux_alpha": 0.198, "aux_rows": 0},
    ]
    assert R._hard_states(health, ledger) == {
        "labels_usd": 2.0, "aux_alpha": 0.198, "aux_rows_trained": 13, "aux_updates": 2,
    }  # fmt: skip
    assert R._hard_states([], None) == {"labels_usd": None, "aux_alpha": None, "aux_rows_trained": 0, "aux_updates": 0}
    built = R.build(tmp_path / "games", ledger=ledger, now=1.0, bootstrap=0)
    assert built["hard_states"]["labels_usd"] == 2.0
