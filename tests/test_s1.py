"""The S1 tool check's data side: picking pilot games to replay, the spec files, and the per-interface metrics."""

import json
from pathlib import Path

from agentenv_zeroad.replay import ReplaySpec
from agentenv_zeroad.s1 import Game, format_summary, game_metrics, load_games, pick_games, replay_spec
from tests.script_loader import load_script

MIN = 60_000


def row(t_min: float, **players):
    return {"t_ms": int(t_min * MIN), "players": players}


def seat_row(pop=12, army=0, phase="village", **extra):
    return {"pop": pop, "army": army, "phase": phase, **extra}


def results(
    *, kind="agent", name="policy v0", interface="v1", outcome="won", level=None, turns=20, failures=(), minutes=20
):
    turn = {"began_ms": 1000, "orders": [{"tool": "train", "arguments": {"role": "villager", "count": 5}}]}
    timeline = [
        row(
            t,
            **{
                "1": seat_row(pop=10 + 2 * t, army=t // 5, phase="town" if t >= 8 else "village"),
                "2": seat_row(pop=30),
            },
        )
        for t in range(0, minutes + 1)
    ]
    focus = {
        "kind": kind,
        "name": name,
        "interface": interface,
        "outcome": outcome,
        "turns": [turn] * turns,
        "failures": [
            {"key": "field", "reason": "no valid spot at near_food: blocked"},
            {"key": "field", "reason": "unknown structure field"},
            {"key": "barracks", "reason": "no valid spot at front: blocked"},
        ][: len(failures) or 3]
        if failures is not None
        else [],
        "metrics": {"match_s": 1200.0, "idle_production_s": 1800.0},
    }
    petra = {"kind": "petra", "name": "Petra 2", "outcome": "lost" if outcome == "won" else "won"}
    if level is not None:
        petra["ai_difficulty"] = level
    return {
        "seats": {"1": focus, "2": petra},
        "timeline": timeline,
        "map": {"name": "mainland", "size": 128, "seed": 9},
        "time_cap_s": 1800,
        "ended_by": "conquest",
        "game_time_s": minutes * 60.0,
    }


def game(tmp_path: Path, game_id: str, box="a", run="pilot", **kwargs) -> Game:
    path = tmp_path / box / run / "rounds" / "r00" / "games" / game_id / "@local" / "zeroad-lb-x-results.json"
    return Game(path, results(**kwargs))


def test_the_focus_seat_is_the_replay_seat_then_the_bot_then_the_policy(tmp_path):
    assert game(tmp_path, "g", kind="replay", name="Replay 1", interface="v2").focus == "1"
    both = results(kind="agent")
    both["seats"]["2"] = {"kind": "bot", "name": "Bot 2", "outcome": "lost"}
    assert Game(tmp_path / "x-results.json", both).focus == "2"
    assert game(tmp_path, "g").group == "agent/v1"
    assert game(tmp_path, "g", kind="replay", name="Replay 1", interface="v2").group == "replay/v2"


def test_the_opponent_is_petras_recorded_level_else_the_rounds_logs_else_the_game_id(tmp_path):
    assert game(tmp_path, "g1", level=3).opponent == "petra-3"
    assert game(tmp_path, "r00-p-petra-2").opponent == "petra-2"
    assert game(tmp_path, "r00-t01").opponent == "petra"
    run = tmp_path / "a" / "pilot"
    (run / "rounds" / "r00").mkdir(parents=True)
    (run / "rounds" / "r00" / "specs.json").write_text(
        json.dumps([{"game_id": "r00-t01", "opponent": {"kind": "petra", "level": 4}}])
    )
    assert game(tmp_path, "r00-t01").opponent == "petra-4"
    (run / "games.jsonl").write_text(json.dumps({"game_id": "r00-t02", "opponent": "petra-5"}) + "\n")
    assert game(tmp_path, "r00-t02").opponent == "petra-5"


def test_a_model_opponent_is_labelled_by_its_name(tmp_path):
    g = game(tmp_path, "g")
    g.results["seats"]["2"] = {"kind": "agent", "name": "Sonnet", "outcome": "lost"}
    assert g.opponent == "sonnet"


def test_games_are_picked_by_version_and_turns_and_spread_over_opponents(tmp_path):
    games = [game(tmp_path, f"p1-{i}", level=1) for i in range(5)] + [
        game(tmp_path, f"p2-{i}", level=2) for i in range(5)
    ]
    games += [game(tmp_path, "short", level=3, turns=3), game(tmp_path, "v3", level=3, name="policy v3")]
    picked = pick_games(games, 4, version=0, min_turns=15, seed=1)
    assert sorted(g.opponent for g in picked) == ["petra-1", "petra-1", "petra-2", "petra-2"]
    assert {g.game_id for g in picked}.isdisjoint({"short", "v3"})
    assert [g.game_id for g in pick_games(games, 4, version=0, seed=1)] == [g.game_id for g in picked]
    assert len(pick_games(games, 50, version=None, min_turns=3)) == 12
    assert {g.opponent for g in pick_games(games, 50, version=3, min_turns=3)} == {"petra-3"}


def test_a_replay_spec_has_the_seats_orders_and_how_to_set_the_game_up_again(tmp_path):
    spec = replay_spec(game(tmp_path, "r00-t05", level=2), delay_ms=5_000)
    assert (spec.source, spec.slot, spec.delay_ms) == ("a-pilot-r00-t05", "1", 5_000) and len(spec.orders) == 20
    assert spec.game["map"] == {"name": "mainland", "size": 128, "seed": 9} and spec.game["opponent"] == "petra-2"
    assert spec.game["seats"]["2"]["kind"] == "petra"
    assert ReplaySpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec


def test_games_that_share_an_id_across_boxes_and_runs_get_distinct_spec_ids(tmp_path):
    same = [
        game(tmp_path, "r00-t02", box=b, run=r, level=2)
        for b, r in (("a", "pilot-1"), ("b", "pilot-1"), ("c", "pilot-solo"))
    ]
    ids = [g.unique_id for g in same]
    assert ids == ["a-pilot-1-r00-t02", "b-pilot-1-r00-t02", "c-pilot-solo-r00-t02"]
    assert {replay_spec(g).source for g in same} == set(ids)
    assert Game(tmp_path / "x-results.json", results()).unique_id == "x-results"


def test_the_metrics_read_failures_timeline_and_idle_production(tmp_path):
    m = game_metrics(game(tmp_path, "g", level=1))
    assert m.field_failures == 1  # only the field's "no valid spot"; another structure or reason doesn't count
    assert m.villagers_15 == 10 + 30 - 3  # pop - army at minute 15: no villager count was recorded
    assert (m.army_10, m.army_15, m.town_minute, m.idle_production) == (2, 3, 8.0, 1.5) and m.points == 1.0


def test_a_recorded_villager_count_is_preferred_and_a_game_shorter_than_the_minute_has_no_value(tmp_path):
    g = game(tmp_path, "g", minutes=12)
    for tick in g.results["timeline"]:
        tick["players"]["1"]["villagers"] = 7
    m = game_metrics(g)
    assert m.villagers_15 is None and m.army_15 is None and m.army_10 == 2
    g = game(tmp_path, "g", minutes=20)
    g.results["timeline"][15]["players"]["1"]["villagers"] = 44
    assert game_metrics(g).villagers_15 == 44


def test_the_summary_has_a_column_per_seat_kind_and_interface_and_a_win_rate_per_opponent(tmp_path):
    games = [
        game(tmp_path, "a", kind="replay", name="Replay 1", interface="v1", level=2, outcome="lost"),
        game(tmp_path, "b", kind="replay", name="Replay 1", interface="v2", level=2, outcome="won"),
        game(tmp_path, "c", kind="replay", name="Replay 1", interface="v2", level=3, outcome="lost"),
    ]
    text = format_summary([game_metrics(g) for g in games])
    header = text.splitlines()[0]
    assert "replay/v1 (1 games)" in header and "replay/v2 (2 games)" in header
    assert "field placement failures per game" in text and "army at 10 min (median)" in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("petra-2"))
    assert "0% (0/1)" in line and "100% (1/1)" in line
    assert next(ln for ln in text.splitlines() if ln.strip().startswith("petra-3")).split()[-1] == "(0/1)"


def write_pilot_game(root: Path, box: str, run: str, game_id: str, **kwargs) -> None:
    folder = root / box / run / "rounds" / "r00" / "games" / game_id / "@local"
    folder.mkdir(parents=True)
    (folder / "zeroad-lb-x-results.json").write_text(json.dumps(results(**kwargs)))


def test_the_two_scripts_write_specs_and_print_the_summary(tmp_path, capsys):
    pilot = tmp_path / "pilot"
    write_pilot_game(pilot, "a", "pilot-1", "r00-t01", level=1)
    specs = load_script("s1_specs")
    out = tmp_path / "specs"
    assert specs.main([str(pilot), "--out", str(out), "--n", "3", "--delay-ms", "5000"]) == 0
    assert "a-pilot-1-r00-t01" in capsys.readouterr().out
    index = json.loads((out / "index.json").read_text())
    assert index == [
        {"id": "a-pilot-1-r00-t01", "game_id": "r00-t01", "opponent": "petra-1", "slot": "1", "orders": 20}
    ]
    spec = ReplaySpec.load(out / "a-pilot-1-r00-t01.json")
    assert spec.delay_ms == 5000 and spec.source == "a-pilot-1-r00-t01"
    summary = load_script("s1_summary")
    assert summary.main([str(pilot)]) == 0
    assert "agent/v1 (1 games)" in capsys.readouterr().out
    assert summary.main([str(tmp_path / "nothing")]) == 1
    assert len(list(load_games(pilot))) == 1


def test_specs_of_games_sharing_an_id_each_keep_their_own_file_and_an_existing_spec_is_never_overwritten(
    tmp_path, capsys
):
    pilot = tmp_path / "pilot"
    for box, level in (("a", 1), ("b", 2), ("c", 3)):
        write_pilot_game(pilot, box, "pilot-1", "r00-t02", level=level)
    specs = load_script("s1_specs")
    out = tmp_path / "specs"
    assert specs.main([str(pilot), "--out", str(out), "--n", "3"]) == 0
    index = json.loads((out / "index.json").read_text())
    assert sorted(r["id"] for r in index) == ["a-pilot-1-r00-t02", "b-pilot-1-r00-t02", "c-pilot-1-r00-t02"]
    assert sorted(p.name for p in out.glob("*-r00-t02.json")) == [
        f"{r['id']}.json" for r in sorted(index, key=lambda r: r["id"])
    ]
    for row in index:
        assert ReplaySpec.load(out / f"{row['id']}.json").source == row["id"]
    before = (out / "a-pilot-1-r00-t02.json").read_text()
    capsys.readouterr()
    assert specs.main([str(pilot), "--out", str(out), "--n", "3"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (out / "a-pilot-1-r00-t02.json").read_text() == before
