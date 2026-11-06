"""scripts/score_rule_check.py: the proxy net worth, who leads at a minute, and the table on a small set of games."""

import json

import pytest

from tests.script_loader import load_script

check = load_script("score_rule_check")
MIN = 60_000


def player(total, *, pop=20, army=0, resources=0, buildings=1, economy=None, military=0.0):
    return {
        "total": total,
        "economy": total if economy is None else economy,
        "military": military,
        "pop": pop,
        "army": army,
        "resources": resources,
        "buildings": buildings,
    }


def test_the_proxy_values_villagers_soldiers_buildings_and_the_bank():
    row = player(0, pop=30, army=10, resources=400, buildings=5)
    assert check.proxy_net_worth(row) == 400 + 50 * 20 + 100 * 10 + 1500 + 150 * 4
    assert check.proxy_net_worth(player(0, pop=0, army=0, resources=0, buildings=0)) == 0


def test_the_leader_is_the_top_slot_and_none_on_a_tie():
    players = {"1": player(100), "2": player(300)}
    assert check.leader(players, check.MEASURES["official total"]) == "2"
    assert check.leader({"1": player(5), "2": player(5)}, check.MEASURES["official total"]) is None


def test_a_timeline_row_is_the_last_one_at_or_before_the_minute_and_none_if_the_game_ended_first():
    timeline = [{"t_ms": t * 10_000, "players": {"1": t}} for t in range(0, 40)]
    assert check.row_at(timeline, 2) == {"1": 12}
    assert check.row_at(timeline, 7) is None


def test_wilson_and_the_sign_test_behave_at_their_edges():
    assert check.wilson(0, 0) == (0.0, 0.0)
    lo, hi = check.wilson(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo < 0.41
    assert check.sign_test(0, 0) == 1.0
    assert check.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert check.sign_test(3, 3) == 1.0


def game(winner, p1, p2, *, minutes=25, ended_by="conquest"):
    timeline = [{"t_ms": t * 30_000, "players": {"1": p1, "2": p2}} for t in range(0, minutes * 2 + 1)]
    seats = {s: {"kind": "agent" if s == "1" else "petra", "outcome": "won" if s == winner else "lost"} for s in "12"}
    return {"ended_by": ended_by, "seats": seats, "timeline": timeline}


def test_the_table_counts_only_conquest_games_still_running_at_each_minute(tmp_path):
    big, small = player(900, pop=40), player(100, pop=20)
    games = [
        game("1", big, small),
        game("2", player(100, pop=60, army=30), player(900, pop=10)),  # the official total says 2, the proxy says 1
        game("1", big, small, ended_by="time_cap"),
        game("1", big, small, minutes=12),
        game("1", player(500), player(500)),  # tied on every measure: left out
    ]
    for i, g in enumerate(games):
        (tmp_path / f"g{i}-results.json").write_text(json.dumps(g))
    data = check.evaluate(tmp_path, (10, 15, 20))
    assert (data["scanned"], data["conquest"]) == (5, 4)
    assert len(data["hits"][(10, "official total")]) == 3 and len(data["hits"][(15, "official total")]) == 2
    assert data["hits"][(15, "official total")] == [True, True]
    assert data["hits"][(15, "proxy net worth")] == [True, False]
    assert data["paired"][15] == [(True, True), (False, True)]
    text = check.format_table(data, (10, 15, 20))
    assert "4 ended by conquest" in text and "Limits of the proxy" in text
    assert "proxy right only 0, official right only 1" in text


def test_the_script_prints_and_saves_the_table(tmp_path, capsys):
    (tmp_path / "a-results.json").write_text(json.dumps(game("1", player(900, pop=40), player(100))))
    out = tmp_path / "out" / "table.txt"
    assert check.main([str(tmp_path), "--out", str(out), "--minutes", "10"]) == 0
    assert out.read_text().startswith("Score rule check") and "official total" in capsys.readouterr().out
