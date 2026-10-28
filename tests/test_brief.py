"""Text and JSON briefs: content, fog, events and the 3k-token budget."""

import copy
import json
from pathlib import Path

from agentenv_zeroad.brief import TOKEN_BUDGET, approx_tokens, json_brief, json_brief_text, text_brief
from agentenv_zeroad.state import Event, Snapshot, diff_events

FIXTURES = Path(__file__).parent / "fixtures"


def raw(seconds: int) -> dict:
    return json.loads((FIXTURES / f"snapshot_2p_{seconds:03d}s.json").read_text())


def test_text_brief_has_clock_economy_roles_and_events():
    prev, cur = Snapshot.from_json(raw(60)), Snapshot.from_json(raw(120))
    events = diff_events(prev.players[1], cur.players[1], cur.time_ms)
    text = text_brief(cur, 1, events)
    assert text.startswith("Time 02:00. You are Rome (rome), player 1")
    assert "Population" in text and "food" in text
    assert "villager" in text and "infantry_melee" in text
    assert "Enemies in sight: none" in text
    assert "[02:00] trained" in text
    assert approx_tokens(text) <= TOKEN_BUDGET


def test_json_brief_is_compact_and_carries_time_and_events():
    cur = Snapshot.from_json(raw(120))
    events = [Event("phase_reached", 119_000, "reached town phase", {"phase": "town"})]
    brief = json_brief(cur, 2, events)
    assert brief["t"] == "02:00" and brief["player"] == 2 and brief["civ"] == "pers"
    assert brief["events"] == [{"kind": "phase_reached", "t": "01:59", "text": "reached town phase", "phase": "town"}]
    assert set(brief["army"]) <= {"villager", "infantry_melee", "infantry_ranged", "cavalry", "siege", "champion"}
    assert brief["enemies"] == {}
    assert approx_tokens(json_brief_text(cur, 2, events)) < 400


def _huge() -> Snapshot:
    data = copy.deepcopy(raw(120))
    me = data["players"]["1"]
    me["buildings"] = [{"id": 5000 + i, "kind": f"Kind{i % 30}", "pos": [i, i], "hp": 100} for i in range(400)]
    me["techs"] = [f"tech_number_{i}" for i in range(120)]
    me["researching"] = [f"tech_in_progress_{i}" for i in range(10)]
    enemy_buildings = [{"id": 9000 + i, "kind": "House", "pos": [i, i]} for i in range(200)]
    me["enemies"] = {
        str(q): {"units": {"cavalry": 40, "champion": 20}, "buildings": enemy_buildings, "nearest": 50}
        for q in range(2, 9)
    }
    return Snapshot.from_json(data)


def test_text_brief_stays_under_budget_in_a_sprawling_late_game():
    events = [Event("tech_researched", i * 1000, f"researched tech_{i}_" + "x" * 80) for i in range(200)]
    text = text_brief(_huge(), 1, events)
    assert approx_tokens(text) <= TOKEN_BUDGET
    assert "+194 more" in text  # enemy structure list capped
    assert "(200 events, 12 shown)" in text and "tech_199_" in text and "tech_187_" not in text


def _loss(t: int, role: str = "infantry_melee") -> Event:
    return Event("unit_lost", t, f"lost 1 unit(s): 1 {role}", {"count": 1, role: 1})


def test_briefs_keep_urgent_event_and_merge_unit_losses_in_a_battle():
    snap = Snapshot.from_json(raw(120))
    lost_cc = Event("building_lost", 100_000, "lost 1 building(s): CivCentre", {"count": 1, "kinds": ["CivCentre"]})
    others = [Event("tech_researched", 100_100 + i, f"researched t{i}", {"tech": f"t{i}"}) for i in range(15)]
    losses = [_loss(100_200 + 200 * i, "cavalry" if i % 2 else "infantry_melee") for i in range(30)]
    events = [lost_cc, *others, *losses]
    text = text_brief(snap, 1, events).split("Since last brief")[1]
    assert "[01:40] lost 1 building(s): CivCentre" in text
    assert "lost 30 unit(s): 15 cavalry, 15 infantry_melee" in text
    assert "(17 events, 12 shown)" in text
    kinds = [e["kind"] for e in json_brief(snap, 1, events)["events"]]
    assert kinds[0] == "building_lost" and kinds.count("unit_lost") == 1 and len(kinds) == 12


def test_text_brief_hard_cap_marks_truncation():
    snap = _huge()
    names = {q: "N" * 3000 for q in range(1, 9)}
    text = text_brief(snap, 1, (), names=names)
    assert approx_tokens(text) <= TOKEN_BUDGET
    assert text.endswith("[brief truncated]")


def test_visible_enemy_appears_in_both_briefs():
    data = copy.deepcopy(raw(120))
    data["players"]["1"]["enemies"] = {
        "2": {"units": {"cavalry": 2}, "buildings": [{"id": 9, "kind": "Barracks", "pos": [10, 20]}], "nearest": 95}
    }
    snap = Snapshot.from_json(data)
    text = text_brief(snap, 1)
    assert "Persia (player 2): units 2 cavalry, nearest 95 m from base" in text
    assert "Barracks@10,20" in text
    assert json_brief(snap, 1)["enemies"] == {"2": {"units": {"cavalry": 2}, "structures": 1, "dist": 95, "contact": 0}}


def test_briefs_report_finished_buildings_by_kind_and_combat_counts():
    r = raw(120)
    me = r["players"]["1"]
    me["buildings"] = [
        {"id": 1, "kind": "CivCentre", "pos": [10, 10]},
        {"id": 2, "kind": "Barracks", "pos": [20, 20]},
        {"id": 3, "kind": "Barracks", "pos": [30, 30], "foundation": True},
    ]
    me["stats"] = {"enemy_units_killed": 7, "units_lost": 4, "buildings_lost": 1, "units_trained": 12}
    snap = Snapshot.from_json(r)
    brief = json_brief(snap, 1)
    assert brief["bld"] == {"Barracks": 1, "CivCentre": 1}
    assert brief["killed"] == 7 and brief["lost"] == 4
    assert "Combat so far: killed 7 enemy units; lost 4 units and 1 buildings." in text_brief(snap, 1)
