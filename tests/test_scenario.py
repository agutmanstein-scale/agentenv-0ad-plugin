import json

import pytest

from agentenv_zeroad.scenario import CONQUEST_TRIGGER_SCRIPTS, Seat, build_scenario, scenario_json


def test_random_map_scenario_has_the_fields_0_28_generate_map_needs():
    cfg = build_scenario([Seat("A", "rome"), Seat("B", "athen")], map_name="mainland", size=128, seed=11)

    assert cfg["mapType"] == "random"
    assert cfg["map"] == "maps/random/mainland"
    assert cfg["script"] == "mainland.js"
    s = cfg["settings"]
    assert s["mapName"] == "mainland"
    assert s["Biome"] == "generic/temperate"
    assert s["PlayerPlacement"] == "circle"
    assert s["CircularMap"] is True
    assert s["Size"] == 128
    assert s["Seed"] == 11 and s["AISeed"] == 11
    assert s["Ceasefire"] == 0
    assert all(set(p["Color"]) == {"r", "g", "b"} for p in s["PlayerData"])


def test_conquest_adds_victory_condition_and_trigger_scripts():
    s = build_scenario([Seat("A", "rome")])["settings"]
    assert s["VictoryConditions"] == ["conquest"]
    assert s["TriggerScripts"] == list(CONQUEST_TRIGGER_SCRIPTS)

    s = build_scenario([Seat("A", "rome")], conquest=False)["settings"]
    assert s["VictoryConditions"] == [] and s["TriggerScripts"] == []


def test_external_seats_have_no_ai_and_petra_seats_carry_difficulty_and_behaviour():
    seats = [
        Seat("Ext", "rome", team=0),
        Seat("Bot", "pers", controller="petra", ai_difficulty=4, ai_behavior="aggressive"),
    ]
    ext, bot = build_scenario(seats)["settings"]["PlayerData"]

    assert ext == {
        "Name": "Ext",
        "Civ": "rome",
        "Color": {"r": 150, "g": 20, "b": 20},
        "AI": "",
        "AIDiff": 0,
        "AIBehavior": "balanced",
        "Team": 0,
    }
    assert bot["AI"] == "petra" and bot["AIDiff"] == 4 and bot["AIBehavior"] == "aggressive" and bot["Team"] == -1


def test_explicit_color_and_ai_seed_override_defaults():
    cfg = build_scenario([Seat("A", "rome", color=(1, 2, 3))], seed=5, ai_seed=9)
    assert cfg["settings"]["PlayerData"][0]["Color"] == {"r": 1, "g": 2, "b": 3}
    assert cfg["settings"]["AISeed"] == 9


def test_scenario_json_round_trips():
    seats = [Seat("A", "rome"), Seat("B", "cart", controller="petra")]
    assert json.loads(scenario_json(seats, placement="river")) == build_scenario(seats, placement="river")


@pytest.mark.parametrize(
    ("seats", "kwargs"),
    [
        ([], {}),
        ([Seat(f"P{i}", "rome") for i in range(9)], {}),
        ([Seat("A", "rome")], {"placement": "spiral"}),
        ([Seat("A", "rome")], {"size": 100}),
        ([Seat("A", "rome")], {"ceasefire_minutes": -1}),
        ([Seat("A", "")], {}),
        ([Seat("A", "rome", controller="human")], {}),  # type: ignore[arg-type]
    ],
)
def test_invalid_scenarios_are_rejected(seats, kwargs):
    with pytest.raises(ValueError):
        build_scenario(seats, **kwargs)
