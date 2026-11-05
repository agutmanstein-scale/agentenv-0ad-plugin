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


def test_the_ai_seed_ceasefire_and_per_seat_behaviour_are_separate_knobs():
    seats = [Seat("Ext", "rome"), Seat("P", "pers", controller="petra", ai_behavior="defensive")]
    s = build_scenario(seats, seed=5, ai_seed=9, ceasefire_minutes=6)["settings"]
    assert (s["Seed"], s["AISeed"], s["Ceasefire"]) == (5, 9, 6) and s["PlayerData"][1]["AIBehavior"] == "defensive"
    assert build_scenario(seats, seed=5)["settings"]["AISeed"] == 5


@pytest.mark.parametrize("behavior", ["random", "balanced", "aggressive", "defensive"])
def test_every_petra_behaviour_the_design_names_is_accepted(behavior):
    seat = Seat("P", "pers", controller="petra", ai_behavior=behavior)
    assert build_scenario([seat])["settings"]["PlayerData"][0]["AIBehavior"] == behavior


def test_an_unknown_behaviour_or_negative_starting_resources_is_refused():
    with pytest.raises(ValueError, match="ai_behavior"):
        build_scenario([Seat("P", "pers", controller="petra", ai_behavior="sneaky")])
    with pytest.raises(ValueError, match="starting_resources"):
        build_scenario([Seat("P", "pers")], starting_resources=-1)


def test_the_experimental_resources_and_population_limit_are_written_only_when_asked_for():
    plain = build_scenario([Seat("A", "rome"), Seat("B", "pers")])["settings"]
    assert "StartingResources" not in plain
    assert all(not {"Resources", "PopulationLimit"} & set(p) for p in plain["PlayerData"])
    seats = [Seat("A", "rome", starting_resources=250, population_limit=75), Seat("B", "pers")]
    s = build_scenario(seats, starting_resources=400)["settings"]
    assert s["StartingResources"] == 400
    assert s["PlayerData"][0]["Resources"] == {"food": 250, "wood": 250, "stone": 250, "metal": 250}
    assert s["PlayerData"][0]["PopulationLimit"] == 75 and "Resources" not in s["PlayerData"][1]
