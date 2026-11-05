"""The score rule (official or net worth), games with no time cap and the safety cap, and the game knobs the match
passes to the scenario: how a time-capped game is decided and what results.json records about it."""

import asyncio
from dataclasses import replace

import pytest

from agentenv_zeroad.grading import SCORE_KEYS, resolve_score_rule, time_cap_outcomes
from agentenv_zeroad.match import SAFETY_CAP_S, LiveMatch, MatchConfig, SeatPlan
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

PETRA = SeatPlan("2", 2, "petra", "pers", "Petra 2")
AGENT = SeatPlan("1", 1, "agent", "rome", "alice")
SEATS = (SeatPlan("1", 1, "bot", "rome", "alice"), PETRA)


def make(tmp_path, **config) -> LiveMatch:
    config = {"speed": 2.0, "time_cap_s": 1.0, "tick_s": 0.05, **config}
    return LiveMatch(Handle(), MatchConfig(seats=SEATS, **config), out_dir=tmp_path, placer=placer())


def official_favours_player_2(match: LiveMatch, net_worth: dict[str, int]) -> None:
    """Player 2 leads on the engine's total (120 vs 90 flipped), player 1 or 2 on net worth as given."""
    match.engine.outcome["1"]["total"] = 90
    match.engine.outcome["2"]["total"] = 120
    match.engine.net_worth = net_worth


async def run_out(match: LiveMatch) -> None:
    await match.setup()
    match.start()
    await asyncio.wait_for(match.wait_end(), 10)


def test_the_score_rules_name_the_measure_each_one_ranks_by():
    assert SCORE_KEYS == {"official": "total", "net_worth": "net_worth"}


def test_the_default_rule_is_the_one_the_agents_prompt_states():
    assert resolve_score_rule(None, ["v2"]) == "net_worth" and resolve_score_rule(None, ["v1", "v1"]) == "official"
    assert resolve_score_rule(None, []) == "official"
    assert resolve_score_rule("net_worth", []) == "net_worth" and resolve_score_rule("official", ["v1"]) == "official"


def test_a_rule_the_prompt_contradicts_is_an_error_but_agents_on_both_interfaces_have_no_one_prompt():
    with pytest.raises(ValueError, match="contradicts the agents' prompt"):
        resolve_score_rule("official", ["v2"])
    with pytest.raises(ValueError, match="contradicts the agents' prompt"):
        resolve_score_rule("net_worth", ["v1"])
    assert resolve_score_rule(None, ["v1", "v2"]) == "official"
    assert resolve_score_rule("net_worth", ["v1", "v2"]) == "net_worth"


def test_a_match_config_resolves_its_rule_from_its_agent_seats_and_ignores_other_seats():
    v2 = (replace(AGENT, interface="v2"), PETRA)
    assert MatchConfig(seats=v2).score_rule == "net_worth"
    assert MatchConfig(seats=(AGENT, PETRA)).score_rule == "official"
    bot_v2 = (SeatPlan("1", 1, "bot", "rome", "Bot 1", interface="v2"), PETRA)
    assert MatchConfig(seats=bot_v2).score_rule == "official"
    assert MatchConfig(seats=bot_v2, score_rule="net_worth").score_rule == "net_worth"
    with pytest.raises(ValueError, match="contradicts"):
        MatchConfig(seats=v2, score_rule="official")


def test_a_draw_margin_is_in_the_chosen_measures_own_units():
    assert time_cap_outcomes({"a": 5000.0, "b": 4950.0}, draw_margin=100) == {"a": "drawn", "b": "drawn"}
    assert time_cap_outcomes({"a": 5000.0, "b": 4950.0}, draw_margin=10) == {"a": "won", "b": "lost"}


async def test_the_official_rule_ranks_by_the_engine_total_whatever_the_net_worth(tmp_path):
    match = make(tmp_path)
    official_favours_player_2(match, {"1": 900, "2": 400})
    await run_out(match)
    assert match.ended_by == "time_cap" and match.outcomes == {"1": "lost", "2": "won"}
    assert match.detail().startswith("Petra 2 won on score at the time cap")


async def test_the_net_worth_rule_ranks_by_net_worth_whatever_the_engine_total(tmp_path):
    match = make(tmp_path, score_rule="net_worth")
    official_favours_player_2(match, {"1": 900, "2": 400})
    await run_out(match)
    assert match.ended_by == "time_cap" and match.outcomes == {"1": "won", "2": "lost"}
    assert match.detail().startswith("alice won on net worth at the time cap")


async def test_a_draw_margin_under_the_net_worth_rule_applies_to_net_worth(tmp_path):
    match = make(tmp_path, score_rule="net_worth", draw_margin=100)
    official_favours_player_2(match, {"1": 5000, "2": 4950})
    await run_out(match)
    assert match.outcomes == {"1": "drawn", "2": "drawn"}


async def test_conquest_decides_first_under_either_rule(tmp_path):
    match = make(tmp_path, score_rule="net_worth", time_cap_s=60)
    official_favours_player_2(match, {"1": 900, "2": 400})
    await match.setup()
    match.start()
    match.engine.outcome["1"]["state"] = "defeated"
    await asyncio.wait_for(match.wait_end(), 5)
    assert match.ended_by == "conquest" and match.outcomes == {"1": "lost", "2": "won"}


async def test_a_defeated_player_stays_lost_under_the_net_worth_rule(tmp_path):
    match = make(tmp_path, score_rule="net_worth", time_cap_s=1.0)
    official_favours_player_2(match, {"1": 5000, "2": 100})
    match.engine.outcome["1"]["state"] = "defeated"
    await match.setup()
    match._check_end(1000)
    assert match.ended_by == "conquest"


async def test_results_record_the_rule_both_measures_and_each_timeline_rows_net_worth(tmp_path):
    match = make(tmp_path, score_rule="net_worth", ai_seed=99, ceasefire_minutes=2)
    official_favours_player_2(match, {"1": 900, "2": 400})
    await run_out(match)
    results = match.results()
    assert results["score_rule"] == "net_worth" and results["ended_by"] == "time_cap"
    assert results["ai_seed"] == 99 and results["ceasefire_minutes"] == 2 and results["safety_cap_s"] == SAFETY_CAP_S
    assert results["seats"]["1"]["score"]["total"] == 90 and results["seats"]["1"]["score"]["net_worth"] == 900
    assert results["seats"]["2"]["score"]["total"] == 120 and results["seats"]["2"]["score"]["net_worth"] == 400
    last = results["timeline"][-1]["players"]
    assert last["1"]["net_worth"] == 900 and last["2"]["net_worth"] == 400
    assert results["seats"]["2"]["ai_difficulty"] == 3 and "ai_difficulty" not in results["seats"]["1"]


async def test_a_net_worth_the_snapshot_did_not_carry_keeps_the_last_known_one(tmp_path):
    match = make(tmp_path, time_cap_s=60)
    match.engine.net_worth = {"1": 700}
    await match.setup()
    assert match.scores["1"]["net_worth"] == 700 and match.scores["2"]["net_worth"] == 0
    match.engine.net_worth = {}
    await match._tick()
    assert match.scores["1"]["net_worth"] == 700


async def test_with_no_time_cap_the_game_runs_to_the_safety_cap_and_is_decided_by_the_rule(tmp_path):
    match = make(tmp_path, time_cap_s=0, safety_cap_s=1.0, score_rule="net_worth")
    official_favours_player_2(match, {"1": 900, "2": 400})
    await run_out(match)
    assert match.ended_by == "safety_cap" and match.outcomes == {"1": "won", "2": "lost"}
    assert match.detail().startswith("alice won on net worth at the safety cap")
    results = match.results()
    assert results["ended_by"] == "safety_cap" and results["time_cap_s"] == 0 and results["safety_cap_s"] == 1.0


async def test_a_game_with_no_time_cap_is_not_ended_before_its_safety_cap_and_shows_that_cap(tmp_path):
    match = make(tmp_path, time_cap_s=0)
    await match.setup()
    match.start()
    await asyncio.sleep(0.3)
    assert not match.over and match.config.cap_s == SAFETY_CAP_S == 45 * 60
    assert "/ cap 45:00" in match.footer("1") and "44:5" in match.brief("1").split("score decides: ")[1]
    await match.stop()


async def test_a_time_cap_ends_the_game_as_time_cap_even_with_a_safety_cap(tmp_path):
    match = make(tmp_path, time_cap_s=1.0, safety_cap_s=30)
    await run_out(match)
    assert match.ended_by == "time_cap" and match.config.cap_s == 1.0


def test_the_game_knobs_reach_the_scenario():
    seats = (
        SeatPlan("1", 1, "agent", "rome", "alice"),
        SeatPlan(
            "2", 2, "petra", "pers", "Petra 2", ai_behavior="aggressive", population_limit=80, starting_resources=900
        ),
    )
    settings = MatchConfig(seats=seats, seed=5, ai_seed=77, ceasefire_minutes=3).scenario()["settings"]
    assert settings["Seed"] == 5 and settings["AISeed"] == 77 and settings["Ceasefire"] == 3
    me, petra = settings["PlayerData"]
    assert petra["AIBehavior"] == "aggressive" and "PopulationLimit" in petra and "Resources" in petra
    assert "PopulationLimit" not in me and "StartingResources" not in settings


def test_without_knobs_the_scenario_is_the_one_the_pilot_ran():
    settings = MatchConfig(seats=SEATS, seed=5).scenario()["settings"]
    assert settings["AISeed"] == 5 and settings["Ceasefire"] == 0
    assert {p["AIBehavior"] for p in settings["PlayerData"]} == {"balanced"}
    assert all("Resources" not in p and "PopulationLimit" not in p for p in settings["PlayerData"])
