"""The interface v2 brief and reference: players by number, own score and army, what is affordable, the executor's
give-ups, the per-civ cost tables, and the prompt-size gate. v1's brief is the frozen one (test_interface)."""

import json
import statistics

import httpx
import pytest

import agentenv_zeroad.match as match_module
from agentenv_zeroad.brief_v2 import NOTICES_HEADER, net_worth, notices_in, reference_text
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import Army, Hold
from agentenv_zeroad.server import ZeroADEnv
from tests.prompt_scenarios import all_prompts, request_chars
from tests.served import served
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

pytestmark = pytest.mark.anyio

CHARS_PER_TOKEN = 3.3  # measured on these prompts with the Qwen3 tokenizer (scripts/prompt_tokens.py)
TOKEN_GATE = 2600
MIDGAME_CEILING = 3400  # the busy mid-game prompt is 3,258 real tokens: over the gate, pinned so it does not grow


def make(tmp_path, interface="v2", seconds=300, engine=None) -> LiveMatch:
    seats = (
        SeatPlan("1", 1, "agent", "rome", "policy v3", interface=interface),
        SeatPlan("2", 2, "agent", "athen", "Opus 5.5", ai_difficulty=3, interface=interface),
    )
    handle = Handle(engine or ScriptedEngine(seconds=seconds))
    return LiveMatch(handle, MatchConfig(seats=seats, time_cap_s=1800), out_dir=tmp_path, placer=placer())


async def test_the_v2_brief_names_players_by_number_only(tmp_path):
    for interface, named in (("v1", True), ("v2", False)):
        match = make(tmp_path, interface)
        await match.setup()
        text = match.brief("1") + match.compact_brief("1")
        for name in ("policy v3", "Opus 5.5"):
            assert (name in text) is named, (interface, name)
    brief = make(tmp_path)
    await brief.setup()
    text = brief.brief("1")
    assert "You are player 1 (rome)" in text and "Opponents: player 2 standing." in text
    assert "level" not in text.lower() and "Petra" not in text
    assert json.loads(brief.compact_brief("1"))["players"] == {"1": "player 1", "2": "player 2"}


async def test_the_v2_brief_shows_own_score_army_affordability_and_the_villager_target(tmp_path):
    engine = ScriptedEngine(seconds=300)
    engine.outcome["1"] = {"state": "active", "total": 999, "economy": 640, "military": 125, "exploration": 234}
    match = make(tmp_path, engine=engine)
    await match.setup()
    match.submit("1", Army(action="attack", target="player_2", min_size=12))
    await match._tick()
    text = match.brief("1")
    assert "Score: economy 640, military 125." in text and "999" not in text and "234" not in text
    assert "Net worth" not in text  # the hook returns None until the net-worth stream lands
    assert "Villagers 28 (target 45)" in text
    assert (
        "Army: 2 infantry_melee, 2 infantry_ranged, 1 cavalry (5 soldiers), home. Orders: attack player_2, waves of 12"
        in text
    )
    assert "of 12 gathered at the rally point (5 soldiers), goes in 90 s whatever its size." in text
    assert "Since your last brief: killed 0 enemy units, lost 0 units." in text
    assert "Can afford now: " in text and "build " in text
    assert "Standing orders: economy 55/40/0/5 (food/wood/stone/metal)" in text and text.count("player 2") >= 1
    assert "Time left" not in text and "get_brief()" not in text  # no v1 footer


async def test_losses_and_kills_are_counted_since_the_last_brief(tmp_path):
    engine = ScriptedEngine(seconds=300)
    stats = engine.raw["snapshot"]["players"]["1"].setdefault("stats", {})
    stats.update({"enemy_units_killed": 2, "units_lost": 1})
    match = make(tmp_path, engine=engine)
    await match.setup()
    assert "killed 2 enemy units, lost 1 units" in match.brief("1")
    stats.update({"enemy_units_killed": 7, "units_lost": 4})
    await match._tick()
    assert "killed 5 enemy units, lost 3 units" in match.brief("1")
    assert "killed 0 enemy units, lost 0 units" in match.brief("1")


async def test_the_net_worth_hook_is_shown_once_the_net_worth_stream_returns_a_number(tmp_path, monkeypatch):
    match = make(tmp_path)
    await match.setup()
    assert net_worth(match.snapshot, 1) is None
    monkeypatch.setattr(match_module, "net_worth", lambda snapshot, player: 4321)
    assert "Score: economy 100, military 0. Net worth 4321." in match.brief("1")


async def test_the_executors_give_ups_appear_in_the_next_v2_brief_once(tmp_path):
    match = make(tmp_path)
    await match.setup()
    executor = match.seats["1"].executor
    from agentenv_zeroad.orders import BuildRequest

    request = BuildRequest("field", "near_food", "structures/rome/field")
    for n in range(3):
        executor.fail_attempt(request, 1000 * n, "build field at near_food", "no valid spot at near_food: blocked")
    first = match.brief("1")
    assert NOTICES_HEADER in first and notices_in(first) == [
        "build field at near_food: no valid spot at near_food: blocked"
    ]
    assert NOTICES_HEADER not in match.brief("1")


async def test_hold_is_recorded_as_an_order_and_the_results_name_the_interface(tmp_path):
    match = make(tmp_path)
    await match.setup()
    result = match.submit("1", Hold())
    assert result.status == "accepted" and match.seats["1"].actions[-1].kind == "hold"
    assert match.seats["1"].actions[-1].what == "hold()"
    assert match.results()["seats"]["1"]["interface"] == "v2"
    assert make(tmp_path, "v1").results()["seats"]["2"]["interface"] == "v1"


async def test_the_reference_is_static_per_civ_and_empty_for_a_v1_seat(tmp_path):
    match = make(tmp_path)
    await match.setup()
    other = make(tmp_path)
    await other.setup()
    ref = match.reference("1")
    executor = match.seats["1"].executor
    assert ref == other.reference("1") == reference_text(executor.roles, executor.last_seat)
    assert ref.startswith("Costs for rome") and "infantry_melee: 50f40w10m" in ref and "Buildings: " in ref
    assert "barracks 200w100s" in ref and ref.endswith("Phase advance: phase_city_generic 750m750s.")
    assert "Time" not in ref and "player" not in ref  # nothing that changes with the match
    village = make(tmp_path, seconds=60)
    await village.setup()
    assert village.reference("1").endswith("Phase advance: phase_town_generic 500f500w.")
    v1 = make(tmp_path, "v1")
    await v1.setup()
    assert v1.reference("1") == ""


async def test_the_reference_route_serves_the_seats_text_at_its_own_address(tmp_path):
    async def start():
        return Handle(ScriptedEngine())

    env = ZeroADEnv(start, tmp_path, tick_s=0.05, match_options={"placer": placer(), "environ": {}})
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        lobby = f"{base}/agentenv/ext/lobby"
        await http.post(f"{lobby}/open", json={"game_settings": {"speed": 2.0, "time_cap_minutes": 0.1}})
        v2 = {"civ": "rome", "interface": "v2"}
        await http.post(f"{lobby}/fill", json={"player_id": "1", "player_kind": "agent", "game_settings": v2})
        await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "agent"})
        assert (await http.post(f"{lobby}/close", json={})).json()["status"] == "closed"
        mine = (await http.get("/players/1/reference")).json()
        theirs = (await http.get("/players/2/reference")).json()
    assert mine["ok"] and mine["text"].startswith("Costs for rome") and theirs == {"ok": True, "text": ""}


def tokens(prompts):
    return sorted(request_chars(p) / CHARS_PER_TOKEN for p in prompts)


def test_the_v2_prompt_is_within_the_token_gate_on_the_recorded_snapshots():
    early = [p for p in all_prompts("v2") if p.name != "midgame"]
    sizes = tokens(early)
    p90 = sizes[min(len(sizes) - 1, round(0.9 * (len(sizes) - 1)))]
    assert len(early) == 4 and p90 <= TOKEN_GATE, (statistics.median(sizes), p90)


def test_the_busy_midgame_prompt_stays_under_its_ceiling_and_carries_what_makes_it_busy():
    [mid] = [p for p in all_prompts("v2") if p.name == "midgame"]
    for needle in ("Wave out:", "Next wave:", NOTICES_HEADER, "Enemies in sight:", "under attack", "executor gave up"):
        assert needle in mid.user, needle
    assert mid.user.count(" -> ") == 18
    [size] = tokens([mid])
    assert size <= MIDGAME_CEILING, size


def test_v2_adds_no_more_than_the_gate_allows_over_v1_and_keeps_the_system_and_tools_near_todays_size():
    v1 = [request_chars(p) for p in all_prompts("v1")]
    v2 = [request_chars(p) for p in all_prompts("v2")]
    assert max(v2) < 1.2 * max(v1)
