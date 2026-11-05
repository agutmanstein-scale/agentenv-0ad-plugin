"""The interface switch: v1 stays byte-identical to the frozen output, v2 is opt-in per seat and per brain."""

import importlib.util
import json
from pathlib import Path

import pytest

from agentenv_zeroad.brains import (
    ORDER_TOOLS,
    ORDER_TOOLS_V2,
    STRATEGIST_SYSTEM,
    STRATEGIST_SYSTEM_V2,
    CommanderSettings,
    SeatLink,
    strategist_system,
)
from agentenv_zeroad.interface import DEFAULT_INTERFACE, parse_interface
from agentenv_zeroad.orders import Army, Hold, SetEconomy
from agentenv_zeroad.pilot.games import PROMPT, PROMPT_V2, GameSpec, PilotSettings, petra, snapshot, task
from agentenv_zeroad.server import SlotSettings, ZeroADEnv
from tests.executor_fixtures import make_executor
from tests.v1_baseline import FIXTURE, _Session, capture

FROZEN = json.loads(FIXTURE.read_text())


def test_the_default_interface_is_v1_everywhere():
    assert DEFAULT_INTERFACE == "v1" and parse_interface(None) == "v1" and parse_interface("") == "v1"
    assert SlotSettings().interface == "v1" and CommanderSettings().interface == "v1"
    assert CommanderSettings().system_prompt == STRATEGIST_SYSTEM and strategist_system("v1") == STRATEGIST_SYSTEM
    assert make_executor()[0].interface == "v1"


def test_a_commanders_default_prompt_follows_its_interface():
    assert CommanderSettings(interface="v2").system_prompt == STRATEGIST_SYSTEM_V2
    assert CommanderSettings(interface="v2", system_prompt="mine").system_prompt == "mine"


def test_an_unknown_interface_is_an_error_not_a_silent_v1():
    assert parse_interface("v2") == "v2"
    with pytest.raises(ValueError, match="ZEROAD_INTERFACE"):
        parse_interface("v3")
    with pytest.raises(ValueError):
        SlotSettings(interface="v3")


def test_v1_prompt_tool_schemas_briefs_results_and_requests_are_byte_identical_to_the_frozen_fixture():
    now = capture()
    assert now["system"] == FROZEN["system"] == STRATEGIST_SYSTEM
    assert now["opening"] == FROZEN["opening"] == PROMPT
    assert json.dumps(now["tools"]) == json.dumps(FROZEN["tools"])
    for seconds, frozen in FROZEN["snapshots"].items():
        got = now["snapshots"][seconds]
        for key in ("results", "brief", "compact_brief", "user"):
            assert got[key] == frozen[key], (seconds, key)
        assert json.dumps(got["request"]) == json.dumps(frozen["request"]), seconds
    assert now["mcp"] == FROZEN["mcp"]  # get_brief and an order's result text over the MCP tool path


def test_the_frozen_v1_tools_are_the_seven_order_tools_without_v2s_parameters():
    names = [t["name"] for t in FROZEN["tools"]]
    assert names == sorted(ORDER_TOOLS)
    props = {t["name"]: set(t["parameters"]["properties"]) for t in FROZEN["tools"]}
    assert props["army"] == {"action", "target"} and props["set_economy"] == {"food", "wood", "stone", "metal"}


async def _tools(interface: str):
    link = SeatLink(_Session(ZeroADEnv().create_app()), "http://env/players/1/mcp", None, interface)  # type: ignore[arg-type]
    return {t.name: t for t in await link.tools()}


@pytest.mark.anyio
async def test_a_v2_seat_is_offered_hold_and_the_wave_and_villager_parameters_in_compact_schemas():
    tools = await _tools("v2")
    assert set(tools) == set(ORDER_TOOLS_V2)
    army, economy = tools["army"].parameters, tools["set_economy"].parameters
    assert {"min_size", "regroup_below"} <= set(army["properties"]) and army["required"] == ["action"]
    assert "villagers" in economy["properties"] and "villagers" not in economy["required"]
    flat = json.dumps(tools["army"].parameters) + json.dumps(economy)
    assert "title" not in flat and "null" not in flat and "minimum" not in flat
    assert tools["army"].description == next(t["description"] for t in FROZEN["tools"] if t["name"] == "army")


@pytest.mark.anyio
async def test_a_v1_seat_is_not_offered_hold():
    assert "hold" not in await _tools("v1")


def test_a_v1_seat_refuses_what_only_v2_has_and_a_v2_seat_takes_it():
    v1, _, _ = make_executor(interface="v1")
    for order in (
        Hold(),
        Army(action="attack", target="player_2", min_size=12),
        SetEconomy(food=25, wood=25, stone=25, metal=25, villagers=60),
    ):
        refused = v1.submit(order)
        assert refused.code == "not_in_v1" and not refused.ok
    v2, _, _ = make_executor(interface="v2")
    assert [v2.submit(o).status for o in (Hold(), Army(action="attack", target="player_2", min_size=12))] == [
        "accepted"
    ] * 2


def test_the_pilot_opening_and_the_commanders_prompt_follow_the_interface():
    assert strategist_system("v2") == STRATEGIST_SYSTEM_V2 and PROMPT_V2 != PROMPT
    for stale in ("exploration", "4:00", "keep the attack standing", "adds score"):
        assert stale not in STRATEGIST_SYSTEM_V2 + PROMPT_V2
    for rule in ("conquest", "net worth", "min_size", "regroup_below", "hold()", "one say()"):
        assert rule in STRATEGIST_SYSTEM_V2


def test_the_lobby_carries_the_slot_interface_into_the_seat_plan_and_refuses_it_for_petra():
    from agentenv_game import PlayerKind, PlayerSlot

    agent = PlayerSlot(player_id="1", player_kind=PlayerKind.AGENT, player_name="a", game_settings={"interface": "v2"})
    assert ZeroADEnv._plan(agent).interface == "v2"
    plain = PlayerSlot(player_id="2", player_kind=PlayerKind.AGENT, player_name="b", game_settings={})
    assert ZeroADEnv._plan(plain).interface == "v1"
    petra = PlayerSlot(
        player_id="2", player_kind=PlayerKind.AI, game_settings={"controller": "petra", "interface": "v2"}
    )
    with pytest.raises(ValueError, match="interface is for seats"):
        ZeroADEnv().slot_rules(petra, None)  # type: ignore[arg-type]


def _steps(opponent, settings):
    spec = GameSpec("g1", 1, "train", 7, opponent, 2, policy_slot="1", policy_civ="rome", opponent_civ="pers")
    return {s["id"]: s for s in task(spec, settings, "/out")}


def test_a_v1_pilot_task_has_no_interface_anywhere_and_a_v2_task_sets_it_on_every_agent_and_slot():
    v1 = _steps(snapshot(1), PilotSettings("http://vllm", "key"))
    v2 = _steps(snapshot(1), PilotSettings("http://vllm", "key", interface="v2"))
    assert "interface" not in json.dumps(v1).lower() and v1["play-policy"]["prompt"] == PROMPT
    assert v2["agent-policy"]["env_vars"]["ZEROAD_INTERFACE"] == "v2"
    assert v2["agent-opponent"]["env_vars"]["ZEROAD_INTERFACE"] == "v2"
    assert v2["slot-1"]["game_settings"]["interface"] == v2["slot-2"]["game_settings"]["interface"] == "v2"
    assert v2["play-policy"]["prompt"] == v2["play-opponent"]["prompt"] == PROMPT_V2
    petra_game = _steps(petra(2), PilotSettings("http://vllm", "key", interface="v2"))
    assert "interface" not in petra_game["slot-2"]["game_settings"]  # Petra has none


def test_the_commander_agent_imports_the_interface_names_it_reads():
    path = Path(__file__).parent.parent / "agents" / "zeroad-commander" / "agent.py"
    spec = importlib.util.spec_from_file_location("zeroad_commander_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ENV_VAR == "ZEROAD_INTERFACE" and module.strategist_system("v2") == STRATEGIST_SYSTEM_V2


def test_the_pilot_runner_takes_the_interface_and_defaults_to_v1(monkeypatch):
    from agentenv_zeroad.pilot import run

    monkeypatch.setenv("VLLM_API_KEY", "k")

    assert run.parse(["--home", "/pilot"]).interface == "v1"
    assert run.parse(["--home", "/pilot", "--interface", "v2"]).interface == "v2"
