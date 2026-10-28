"""The env in process on a scripted engine: lobby rules, the start gate, the per-seat tools, brain-call reports, the
match report, grading and files."""

import asyncio
import json
from contextlib import contextmanager

import httpx
import pytest
from agentenv_game import GameError, SlotRequest
from agentenv_game.grade import GradeMatchTaskStep
from agentenv_game.routing import PLAYER

from agentenv_zeroad.orders import ORDER_TYPES
from agentenv_zeroad.server import ZeroADEnv
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

FAST = {"speed": 2.0, "time_cap_minutes": 60, "start_grace_seconds": 30}


@pytest.fixture
def handles():
    return []


@pytest.fixture
async def env(tmp_path, handles):
    async def start():
        handles.append(Handle())
        return handles[-1]

    e = ZeroADEnv(start, tmp_path, tick_s=0.05, match_options={"placer": placer()})
    e.create_app()
    yield e
    if e.live is not None:
        await e.live.stop()


@contextmanager
def playing(player_id):
    token = PLAYER.set(player_id)
    try:
        yield
    finally:
        PLAYER.reset(token)


async def call(env, player_id, tool, **args) -> str:
    with playing(player_id):
        result = await env.mcp.call_tool(tool, args)
    content = result[0] if isinstance(result, tuple) else result
    return content[0].text


async def agent_vs_petra(env, **settings):
    env.new_lobby({**FAST, **settings})
    env.fill_slot(SlotRequest(player_id="1", player_kind="agent", player_name="alice", game_settings={"civ": "rome"}))
    env.fill_slot(SlotRequest(player_id="2", player_kind="ai", game_settings={"civ": "pers"}))
    await env.close_lobby()


async def test_player_slots_are_the_engines_player_numbers(env):
    env.new_lobby(FAST)
    with pytest.raises(GameError, match="player numbers 1 to 8"):
        env.fill_slot(SlotRequest(player_id="red", player_kind="ai"))
    with pytest.raises(GameError, match="controller is for ai slots"):
        bot = {"controller": "bot"}
        env.fill_slot(SlotRequest(player_id="1", player_kind="agent", player_name="a", game_settings=bot))
    with pytest.raises(GameError, match="civ must be one of"):
        env.fill_slot(SlotRequest(player_id="1", player_kind="ai", game_settings={"civ": "klingon"}))
    with pytest.raises(GameError, match="speed"):
        env.new_lobby({"speed": 3.0})


async def test_slots_with_a_gap_fail_the_lobby_and_start_no_engine(env, handles):
    env.new_lobby(FAST)
    env.fill_slot(SlotRequest(player_id="1", player_kind="ai"))
    env.fill_slot(SlotRequest(player_id="3", player_kind="ai"))
    with pytest.raises(ValueError, match="1 to 2 without gaps"):
        await env.close_lobby()
    assert env.lobby.status.value == "failed" and handles == []


async def test_a_match_of_the_games_own_ai_starts_as_the_lobby_closes(env, handles):
    env.new_lobby(FAST)
    env.fill_slot(SlotRequest(player_id="1", player_kind="ai", game_settings={"controller": "bot"}))
    env.fill_slot(SlotRequest(player_id="2", player_kind="ai"))
    await env.close_lobby()
    assert env.match.status.value == "started" and env.live.started
    assert [(s.plan.kind, s.plan.civ, s.plan.name) for s in env.live.seats.values()] == [
        ("bot", "rome", "Bot 1"),
        ("petra", "athen", "Petra 2"),
    ]
    assert env.match.player_states["1"].moves is None


async def test_an_agents_first_call_starts_the_match(env):
    await agent_vs_petra(env)
    assert env.match.status.value == "not_started" and not env.live.started
    brief = await call(env, "1", "get_brief")
    assert brief.startswith("Time 00:00.") and env.match.status.value == "started" and env.live.started


async def test_a_silent_agent_does_not_hold_the_match_past_the_grace_period(env):
    await agent_vs_petra(env, start_grace_seconds=0.1)
    await asyncio.sleep(0.3)
    match = env.match
    assert match.status.value == "started" and match.status_detail == "started after 0 s without 1"


async def test_an_order_returns_its_status_message_and_the_brief_footer(env):
    await agent_vs_petra(env)
    result = json.loads(await call(env, "1", "train", role="villager", count=5))
    assert result["status"] == "accepted" and "Queued 5 villager" in result["message"]
    assert result["footer"] == env.live.footer("1")
    assert env.live.seats["1"].executor.book.pending_count("villager") == 5


async def test_a_bad_order_is_rejected_in_protocol_style(env):
    await agent_vs_petra(env)
    bad = json.loads(await call(env, "1", "set_economy", food=50, wood=40, stone=0, metal=0))
    assert bad["status"] == "rejected" and bad["code"] == "bad_arguments" and "must be 100" in bad["message"]
    where = json.loads(await call(env, "1", "build", structure="barracks", where="downtown"))
    assert where["code"] == "bad_arguments" and "toward_<player>" in where["message"]
    unknown = json.loads(await call(env, "1", "train", role="champion", count=1, at="moon"))
    assert unknown["status"] == "rejected" and unknown["code"] == "bad_building"
    assert unknown["alternatives"] == ["barracks", "fortress", "stable"] and unknown["footer"] == env.live.footer("1")


async def test_the_envs_own_address_and_an_ai_seat_play_no_one(env):
    await agent_vs_petra(env)
    own = json.loads(await call(env, None, "get_brief"))
    assert own["code"] == "no_seat" and own["alternatives"] == ["/players/1/mcp"]
    ai = json.loads(await call(env, "2", "army", action="attack", target="player_1"))
    assert ai["code"] == "unknown_player"


async def test_every_order_tool_takes_its_order_models_fields(env):
    tools = {t.name: t.inputSchema for t in await env.mcp.list_tools()}
    assert set(tools) == {"get_brief"} | {m.model_fields["kind"].default for m in ORDER_TYPES}
    for model in ORDER_TYPES:
        expected = {k: v for k, v in model.model_json_schema()["properties"].items() if k != "kind"}
        got = tools[model.model_fields["kind"].default].get("properties", {})
        assert set(got) == set(expected)
        for name, prop in expected.items():
            for key in ("type", "enum", "minimum", "maximum", "minLength", "maxLength", "default"):
                assert got[name].get(key) == prop.get(key), (model.__name__, name, key)


async def test_tool_calls_are_timed_per_seat(env):
    await agent_vs_petra(env)
    await call(env, "1", "get_brief")
    await call(env, "1", "army", action="defend", target="home")
    calls = env.live.seats["1"].tool_calls
    assert [c.tool for c in calls] == ["get_brief", "army"] and all(c.seconds >= 0 for c in calls)
    assert env.match.player_states["1"].moves.model_dump() == {"made": 2, "missed": 0}


async def test_brain_calls_reported_at_the_seats_address_drive_its_moves_and_the_forfeit_rule(env):
    await agent_vs_petra(env, time_cap_minutes=0.02)
    app = env.mcp.streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://env") as http:
        for body in ({"seconds": 12.5}, {"seconds": 3, "error": "rate limited"}, {"seconds": 150}):
            assert (await http.post("/players/1/brain_calls", json=body)).json()["ok"]
        assert (await http.post("/players/2/brain_calls", json={"seconds": 1})).status_code == 404
        assert (await http.post("/brain_calls", json={"seconds": 1})).status_code == 400
        assert (await http.post("/players/1/brain_calls", json={"secs": 1})).json()["error"]["code"] == "bad_request"
    assert env.match.player_states["1"].moves.model_dump() == {"made": 1, "missed": 2}
    await call(env, "1", "get_brief")
    match = await asyncio.wait_for(env.finish_match(), 10)
    assert match.status.value == "finished" and match.player_states["1"].status.value == "won"
    step = GradeMatchTaskStep("grade", None, env_id="zeroad", max_missed=0.5)
    rows = step.grade(match.model_dump(mode="json"), env.lobby.model_dump(mode="json"))["grade"]
    assert next(r for r in rows if r["name"] == "missed")["result"] is False
    results = json.loads(env.live.files["results"].read_text())["seats"]["1"]
    assert results["forfeited"] and results["grade"] == 0.0 and results["outcome"] == "won"


async def test_finish_plays_the_match_out_and_its_files_are_the_results_and_the_replay(env, handles):
    await agent_vs_petra(env, time_cap_minutes=0.02)
    await call(env, "1", "get_brief")
    match = await asyncio.wait_for(env.finish_match(), 10)
    assert match.status.value == "finished" and match.status_detail.startswith("alice won on score")
    agent = json.loads(env.live.files["results"].read_text())["seats"]["1"]
    # an agent whose brain reports no calls is graded by its tool calls
    assert agent["outcome"] == "won" and agent["grade"] == 1.0 and agent["forfeited"] is False
    assert [s.name for s in match.player_states["2"].scores] == ["score", "economy", "military", "exploration"]
    late = json.loads(await call(env, "1", "advance_phase"))
    assert late["code"] in ("match_over", "no_match") or late["status"] == "rejected"
    listed = await env.list_match_files()
    lobby_id = env.lobby.lobby_id
    assert [(f["kind"], f["name"]) for f in listed["files"]] == [
        ("results", f"zeroad-{lobby_id}-results.json"),
        ("replay", f"zeroad-{lobby_id}-commands.txt"),
    ]
    assert handles[0].closed == 1
    with pytest.raises(GameError):
        await env.list_match_files(["video"])


async def test_cancel_stops_the_engine_and_a_new_lobby_replaces_the_match(env, handles):
    await agent_vs_petra(env)
    env.cancel_match()
    await asyncio.sleep(0.1)
    assert env.match.status.value == "cancelled" and handles[0].closed == 1
    await agent_vs_petra(env)
    assert len(handles) == 2 and env.live.handle is handles[1]
    env.new_lobby(FAST)
    await asyncio.sleep(0.1)
    assert env.live is None and handles[1].closed == 1


async def test_a_brain_call_that_is_not_a_finite_non_negative_time_is_refused(env):
    await agent_vs_petra(env)
    app = env.mcp.streamable_http_app()
    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://env") as http:
        for raw in ('{"seconds": NaN}', '{"seconds": Infinity}', '{"seconds": -5}', '{"seconds": 1, "error": 7}'):
            answer = await http.post("/players/1/brain_calls", content=raw, headers=headers)
            assert answer.status_code == 400 and answer.json()["error"]["code"] == "bad_request", raw
    assert env.live.seats["1"].brain_calls == []


async def test_no_order_is_taken_while_the_match_is_played_out(env):
    await agent_vs_petra(env)
    await call(env, "1", "get_brief")
    finishing = asyncio.create_task(env.finish_match())
    await asyncio.sleep(0.2)
    assert env.playing_out and not env.live.over
    late = json.loads(await call(env, "1", "train", role="villager", count=5))
    assert late["status"] == "rejected" and late["code"] == "match_over"
    assert env.live.seats["1"].executor.book.pending_count("villager") == 0
    finishing.cancel()
    await asyncio.gather(finishing, return_exceptions=True)


async def test_an_engine_whose_setup_fails_is_stopped_and_no_match_is_kept(tmp_path):
    handles = []

    async def start():
        handle = Handle()

        async def broken(scenario, *, save_replay=False):
            raise RuntimeError("reset failed")

        handle.engine.reset = broken
        handles.append(handle)
        return handle

    env = ZeroADEnv(start, tmp_path, tick_s=0.05, match_options={"placer": placer()})
    env.create_app()
    with pytest.raises(Exception, match="reset failed"):
        await agent_vs_petra(env)
    assert len(handles) == 1 and handles[0].closed == 1 and env.live is None
