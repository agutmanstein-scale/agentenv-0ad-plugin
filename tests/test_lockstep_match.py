"""A lockstep match on a scripted engine: the seat tick in game time, a Commander run as a library through the
in-process link, missed and failed decisions, and a dead engine."""

import asyncio
import json
import time

import pytest

from agentenv_zeroad.brains import Commander, CommanderSeat, CommanderSettings, InProcessSeatLink
from agentenv_zeroad.grading import BRAIN_TIMEOUT_S
from agentenv_zeroad.match import MatchConfig, SeatPlan
from agentenv_zeroad.server import tool_specs
from agentenv_zeroad.spend import SpendCapReached
from agentenv_zeroad.train.game import LockstepMatch
from tests.fake_models import FakeStrategist
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

pytestmark = pytest.mark.anyio

AGENT_VS_PETRA = (SeatPlan("1", 1, "agent", "rome", "alice"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))
BOT_VS_PETRA = (SeatPlan("1", 1, "bot", "rome", "Bot 1"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))


def make(
    tmp_path, seats=AGENT_VS_PETRA, *, handle=None, cap_s=30.0, timeout=BRAIN_TIMEOUT_S, **kwargs
) -> LockstepMatch:
    config = MatchConfig(seats=seats, time_cap_s=cap_s)
    return LockstepMatch(
        handle or Handle(), config, out_dir=tmp_path, placer=placer(), brain_timeout_s=timeout, **kwargs
    )


async def seat_of(match, model, *, slot="1", **settings) -> CommanderSeat:
    link = InProcessSeatLink(match, slot, await tool_specs())
    commander = Commander(model, CommanderSettings(game_clock=True, log_turns=True, **settings))
    seat = CommanderSeat(commander, link, slot, await link.tools())
    match.lockstep.add_brain(seat)
    return seat


async def play(match) -> dict:
    await match.setup()
    match.start()
    await match.wait_end()
    return json.loads(match.files["results"].read_text())


async def test_a_match_without_brains_runs_to_the_time_cap_at_engine_speed_and_ticks_once_per_game_second(tmp_path):
    handle = Handle()
    match = make(tmp_path, BOT_VS_PETRA, handle=handle, cap_s=60.0)
    started = time.monotonic()
    results = await play(match)
    assert time.monotonic() - started < 10  # a live match takes the 60 game seconds
    assert match.ended_by == "time_cap" and results["status"] == "finished" and results["ended_by"] == "time_cap"
    assert match.clock.steps == 300 and handle.engine.raw_steps == 300
    assert match.ticks == 61  # setup's tick, then one at every game second up to the cap
    assert results["clock"]["lockstep"] is True and results["clock"]["latency_ms"] == 5000
    assert results["clock"]["overruns"] == 0 and handle.closed == 1
    assert [row["t_ms"] for row in results["timeline"]][:3] == [0, 10_000, 20_000]


async def test_a_game_without_a_time_cap_ends_at_the_safety_cap_and_its_briefs_count_down_to_it(tmp_path):
    config = MatchConfig(seats=BOT_VS_PETRA, time_cap_s=0, safety_cap_s=20.0)
    match = LockstepMatch(Handle(), config, out_dir=tmp_path, placer=placer())
    results = await play(match)
    assert results["ended_by"] == "safety_cap" and results["status"] == "finished"
    assert results["time_cap_s"] == 0 and results["safety_cap_s"] == 20.0 and results["score_rule"] == "official"
    assert "at the safety cap" in results["detail"]
    assert match.cap_ms == 20_000
    brief = match.brief("1")
    assert "Time left before the score decides: 00:00." in brief and " / cap 00:20 " in brief.splitlines()[-1]


async def test_a_capped_game_still_tells_its_brief_how_long_is_left_and_ends_as_time_cap(tmp_path):
    match = make(tmp_path, BOT_VS_PETRA, cap_s=20.0)
    results = await play(match)
    assert results["ended_by"] == "time_cap" and results["time_cap_s"] == 20.0
    brief = match.brief("1")
    assert "Time left before the score decides: " in brief and " / cap 00:20 " in brief.splitlines()[-1]


async def test_every_command_sent_to_the_engine_is_in_the_results_by_the_step_it_went_out_with(tmp_path):
    match = make(tmp_path, BOT_VS_PETRA, cap_s=10.0)
    await match.setup()
    stop = {"type": "stop", "entities": [5]}
    match.clock.queue(2, stop)
    match.clock.queue(1, {"type": "stop", "entities": [4]})
    match.start()
    await match.wait_end()
    results = json.loads(match.files["results"].read_text())
    assert results["commands"] == {"0": [[1, {"type": "stop", "entities": [4]}], [2, stop]]}
    assert match.engine.steps[0] == [(1, {"type": "stop", "entities": [4]}), (2, stop)]


async def test_a_commanders_orders_land_five_game_seconds_after_its_brief_whatever_the_model_takes(tmp_path):
    match = make(tmp_path, cap_s=60.0)
    model = FakeStrategist([("set_economy", {"food": 50, "wood": 30, "stone": 10, "metal": 10})], delay_s=0.05)
    await seat_of(match, model)
    results = await play(match)
    seat = results["seats"]["1"]
    first = results["lockstep"]["decisions"][0]
    assert first == {"slot": "1", "frozen_ms": 0, "due_ms": 5000, "applied_ms": 5000, "missed": False}
    assert [d["frozen_ms"] for d in results["lockstep"]["decisions"]] == [0, 25_000, 50_000]
    assert seat["actions"][0]["time_ms"] == 5000 and seat["actions"][0]["status"] == "accepted"
    turn = seat["turns"][0]
    assert turn["began_ms"] == 0 and turn["reported_ms"] == 5000
    assert turn["prompt_token_ids"] == [1, 2, 3] and turn["token_ids"] == [7, 8]
    assert turn["logprobs"] == [-0.5, -1.25] and len(turn["logprobs"]) == len(turn["token_ids"])
    assert seat["moves"] == {"made": 3, "missed": 0} and seat["metrics"]["decisions"] == 3
    assert results["clock"]["decisions"] == 3 and results["clock"]["applied"] == 3


async def test_a_decisions_brief_is_frozen_when_it_falls_due_not_when_the_reply_arrives(tmp_path):
    match = make(tmp_path, cap_s=30.0)
    model = FakeStrategist(delay_s=0.02)
    await seat_of(match, model)
    await play(match)
    assert [user.split(".")[0] for _, user, _ in model.requests] == ["Time 00:00", "Time 00:25"]
    assert model.requests[0][0].startswith("You are the strategist")


async def test_a_reply_that_never_comes_is_a_missed_call_and_the_match_goes_on_to_its_cap(tmp_path):
    match = make(tmp_path, cap_s=20.0, timeout=0.05)
    seat = await seat_of(match, FakeStrategist(gate=asyncio.Event()))
    results = await play(match)
    seat_results = results["seats"]["1"]
    assert results["ended_by"] == "time_cap" and results["game_time_s"] >= 20
    assert seat_results["moves"] == {"made": 0, "missed": 1} and seat_results["brain_calls"][0]["error"]
    assert "no reply within 0.05 s" in seat_results["brain_calls"][0]["error"]
    assert results["clock"]["missed"] == 1 and seat.commander.record.errors == 1
    assert seat_results["forfeited"] is True


async def test_a_model_that_fails_costs_one_call_and_the_next_decision_tries_again(tmp_path):
    match = make(tmp_path, cap_s=60.0)
    seat = await seat_of(match, FakeStrategist(fail=RuntimeError("503")))
    results = await play(match)
    assert seat.commander.record.calls == 3 and seat.commander.record.errors == 3
    assert results["seats"]["1"]["moves"] == {"made": 0, "missed": 3} and results["ended_by"] == "time_cap"


async def test_the_spend_cap_ends_the_seats_decisions_but_not_the_match(tmp_path):
    match = make(tmp_path, cap_s=60.0)
    seat = await seat_of(match, FakeStrategist(fail=SpendCapReached("cap")))
    results = await play(match)
    assert seat.commander.record.end_reason == "spend_cap (match_over)" and seat.commander.record.calls == 0
    assert results["ended_by"] == "time_cap" and len(results["lockstep"]["decisions"]) == 1


async def test_two_commanders_decide_at_the_same_game_time_and_both_land_at_the_deadline(tmp_path):
    seats = (SeatPlan("1", 1, "agent", "rome", "alice"), SeatPlan("2", 2, "agent", "pers", "bob"))
    match = make(tmp_path, seats, cap_s=30.0)
    await seat_of(match, FakeStrategist(delay_s=0.03), slot="1")
    await seat_of(match, FakeStrategist(delay_s=0.0), slot="2")
    results = await play(match)
    first = [d for d in results["lockstep"]["decisions"] if d["frozen_ms"] == 0]
    assert {d["slot"] for d in first} == {"1", "2"} and all(d["applied_ms"] == 5000 for d in first)


async def test_an_engine_that_dies_fails_the_match_and_still_writes_its_results(tmp_path):
    engine = ScriptedEngine()
    original = engine.step_raw
    calls = 0

    async def dying(commands=()):
        nonlocal calls
        calls += 1
        if calls > 12:
            raise RuntimeError("engine gone")
        await original(commands)

    engine.step_raw = dying
    match = make(tmp_path, BOT_VS_PETRA, handle=Handle(engine), cap_s=600.0)
    results = await play(match)
    assert results["status"] == "failed" and "engine gone" in results["detail"] and results["ended_by"] == "failed"


async def test_a_lockstep_match_refuses_a_luna_reflex_seat(tmp_path):
    seats = (SeatPlan("1", 1, "agent", "rome", "alice", reflex="luna"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))
    with pytest.raises(ValueError, match="no Luna reflex"):
        make(tmp_path, seats)
