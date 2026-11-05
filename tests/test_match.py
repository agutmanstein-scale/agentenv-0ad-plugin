"""LiveMatch on a scripted engine: setup (replay, fog), the wall-clock schedule, the end rules, results and files."""

import asyncio
import dataclasses
import json
import time

import pytest

from agentenv_zeroad.grading import BrainCall
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import SetEconomy
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

AGENT_VS_PETRA = (SeatPlan("1", 1, "agent", "rome", "alice"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))
BOT_VS_PETRA = (SeatPlan("1", 1, "bot", "rome", "Bot 1"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))


def make(tmp_path, seats=AGENT_VS_PETRA, *, handle=None, **config) -> LiveMatch:
    config = {"speed": 2.0, "time_cap_s": 3600, "tick_s": 0.05, **config}
    return LiveMatch(handle or Handle(), MatchConfig(seats=seats, **config), out_dir=tmp_path, placer=placer())


async def test_setup_resets_with_the_replay_on_and_turns_each_players_reveal_flag_off(tmp_path):
    match = make(tmp_path)
    await match.setup()
    engine = match.engine
    assert engine.saved_replay is True and engine.resets == 1 and engine.fog_calls == 1
    scenario = engine.scenario["settings"]["PlayerData"]
    assert [(p["Civ"], p["AI"]) for p in scenario] == [("rome", ""), ("pers", "petra")]
    run = ["\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("//")) for code in engine.evaluated]
    assert not any("SetLosRevealAll(-1, true)" in code for code in run)
    assert match.fog == {"global": False, "own": [False, False, False], "observer": True}
    assert match.snapshot is not None and match.seats["1"].executor is not None
    assert match.seats["2"].executor is None


async def test_without_fog_no_reveal_flag_is_touched(tmp_path):
    match = make(tmp_path, fog=False)
    await match.setup()
    assert match.engine.fog_calls == 0 and match.fog is None


async def test_the_clock_keeps_its_wall_clock_schedule_while_seat_ticks_are_slow(tmp_path):
    match = make(tmp_path, speed=1.0)
    await match.setup()
    view = match._view

    async def slow_view():
        await asyncio.sleep(0.5)  # a slow executor tick (or anything a brain does) never holds the clock
        return await view()

    match._view = slow_view
    match.start()
    started = time.monotonic()
    await asyncio.sleep(1.2)
    steps = match.clock.steps
    await match.stop()
    elapsed = time.monotonic() - started
    assert abs(steps - elapsed / 0.2) <= 2
    assert match.clock.overruns == 0


async def test_the_time_cap_ends_the_match_with_the_highest_score_winning_and_keeps_its_files(tmp_path):
    handle = Handle()
    match = make(tmp_path, handle=handle, time_cap_s=1.0)
    await match.setup()
    match.start()
    await asyncio.wait_for(match.wait_end(), 10)
    assert match.ended_by == "time_cap" and match.outcomes == {"1": "won", "2": "lost"}
    assert match.detail().startswith("alice won on score at the time cap")
    assert handle.closed == 1 and match.clock.steps >= 5
    results = json.loads(match.files["results"].read_text())
    assert results["status"] == "finished" and results["seats"]["1"]["outcome"] == "won"
    assert results["seats"]["1"]["score"]["total"] == 120 and results["clock"]["steps"] == match.clock.steps
    assert match.files["replay"].read_text().startswith('{"settings"')


async def test_a_tie_on_top_at_the_time_cap_is_a_draw(tmp_path):
    match = make(tmp_path, time_cap_s=0.6)
    await match.setup()
    match.engine.outcome["2"]["total"] = 120
    match.start()
    await asyncio.wait_for(match.wait_end(), 10)
    assert match.outcomes == {"1": "drawn", "2": "drawn"}
    assert match.detail().startswith("a draw between alice, Petra 2")


async def test_conquest_ends_the_match_when_the_engine_defeats_a_player(tmp_path):
    match = make(tmp_path)
    await match.setup()
    match.start()
    await asyncio.sleep(0.2)
    assert not match.over
    match.engine.outcome["1"]["state"] = "defeated"
    await asyncio.wait_for(match.wait_end(), 5)
    assert match.ended_by == "conquest" and match.outcomes == {"1": "lost", "2": "won"}
    assert match.detail().startswith("Petra 2 won by conquest")


async def test_each_executor_learns_which_opponents_the_engine_has_defeated(tmp_path):
    handle = Handle()
    handle.engine.outcome["3"] = {"state": "active", "total": 50, "economy": 40, "military": 0, "exploration": 10}
    match = make(tmp_path, (*AGENT_VS_PETRA, SeatPlan("3", 3, "petra", "cart", "Petra 3")), handle=handle)
    await match.setup()
    match.start()
    await asyncio.sleep(0.2)
    match.engine.outcome["2"]["state"] = "defeated"
    await asyncio.sleep(0.3)
    assert not match.over and match.seats["1"].executor.fallen == {2}
    assert match.status("2")["defeated"] and not match.status("1")["defeated"]


async def test_results_keep_each_logged_turn_and_a_timeline_of_every_players_score(tmp_path):
    match = make(tmp_path, time_cap_s=0.6)
    await match.setup()
    match.record_brain_call("1", BrainCall(2.0, game_ms=1000), model="local/qwen3-8b", turn={"began_ms": 0, "user": "b"})
    match.record_brain_call("1", BrainCall(2.0))  # an unlogged call keeps no turn
    match.start()
    await asyncio.wait_for(match.wait_end(), 10)
    results = match.results()
    [turn] = results["seats"]["1"]["turns"]
    assert turn == {"began_ms": 0, "user": "b", "reported_ms": 1000, "seconds": 2.0, "model": "local/qwen3-8b"}
    assert "turns" not in results["seats"]["2"]
    first = results["timeline"][0]
    assert first["t_ms"] == 0 and set(first["players"]) == {"1", "2"}
    assert first["players"]["1"]["total"] == 120 and "army" in first["players"]["1"]


async def test_an_engine_that_stops_answering_fails_the_match_and_is_released(tmp_path):
    handle = Handle()
    match = make(tmp_path, handle=handle)
    await match.setup()

    async def broken(commands=()):
        raise RuntimeError("engine died")

    match.engine.step = broken
    match.start()
    await asyncio.wait_for(match.wait_end(), 5)
    assert match.failed and "engine died" in match.failed
    assert json.loads(match.files["results"].read_text())["status"] == "failed"
    assert handle.closed == 1


async def test_a_bot_seat_plays_through_its_executor(tmp_path):
    match = make(tmp_path, BOT_VS_PETRA)
    await match.setup()
    bot = match.seats["1"].bot
    assert bot is not None and bot.log
    assert bot.log[0][1].kind == "set_economy" and match.seats["1"].executor.book.split["food"] == 55


async def test_results_carry_each_seats_orders_tool_timings_brain_calls_and_metrics(tmp_path):
    match = make(tmp_path, time_cap_s=0.4)
    await match.setup()
    assert match.submit("1", SetEconomy(food=40, wood=40, stone=10, metal=10)).status == "accepted"
    match.record_tool_call("1", "set_economy", 0.004, "accepted")
    match.record_brain_call("1", BrainCall(12.0))
    match.record_brain_call("1", BrainCall(130.0))
    match.start()
    await asyncio.wait_for(match.wait_end(), 10)
    seat = match.results()["seats"]["1"]
    assert seat["orders"]["economy"] == {"food": 40, "wood": 40, "stone": 10, "metal": 10}
    assert seat["tool_calls"]["n"] == 1 and seat["tool_calls"]["by_tool"] == {"set_economy": 1}
    assert seat["moves"] == {"made": 1, "missed": 1} and not seat["forfeited"]
    assert seat["metrics"]["decisions"] == 1 and seat["metrics"]["missed"] == 1
    assert seat["metrics"]["latency_s"]["max"] == 130.0
    assert seat["grade"] == 1.0


async def test_the_brief_drains_events_and_ends_with_the_footer(tmp_path):
    match = make(tmp_path)
    await match.setup()
    brief = match.brief("1")
    assert brief.startswith("Time 00:00.") and "Standing orders:" in brief
    assert brief.splitlines()[-1] == match.footer("1")
    assert "/ cap 60:00" in match.footer("1") and "get_brief()" in match.footer("1")


THREE_PETRAS = tuple(
    SeatPlan(str(p), p, "petra", civ, f"Petra {p}") for p, civ in ((1, "rome"), (2, "pers"), (3, "cart"))
)


async def test_a_player_conquered_before_the_time_cap_stays_lost_even_with_the_top_score(tmp_path):
    match = make(tmp_path, THREE_PETRAS, time_cap_s=60)
    totals = {"1": ("defeated", 900), "2": ("active", 500), "3": ("active", 400)}
    match._read_outcome({p: {"state": s, "total": t} for p, (s, t) in totals.items()})
    match._check_end(60_000)
    assert match.ended_by == "time_cap" and match.outcomes == {"1": "lost", "2": "won", "3": "lost"}
    assert match.detail().startswith("Petra 2 won on score")


async def test_a_tie_among_the_players_still_standing_at_the_time_cap_is_their_draw(tmp_path):
    match = make(tmp_path, THREE_PETRAS, time_cap_s=60)
    totals = {"1": ("defeated", 900), "2": ("active", 500), "3": ("active", 500)}
    match._read_outcome({p: {"state": s, "total": t} for p, (s, t) in totals.items()})
    match._check_end(60_000)
    assert match.outcomes == {"1": "lost", "2": "drawn", "3": "drawn"}


async def test_the_time_cap_still_ends_the_match_when_every_engine_query_keeps_failing(tmp_path):
    match = make(tmp_path, time_cap_s=1.0)
    await match.setup()

    async def broken(code):
        raise RuntimeError("query failed")

    match.engine.evaluate = broken
    match.start()
    await asyncio.wait_for(match.wait_end(), 5)
    assert match.ended_by == "time_cap" and match.outcomes == {"1": "won", "2": "lost"}  # on the last known scores
    assert match.tick_errors > 0 and match.files["results"].exists()


async def test_a_seat_loop_that_keeps_failing_fails_the_match_after_it_once_ticked(tmp_path):
    handle = Handle()
    match = make(tmp_path, handle=handle)
    await match.setup()
    assert match.ticks == 1

    async def broken(code):
        raise RuntimeError("query failed")

    match.engine.evaluate = broken
    match.start()
    await asyncio.wait_for(match.wait_end(), 5)
    assert match.failed and "ticks in a row" in match.failed and "query failed" in match.failed
    assert json.loads(match.files["results"].read_text())["status"] == "failed" and handle.closed == 1


def at(match: LiveMatch, seconds: float) -> None:
    match.snapshot = dataclasses.replace(match.snapshot, time_ms=int(seconds * 1000))
    match.metrics.observe(match.snapshot)


async def test_a_brains_wait_counts_the_game_time_from_begin_to_end_and_a_stray_end_is_ignored(tmp_path):
    match = make(tmp_path)
    await match.setup()
    at(match, 0)
    match.record_waiting("1", "end")  # nothing open: ignored
    at(match, 10)
    match.record_waiting("1", "begin")
    at(match, 20)
    match.record_waiting("1", "end")
    at(match, 40)
    seat = match.metrics.seats[1]
    assert seat.waiting_s == 10.0 and seat.waiting_share() == 0.25


async def test_only_a_brain_that_polls_status_has_its_no_waits_read_as_a_measured_zero(tmp_path):
    match = make(tmp_path)
    await match.setup()
    at(match, 0)
    match.record_brain_call("1", BrainCall(5.0))  # any MCP agent may report its calls: that says nothing of waits
    at(match, 30)
    assert match.metrics.seats[1].summary()["waiting_share"] is None
    match.status("1")  # the shipped brains poll status and gate their waits on `open`
    assert match.metrics.seats[1].summary()["waiting_share"] == 0.0


async def test_status_says_which_decisions_are_open_for_the_seat(tmp_path):
    match = make(tmp_path)
    await match.setup()
    assert match.status("1")["open"] == []
    raw = match.snapshot
    me = dataclasses.replace(raw.players[1], idle_units={"infantry_melee": 4}, enemies_near_base=3)
    match.snapshot = dataclasses.replace(raw, players={**raw.players, 1: me})
    assert "army" in match.status("1")["open"]
