"""U5c: one action record per seat of who changed what (strategist or Luna through the tools, the Luna reflex, the
scripted bot, the autopilot carrying out a default Luna held past its deadline), read by results.json, the brief
footer and the overlay; and the held defaults themselves."""

from dataclasses import replace

import httpx
import pytest

from agentenv_zeroad import grading
from agentenv_zeroad.autopilot import HELD_DEADLINES, held_default
from agentenv_zeroad.live import overlay_state
from agentenv_zeroad.match import HOLD_REPEAT_MS, LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import AdvancePhase, Army, Train
from agentenv_zeroad.reflex import ReflexAction
from agentenv_zeroad.server import ZeroADEnv
from tests.executor_fixtures import load
from tests.served import served
from tests.test_live import _handle, agent_vs_petra
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

CAP_MS = 600_000


def test_a_held_default_waits_for_its_deadline_and_never_repeats_what_already_stands():
    kw = dict(cap_ms=CAP_MS, phase="village", advancing=False, army_action="gather", soldiers=10)
    enemies = {1: ("Opus", 900.0), 3: ("Petra", 1200.0)}
    phase_due = int(HELD_DEADLINES["phase"] * CAP_MS)
    assert held_default("phase", time_ms=phase_due - 1, enemies=enemies, **kw) is None
    assert held_default("phase", time_ms=phase_due, enemies=enemies, **kw) == (
        AdvancePhase(),
        "ordered the advance to town",
    )
    assert held_default("phase", time_ms=phase_due, enemies=enemies, threat=2, **kw) is None  # defend first
    assert held_default("phase", time_ms=phase_due, enemies=enemies, **{**kw, "advancing": True}) is None
    assert held_default("phase", time_ms=phase_due, enemies=enemies, **{**kw, "phase": "town"}) is None
    attack_due = int(HELD_DEADLINES["attack"] * CAP_MS)
    assert held_default("attack", time_ms=attack_due - 1, enemies=enemies, **kw) is None
    order, line = held_default("attack", time_ms=attack_due, enemies=enemies, **kw)
    assert order == Army(action="attack", target="player_3") and line == "sent the army to attack Petra"  # the leader
    assert held_default("attack", time_ms=attack_due, enemies=enemies, **{**kw, "army_action": "raid"}) is None
    assert held_default("attack", time_ms=attack_due, enemies=enemies, **{**kw, "soldiers": 3}) is None
    assert held_default("attack", time_ms=attack_due, enemies={}, **kw) is None
    assert held_default("attack", time_ms=attack_due, enemies=enemies, threat=2, **kw) is None  # defend first


async def match_with_luna(tmp_path) -> LiveMatch:
    seats = (SeatPlan("1", 1, "agent", "rome", "Opus 5.5"), SeatPlan("2", 2, "agent", "pers", "GPT-6 Luna"))
    config = MatchConfig(seats=seats, time_cap_s=CAP_MS / 1000, tick_s=3600)  # the test sets the snapshot itself
    match = LiveMatch(Handle(), config, out_dir=tmp_path, placer=placer())
    await match.setup()
    match.start()
    match.record_brain_call("1", grading.BrainCall(seconds=9.0), model="anthropic/claude-opus-5-5")
    match.record_brain_call("2", grading.BrainCall(seconds=1.0), model="openai/gpt-6-luna")
    return match


def at(match: LiveMatch, time_ms: int) -> None:
    match.snapshot = replace(load(300)[1], time_ms=time_ms)  # player 2: village phase, 17 soldiers


async def test_past_its_deadline_a_luna_hold_is_carried_out_by_the_autopilot_and_never_credited_to_luna(tmp_path):
    match = await match_with_luna(tmp_path)
    at(match, 100_000)
    assert match.record_hold("2", "phase")["applied"] is False  # before the deadline "not yet" is Luna's choice
    assert not match.seats["2"].actions
    at(match, 200_000)
    done = match.record_hold("2", "phase")
    assert done["applied"] is True and done["order"] == "advance_phase()"  # partial: it saves up for it
    assert match.seats["2"].executor.book.advance_phase
    assert match.record_hold("2", "phase")["applied"] is False  # once, not every second Luna holds
    assert match.record_hold("2", "attack")["applied"] is False  # the attack deadline is later
    at(match, 300_000)
    assert match.record_hold("2", "attack")["applied"] is True
    assert match.seats["2"].executor.book.army.action == "attack"
    seat = match.results()["seats"]["2"]
    assert [(a["by"], a["who"], a["line"]) for a in seat["actions"]] == [
        ("autopilot", "autopilot (Luna held)", "ordered the advance to town"),
        ("autopilot", "autopilot (Luna held)", "sent the army to attack Opus 5.5"),
    ]
    assert [a["kind"] for a in seat["actions"]] == ["held:phase", "held:attack"]
    assert seat["action_counts"]["autopilot"] == 2 and seat["action_counts"]["brain"] == 0
    assert "autopilot (Luna held): sent the army to attack Opus 5.5" in match.footer("2")
    recent = overlay_state(match)["seats"][1]["recent"]
    assert [(o["who"], o["line"]) for o in recent][0] == ("autopilot (Luna held)", "sent the army to attack Opus 5.5")
    await match.stop()


async def test_one_record_holds_every_hand_on_the_orders_resolved_by_the_brain_that_reported(tmp_path):
    match = await match_with_luna(tmp_path)
    at(match, 60_000)
    match.submit("1", Train(role="cavalry", count=5))  # Opus's tool order
    match.submit("2", Army(action="defend", target="home"))  # Luna player's tool order
    match.submit("2", AdvancePhase())
    results = match.results()["seats"]
    assert [(a["by"], a["who"], a["what"]) for a in results["1"]["actions"]] == [
        ("strategist", "strategist", "train(role='cavalry', count=5)")
    ]
    assert [(a["by"], a["who"]) for a in results["2"]["actions"]] == [("luna", "Luna"), ("luna", "Luna")]
    assert all("what" not in c for c in results["1"]["tool_calls"]["calls"])  # the record is the one place
    await match.stop()


async def test_a_seat_with_a_reflex_shows_its_reflex_lines_in_the_footer_brief_and_overlay_under_luna(tmp_path):
    seats = (SeatPlan("1", 1, "agent", "rome", "Opus 5.5", reflex="luna"), SeatPlan("2", 2, "petra", "pers", "P"))
    match = LiveMatch(Handle(), MatchConfig(seats=seats, time_cap_s=600), out_dir=tmp_path, placer=placer())
    await match.setup()
    line = "pulled the cavalry back home out of the fight"
    match.seats["1"].reflex.on_action(ReflexAction(95_000, "engage:fall_back", line, "army(...)", "accepted"))
    assert f"last 01:35 Luna: {line}" in match.footer("1")
    assert f"[01:35] Luna: {line}" in match.brief("1")
    assert overlay_state(match)["seats"][0]["recent"][0] == {
        "at": "01:35",
        "who": "Luna",
        "by": "reflex",
        "line": line,
        "status": "accepted",
    }
    await match.stop()


async def test_the_held_route_takes_a_brains_hold_at_its_seats_address(tmp_path):
    env = ZeroADEnv(lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer()})
    async with served(env) as url, httpx.AsyncClient() as http:
        await agent_vs_petra(env)
        ok = await http.post(f"{url}/players/1/held", json={"decision": "phase"})
        assert ok.status_code == 200 and ok.json()["ok"] is True and ok.json()["applied"] is False
        bad = await http.post(f"{url}/players/1/held", json={"decision": "everything"})
        assert bad.status_code == 400
        nowhere = await http.post(f"{url}/held", json={"decision": "phase"})
        assert nowhere.status_code == 400
    await env.live.stop()


async def test_a_held_attack_never_pulls_the_army_off_a_base_under_attack(tmp_path):
    match = await match_with_luna(tmp_path)
    at(match, 300_000)
    me = match.snapshot.players[2]
    match.snapshot.players[2] = replace(me, enemies_near_base=6)  # raiders arrived after Luna's brief was taken
    match.submit("2", Army(action="defend", target="home"))
    done = match.record_hold("2", "attack")
    assert done["applied"] is False
    assert match.seats["2"].executor.book.army.action == "defend"
    assert [a["by"] for a in match.results()["seats"]["2"]["actions"]] == ["luna"]
    await match.stop()


async def test_a_held_phase_never_starts_saving_for_town_while_raiders_are_at_the_base(tmp_path):
    match = await match_with_luna(tmp_path)
    at(match, 200_000)
    me = match.snapshot.players[2]
    match.snapshot.players[2] = replace(me, enemies_near_base=6)  # raiders arrived after Luna's brief was taken
    done = match.record_hold("2", "phase")
    assert done["applied"] is False
    assert not match.seats["2"].executor.book.advance_phase  # no cost set aside, no buildings added mid-raid
    assert not match.results()["seats"]["2"]["actions"]
    match.snapshot.players[2] = me  # the raid is over: the default is carried out on the next hold
    assert match.record_hold("2", "phase")["applied"] is True
    await match.stop()


async def test_a_held_attack_goes_at_the_leading_enemy_still_standing_never_at_the_seat_itself(tmp_path):
    seats = (
        SeatPlan("1", 1, "agent", "rome", "Opus 5.5"),
        SeatPlan("2", 2, "agent", "pers", "GPT-6 Luna"),
        SeatPlan("3", 3, "petra", "athen", "Petra"),
    )
    config = MatchConfig(seats=seats, time_cap_s=CAP_MS / 1000, tick_s=3600)
    match = LiveMatch(Handle(), config, out_dir=tmp_path, placer=placer())
    await match.setup()
    match.start()
    match.record_brain_call("2", grading.BrainCall(seconds=1.0), model="openai/gpt-6-luna")
    at(match, 300_000)
    match.scores = {"1": {"total": 500.0}, "2": {"total": 2000.0}, "3": {"total": 3000.0}}
    match.states["3"] = "defeated"
    done = match.record_hold("2", "attack")
    assert done["applied"] is True and done["order"] == "army(action='attack', target='player_1')"
    await match.stop()


async def test_a_held_default_that_cleared_is_carried_out_again_only_after_the_repeat_gap(tmp_path):
    match = await match_with_luna(tmp_path)
    at(match, 200_000)
    assert match.record_hold("2", "phase")["applied"] is True
    match.seats["2"].executor.book.advance_phase = False  # the advance was dropped (say, it could not be afforded)
    at(match, 200_000 + HOLD_REPEAT_MS - 1)
    assert match.record_hold("2", "phase") == {"applied": False, "reason": "carried out moments ago"}
    at(match, 200_000 + HOLD_REPEAT_MS)
    assert match.record_hold("2", "phase")["applied"] is True
    assert [a["kind"] for a in match.results()["seats"]["2"]["actions"]] == ["held:phase", "held:phase"]
    await match.stop()
