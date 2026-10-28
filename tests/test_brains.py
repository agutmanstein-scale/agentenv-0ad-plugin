"""The brains play a seat through the env's real endpoints over HTTP, against a fake LiteLLM (no network, no keys):
the strategist (with Setup A's Luna reflex in the env) and the Luna player (Setup B). Their model calls, spend and
waits land in results.json, and the key handed to the reflex is never written anywhere."""

import json

import httpx
import pytest

from agentenv_zeroad.brains import Commander, CommanderSettings, LunaPlayer, LunaSettings, luna_orders, luna_questions
from agentenv_zeroad.brains import seat_link as link_to
from agentenv_zeroad.decisions import Answer, DecisionsClient
from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.server import ZeroADEnv
from agentenv_zeroad.spend import SpendGuard
from tests.fake_models import KEY, FakeLiteLLM
from tests.served import served
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

pytestmark = pytest.mark.anyio

SETTINGS = {"speed": 2.0, "time_cap_minutes": 0.1, "start_grace_seconds": 30}


def busy_engine() -> ScriptedEngine:
    """A scripted engine whose player 1 has idle soldiers and a free queue, so the reflex has questions to ask."""
    engine = ScriptedEngine()
    me = engine.raw["snapshot"]["players"]["1"]
    me["idle_units"] = {"infantry_melee": 4}
    me["enemies_near_base"] = 2
    return engine


def make_env(tmp_path, fake: FakeLiteLLM | None = None) -> ZeroADEnv:
    options = {"placer": placer(), "environ": {}}
    if fake is not None:
        options["decider_for"] = lambda plan, base, key, guard: DecisionsClient(base, key, guard, http=fake.http())

    async def start():
        return Handle(busy_engine())

    return ZeroADEnv(start, tmp_path, tick_s=0.05, match_options=options)


async def open_match(http, base, slot1: dict, slot2: dict | None = None) -> None:
    lobby = f"{base}/agentenv/ext/lobby"
    assert (await http.post(f"{lobby}/open", json={"game_settings": SETTINGS})).status_code == 200
    filled = await http.post(f"{lobby}/fill", json={"player_id": "1", "player_kind": "agent", **slot1})
    assert filled.status_code == 200, filled.text
    await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai", **(slot2 or {})})
    assert (await http.post(f"{lobby}/close", json={})).json()["status"] == "closed"


async def finish(http, base) -> dict:
    match = (await http.post(f"{base}/agentenv/ext/match/finish", json={})).json()
    assert match["status"] == "finished"
    files = (await http.post(f"{base}/agentenv/ext/match/files", json={})).json()["files"]
    return (await http.get(next(f["path"] for f in files if f["kind"] == "results"))).json()


async def test_the_commander_plays_its_seat_and_hands_its_endpoint_to_the_seats_luna_reflex(tmp_path):
    fake = FakeLiteLLM(
        tool_calls=[("set_economy", {"food": 50, "wood": 30, "stone": 10, "metal": 10}), ("say", {"text": "Ave!"})],
        choose=lambda q: "hold" if q["name"] == "army" else q["choices"][0]["value"],
        delay_s=0.2,  # model time the seat waits through, in game time
    )
    env = make_env(tmp_path, fake)
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        settings = {"civ": "rome", "reflex": "luna", "reflex_max_usd": 0.5}
        await open_match(
            http, base, {"player_name": "opus", "game_settings": settings}, {"game_settings": {"controller": "bot"}}
        )
        guard = SpendGuard("commander")
        model = StrategistModel("anthropic/claude-opus-5-5", "https://proxy.example/v1", KEY, guard, http=fake.http())
        commander = Commander(model, CommanderSettings(interval_s=0.5, min_gap_s=0.5, poll_s=0.05))
        async with link_to({"url": f"{base}/players/1/mcp"}) as link:
            record = await commander.play(link, reflex_key=("https://proxy.example/v1", KEY))
        results = await finish(http, base)

    assert record.end_reason == "match_over" and record.calls >= 2 and record.errors == 0
    assert record.orders == 2 * record.calls and record.rejected == 0
    assert len(fake.bodies("/v1/chat/completions")) == record.calls
    seat = results["seats"]["1"]
    assert len(seat["brain_calls"]) == record.calls and seat["model"] == "anthropic/claude-opus-5-5"
    assert seat["metrics"]["decisions"] == record.calls and seat["metrics"]["latency_s"]["p50"] is not None
    assert seat["metrics"]["waiting_share"] is not None and 0 < seat["metrics"]["waiting_share"] <= 1
    assert seat["spend"]["brain"]["usd_est"] == pytest.approx(guard.spent_usd, abs=1e-5)
    assert seat["said"] and seat["said"][0][1] == "Ave!"
    reflex = seat["reflex"]
    assert reflex["answered"] >= 1 and reflex["orders"] >= 1 and len(fake.bodies("/openai/v1/decisions")) >= 1
    assert seat["spend"]["reflex"]["usd_est"] > 0 and seat["spend"]["reflex"]["cap_usd"] == 0.5
    assert results["spend_usd_est"] == pytest.approx(
        seat["spend"]["brain"]["usd_est"] + seat["spend"]["reflex"]["usd_est"]
    )
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(errors="ignore"), path


async def test_a_seat_without_reflex_turns_the_endpoint_down_and_petra_takes_no_reflex(tmp_path):
    env = make_env(tmp_path)
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        lobby = f"{base}/agentenv/ext/lobby"
        await http.post(f"{lobby}/open", json={"game_settings": SETTINGS})
        refused = await http.post(
            f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai", "game_settings": {"reflex": "luna"}}
        )
        assert refused.status_code >= 400 and "Petra plays on its own" in refused.text
        await http.post(f"{lobby}/fill", json={"player_id": "1", "player_kind": "agent", "player_name": "a"})
        await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai"})
        await http.post(f"{lobby}/close", json={})
        answer = await http.post("/players/1/reflex", json={"base_url": "https://p", "api_key": KEY})
        assert answer.json() == {"ok": True, "reflex": None} and KEY not in answer.text
        assert (await http.post("/players/1/waiting", json={"state": "maybe"})).status_code == 400
        assert (await http.post("/players/1/waiting", json={"state": "begin"})).json() == {"ok": True}
        status = (await http.get("/players/1/status")).json()
        assert status["ok"] and status["over"] is False and {"game_ms", "events", "urgent", "started"} <= set(status)
        await finish(http, base)


async def test_the_luna_player_plays_its_seat_every_interval_through_the_order_tools(tmp_path):
    fake = FakeLiteLLM(
        choose=lambda q: {"economy": "military", "phase": "advance"}.get(q["name"], q["choices"][0]["value"])
    )
    env = make_env(tmp_path)
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        await open_match(http, base, {"player_name": "luna", "game_settings": {"civ": "rome"}})
        guard = SpendGuard("luna")
        player = LunaPlayer(DecisionsClient("https://proxy.example", KEY, guard, http=fake.http()), LunaSettings(0.2))
        async with link_to({"url": f"{base}/players/1/mcp"}) as link:
            record = await player.play(link)
        results = await finish(http, base)
    assert record.end_reason == "match_over" and record.calls >= 5 and record.errors == 0
    assert any("set_economy" in o for t in record.turns for o in t["orders"])
    seat = results["seats"]["1"]
    assert seat["orders"]["economy"] == {"food": 40, "wood": 30, "stone": 5, "metal": 25}
    assert seat["metrics"]["decisions"] == record.calls and seat["metrics"]["decisions_per_min"] > 0
    assert seat["metrics"]["waiting_share"] is not None and seat["spend"]["brain"]["usd_est"] > 0
    assert seat["grade"] is not None and not seat["forfeited"]
    body = fake.bodies("/openai/v1/decisions")[0]
    names = [q["name"] for q in body["questions"]]
    assert {"economy", "army", "aggression", "build"} <= set(names)
    assert json.loads(body["input"].split("Empire now: ", 1)[1])["player"] == 1


def test_luna_answers_only_send_orders_that_change_the_standing_orders():
    brief = {
        "player": 1,
        "phase": "village",
        "players": {"1": "Luna", "2": "Opus"},
        "orders": {
            "economy": {"food": 40, "wood": 30, "stone": 5, "metal": 25},
            "army": "attack player_2",
            "train": {"cavalry": 10},
            "build": ["house@base"],
        },
        "idle": {"production": [150]},
        "pop": [20, 30],
    }
    questions = luna_questions(brief)
    names = [q.question.name for q in questions]
    assert names == ["economy", "phase", "army", "aggression", "train", "build"]
    build = next(q for q in questions if q.question.name == "build")
    assert "house" not in [c.value for c in build.question.choices]  # already ordered
    answers = {
        "economy": Answer("economy", "choice", choice="military"),
        "army": Answer("army", "choice", choice="press"),
        "train": Answer("train", "choice", choice="cavalry"),
        "aggression": Answer("aggression", "score", score=2.0),
        "phase": Answer("phase", "choice", choice="advance"),
        "build": Answer("build", "choice", choice="barracks"),
    }
    orders = luna_orders(brief, questions, answers)
    assert orders == {
        "phase": [("advance_phase", {})],
        "build": [("build", {"structure": "barracks", "where": "base"})],
    }  # the split, the attack and 10 >= 10 cavalry are standing already


async def test_a_brain_call_with_only_one_token_count_is_refused_and_records_nothing(tmp_path):
    env = make_env(tmp_path)
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        await open_match(http, base, {"player_name": "a"})
        refused = await http.post("/players/1/brain_calls", json={"seconds": 2.0, "input_tokens": 100})
        assert refused.status_code == 400
        assert env.live.seats["1"].brain_calls == [] and env.live.metrics.seats[1].decisions == 0
        both = {"seconds": 2.0, "input_tokens": 100, "output_tokens": 5}
        assert (await http.post("/players/1/brain_calls", json=both)).status_code == 200
        assert env.live.seats["1"].brain_spend["output_tokens"] == 5
        await finish(http, base)


def luna_brief(**changes):
    brief = {
        "player": 1,
        "phase": "village",
        "t": "02:00",
        "time_ms": 120_000,
        "cap": "10:00",
        "cap_ms": 600_000,
        "players": {"1": "Luna", "2": "Opus"},
        "orders": {"economy": {"food": 55, "wood": 45, "stone": 0, "metal": 0}, "army": "gather home", "train": {}},
        "army": {"villager": 20, "infantry_melee": 4, "infantry_ranged": 4},
        "res": {"food": 450, "wood": 420, "stone": 100, "metal": 100},
        "next_phase": {"tech": "phase_town_generic", "ready": True, "cost": {"food": 500, "wood": 500}},
        "bld": {"CivCentre": 1, "House": 4},
        "idle": {"production": []},
        "pop": [28, 40],
        "threat": 0,
    }
    brief.update(changes)
    return brief


def asked(brief, name):
    return next((q for q in luna_questions(brief) if q.question.name == name), None)


def test_luna_asks_about_the_next_phase_once_it_is_nearly_affordable_and_city_only_with_time_left():
    assert asked(luna_brief(res={"food": 100, "wood": 100}), "phase") is None
    q = asked(luna_brief(), "phase")
    assert [c.value for c in q.question.choices] == ["advance", "wait"] and "town" in q.question.instructions
    assert q.to_orders("advance") == [("advance_phase", {})] and q.to_orders("wait") == []
    late_town = luna_brief(phase="town", time_ms=420_000, next_phase={"cost": {"food": 10}})
    assert asked(late_town, "phase") is None  # three minutes left: city doesn't pay off
    assert asked(luna_brief(phase="town", next_phase={"cost": {"food": 10}}), "phase") is not None


def test_luna_keeps_not_yet_for_its_attack_all_match_and_past_the_deadline_a_wait_is_a_hold():
    early = asked(luna_brief(), "army")
    assert {c.value for c in early.question.choices} == {"attack_2", "raid_2", "wait"}
    assert early.to_orders("attack_2") == [("army", {"action": "attack", "target": "player_2"})]
    assert early.to_orders("wait") == [] and early.held is None
    late = asked(luna_brief(time_ms=300_000, t="05:00"), "army")
    assert {c.value for c in late.question.choices} == {"attack_2", "raid_2", "wait"}
    assert late.held == "attack" and "the autopilot attacks for you" in late.question.instructions
    assert late.to_orders("wait") == []  # Luna's hold is no order of Luna's
    assert asked(luna_brief(army={"villager": 20, "cavalry": 1}), "army") is None  # nothing to attack with yet


def test_luna_defends_a_raided_base_and_under_an_attack_only_regroups_a_spent_army():
    q = asked(luna_brief(threat=3), "army")
    assert [c.value for c in q.question.choices] == ["defend", "retreat"]
    assert q.to_orders("defend") == [("army", {"action": "defend", "target": "home"})]
    attacking = luna_brief(orders={"army": "attack player_2", "train": {}})
    assert asked(attacking, "army") is None  # the attack stands while the army is strong
    spent = asked(dict(attacking, army={"infantry_melee": 2}), "army")
    assert [c.value for c in spent.question.choices] == ["press", "regroup"]
    assert spent.to_orders("regroup") == [("army", {"action": "retreat", "target": "home"})]
    assert spent.to_orders("press") == []


def test_luna_is_told_when_it_has_no_barracks():
    q = asked(luna_brief(), "build")
    barracks = next(c for c in q.question.choices if c.value == "barracks")
    assert "we have none" in barracks.description
    q = asked(luna_brief(bld={"Barracks": 1}), "build")
    assert "we have none" not in next(c for c in q.question.choices if c.value == "barracks").description


def test_from_the_phase_deadline_luna_may_still_say_not_yet_which_is_reported_as_a_hold():
    poor = luna_brief(time_ms=150_000, res={"food": 100, "wood": 100})
    q = asked(poor, "phase")
    assert [c.value for c in q.question.choices] == ["advance", "advance_boom", "wait"]
    assert q.held == "phase" and "the autopilot advances for you" in q.question.instructions
    assert q.to_orders("advance_boom") == [
        ("set_economy", {"food": 55, "wood": 45, "stone": 0, "metal": 0}),
        ("advance_phase", {}),
    ]
    assert q.to_orders("wait") == []
    before = asked(luna_brief(), "phase")  # affordable before the deadline: asked, "not yet" is no hold
    assert "wait" in [c.value for c in before.question.choices] and before.held is None
    assert asked(dict(poor, orders={"advance_phase": True, "army": "gather home"}), "phase") is None


def test_past_the_phase_deadline_a_not_yet_with_raiders_at_the_base_is_luna_s_choice_and_no_hold():
    raided = asked(luna_brief(time_ms=150_000, res={"food": 100, "wood": 100}, threat=4), "phase")
    assert "wait" in [c.value for c in raided.question.choices]
    assert raided.held is None  # the plan defends first: the autopilot must not advance on this "not yet"


def test_luna_is_not_asked_to_attack_with_its_starting_soldiers_during_the_boom():
    start = luna_brief(t="00:03", time_ms=3_200, army={"villager": 4, "infantry_melee": 2, "infantry_ranged": 2})
    assert asked(start, "army") is None
    small = luna_brief(t="03:00", time_ms=180_000, army={"villager": 20, "infantry_melee": 3, "infantry_ranged": 2})
    assert asked(small, "army") is None  # before the deadline only a real army is asked to attack
    assert asked(luna_brief(t="01:30", time_ms=90_000), "army") is None  # eight soldiers, but still the boom


def test_past_the_deadline_an_army_too_small_to_press_is_not_sent_straight_back_out():
    late = dict(t="06:00", time_ms=360_000, army={"villager": 20, "infantry_melee": 3})
    spent = asked(luna_brief(orders={"army": "attack player_2", "train": {}}, **late), "army")
    assert spent.to_orders("regroup") == [("army", {"action": "retreat", "target": "home"})]
    assert asked(luna_brief(orders={"army": "retreat home", "train": {}}, **late), "army") is None
    four = dict(late, army={"villager": 20, "infantry_melee": 4})
    assert asked(luna_brief(orders={"army": "retreat home", "train": {}}, **four), "army") is not None


def test_luna_with_no_other_player_is_asked_nothing_about_attacking():
    alone = {"players": {"1": "Luna"}}
    assert asked(luna_brief(**alone), "army") is None
    assert asked(luna_brief(time_ms=300_000, t="05:00", **alone), "army") is None
