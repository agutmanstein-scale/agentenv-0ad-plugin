"""The strategist as a library under the lockstep clock: the due-time rule in game seconds (the 25 s interval and the
8 s gap after urgent events), a seat's wait and its end, and the in-process seat link showing the model exactly what
the MCP link does."""

import asyncio
import json

import httpx
import pytest

from agentenv_zeroad.brains import Commander, CommanderSeat, CommanderSettings, InProcessSeatLink, decision_due
from agentenv_zeroad.brains import seat_link as mcp_link
from agentenv_zeroad.clock import LockstepClock
from agentenv_zeroad.engine import FakeEngine
from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.server import ZeroADEnv, tool_specs
from agentenv_zeroad.spend import SpendGuard
from tests.fake_models import KEY, FakeLiteLLM, FakeStrategist
from tests.served import served
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

pytestmark = pytest.mark.anyio


class FakeLink:
    """What a CommanderSeat needs of a seat link, scripted: `status` fields change by test, every report is kept."""

    def __init__(self, engine: FakeEngine, **status):
        self.engine = engine
        self.status_fields = {"started": True, "over": False, "defeated": False, "urgent": 0, "open": []} | status
        self.briefs: list[int] = []
        self.reports: list[tuple[float, str | None]] = []
        self.waits: list[str] = []

    async def tools(self):
        return []

    async def brief(self, *, compact=False):
        self.briefs.append(self.engine.time_ms)
        return f"brief at {self.engine.time_ms}"

    async def call(self, name, arguments):
        return json.dumps({"status": "accepted", "message": "ok"})

    async def status(self):
        return dict(self.status_fields)

    async def report_call(self, seconds, error, **extra):
        self.reports.append((seconds, error))

    async def waiting(self, state):
        self.waits.append(state)


def rig(model, *, timeout=120.0, settings=None, **status):
    engine = FakeEngine()

    async def noop() -> bool:
        return True

    async def act() -> None:
        return None

    clock = LockstepClock(engine, observe=noop, act=act, done=lambda: False, brain_timeout_s=timeout)
    link = FakeLink(engine, **status)
    seat = CommanderSeat(Commander(model, settings or CommanderSettings(game_clock=True)), link, "1", [])
    clock.add_brain(seat)
    return clock, link, seat


@pytest.mark.parametrize(
    ("now", "last", "urgent", "due"),
    [
        (0, None, 0, True),
        (24.9, 0, 0, False),
        (25, 0, 0, True),
        (7.9, 0, 3, False),
        (8, 0, 3, True),
        (9, 0, 0, False),
    ],
)
def test_a_strategist_call_falls_due_every_25_s_or_8_s_after_the_last_when_something_is_urgent(now, last, urgent, due):
    assert decision_due(now, last, urgent, interval_s=25, min_gap_s=8) is due
    assert decision_due(now, last, urgent, interval_s=25, min_gap_s=8, started=False) is False


async def test_the_seat_decides_every_25_game_seconds_with_nothing_urgent():
    clock, link, seat = rig(FakeStrategist())
    await clock.run(max_steps=160)
    assert link.briefs == [0, 25_000]
    assert seat.commander.record.calls == 2 and [d["applied_ms"] for d in clock.decisions] == [5000, 30_000]


async def test_urgent_events_bring_a_decision_but_never_closer_than_8_game_seconds_to_the_last():
    clock, link, seat = rig(FakeStrategist(), urgent=2)
    await clock.run(max_steps=160)
    assert link.briefs == [0, 8000, 16_000, 24_000]
    assert seat.commander.record.calls == 4


async def test_a_seat_waits_on_its_decision_only_when_one_was_open_and_the_wait_spans_the_latency():
    clock, link, _ = rig(FakeStrategist(), open=["economy"])
    await clock.run(max_steps=40)
    assert link.waits == ["begin", "end"]
    clock, link, _ = rig(FakeStrategist())
    await clock.run(max_steps=40)
    assert link.waits == []


async def test_a_missed_decision_is_reported_with_its_error_and_the_wait_ends():
    clock, link, seat = rig(FakeStrategist(gate=asyncio.Event()), timeout=0.05, open=["economy"])
    await clock.run(max_steps=40)
    assert link.waits == ["begin", "end"]
    assert [(round(s), e) for s, e in link.reports][0][1] == "no reply within 0.05 s"
    assert seat.commander.record.calls == 1 and seat.commander.record.errors == 1


async def test_a_defeated_or_finished_seat_decides_nothing_more():
    clock, link, seat = rig(FakeStrategist(), defeated=True)
    await clock.run(max_steps=40)
    assert link.briefs == [] and seat.commander.record.end_reason == "defeated (match_over)"
    clock, link, seat = rig(FakeStrategist(), over=True)
    await clock.run(max_steps=40)
    assert link.briefs == [] and seat.commander.record.end_reason == "match_over"


async def test_max_calls_ends_the_seat_after_that_many_answered_decisions():
    settings = CommanderSettings(game_clock=True, max_calls=2)
    clock, link, seat = rig(FakeStrategist(), settings=settings, urgent=1)
    await clock.run(max_steps=200)
    assert seat.commander.record.calls == 2 and seat.commander.record.end_reason == "match_over"


# ---- the in-process link against the MCP link ---------------------------------------------------------------------
SETTINGS = {"speed": 2.0, "time_cap_minutes": 5, "start_grace_seconds": 300}


@pytest.fixture
async def env(tmp_path):
    async def start():
        return Handle(ScriptedEngine())

    e = ZeroADEnv(start, tmp_path, tick_s=0.05, match_options={"placer": placer(), "environ": {}})
    yield e
    if e.live is not None:
        await e.live.stop()


async def open_match(http, base) -> None:
    lobby = f"{base}/agentenv/ext/lobby"
    assert (await http.post(f"{lobby}/open", json={"game_settings": SETTINGS})).status_code == 200
    seat = {"player_id": "1", "player_kind": "agent", "player_name": "alice", "game_settings": {"civ": "rome"}}
    assert (await http.post(f"{lobby}/fill", json=seat)).status_code == 200
    await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai", "game_settings": {"civ": "pers"}})
    assert (await http.post(f"{lobby}/close", json={})).json()["status"] == "closed"


async def test_the_model_sees_the_same_system_prompt_tools_and_user_message_over_both_links(env):
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        await open_match(http, base)
        live = env.live
        live.start = lambda: None  # the match stays at its first snapshot, so both links read one state
        bodies = {}
        async with mcp_link({"url": f"{base}/players/1/mcp"}) as over_mcp:
            in_process = InProcessSeatLink(live, "1", await tool_specs())
            await in_process.brief()  # drains the opening events, as the MCP link's first brief did
            for name, link in (("mcp", over_mcp), ("in_process", in_process)):
                fake = FakeLiteLLM()
                model = StrategistModel(
                    "anthropic/claude-opus-5-5", "https://p.example", KEY, SpendGuard("m"), http=fake.http()
                )
                commander = Commander(model, CommanderSettings(opening="Win.", game_clock=True))
                tools = await link.tools()
                user = await commander.prepare(link, False)
                await commander.ask(user, tools)
                bodies[name] = fake.bodies("/v1/chat/completions")[0]
            assert await over_mcp.tools() == await in_process.tools()

    assert bodies["mcp"] == bodies["in_process"]
    assert json.dumps(bodies["mcp"], sort_keys=True) == json.dumps(bodies["in_process"], sort_keys=True)
    sent_tools = [t["function"]["name"] for t in bodies["mcp"]["tools"]]
    assert sent_tools == ["advance_phase", "army", "build", "research", "say", "set_economy", "train"]


async def test_an_order_gets_the_same_result_text_over_both_links_when_its_arguments_are_refused(env):
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        await open_match(http, base)
        env.live.start = lambda: None
        async with mcp_link({"url": f"{base}/players/1/mcp"}) as over_mcp:
            in_process = InProcessSeatLink(env.live, "1", await tool_specs())
            bad = {"action": "attack", "target": "the moon"}
            assert await over_mcp.call("army", bad) == await in_process.call("army", bad)
            good = json.loads(await in_process.call("train", {"role": "villager", "count": 5}))
            assert good["status"] == "accepted" and good["footer"] == env.live.footer("1")
            assert env.live.seats["1"].executor.book.pending_count("villager") == 5


async def test_the_in_process_link_reports_calls_and_waits_into_the_matchs_seat_record(env):
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        await open_match(http, base)
        link = InProcessSeatLink(env.live, "1", await tool_specs())
        await link.report_call(1.5, None, usd=0.01, input_tokens=100, output_tokens=20, model="local/q", turn={"a": 1})
        await link.report_call(0.5, "boom")
        seat = env.live.seats["1"]
        assert [(c.seconds, c.error) for c in seat.brain_calls] == [(1.5, None), (0.5, "boom")]
        assert seat.brain_model == "local/q" and seat.brain_spend["input_tokens"] == 100
        assert seat.turns[0]["a"] == 1 and seat.turns[0]["model"] == "local/q"
        await link.waiting("begin")
        assert seat.thinking_since is not None
        await link.waiting("end")
        assert seat.thinking_since is None and (await link.status())["playing_out"] is False
