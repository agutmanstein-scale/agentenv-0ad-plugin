"""The brains' loops against a scripted seat (no env, no network): when the strategist calls (interval or urgent
event), when a seat counts as waiting on its brain (only while something is open), that a wait always closes, that a
failed call reports what it was charged, and the Luna player's per-question cooldown."""

import json

import pytest

from agentenv_zeroad.brains import COOLDOWN_S, Commander, CommanderSettings, LunaPlayer
from agentenv_zeroad.decisions import DecisionsClient
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendGuard
from tests.fake_models import KEY, FakeLiteLLM

pytestmark = pytest.mark.anyio


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.now += max(s, 0.01)


class FakeLink:
    """A seat: `status_at(t, calls)` scripts what /status says; every route the brain uses is recorded."""

    def __init__(self, clock: Clock, status_at, *, fail_tool: bool = False):
        self.clock = clock
        self.status_at = status_at
        self.fail_tool = fail_tool
        self.call_times: list[float] = []
        self.waits: list[str] = []
        self.reports: list[dict] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.holds: list[str] = []

    async def hold(self, decision: str) -> None:
        self.holds.append(decision)

    async def tools(self):
        return [ToolSpec("army", "army", {"type": "object", "properties": {}})]

    async def brief(self, *, compact: bool = False) -> str:
        return json.dumps({"player": 1, "phase": "village", "orders": {}, "players": {"1": "a", "2": "b"}})

    async def status(self):
        return self.status_at(self.clock.now, len(self.call_times))

    async def report_call(self, seconds, error, **extra):
        self.reports.append({"seconds": seconds, "error": error, **extra})

    async def waiting(self, state: str) -> None:
        self.waits.append(state)

    async def offer_reflex(self, base_url, api_key) -> bool:
        return False

    async def call(self, name, arguments):
        if self.fail_tool:
            raise RuntimeError("the MCP session dropped")
        self.tool_calls.append((name, arguments))
        return json.dumps({"status": "accepted"})


def commander(fake: FakeLiteLLM, clock: Clock, **settings) -> Commander:
    model = StrategistModel("anthropic/claude-opus-5-5", "https://p/v1", KEY, SpendGuard("t"), http=fake.http())
    commander = Commander(model, CommanderSettings(poll_s=1.0, **settings), clock=clock, sleep=clock.sleep)
    real_turn = commander.turn

    async def timed_turn(link, tools, **kw):
        link.call_times.append(clock.now)
        return await real_turn(link, tools, **kw)

    commander.turn = timed_turn
    return commander


async def test_an_urgent_event_brings_the_next_call_forward_but_not_before_the_min_gap():
    clock = Clock()

    def status(t, calls):
        if t >= 60:
            return {"started": True, "over": True}
        return {"started": True, "over": False, "urgent": 1 if calls >= 1 else 0, "open": ["urgent"] if calls else []}

    link = FakeLink(clock, status)
    c = commander(FakeLiteLLM(tool_calls=[("army", {"action": "defend", "target": "home"})]), clock)
    await c.play(link)
    first, second = link.call_times[:2]
    assert second - first >= 8.0 and second - first < 25.0  # min_gap 8 s, interval 25 s


async def test_without_an_urgent_event_the_strategist_calls_on_its_interval():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": t >= 60, "urgent": 0, "open": []})
    c = commander(FakeLiteLLM(), clock)
    await c.play(link)
    first, second = link.call_times[:2]
    assert second - first >= 25.0


async def test_on_the_game_clock_the_interval_counts_game_seconds_at_any_speed():
    clock = Clock()
    status = lambda t, calls: {"started": True, "over": t >= 60, "game_ms": int(t * 4000), "open": []}  # noqa: E731
    link = FakeLink(clock, status)
    await commander(FakeLiteLLM(), clock, game_clock=True).play(link)
    first, second = link.call_times[:2]
    assert 6.0 <= second - first < 8.0  # 25 game seconds at 4x


async def test_a_logged_turn_reports_its_start_the_user_message_its_orders_and_token_ids():
    clock = Clock()
    status = lambda t, calls: {"started": True, "over": t >= 10, "game_ms": int(t * 1000), "open": []}  # noqa: E731
    link = FakeLink(clock, status)
    c = commander(FakeLiteLLM(token_ids=True, tool_calls=[("army", {"action": "defend"})]), clock, log_turns=True)
    await c.play(link)
    turn = link.reports[0]["turn"]
    assert turn["began_ms"] == 0 and "Decide your orders now" in turn["user"] and turn["urgent"] is False
    assert turn["orders"] == [{"tool": "army", "arguments": {"action": "defend"}, "status": "accepted"}]
    assert turn["prompt_token_ids"] == [1, 2, 3] and turn["token_ids"] == [7, 8]
    unlogged = Clock()
    link = FakeLink(unlogged, status)
    await commander(FakeLiteLLM(), unlogged).play(link)
    assert link.reports[0].get("turn") is None  # SeatLink drops a None extra from the body


async def test_a_defeated_seats_strategist_stops_calling_its_model_and_waits_out_the_match():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": t >= 300, "defeated": t >= 30, "open": []})
    c = commander(FakeLiteLLM(), clock)
    record = await c.play(link)
    assert len(link.call_times) == 2 and record.end_reason == "defeated (match_over)" and clock.now >= 300


async def test_a_call_while_nothing_is_open_is_no_wait_and_one_while_something_is_open_is():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": t >= 30, "urgent": 0, "open": []})
    await commander(FakeLiteLLM(), clock).play(link)
    assert len(link.call_times) == 2 and link.waits == []
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": t >= 30, "urgent": 0, "open": ["army"]})
    await commander(FakeLiteLLM(), clock).play(link)
    assert link.waits == ["begin", "end", "begin", "end"]


async def test_a_raising_order_call_is_rejected_the_wait_closes_and_the_strategist_plays_on():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": t >= 60, "open": ["army"]}, fail_tool=True)
    c = commander(FakeLiteLLM(tool_calls=[("army", {"action": "defend", "target": "home"})]), clock)
    record = await c.play(link)
    assert record.end_reason == "match_over" and len(link.call_times) == 3
    assert record.rejected == record.orders == 3 and "dropped" in record.last_error
    assert link.waits == ["begin", "end"] * 3


async def test_a_failed_strategist_call_reports_the_bound_it_was_charged():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": True, "over": False, "open": []})
    c = commander(FakeLiteLLM(fail=500), clock)
    assert await c.turn(link, await link.tools())
    assert c.model.guard.spent_usd > 0
    assert link.reports[0]["error"] and link.reports[0]["usd"] == pytest.approx(c.model.guard.spent_usd, abs=1e-6)


def luna(fake: FakeLiteLLM, clock: Clock) -> LunaPlayer:
    client = DecisionsClient("https://p", KEY, SpendGuard("luna"), http=fake.http())
    return LunaPlayer(client, clock=clock, sleep=clock.sleep)


BRIEF = {"player": 1, "phase": "city", "orders": {"train": {}}, "players": {"1": "a", "2": "b"}, "idle": {}}


async def test_the_luna_players_cooldown_keeps_a_steady_answer_from_repeating_its_order():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {})
    player = luna(FakeLiteLLM(choose=lambda q: "cavalry" if q["name"] == "train" else "none"), clock)
    await player.turn(link, BRIEF)
    clock.now += COOLDOWN_S["train"] / 2
    await player.turn(link, BRIEF)
    assert [n for n, _ in link.tool_calls].count("train") == 1
    clock.now += COOLDOWN_S["train"]
    await player.turn(link, BRIEF)
    assert [n for n, _ in link.tool_calls].count("train") == 2


async def test_the_luna_player_waits_only_while_something_is_open_and_always_closes_the_wait():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {})
    player = luna(FakeLiteLLM(), clock)
    await player.turn(link, BRIEF)
    assert link.waits == []
    await player.turn(link, BRIEF, waiting=True)
    assert link.waits == ["begin", "end"]
    failing = FakeLink(clock, lambda t, calls: {}, fail_tool=True)
    player = luna(FakeLiteLLM(choose=lambda q: "cavalry" if q["name"] == "train" else "none"), clock)
    assert await player.turn(failing, BRIEF, waiting=True)  # a raising order call is rejected, not fatal
    assert failing.waits == ["begin", "end"] and player.record.rejected == 1


async def test_a_failed_luna_call_reports_the_bound_it_was_charged():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {})
    player = luna(FakeLiteLLM(fail=500), clock)
    assert await player.turn(link, BRIEF)
    assert link.reports[0]["usd"] == pytest.approx(player.client.guard.spent_usd, abs=1e-6) and link.reports[0]["usd"]


async def test_the_luna_player_asks_nothing_before_the_match_starts():
    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {"started": t >= 5, "over": t >= 8, "open": []})
    briefs, reported = [], []
    real_brief, real_report = link.brief, link.report_call

    async def brief(*, compact=False):
        briefs.append(clock.now)
        return await real_brief(compact=compact)

    async def report_call(seconds, error, **extra):
        reported.append(clock.now)
        await real_report(seconds, error, **extra)

    link.brief, link.report_call = brief, report_call
    player = luna(FakeLiteLLM(choose=lambda q: "advance" if q["name"] == "phase" else "none"), clock)
    await player.play(link)
    assert briefs[0] == 0.0  # the first brief still marks the seat ready
    assert [t for t in briefs[1:] if t < 5] == [] and [t for t in reported if t < 5] == []
    assert player.record.calls >= 1 and link.tool_calls  # it plays once the match has started


async def test_a_luna_player_that_says_not_yet_past_a_deadline_reports_a_hold_and_sends_no_order_for_it():
    from tests.test_brains import luna_brief

    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {})
    late = luna_brief(time_ms=300_000, t="05:00", res={"food": 100, "wood": 100})
    player = luna(FakeLiteLLM(choose=lambda q: "wait" if q["name"] in ("phase", "army") else "keep"), clock)
    await player.turn(link, late)
    assert link.holds == ["phase", "attack"]
    assert not [c for c in link.tool_calls if c[0] in ("advance_phase", "army")]
    await player.turn(link, late)  # a steady hold is reported again only after a pause
    assert link.holds == ["phase", "attack"]
    clock.now += 10
    await player.turn(link, late)
    assert link.holds == ["phase", "attack", "phase", "attack"]
    early = luna_brief(res={"food": 450, "wood": 420})  # before the deadlines "not yet" is just Luna's choice
    clock.now += 10
    await player.turn(link, early)
    assert link.holds == ["phase", "attack", "phase", "attack"]


async def test_a_luna_player_that_chooses_past_a_deadline_sends_its_own_order_and_reports_no_hold():
    from tests.test_brains import luna_brief

    clock = Clock()
    link = FakeLink(clock, lambda t, calls: {})
    late = luna_brief(time_ms=300_000, t="05:00", res={"food": 100, "wood": 100})
    answers = {"phase": "advance", "army": "attack_2"}
    player = luna(FakeLiteLLM(choose=lambda q: answers.get(q["name"], "keep")), clock)
    await player.turn(link, late)
    assert link.holds == []
    assert ("advance_phase", {}) in link.tool_calls
    assert ("army", {"action": "attack", "target": "player_2"}) in link.tool_calls
