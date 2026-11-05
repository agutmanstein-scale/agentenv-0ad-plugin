"""Interface v2 order handling: an ignored `at`, the villager lever, hold, the executor's give-up notices, and what the
commander keeps in its memory (the game's answers, the notices), its one say() a turn and its per-civ system prompt."""

import asyncio
import json
from dataclasses import replace

import pytest

from agentenv_zeroad.brains import MEMORY_MESSAGE_CHARS, STRATEGIST_SYSTEM_V2, _answer_of
from agentenv_zeroad.brief_v2 import NOTICES_HEADER, notices_in, notices_lines
from agentenv_zeroad.executor import Failure
from agentenv_zeroad.llm import ToolSpec
from agentenv_zeroad.orders import BuildRequest, Hold, SetEconomy, Train
from tests.executor_fixtures import load, make_executor
from tests.fake_models import FakeLiteLLM
from tests.test_brain_loops import Clock, FakeLink, commander

RICH = {"food": 2000, "wood": 2000, "stone": 2000, "metal": 2000}


def trained_villagers(executor, seat):
    cc = replace(seat.structure(150), queue=0)
    free = replace(seat, resources=RICH, pop_cap=100, structures=(cc,) + seat.structures[1:])
    commands = asyncio.run(executor.tick(free, None))
    return [c for c in commands if c["type"] == "train" and "support" in c["template"]]


def test_a_wrong_at_is_rejected_in_v1_but_ignored_and_auto_routed_in_v2_with_a_message_saying_where():
    seat, _ = load(300)
    v1, _, _ = make_executor(interface="v1")
    v1.last_seat = seat
    bad = v1.submit(Train(role="infantry_melee", count=3, at="stable"))
    assert bad.code == "bad_building" and v1.book.train == []

    v2, _, _ = make_executor(interface="v2")
    v2.last_seat = seat
    ok = v2.submit(Train(role="infantry_melee", count=3, at="stable"))
    assert ok.status == "accepted" and ok.message.startswith("Ignored at='stable'")
    assert "barracks" in ok.message and "civil_centre" in ok.message
    assert [(t.role, t.remaining, t.at) for t in v2.book.train] == [("infantry_melee", 3, None)]


def test_a_wrong_at_for_a_role_nothing_built_can_train_is_partial_with_a_building_to_build():
    seat, _ = load(300)
    v2, _, _ = make_executor(interface="v2")
    v2.last_seat = seat
    result = v2.submit(Train(role="siege", count=1, at="stable"))
    assert result.status == "partial" and result.message.startswith("Ignored at='stable'")
    assert result.suggest and result.suggest.startswith("build(structure=")


def test_a_valid_at_and_no_at_answer_as_before_in_both_interfaces():
    seat, _ = load(300)
    for interface in ("v1", "v2"):
        executor, _, _ = make_executor(interface=interface)
        executor.last_seat = seat
        plain = executor.submit(Train(role="infantry_melee", count=2))
        valid = executor.submit(Train(role="infantry_melee", count=2, at="barracks"))
        assert (
            plain.message
            == valid.message
            == "Queued 2 infantry_melee; production starts as resources and houses allow."
        )


def test_the_villager_lever_sets_the_target_the_autopilot_trains_up_to():
    seat, _ = load(300)
    default, _, _ = make_executor(interface="v2")
    assert default.book.villager_target == 45 and trained_villagers(default, seat)
    lowered, _, _ = make_executor(interface="v2")
    result = lowered.submit(SetEconomy(food=50, wood=50, stone=0, metal=0, villagers=20))
    assert lowered.book.villager_target == 20 and "Villager target 20" in result.message
    assert trained_villagers(lowered, seat) == []
    raised, _, _ = make_executor(interface="v2")
    raised.submit(SetEconomy(food=50, wood=50, stone=0, metal=0, villagers=100))
    assert raised.book.villager_target == 100
    kept, _, _ = make_executor(interface="v2")
    kept.submit(SetEconomy(food=50, wood=50, stone=0, metal=0, villagers=70))
    kept.submit(SetEconomy(food=40, wood=40, stone=10, metal=10))
    assert kept.book.villager_target == 70  # leaving it out keeps it


def test_the_villager_target_is_bounded_to_20_through_100():
    for bad in (19, 101):
        with pytest.raises(ValueError):
            SetEconomy(food=100, wood=0, stone=0, metal=0, villagers=bad)


def test_hold_is_an_accepted_no_op_that_leaves_the_book_as_it_was():
    executor, _, _ = make_executor(interface="v2")
    executor.submit(Train(role="villager", count=5))
    before = executor.book.summary()
    result = executor.submit(Hold())
    assert result.status == "accepted" and executor.book.summary() == before


def test_the_executors_give_ups_are_kept_once_for_the_brief():
    executor, _, _ = make_executor(interface="v2")
    request = BuildRequest("field", "near_food", "structures/rome/field")
    for n in range(3):
        executor.fail_attempt(request, 1000 * n, "build field at near_food", "no valid spot")
    drained = executor.drain_notices()
    assert [f.what for f in drained] == ["build field at near_food"] and executor.drain_notices() == []
    assert executor.failures == drained  # the results still keep every give-up


def test_notices_merge_by_order_and_reason_are_capped_and_read_back_from_a_brief():
    failures = [
        Failure(1000 * i, 1, "field", "build field at near_food", "no valid spot at near_food", "near_food")
        for i in range(7)
    ]
    failures += [Failure(9000, 1, "barracks", "build barracks at front", "x" * 300, "front")]
    lines = notices_lines(failures)
    assert lines[0] == NOTICES_HEADER and "(x7)" in lines[1] and len(lines[1]) <= 120 and lines[2].endswith("…")
    assert notices_in("Time 05:00\n" + "\n".join(lines) + "\nSince last brief: nothing") == [
        ln.strip() for ln in lines[1:]
    ]
    assert notices_lines([]) == [] and notices_in("no section here") == []


def test_the_answer_kept_in_memory_is_the_message_and_the_suggestion_clipped():
    rejected = json.dumps(
        {
            "status": "rejected",
            "message": "m" * 400,
            "code": "x",
            "suggest": "build(structure='barracks', where='base')",
        }
    )
    answer = _answer_of(rejected)
    assert answer.startswith("m" * (MEMORY_MESSAGE_CHARS - 1) + "… (try build(") and len(answer) < 220
    assert _answer_of(json.dumps({"status": "accepted", "message": "Said."})) == "Said."
    assert _answer_of("not json") == "not json"


class AnsweringLink(FakeLink):
    """A v2 seat's link: answers each order as the env would, and offers the per-civ reference."""

    def __init__(self, clock, brief_text):
        super().__init__(clock, lambda t, calls: {"started": True, "over": calls >= 1})
        self.brief_text = brief_text
        self.answers = {
            "army": {
                "status": "rejected",
                "message": "Player 4 is not an opponent in this match.",
                "code": "bad_target",
                "suggest": "army(action='attack', target='player_2')",
            },
            "set_economy": {"status": "accepted", "message": "Economy split set: food 50%."},
        }

    async def tools(self):
        return [ToolSpec("army", "army", {"type": "object", "properties": {}})]

    async def reference(self):
        return "Costs for rome: villager 50f."

    async def brief(self, *, compact=False):
        return self.brief_text()

    async def call(self, name, arguments):
        self.tool_calls.append((name, arguments))
        return json.dumps(self.answers.get(name, {"status": "accepted", "message": "ok"}))


@pytest.mark.anyio
async def test_the_v2_commander_keeps_the_games_answers_and_notices_in_its_memory_and_one_say_a_turn():
    clock = Clock()
    notice = NOTICES_HEADER + "\n  build field at near_food (x3): no valid spot at near_food"
    briefs = iter(f"Time 00:30\n{n}\nSince last brief: nothing" for n in ("", notice, ""))
    link = AnsweringLink(clock, lambda: next(briefs))
    fake = FakeLiteLLM(
        tool_calls=[
            ("set_economy", {"food": 50, "wood": 50, "stone": 0, "metal": 0}),
            ("army", {"action": "attack", "target": "player_4"}),
            ("say", {"text": "one"}),
            ("say", {"text": "two"}),
            ("hold", {}),
            ("dance", {}),
        ]
    )
    c = commander(fake, clock, interface="v2", system_prompt=STRATEGIST_SYSTEM_V2)
    c._reference = await link.reference()
    tools = await link.tools()
    await c.turn(link, tools)
    assert [n for n, _ in link.tool_calls] == [
        "set_economy",
        "army",
        "say",
        "hold",
    ]  # a second say, a made-up tool: not sent
    await c.turn(link, tools)
    await c.turn(link, tools)

    first, second, third = fake.bodies("/v1/chat/completions")
    assert first["messages"][0]["content"] == STRATEGIST_SYSTEM_V2 + "\n\nCosts for rome: villager 50f."
    assert "Your orders in your last turns" not in first["messages"][1]["content"]
    memory = second["messages"][1]["content"]
    assert "-> rejected: Player 4 is not an opponent in this match. (try army(action='attack'" in memory
    assert "-> accepted: Economy split set: food 50%." in memory
    assert 'say({"text":"two"}) -> rejected: Only one say() per turn.' in memory
    assert "dance({}) -> rejected: unknown tool dance" in memory
    assert "executor gave up" not in memory  # the first brief had no notices
    assert (
        "executor gave up: build field at near_food (x3): no valid spot at near_food" in third["messages"][1]["content"]
    )
    assert len(c._memory) == 3 and all(" | " in turn for turn in c._memory)


@pytest.mark.anyio
async def test_a_v1_commander_never_asks_for_the_reference_and_keeps_its_old_memory_lines():
    clock = Clock()
    link = AnsweringLink(clock, lambda: "Time 00:30")

    async def refuse():
        raise AssertionError("a v1 seat has no reference")

    link.reference = refuse
    fake = FakeLiteLLM(tool_calls=[("army", {"action": "attack", "target": "player_4"}), ("say", {"text": "a"})])
    c = commander(fake, clock)
    await c.play(link)
    assert c.system == c.settings.system_prompt
    assert c._memory[-1] == 'army({"action":"attack","target":"player_4"}) -> rejected; say({"text":"a"}) -> accepted'
