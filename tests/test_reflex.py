"""The Luna reflex: which tactical questions open, choices limited by the standing orders, answers as orders, one
request in flight, failures skipped without stopping anything, and the seat's waits reported to its metrics."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from agentenv_zeroad.decisions import Answer, Decision, Question
from agentenv_zeroad.match import LiveMatch
from agentenv_zeroad.metrics import SeatMetrics
from agentenv_zeroad.orders import Army, OrderBook, SetEconomy
from agentenv_zeroad.reflex import Reflex, exhausted_question, open_questions
from agentenv_zeroad.spend import SpendCapReached, SpendGuard
from agentenv_zeroad.state import Snapshot
from tests.executor_fixtures import load_raw

pytestmark = pytest.mark.anyio


def snapshot(**changes) -> Snapshot:
    raw = copy.deepcopy(load_raw(60)["snapshot"])
    raw["players"]["1"].update(changes)
    return Snapshot.from_json(raw)


def me(**changes):
    return snapshot(**changes).players[1]


def test_nothing_is_asked_while_no_tactical_decision_is_open():
    assert open_questions(me(), OrderBook(1)) == []
    assert open_questions(me(idle_units={"infantry_melee": 6}, idle_production=[150]), OrderBook(1)) == []


def test_an_exhausted_resource_moves_its_share_to_a_chosen_resource_of_the_split():
    book = OrderBook(1)
    book.set_split({"food": 50, "wood": 40, "stone": 0, "metal": 10})
    nearby = {"food": {"amount": 0, "sources": 0}, "wood": {"amount": 900, "sources": 9}}
    q = exhausted_question(me(nearby_resources=nearby), book, ran_out=["food"])
    assert [c.value for c in q.question.choices] == ["wood", "metal", "keep"]  # stone isn't in the split
    assert q.apply("wood") == [SetEconomy(food=0, wood=90, stone=0, metal=10)]
    assert not q.apply("keep")


class FakeDecider:
    def __init__(self, answers=None, *, fail=False, wait: asyncio.Event | None = None):
        self.answers = answers or {}
        self.fail = fail
        self.wait = wait
        self.asked: list[list[Question]] = []

    async def decide(self, context, questions):
        self.asked.append(questions)
        if self.wait is not None:
            await self.wait.wait()
        if self.fail:
            raise TimeoutError("slow")
        return Decision({k: Answer(k, "choice", choice=v) for k, v in self.answers.items()}, 0.2, 0.0001)


def reflex(decider, snap, book=None):
    book = book or OrderBook(1)
    submitted = []

    def submit(order):
        submitted.append(order)
        if isinstance(order, Army):  # like Executor._submit_army: every army order bumps the book's seq
            book.army.action, book.army.target = order.action, order.target
            book.army.seq += 1

    metrics = SeatMetrics(1)
    metrics.observe(snap)
    r = Reflex(1, book, submit, lambda: snap, metrics, decider)
    return r, submitted, metrics


async def test_answers_become_orders_and_the_wait_is_reported():
    snap = snapshot(idle_units={"infantry_melee": 5}, enemies_near_base=3)
    r, submitted, metrics = reflex(FakeDecider({"threat": "defend"}), snap)
    r.tick()
    await r._inflight
    assert submitted == [Army(action="defend", target="home")]
    assert r.stats.answered == 1 and r.stats.orders == 1 and r.stats.questions == 1
    assert metrics.waits_reported and metrics.summary()["waiting_share"] is not None


async def test_one_request_in_flight_a_busy_tick_is_skipped_and_returns_at_once():
    gate = asyncio.Event()
    snap = snapshot(idle_units={"infantry_melee": 5}, enemies_near_base=2)
    decider = FakeDecider({"threat": "ignore"}, wait=gate)  # no change: the question does not rest
    r, _, _ = reflex(decider, snap)
    r.tick()
    r.tick()
    r.tick()
    assert len(decider.asked) <= 1 and r.stats.skipped_busy == 2
    await asyncio.sleep(0)
    assert len(decider.asked) == 1
    gate.set()
    await r._inflight
    r.tick()
    await r._inflight
    assert len(decider.asked) == 2


async def test_a_failed_or_slow_request_is_skipped_and_the_next_interval_asks_again():
    snap = snapshot(idle_units={"infantry_melee": 5}, enemies_near_base=2)
    r, submitted, metrics = reflex(FakeDecider(fail=True), snap)
    r.tick()
    await r._inflight
    assert r.stats.failed == 1 and submitted == [] and "TimeoutError" in r.stats.last_error
    assert metrics._waits == 0  # the wait ended with the failure
    r.tick()
    await r._inflight
    assert r.stats.requests == 2


async def test_a_seat_without_credentials_asks_nothing():
    r, _, _ = reflex(None, snapshot(idle_units={"infantry_melee": 5}))
    r.tick()
    assert r._inflight is None and r.stats.skipped_no_key == 1


def nearby(**amounts):
    return {r: {"amount": a, "sources": 1 if a else 0} for r, a in amounts.items()}


async def test_exhausted_opens_only_when_a_resource_runs_out_not_when_it_was_never_there_or_is_farmed():
    book = OrderBook(1)
    book.set_split({"food": 50, "wood": 30, "stone": 10, "metal": 10})
    never = snapshot(nearby_resources=nearby(food=0, wood=900, stone=0, metal=500))
    r, _, _ = reflex(FakeDecider({"exhausted": "wood"}), never, book)
    r.tick()
    assert r._inflight is None  # food and stone were never seen near the base: nothing ran out
    farmed = copy.deepcopy(load_raw(60)["snapshot"])
    farmed["players"]["1"]["nearby_resources"] = nearby(food=300, wood=900, stone=0, metal=500)
    farm = {"id": 999, "kind": "Field", "pos": None, "hp": 100}
    farmed["players"]["1"]["buildings"] = [*farmed["players"]["1"].get("buildings", []), farm]
    seen = Snapshot.from_json(farmed)
    farmed["players"]["1"]["nearby_resources"] = nearby(food=0, wood=900, stone=0, metal=500)
    gone = Snapshot.from_json(farmed)
    current = [seen]
    r = Reflex(1, book, lambda o: None, lambda: current[0], SeatMetrics(1), FakeDecider({"exhausted": "wood"}))
    r.tick()
    current[0] = gone
    r.tick()
    assert r._inflight is None  # the hunt ran out, but the seat lives on its fields


async def test_a_resource_that_ran_out_is_asked_once_and_again_only_after_supply_returns():
    book = OrderBook(1)
    book.set_split({"food": 50, "wood": 40, "stone": 0, "metal": 10})
    current = [snapshot(nearby_resources=nearby(food=0, wood=500, stone=0, metal=500))]
    decider = FakeDecider({"exhausted": "keep"})
    r = Reflex(1, book, lambda o: None, lambda: current[0], SeatMetrics(1), decider)
    r.tick()
    current[0] = snapshot(nearby_resources=nearby(food=0, wood=0, stone=0, metal=500))
    r.tick()
    await r._inflight
    assert [q.name for q in decider.asked[0]] == ["exhausted"] and "wood has run out" in decider.asked[0][
        0
    ].instructions
    r.tick()
    assert r._inflight.done() and len(decider.asked) == 1  # 'keep' was answered: not asked again every second
    current[0] = snapshot(nearby_resources=nearby(food=0, wood=200, stone=0, metal=500))
    r.tick()
    current[0] = snapshot(nearby_resources=nearby(food=0, wood=0, stone=0, metal=500))
    r.tick()
    await r._inflight
    assert len(decider.asked) == 2


class CappedDecider(FakeDecider):
    async def decide(self, context, questions):
        self.asked.append(questions)
        raise SpendCapReached("the next call could pass the cap")


async def test_a_reflex_stopped_by_its_cap_asks_nothing_more_and_is_not_rearmed():
    decider = CappedDecider()
    r, _, _ = reflex(decider, snapshot(idle_units={"infantry_melee": 5}, enemies_near_base=2))
    r.tick()
    await r._inflight
    assert r.decider is None and "cap" in r.stats.last_error
    r.tick()
    assert len(decider.asked) == 1 and r.stats.skipped_no_key == 1
    guard = SpendGuard("reflex", cap_usd=0.01)
    guard.stopped = "cap reached"
    seat = SimpleNamespace(reflex=r, reflex_guard=guard, plan=None)
    fake_match = SimpleNamespace(seats={"1": seat}, _decider_for=lambda *a: FakeDecider())
    assert LiveMatch.set_reflex_credentials(fake_match, "1", "https://p", "k") is True
    assert r.decider is None


async def test_a_defeated_seat_is_not_asked():
    decider = FakeDecider({"threat": "defend"})
    r, _, _ = reflex(decider, snapshot(state="defeated", idle_units={"infantry_melee": 5}, enemies_near_base=2))
    r.tick()
    assert r._inflight is None and decider.asked == []


def test_an_exhausted_answer_is_dropped_when_the_strategist_changed_the_split_meanwhile():
    book = OrderBook(1)
    book.set_split({"food": 50, "wood": 50, "stone": 0, "metal": 0})
    q = exhausted_question(me(), book, ran_out=["wood"])
    book.set_split({"food": 20, "wood": 30, "stone": 0, "metal": 50})  # lands while the request is in flight
    assert not q.apply("food")
    assert book.split == {"food": 20, "wood": 30, "stone": 0, "metal": 50}


def with_ccs(stone: int, cc_ids: list[int]) -> Snapshot:
    raw = copy.deepcopy(load_raw(60)["snapshot"])
    p = raw["players"]["1"]
    p["buildings"] = [b for b in p.get("buildings", []) if b["kind"] != "CivCentre"]
    p["buildings"] += [{"id": i, "kind": "CivCentre", "pos": [900, 900], "hp": 100} for i in cc_ids]
    p["nearby_resources"] = nearby(food=800, wood=5000, stone=stone, metal=3000)
    return Snapshot.from_json(raw)


async def test_losing_a_civic_centre_is_not_read_as_a_resource_running_out():
    book = OrderBook(1)
    book.set_split({"food": 40, "wood": 30, "stone": 20, "metal": 10})
    current = [with_ccs(2000, [1, 777])]
    decider = FakeDecider({"exhausted": "metal"})
    r = Reflex(1, book, lambda o: None, lambda: current[0], SeatMetrics(1), decider)
    r.tick()
    current[0] = with_ccs(0, [1])  # the CC whose area held the stone was lost; the stone is still there
    r.tick()
    assert r._inflight is None and decider.asked == []
    current[0] = with_ccs(1500, [1])
    r.tick()
    current[0] = with_ccs(0, [1])  # same base, the stone really runs out
    r.tick()
    await r._inflight
    assert [q.name for q in decider.asked[0]] == ["exhausted"]


async def test_a_strategist_order_in_the_same_tick_as_a_reflex_departure_stands():
    snap = snapshot(idle_units={"infantry_melee": 5}, enemies_near_base=2)
    book = OrderBook(1)
    book.army.action, book.army.target, book.army.since_ms = "attack", "player_2", 1000
    decider = FakeDecider({"threat": "defend"})
    r, submitted, _ = reflex(decider, snap, book)
    submit = r.submit

    def submit_in_one_tick(order):  # the executor stamps since_ms with its last view time: the same in one tick
        submit(order)
        book.army.since_ms = 60000

    r.submit = submit_in_one_tick
    r.tick()
    await r._inflight  # the reflex departs: defend home
    r.submit(Army(action="defend", target="home"))  # the strategist orders defend home in the same executor tick
    decider.answers = {"threat": "press_on"}
    r.tick()
    if r._inflight is not None:
        await r._inflight
    assert (book.army.action, book.army.target) == ("defend", "home")
    assert submitted[-1] == Army(action="defend", target="home")
