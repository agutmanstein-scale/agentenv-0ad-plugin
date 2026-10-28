"""U5b: the reflex asks at real moments (threat, regroup, engage, target, counter) where its answer can differ from
the standing orders, each changing answer becomes executor orders (or the army's focus) and is logged by kind; the
army driver honours the focus. All with a fake Decisions client (no network, no keys)."""

import asyncio
import copy
from dataclasses import replace

import pytest

from agentenv_zeroad.decisions import Answer, Decision
from agentenv_zeroad.metrics import SeatMetrics
from agentenv_zeroad.orders import Army, OrderBook, Train, TrainRequest, partial, rejected
from agentenv_zeroad.reflex import (
    COUNTER_TOP,
    QUIET_MS,
    Focus,
    Reflex,
    counter_question,
    engage_question,
    open_decisions,
    open_questions,
    regroup_question,
    target_question,
    threat_question,
)
from agentenv_zeroad.state import Snapshot
from tests.executor_fixtures import load, load_raw, make_executor

pytestmark = pytest.mark.anyio

ENEMY = {
    "units": {"infantry_melee": 4, "cavalry": 2, "villager": 6},
    "buildings": [{"id": 900, "kind": "House", "pos": [300, 200]}],
    "nearest": 150,
    "soldiers_at": [320, 210, 6],
    "workers_at": [280, 180, 6],
    "contact": 0,
}


def snapshot(time_ms=60_000, enemy=None, **changes) -> Snapshot:
    raw = copy.deepcopy(load_raw(60)["snapshot"])
    raw["time_ms"] = time_ms
    raw["players"]["1"].update(changes)
    if enemy is not None:
        raw["players"]["1"]["enemies"] = {"2": enemy}
    return Snapshot.from_json(raw)


def me(**kw):
    return snapshot(**kw).players[1]


def attacking(target="player_2") -> OrderBook:
    book = OrderBook(1)
    book.army.action, book.army.target, book.army.seq = "attack", target, 1
    return book


def values(q):
    return [c.value for c in q.question.choices]


# ---- the questions -------------------------------------------------------------------------------------------------
def test_a_raid_on_a_working_base_asks_defend_or_ignore_and_defend_changes_the_order():
    book = OrderBook(1)
    q = threat_question(me(enemies_near_base=3, enemy=ENEMY), book, ("gather", "home"))
    assert values(q) == ["defend", "ignore"] and "3 enemy soldiers are at our base" in q.question.instructions
    assert q.apply("defend") == [Army(action="defend", target="home")]
    assert q.apply("ignore") == []  # gather home is the standing order already
    assert q.texts["defend"] == "sent the soldiers to defend the base"


def test_a_raid_while_our_attack_is_out_asks_press_on_defend_or_fall_back():
    book = attacking()
    q = threat_question(me(enemies_near_base=2), book, ("attack", "player_2"))
    assert values(q) == ["press_on", "defend", "fall_back"]
    assert q.apply("press_on") == []
    assert q.apply("defend") == [Army(action="defend", target="home")]
    assert q.apply("fall_back") == [Army(action="retreat", target="home")]


PEACEFUL = [("gather", "home"), ("scout", "player_2"), ("retreat", "home"), ("defend", "front")]


@pytest.mark.parametrize("posture", PEACEFUL)
def test_no_question_offers_an_attack_the_strategist_did_not_order(posture):
    book = OrderBook(1)
    book.army.action, book.army.target = posture
    fight = dict(ENEMY, contact=4)
    m = me(enemies_near_base=2, idle_production=[150], enemy=fight)
    for q in open_questions(m, book, standing=posture) + open_questions(me(enemy=fight), book, standing=posture):
        for value in values(q):
            for act in q.apply(value) or []:
                assert not (isinstance(act, Army) and act.action in ("attack", "raid")), (q.question.name, value)
                assert not isinstance(act, Focus)


def test_defending_at_home_already_leaves_no_threat_question():
    book = OrderBook(1)
    book.army.action = "defend"
    assert threat_question(me(enemies_near_base=2), book, ("defend", "home")) is None


def test_regroup_offers_the_strategists_order_back_once_the_danger_has_passed():
    book = attacking()
    book.army.action, book.army.target = "defend", "home"  # the reflex's departure
    assert regroup_question(me(enemies_near_base=1), book, ("attack", "player_2")) is None
    q = regroup_question(me(), book, ("attack", "player_2"))
    assert values(q) == ["resume", "stay"]
    assert q.apply("resume") == [Army(action="attack", target="player_2")] and q.apply("stay") == []
    assert regroup_question(me(), attacking(), ("attack", "player_2")) is None  # nothing to go back to


def test_contact_during_an_attack_asks_fight_push_or_fall_back_and_fight_sets_the_focus():
    book = attacking()
    m = me(enemy=dict(ENEMY, contact=3))
    q = engage_question(m, book)
    assert values(q) == ["fight", "push", "fall_back"]
    assert q.apply("fight") == [Focus("army")] and q.apply("push") == [Focus("buildings")]
    assert q.apply("fall_back") == [Army(action="retreat", target="home")]
    book.army.seq += 1  # the strategist gave a new army order meanwhile
    assert q.apply("fight") == []
    assert engage_question(me(enemy=dict(ENEMY, contact=1)), attacking()) is None  # a lone scout is no fight
    # enemies at the base: the threat question owns that moment
    assert engage_question(me(enemies_near_base=2, enemy=dict(ENEMY, contact=3)), attacking()) is None


def test_contact_under_a_peaceful_order_only_holds_or_falls_back():
    q = engage_question(me(enemy=dict(ENEMY, contact=3)), OrderBook(1))
    assert values(q) == ["hold", "fall_back"]
    assert q.apply("hold") == [] and q.apply("fall_back") == [Army(action="retreat", target="home")]


def test_contact_during_an_attack_on_a_place_not_a_player_only_holds_or_falls_back():
    q = engage_question(me(enemy=dict(ENEMY, contact=3)), attacking(target="center"))
    assert values(q) == ["hold", "fall_back"]  # the driver aims a focus only at a player's army or buildings
    assert q.apply("hold") == []


def test_an_attack_in_sight_of_the_enemy_picks_what_to_hit_first():
    book = attacking()
    q = target_question(me(enemy=ENEMY), book)
    assert values(q) == ["army", "workers", "buildings"]
    assert q.apply("workers") == [Focus("workers")]
    book.army.focus = "workers"
    assert q.apply("workers") == []  # already aimed there: no change, no order
    assert target_question(me(enemy=ENEMY), OrderBook(1)) is None  # no attack out
    book.army.seq += 1  # the strategist gave a new army order while the request was in flight
    assert q.apply("army") == []
    assert target_question(me(enemy=dict(ENEMY, contact=5)), book) is None  # a fight is on: engage asks instead
    only_houses = {"units": {}, "buildings": ENEMY["buildings"], "nearest": 150}
    assert target_question(me(enemy=only_houses), book) is None  # one kind of target is no choice


def ordering(*roles, book=None, remaining=2) -> OrderBook:
    book = book or OrderBook(1)
    book.train = [TrainRequest(r, remaining) for r in roles]
    return book


def test_a_free_queue_during_a_fight_trains_more_of_an_ordered_soldier_role_unless_enough_are_ordered():
    book = ordering("infantry_melee", "cavalry")
    assert counter_question(me(idle_production=[150]), book) is None  # no fight
    q = counter_question(me(idle_production=[150], enemies_near_base=2, enemy=ENEMY), book)
    assert values(q) == ["infantry_melee", "cavalry", "none"]
    assert "4 infantry_melee, 2 cavalry" in q.question.instructions
    assert q.apply("infantry_melee") == [Train(role="infantry_melee", count=3)] and q.apply("none") == []
    book.train = [TrainRequest("infantry_melee", COUNTER_TOP)]
    assert q.apply("infantry_melee") == []  # topped up already
    book.train = [TrainRequest("infantry_melee", COUNTER_TOP - 1)]
    assert q.apply("infantry_melee") == [Train(role="infantry_melee", count=1)]


def test_counter_offers_only_the_soldier_roles_the_strategist_ordered():
    fight = me(idle_production=[150], enemies_near_base=2, enemy=ENEMY)
    assert values(counter_question(fight, ordering("cavalry", "villager"))) == ["cavalry", "none"]
    assert counter_question(fight, ordering("villager")) is None  # no soldiers ordered: nothing to add to
    assert counter_question(fight, OrderBook(1)) is None
    assert counter_question(fight, ordering("cavalry", remaining=COUNTER_TOP)) is None  # nothing left to top up


def test_counter_needs_a_fight_not_just_an_attack_with_enemies_in_sight():
    book = ordering("cavalry", book=attacking())
    assert counter_question(me(idle_production=[150], enemy=ENEMY), book) is None
    assert counter_question(me(idle_production=[150], enemy=dict(ENEMY, contact=3)), book) is not None


def test_soldiers_in_contact_count_as_an_open_decision():
    assert "contact" in open_decisions(me(enemy=dict(ENEMY, contact=3)), OrderBook(1), [])
    assert "contact" not in open_decisions(me(enemy=ENEMY), OrderBook(1), [])


# ---- the reflex loop -----------------------------------------------------------------------------------------------
class FakeDecider:
    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    async def decide(self, context, questions):
        self.asked.append([q.name for q in questions])
        return Decision({k: Answer(k, "choice", choice=v) for k, v in self.answers.items()}, 0.2, 0.0001)


def run_reflex(decider, book, current):
    submitted = []

    def submit(order):
        submitted.append(order)
        if isinstance(order, Army):
            book.army.action, book.army.target, book.army.focus = order.action, order.target, "any"
            book.army.seq += 1
        elif isinstance(order, Train):
            book.train.append(TrainRequest(order.role, order.count))

    r = Reflex(1, book, submit, lambda: current[0], SeatMetrics(1), decider)
    r.actions = []
    r.on_action = r.actions.append
    return r, submitted


async def step(r):
    r.tick()
    if r._inflight is not None:
        await r._inflight


async def test_a_raid_defended_then_resumed_is_two_logged_orders_with_overlay_lines():
    book = attacking()
    current = [snapshot(60_000, enemies_near_base=3, enemy=ENEMY)]
    decider = FakeDecider({"threat": "defend", "regroup": "resume"})
    r, submitted = run_reflex(decider, book, current)
    await step(r)
    assert submitted == [Army(action="defend", target="home")]
    current[0] = snapshot(70_000, enemy=ENEMY)
    await step(r)
    assert submitted[-1] == Army(action="attack", target="player_2")
    assert r.stats.orders == 2 and dict(r.stats.by_kind) == {"threat:defend": 1, "regroup:resume": 1}
    assert [a.line for a in r.actions] == [
        "pulled the army home to defend the base",
        "sent the army back to attack player_2",
    ]
    assert r.actions[0].time_ms == 60_000 and r.actions[0].kind == "threat:defend"
    assert r.actions[0].what == "army(action='defend', target='home')" and r.actions[0].status == "accepted"
    summary = r.stats.summary(600)
    assert summary["by_kind"] == {"threat:defend": 1, "regroup:resume": 1} and "log" not in summary
    assert summary["asked"]["threat"] == 1


async def test_a_focus_answer_changes_the_army_without_an_order_call_and_counts():
    book = attacking()
    current = [snapshot(enemy=ENEMY)]
    r, submitted = run_reflex(FakeDecider({"target": "workers"}), book, current)
    await step(r)
    assert submitted == [] and book.army.focus == "workers"
    assert r.stats.orders == 1 and r.actions[0].line == "sent the attack after their gatherers"
    assert r.actions[0].what == "focus('workers')"


async def test_a_question_that_just_changed_something_rests_before_it_is_asked_again():
    book = ordering("cavalry")
    current = [snapshot(60_000, enemies_near_base=2, idle_production=[150], enemy=ENEMY)]
    decider = FakeDecider({"counter": "cavalry"})
    r, submitted = run_reflex(decider, book, current)
    await step(r)
    assert Train(role="cavalry", count=3) in submitted and "counter" in decider.asked[0]
    current[0] = snapshot(60_000 + QUIET_MS["counter"] - 1000, enemies_near_base=2, idle_production=[150], enemy=ENEMY)
    await step(r)
    assert "counter" not in decider.asked[-1]
    current[0] = snapshot(60_000 + QUIET_MS["counter"] + 1000, enemies_near_base=2, idle_production=[150], enemy=ENEMY)
    await step(r)
    assert "counter" in decider.asked[-1]


async def test_an_answer_that_keeps_the_standing_order_is_not_counted():
    book = attacking()
    current = [snapshot(enemies_near_base=2)]
    r, submitted = run_reflex(FakeDecider({"threat": "press_on"}), book, current)
    await step(r)
    assert submitted == [] and r.stats.orders == 0 and r.stats.answered == 1


async def test_an_order_the_executor_rejects_is_not_counted_logged_or_rested():
    book = ordering("cavalry")
    current = [snapshot(60_000, enemies_near_base=2, idle_production=[150], enemy=ENEMY)]
    decider = FakeDecider({"counter": "cavalry", "threat": "ignore"})
    submitted = []

    def submit(order):
        submitted.append(order)
        return rejected("no_building", "nothing can train cavalry")

    actions = []
    r = Reflex(1, book, submit, lambda: current[0], SeatMetrics(1), decider, on_action=actions.append)
    await step(r)
    assert submitted == [Train(role="cavalry", count=3)]
    assert r.stats.orders == 0 and not actions and "counter" not in r._quiet_until


class GatedDecider(FakeDecider):
    def __init__(self, answers):
        super().__init__(answers)
        self.gate = asyncio.Event()

    async def decide(self, context, questions):
        await self.gate.wait()
        return await super().decide(context, questions)


async def asked_alone(snap, book, answers):
    """Tick once with a gated decider; return the questions asked and whether the seat counts as waiting."""
    decider = GatedDecider(answers)
    metrics = SeatMetrics(1)
    r = Reflex(1, book, lambda order: None, lambda: snap, metrics, decider)
    r.tick()
    await asyncio.sleep(0)
    waiting = metrics._waits
    decider.gate.set()
    await r._inflight
    return decider.asked[0], waiting


async def test_a_fight_in_flight_counts_as_waiting_and_aiming_an_attack_does_not():
    names, waiting = await asked_alone(snapshot(enemy=dict(ENEMY, contact=3)), attacking(), {})
    assert names == ["engage"] and waiting == 1
    names, waiting = await asked_alone(snapshot(enemy=ENEMY), attacking(), {})
    assert names == ["target"] and waiting == 0
    names, waiting = await asked_alone(
        snapshot(idle_production=[150], enemy=dict(ENEMY, contact=3)), ordering("cavalry"), {}
    )
    assert names == ["engage", "counter"] and waiting == 1  # engage waits; counter rides along
    book = attacking()
    book.army.action, book.army.target = "defend", "home"  # the reflex's own departure; the danger has passed
    decider = GatedDecider({})
    metrics = SeatMetrics(1)
    r = Reflex(1, book, lambda order: None, lambda: snapshot(), metrics, decider)
    r._strategist_army, r._departure = ("attack", "player_2"), book.army.seq
    r.tick()
    await asyncio.sleep(0)
    assert decider.asked == [] and metrics._waits == 0  # regroup is in flight (the gate holds it): no wait
    decider.gate.set()
    await r._inflight
    assert decider.asked == [["regroup"]]


# ---- the army driver honours the focus -----------------------------------------------------------------------------
def tick(executor, seat, snap=None):
    return [c for c in asyncio.run(executor.tick(seat, snap)) if c["type"] == "attack-walk"]


def test_an_attack_focused_on_workers_walks_at_their_gatherers_attacking_only_workers():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    enemy = snapshot(enemy=ENEMY).players[1].enemies[2]
    view = replace(snap.players[1], enemies={2: enemy})
    executor.submit(Army(action="attack", target="player_2"))
    executor.book.army.focus = "workers"
    [cmd] = tick(executor, seat, view)
    assert (cmd["x"], cmd["z"]) == (280.0, 180.0) and cmd["targetClasses"] == {"attack": ["Support", "Worker"]}


def test_an_attack_focused_on_their_army_walks_at_their_soldiers_and_a_new_order_resets_the_focus():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    view = replace(snap.players[1], enemies={2: snapshot(enemy=ENEMY).players[1].enemies[2]})
    executor.submit(Army(action="attack", target="player_2"))
    executor.book.army.focus = "army"
    [cmd] = tick(executor, seat, view)
    assert (cmd["x"], cmd["z"]) == (320.0, 210.0) and cmd["targetClasses"] == {"attack": ["Unit", "Structure"]}
    executor.submit(Army(action="raid", target="player_2"))
    assert executor.book.army.focus == "any"


@pytest.mark.parametrize("action", ["attack", "raid"])
def test_a_buildings_focus_walks_at_their_nearest_building_attacking_only_structures(action):
    seat, snap = load(300)
    executor, _, _ = make_executor()
    view = replace(snap.players[1], enemies={2: snapshot(enemy=ENEMY).players[1].enemies[2]})
    executor.submit(Army(action=action, target="player_2"))
    default = tick(executor, seat, view)
    executor, _, _ = make_executor()
    executor.submit(Army(action=action, target="player_2"))
    executor.book.army.focus = "buildings"
    [cmd] = tick(executor, seat, view)
    assert (cmd["x"], cmd["z"]) == (300.0, 200.0) and cmd["targetClasses"] == {"attack": ["Structure"]}
    assert all(c["targetClasses"] != cmd["targetClasses"] for c in default)  # it differs from no focus


@pytest.mark.parametrize("doing", ["move", "fight"])
@pytest.mark.parametrize("action", ["attack", "raid"])
def test_a_focus_change_mid_attack_re_aims_soldiers_already_walking_or_fighting(action, doing):
    seat, snap = load(300)
    executor, _, _ = make_executor()
    view = replace(snap.players[1], enemies={2: snapshot(enemy=ENEMY).players[1].enemies[2]})
    executor.submit(Army(action=action, target="player_2"))
    assert tick(executor, seat, view)  # the attack goes out with no focus
    executor.book.army.focus = "buildings"  # the reflex answers push / buildings
    soldiers = {u.id for u in seat.soldiers}
    units = tuple(replace(u, doing=doing) if u.id in soldiers else u for u in seat.units)
    later = replace(seat, units=units, time_ms=seat.time_ms + 5000)
    [cmd] = tick(executor, later, view)
    assert cmd["targetClasses"] == {"attack": ["Structure"]}
    assert tick(executor, replace(later, time_ms=later.time_ms + 1000), view) == []  # sent once, not every tick


def test_a_focus_with_nothing_of_that_kind_in_sight_keeps_the_default_target():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    houses = {"units": {}, "buildings": ENEMY["buildings"], "nearest": 150}
    view = replace(snap.players[1], enemies={2: snapshot(enemy=houses).players[1].enemies[2]})
    executor.submit(Army(action="attack", target="player_2"))
    executor.book.army.focus = "workers"
    [cmd] = tick(executor, seat, view)
    assert (cmd["x"], cmd["z"]) == (300.0, 200.0)


# ---- what the seat shows: footer, brief, results -------------------------------------------------------------------
async def test_reflex_actions_show_in_the_footer_the_brief_and_results_with_combat_and_order_texts(tmp_path):
    from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
    from agentenv_zeroad.reflex import ReflexAction
    from tests.zeroad_fakes import Handle, placer

    seats = (
        SeatPlan("1", 1, "agent", "rome", "alice", reflex="luna"),
        SeatPlan("2", 2, "petra", "pers", "Petra 2"),
    )
    match = LiveMatch(Handle(), MatchConfig(seats=seats, time_cap_s=600), out_dir=tmp_path, placer=placer())
    await match.setup()
    reflex = match.seats["1"].reflex
    line = "pulled the army home to defend the base"
    what = "army(action='defend', target='home')"
    reflex.on_action(ReflexAction(64_000, "threat:defend", line, what, "accepted"))
    reflex.stats.acted("threat:defend")
    assert f"1 reflex/autopilot actions, last 01:04 Luna: {line}" in match.footer("1")
    brief = match.brief("1")
    assert f"Your Luna reflexes and autopilot acted 1 times; recently: [01:04] Luna: {line}" in brief
    assert "Time left before the score decides:" in brief
    compact = __import__("json").loads(match.compact_brief("1"))
    assert compact["cap_ms"] == 600_000
    match.seats["1"].executor.last_seat = load(300)[0]  # a town-phase seat whose civic centre offers city
    compact = __import__("json").loads(match.compact_brief("1"))
    assert compact["next_phase"] == {"tech": "phase_city_generic", "ready": False, "cost": {"metal": 750, "stone": 750}}
    match.submit("1", Army(action="attack", target="player_2"))
    seat = match.results()["seats"]["1"]
    assert seat["reflex"]["by_kind"] == {"threat:defend": 1}
    assert set(seat["combat"]) == {"units_lost", "enemy_units_killed", "buildings_lost", "units_trained"}
    reflexes, ordered = seat["actions"]
    assert reflexes["by"] == "reflex" and reflexes["t"] == "01:04" and reflexes["line"] == line
    assert ordered["by"] == "brain" and ordered["what"] == "army(action='attack', target='player_2')"
    assert seat["action_counts"] == {"brain": 1, "reflex": 1, "autopilot": 0, "bot": 0}


def test_the_strategists_are_told_the_match_is_short_and_passivity_loses():
    from agentenv_zeroad.brains import STRATEGIST_SYSTEM

    assert "passivity loses" in STRATEGIST_SYSTEM and "army(attack, player_<n>)" in STRATEGIST_SYSTEM
    assert "don't flip it back" in STRATEGIST_SYSTEM


async def test_an_order_the_executor_takes_only_in_part_is_logged_with_its_partial_status():
    book = ordering("cavalry")
    current = [snapshot(60_000, enemies_near_base=2, idle_production=[150], enemy=ENEMY)]
    decider = FakeDecider({"counter": "cavalry", "threat": "ignore"})
    actions = []
    r = Reflex(
        1,
        book,
        lambda order: partial("queued 1 of 3"),
        lambda: current[0],
        SeatMetrics(1),
        decider,
        on_action=actions.append,
    )
    await step(r)
    assert [(a.kind, a.status) for a in actions] == [("counter:cavalry", "partial")]
