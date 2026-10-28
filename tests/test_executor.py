import asyncio
from dataclasses import replace

from agentenv_zeroad.autopilot import Autopilot
from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.engine import FakeEngine
from agentenv_zeroad.executor import VERIFY_MS, Executor, ExecutorLoop, split_targets
from agentenv_zeroad.orders import AdvancePhase, Army, Build, Research, Say, SetEconomy, Train
from agentenv_zeroad.placement import Placement
from agentenv_zeroad.state import Building, EnemyView
from agentenv_zeroad.view import ExecutorView, Unit, ViewQuery
from tests.executor_fixtures import FakePlacer, load, load_raw, make_executor, rome_table

RICH = {"food": 2000, "wood": 2000, "stone": 2000, "metal": 2000}


def tick(executor, seat, snap=None):
    return asyncio.run(executor.tick(seat, snap))


def idle(seat, ids):
    return replace(
        seat, units=tuple(replace(u, doing="idle", gathering=None) if u.id in ids else u for u in seat.units)
    )


def later(seat, ms, **changes):
    return replace(seat, time_ms=seat.time_ms + ms, **changes)


def of_type(commands, kind):
    return [c for c in commands if c["type"] == kind]


def test_split_targets_add_up_and_follow_the_percentages():
    assert split_targets({"food": 50, "wood": 50, "stone": 0, "metal": 0}, 7) in (
        {"food": 4, "wood": 3, "stone": 0, "metal": 0},
        {"food": 3, "wood": 4, "stone": 0, "metal": 0},
    )
    assert sum(split_targets({"food": 33, "wood": 33, "stone": 17, "metal": 17}, 31).values()) == 31


def jobs(commands):
    return [e for c in commands if c["type"] in ("gather", "repair", "construct") for e in c["entities"]]


def test_idle_villagers_go_to_work_once_and_not_again_while_the_order_lands():
    seat, snap = load(300)
    executor, queued, _ = make_executor()
    seat = idle(seat, {1962, 1964, 1966})
    first = tick(executor, seat, snap.players[1])
    assert {1962, 1964, 1966} <= set(jobs(first)) and len(jobs(first)) == len(set(jobs(first)))
    assert [c for _, c in queued] == first and all(p == 1 for p, _ in queued)
    sites = {s.id for r in seat.sites.values() for s in r} | {s.id for s in seat.finished("Field")}
    assert all(c["target"] in sites and c["queued"] is False for c in of_type(first, "gather"))
    # The next view still shows them idle (the step hasn't applied the command yet): nothing is re-sent.
    assert not {1962, 1964, 1966} & set(jobs(tick(executor, later(seat, 1000), snap.players[1])))
    # Still idle after the cooldown: they get a new job.
    again = tick(executor, later(seat, 4000), snap.players[1])
    assert {1962, 1964, 1966} <= set(jobs(again))


def test_idle_workers_follow_the_economy_split():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    executor.submit(SetEconomy(food=0, wood=0, stone=100, metal=0))
    commands = of_type(tick(executor, idle(seat, {1962, 1966}), snap.players[1]), "gather")
    stone_sites = {s.id for s in seat.sites["stone"]}
    assert commands and all(c["target"] in stone_sites for c in commands)


def test_house_goes_up_before_population_caps_with_builders_and_the_engine_checked_spot():
    seat, snap = load(300)
    seat = replace(seat, structures=tuple(s for s in seat.structures if s.id != 2007), resources=RICH)
    executor, _, placer = make_executor()
    commands = tick(executor, seat, snap.players[1])
    construct = of_type(commands, "construct")
    assert construct and construct[0]["template"] == "structures/rome/house"
    assert placer.requests[0][1] == "structures/rome/house" and placer.requests[0][2].anchors[0] == seat.base
    assert 1 <= len(construct[0]["entities"]) <= 3 and construct[0]["autorepair"] is True
    builders = set(construct[0]["entities"])
    assert not any(builders & set(c.get("entities", ())) for c in of_type(commands, "gather"))


def test_construct_that_never_shows_is_logged_as_rejected_and_retried():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.submit(Build(structure="barracks", where="base"))
    rich = replace(seat, resources=RICH)
    tick(executor, rich)
    assert len(executor.pending) >= 1
    retry = of_type(tick(executor, later(rich, VERIFY_MS + 200)), "construct")
    assert any(r.what == "construct barracks at base" for r in executor.rejections)
    req = next(r for r in executor.book.build if r.structure == "barracks")
    assert req.attempts == 1
    # The same tick tries again; this time a foundation shows at the chosen spot and the placement is confirmed.
    spot = next(c for c in retry if c["template"].endswith("barracks"))
    foundation = replace(
        seat.structures[0],
        id=9999,
        template=spot["template"],
        kind="Barracks",
        foundation=True,
        pos=(spot["x"], spot["z"]),
        queue=None,
    )
    tick(executor, later(rich, VERIFY_MS + 2400, structures=seat.structures + (foundation,)))
    assert ("barracks", "base", rich.time_ms + VERIFY_MS + 2400) in executor.placed
    assert not any(r.structure == "barracks" for r in executor.book.build)


def test_placement_that_finds_no_spot_fails_the_order_after_three_tries():
    seat, _ = load(300)
    executor, _, _ = make_executor(ok=False)
    executor.submit(Build(structure="barracks", where="front"))
    rich = replace(seat, resources=RICH)
    for i in range(3):
        tick(executor, later(rich, 1000 * i))
    assert not any(r.structure == "barracks" for r in executor.book.build)


def test_train_order_queues_a_batch_at_a_building_that_trains_the_role_and_counts_it_down():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    assert executor.submit(Train(role="infantry_melee", count=3, at="barracks")).status == "accepted"
    commands = of_type(tick(executor, replace(seat, resources=RICH)), "train")
    melee = [c for c in commands if c["entities"] == [1988]]
    assert melee and melee[0]["count"] == 3 and melee[0]["template"] in executor.roles.templates_for("infantry_melee")
    assert executor.book.pending_count("infantry_melee") == 0
    assert of_type(tick(executor, later(replace(seat, resources=RICH), 1000)), "train") == []


def test_villagers_train_up_to_the_target_only():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    cc = replace(seat.structure(150), queue=0)
    free = replace(seat, resources=RICH, pop_cap=100, structures=(cc,) + seat.structures[1:])
    trains = of_type(tick(executor, free), "train")
    assert [c["entities"] for c in trains] == [[150]] and "support" in trains[0]["template"]
    executor2, _, _ = make_executor()
    executor2.book.villager_target = len(seat.villagers)
    assert of_type(tick(executor2, free), "train") == []


def test_advance_phase_reserves_its_cost_so_villagers_wait():
    seat, _ = load(60)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    assert executor.submit(AdvancePhase()).status == "partial"  # five village structures not there yet
    cc = seat.structure(150)
    ready = replace(cc, queue=0, techs=tuple((n, True, c) for n, _, c in cc.techs))
    poor = replace(
        seat,
        resources={"food": 400, "wood": 400, "stone": 0, "metal": 0},
        pop_cap=100,
        structures=(ready,) + seat.structures[1:],
    )
    commands = tick(executor, poor)
    assert of_type(commands, "research") == [] and of_type(commands, "train") == []
    rich = later(poor, 1000, resources=RICH)
    research = of_type(tick(executor, rich), "research")
    assert research == [{"type": "research", "entity": 150, "template": "phase_town_generic"}]


def test_requirements_unmet_phase_advance_adds_a_house():
    seat, _ = load(60)
    executor, _, _ = make_executor()
    executor.submit(AdvancePhase())
    tick(executor, seat)
    assert any(r.structure == "house" for r in executor.book.build)


def test_research_order_starts_once_and_is_verified():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    assert executor.submit(Research(tech="gather_lumbering_ironaxes")).ok
    commands = of_type(tick(executor, replace(seat, resources=RICH)), "research")
    assert commands == [{"type": "research", "entity": 1930, "template": "gather_lumbering_ironaxes"}]
    assert of_type(tick(executor, later(replace(seat, resources=RICH), 1000)), "research") == []


def test_attack_sends_every_soldier_toward_the_guessed_enemy_base_then_waits():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    assert executor.submit(Army(action="attack", target="player_2")).status == "accepted"
    assert executor.book.army.seq == 1  # every army order is counted, so the reflex can tell its own apart
    commands = of_type(tick(executor, seat, snap.players[1]), "attack-walk")
    sent = {e for c in commands for e in c["entities"]}
    assert sent == {u.id for u in seat.soldiers}
    x, z = commands[0]["x"], commands[0]["z"]
    assert abs(x - 268) < 3 and abs(z - 140) < 3  # opposite the own base across the map centre
    moving = replace(
        seat, time_ms=seat.time_ms + 1000, units=tuple(replace(u, doing="move") if u.soldier else u for u in seat.units)
    )
    assert of_type(tick(executor, moving, snap.players[1]), "attack-walk") == []


def test_attack_targets_a_seen_enemy_structure_over_the_guess():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    me = replace(snap.players[1], enemies={2: EnemyView(2, {}, (Building(777, "House", (300, 200), 100),), 400)})
    executor.submit(Army(action="attack", target="2"))
    commands = of_type(tick(executor, seat, me), "attack-walk")
    assert (commands[0]["x"], commands[0]["z"]) == (300.0, 200.0)


def test_an_attack_on_a_defeated_player_is_refused_naming_the_players_still_standing():
    seat, _ = load(300)
    executor, _, _ = make_executor(players=(1, 2, 3))
    executor.last_seat = seat
    executor.fallen = frozenset({2})
    army = executor.submit(Army(action="attack", target="player_2"))
    assert army.code == "defeated" and "player_3" in army.alternatives and "player_2" not in army.alternatives


def test_an_attack_on_a_player_who_falls_turns_to_the_nearest_one_still_standing():
    seat, snap = load(300)
    executor, _, _ = make_executor(players=(1, 2, 3))
    executor.submit(Army(action="attack", target="player_2"))
    executor.fallen = frozenset({2})
    commands = of_type(tick(executor, seat, snap.players[1]), "attack-walk")
    assert executor.book.army.target == "player_3" and list(executor.enemy_bases(seat)) == [3]
    assert (commands[0]["x"], commands[0]["z"]) == tuple(round(v, 1) for v in executor.enemy_bases(seat)[3])


def test_retreat_walks_soldiers_home_and_scout_sends_one_cavalryman():
    seat, _ = load(300)
    away = replace(seat, units=tuple(replace(u, pos=(300.0, 300.0)) if u.soldier else u for u in seat.units))
    executor, _, _ = make_executor()
    executor.submit(Army(action="retreat"))
    walks = of_type(tick(executor, away), "walk")
    assert walks and (walks[0]["x"], walks[0]["z"]) == seat.base
    scout, _, _ = make_executor()
    scout.submit(Army(action="scout", target="player_2"))
    walks = of_type(tick(scout, seat), "walk")
    assert [w["entities"] for w in walks] == [[159]]


def test_soldiers_get_the_stance_for_the_posture_once():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.submit(Army(action="attack", target="player_2"))
    stances = of_type(tick(executor, seat), "stance")
    assert stances and stances[0]["name"] == "aggressive"
    assert set(stances[0]["entities"]) == {u.id for u in seat.soldiers}
    assert of_type(tick(executor, later(seat, 1000)), "stance") == []


def gathering_at(seat, resource, pos):
    """Every unit gathering `resource` stands at `pos` (a few metres apart)."""
    workers = [u.id for u in seat.units if u.gathering == resource and u.doing == "gather"]
    moved = {uid: (pos[0] + i, pos[1]) for i, uid in enumerate(workers)}
    return replace(seat, units=tuple(replace(u, pos=moved[u.id]) if u.id in moved else u for u in seat.units))


def test_wood_line_served_only_by_a_farmstead_gets_a_storehouse():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    # Wood workers next to storehouse 1930 are served: no new storehouse.
    tick(executor, replace(gathering_at(seat, "wood", (481, 690)), resources=RICH))
    assert not any(r.structure == "storehouse" and r.where == "near_wood" for r in executor.book.build)
    # Next to farmstead 1961 (food only), 78 m from the nearest wood drop-site: they need a storehouse.
    executor, _, _ = make_executor()
    tick(executor, replace(gathering_at(seat, "wood", (423, 616)), resources=RICH))
    assert any(r.structure == "storehouse" and r.where == "near_wood" for r in executor.book.build)


def test_submit_rejects_with_reasons_and_alternatives():
    seat, _ = load(60)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    unknown = executor.submit(Build(structure="moon_base"))
    assert unknown.status == "rejected" and "house" in unknown.alternatives
    temple = executor.submit(Build(structure="temple"))
    assert temple.code == "phase_required" and temple.suggest == "advance_phase()"
    toward = executor.submit(Build(structure="house", where="toward_5"))
    assert toward.code == "bad_target" and "toward_player_2" in toward.alternatives
    bad_at = executor.submit(Train(role="cavalry", count=2, at="house"))
    assert bad_at.code == "bad_building" and "stable" in bad_at.alternatives
    tech = executor.submit(Research(tech="laser_beams"))
    assert tech.code == "unknown_tech" and "gather_lumbering_ironaxes" in tech.alternatives
    army = executor.submit(Army(action="attack", target="player_4"))
    assert army.code == "bad_target" and "player_2" in army.alternatives
    assert executor.submit(Say(text="For Rome!")).ok and executor.book.said[-1][1] == "For Rome!"
    city = replace(seat, phase="city")
    executor.last_seat = city
    assert executor.submit(AdvancePhase()).code == "already_max"


def test_train_without_a_building_for_the_role_is_partial_and_suggests_one():
    seat, _ = load(60)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    result = executor.submit(Train(role="champion", count=2))
    assert result.status == "partial" and result.suggest.startswith("build(")


def test_executor_loop_queues_commands_on_the_clock_without_stepping_it():
    raw = load_raw(300)["view"]
    raw = {**raw, "players": {"1": raw["players"]["1"]}}
    seat, _ = load(300)
    executor, _, _ = make_executor()
    engine = FakeEngine()
    query = ViewQuery(engine, [1])
    engine._evaluations[query.code] = raw
    clock = GameClock(engine)
    executor.queue = clock.queue
    executor.submit(Army(action="attack", target="player_2"))
    loop = ExecutorLoop(clock, {1: executor}, view=query)
    asyncio.run(loop.tick_once())
    assert engine.steps == [] and clock.pending(1) > 0
    asyncio.run(clock.step_once())
    sent = engine.steps[0]
    assert sent and all(p == 1 for p, _ in sent) and any(c["type"] == "attack-walk" for _, c in sent)


def test_a_build_that_finds_no_spot_is_reported_failed_with_its_reason():
    seat, _ = load(300)
    executor, _, _ = make_executor(ok=False)
    executor.submit(Build(structure="barracks", where="front"))
    for i in range(3):
        tick(executor, later(replace(seat, resources=RICH), 1000 * i))
    failed = [f for f in executor.failures if f.key == "barracks"]
    assert len(failed) == 1 and failed[0].what == "build barracks at front" and "no valid spot" in failed[0].reason


class RefusesHouses(FakePlacer):
    async def find(self, player, template, search):
        if template.endswith("/house"):
            self.requests.append((player, template, search))
            return Placement(False, tried=5, reasons={"it cannot be built here": 5})
        return await super().find(player, template, search)


def test_a_house_with_no_spot_does_not_hold_back_other_buildings_and_backs_off():
    seat, _ = load(300)  # pop 35 of 40: with no house foundation on the way, the autopilot wants a house
    seat = replace(seat, structures=tuple(s for s in seat.structures if not (s.foundation and s.kind == "House")))
    executor, _, _ = make_executor()
    executor.placer = RefusesHouses()
    executor.submit(Build(structure="barracks", where="base"))
    rich = replace(seat, resources=RICH)
    first = of_type(tick(executor, rich), "construct")
    assert [c["template"] for c in first] == ["structures/rome/barracks"]
    for i in range(1, 6):
        tick(executor, later(rich, 1000 * i))
    houses = [f for f in executor.failures if f.key == "house"]
    assert len(houses) == 1

    def house_searches():
        return sum(1 for _, tpl, _ in executor.placer.requests if tpl.endswith("/house"))

    tries = house_searches()
    assert tries == 3
    tick(executor, later(rich, 10_000))
    assert house_searches() == tries  # no new house request within RETRY_MS


def test_villagers_queued_in_the_civic_centre_count_toward_the_target():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.book.villager_target = len(seat.villagers) + 2
    cc = replace(seat.structure(150), queue=0)
    free = replace(seat, resources=RICH, pop_cap=100, structures=(cc,) + seat.structures[1:])
    first = [c for c in of_type(tick(executor, free), "train") if "support" in c["template"]]
    assert sum(c["count"] for c in first) == 2
    # The engine took the batch: the queue shows it, no villager is born yet.
    queued = replace(cc, queue=1, queued=((first[0]["template"], 2),))
    again = tick(executor, later(free, VERIFY_MS + 500, structures=(queued,) + seat.structures[1:]))
    assert [c for c in of_type(again, "train") if "support" in c["template"]] == []
    assert executor.rejections == []


def train_melee(executor, seat):
    executor.last_seat = seat
    executor.submit(Train(role="infantry_melee", count=3, at="barracks"))
    return [c for c in of_type(tick(executor, seat), "train") if c["entities"] == [1988]]


def test_a_train_the_engine_refused_is_logged_put_back_and_failed_after_three_tries():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    rich = replace(seat, resources=RICH, pop_cap=100)
    assert train_melee(executor, rich)
    for i in range(1, 4):
        tick(executor, later(rich, i * (VERIFY_MS + 100)))
        refused = [r for r in executor.rejections if r.what.startswith("train 3 units/rome/infantry")]
        assert len(refused) == i
    assert [f.key for f in executor.failures] == ["infantry_melee"]
    assert executor.book.pending_count("infantry_melee") == 0 and not executor.book.train


def test_a_train_whose_queue_grew_is_not_a_rejection():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    rich = replace(seat, resources=RICH, pop_cap=100)
    [command] = train_melee(executor, rich)
    batch = ((command["template"], command["count"]),)
    grown = tuple(replace(s, queue=1, queued=batch) if s.id == 1988 else s for s in rich.structures)
    tick(executor, later(rich, VERIFY_MS + 100, structures=grown))
    assert executor.rejections == [] and executor.book.pending_count("infantry_melee") == 0


def test_another_batch_finishing_while_ours_is_queued_is_not_a_rejection():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    busy = tuple(
        replace(s, queue=1, queued=(("units/rome/infantry_javelineer_b", 5),)) if s.id == 1988 else s
        for s in seat.structures
    )
    rich = replace(seat, resources=RICH, pop_cap=100, structures=busy)
    [command] = train_melee(executor, rich)
    # The javelineers walk out as the swordsmen go in: the queue length stays 1.
    swapped = tuple(replace(s, queued=((command["template"], 3),)) if s.id == 1988 else s for s in busy)
    tick(executor, later(rich, VERIFY_MS + 100, structures=swapped))
    assert executor.rejections == []


def test_a_new_unit_of_another_role_does_not_hide_a_refused_train():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    rich = replace(seat, resources=RICH, pop_cap=100)
    assert train_melee(executor, rich)
    newborn = replace(seat.villagers[0], id=99_999)
    tick(executor, later(rich, VERIFY_MS + 100, units=rich.units + (newborn,)))
    assert any(r.what.startswith("train 3 units/rome/infantry") for r in executor.rejections)


def test_research_the_engine_refused_is_logged_and_dropped_after_three_tries():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    tech = "gather_lumbering_ironaxes"
    executor.submit(Research(tech=tech))
    rich = replace(seat, resources=RICH)
    tick(executor, rich)
    started = tick(executor, later(rich, VERIFY_MS + 100, researching=(tech,)))
    assert executor.rejections == [] and of_type(started, "research") == []
    executor2, _, _ = make_executor()
    executor2.last_seat = seat
    executor2.submit(Research(tech=tech))
    tick(executor2, rich)
    for i in range(1, 4):
        tick(executor2, later(rich, i * (VERIFY_MS + 100)))
    assert [r.what for r in executor2.rejections] == [f"research {tech}"] * 3
    assert tech not in executor2.book.research and [f.key for f in executor2.failures] == [tech]


def test_an_unaffordable_building_holds_its_cost_back_from_lower_priorities():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    barracks = executor.roles.structure("barracks")
    executor.submit(Build(structure="barracks", where="base"))
    executor.submit(Train(role="infantry_ranged", count=2, at="barracks"))
    short = {**RICH, **{r: v for r, v in barracks.cost.items()}, "wood": barracks.cost["wood"] - 1}
    commands = tick(executor, replace(seat, resources=short, pop_cap=100))
    assert [c for c in of_type(commands, "train") if c["entities"] == [1988]] == []
    assert of_type(commands, "construct") == []


class Breaks(Autopilot):
    def gatherers(self, ex, t):
        raise RuntimeError("bug in a late step")


def test_one_seats_failed_tick_sends_nothing_logs_no_rejection_and_spares_the_other_seat():
    raw = load_raw(300)["view"]
    view = ExecutorView.from_json(raw)
    rich = {p: replace(s, resources=RICH) for p, s in view.seats.items()}
    broken, q1, _ = make_executor()
    broken.autopilot = Breaks()
    broken.last_seat = rich[1]
    broken.submit(Research(tech="gather_lumbering_ironaxes"))
    broken.submit(Build(structure="barracks", where="base"))
    healthy = Executor(2, rome_table(), queue=lambda p, c: q2.append(c), placer=FakePlacer(), players=(1, 2))
    q2: list = []
    ticks = {"n": 0}

    async def views():
        ticks["n"] += 1
        return replace(view, seats={p: later(s, 1000 * ticks["n"]) for p, s in rich.items()})

    loop = ExecutorLoop(None, {1: broken, 2: healthy}, view=views)
    for _ in range(6):
        asyncio.run(loop.tick_once())
    assert loop.errors == 6 and loop.ticks == 6
    assert q1 == [] and broken.pending == [] and broken.rejections == []
    assert all(r.placed_id is None for r in broken.book.build)
    assert q2


def test_executor_loop_run_counts_a_failed_view_and_carries_on():
    seat, _ = load(300)
    executor, _, _ = make_executor()
    calls = {"n": 0}

    async def views():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("engine busy")
        if calls["n"] >= 3:
            loop.stop()
        return ExecutorView(seat.time_ms, seat.map_size, {1: seat})

    loop = ExecutorLoop(None, {1: executor}, view=views, period=0)
    asyncio.run(loop.run())
    assert loop.errors == 1 and loop.ticks == 2


class RefusesKind(FakePlacer):
    """Refuses every spot for templates containing `kind` (unexplored ground); places the rest."""

    def __init__(self, kind):
        super().__init__(True)
        self.kind = kind

    async def find(self, player, template, search):
        if self.kind not in template:
            return await super().find(player, template, search)
        self.requests.append((player, template, search))
        return Placement(False, tried=10, reasons={"it cannot be built in unexplored area": 10})


def ten_minutes_with_a_refused(kind, seat):
    executor, _, _ = make_executor()
    executor.placer = RefusesKind(kind)
    for i in range(600):
        tick(executor, later(seat, 1000 * i))
    searches = sum(1 for _, tpl, _ in executor.placer.requests if kind in tpl)
    return searches, [f for f in executor.failures if kind in f.key]


def test_a_field_that_cannot_be_placed_backs_off_instead_of_searching_every_tick():
    seat, _ = load(300)
    seat = replace(seat, resources=RICH, structures=tuple(s for s in seat.structures if s.id != 2013))
    searches, failures = ten_minutes_with_a_refused("field", seat)
    # Back-off 30 s, 60 s, 120 s, 240 s: five failures (three searches each) in ten minutes, not one every 3 s.
    assert len(failures) <= 5 and searches <= 15, (len(failures), searches)
    assert failures[0].where == "near_food"


def test_a_storehouse_that_cannot_be_placed_backs_off_instead_of_searching_every_tick():
    seat, _ = load(300)
    far = tuple(replace(u, pos=(5.0, 5.0)) if u.gathering == "wood" and u.doing == "gather" else u for u in seat.units)
    searches, failures = ten_minutes_with_a_refused("storehouse", replace(seat, resources=RICH, units=far))
    assert len(failures) <= 5 and searches <= 15, (len(failures), searches)
    assert {f.where for f in failures} == {"near_wood"}


def test_a_refused_batch_is_not_taken_for_an_earlier_batch_of_that_role_finishing():
    seat, _ = load(300)
    spear = "units/rome/infantry_spearman_b"
    earlier = tuple(replace(s, queue=1, queued=((spear, 5),)) if s.id == 1988 else s for s in seat.structures)
    rich = replace(seat, resources=RICH, pop_cap=100, structures=earlier)
    executor, _, _ = make_executor()
    assert train_melee(executor, rich)
    # 3 s on: the earlier 5 spearmen are out and the barracks queue is empty, so ours was refused.
    born = tuple(Unit(90_000 + i, "infantry_melee", (500.0, 500.0), "idle", None, "aggressive") for i in range(5))
    done = tuple(replace(s, queue=0, queued=()) if s.id == 1988 else s for s in earlier)
    retried = of_type(tick(executor, later(rich, VERIFY_MS + 100, units=rich.units + born, structures=done)), "train")
    assert any(r.what.startswith("train 3 units/rome/infantry") for r in executor.rejections)
    assert [c["count"] for c in retried if c["entities"] == [1988]] == [3]  # put back and sent again


def test_units_waiting_in_queues_count_against_population_room():
    seat, _ = load(300)
    spear = "units/rome/infantry_spearman_b"
    waiting = ((spear, 5), (spear, 5), (spear, 5))  # one batch in production (already in pop), ten units waiting
    structures = tuple(
        replace(s, queue=3, queued=waiting) if s.id == 1988 else replace(s, queue=0, queued=()) for s in seat.structures
    )
    full = replace(seat, resources=RICH, pop_cap=seat.pop + 10, structures=structures)
    executor, _, _ = make_executor()
    executor.last_seat = full
    executor.submit(Train(role="infantry_melee", count=5))
    assert of_type(tick(executor, full), "train") == []
    roomy = replace(full, pop_cap=seat.pop + 13)
    [command] = [c for c in of_type(tick(executor, later(roomy, VERIFY_MS + 100)), "train") if c["count"] != 2]
    assert command["count"] == 3
