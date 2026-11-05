"""Army waves (interface v2): gather at the rally point, launch together, regroup, and a standing order that survives
being repeated. Soldiers are placed on the recorded Rome view; v1 and the other postures are checked against v1."""

import asyncio
from dataclasses import replace

from agentenv_zeroad.army import GATHER_M, WAVE_WAIT_MS
from agentenv_zeroad.orders import Army
from agentenv_zeroad.view import Unit
from tests.executor_fixtures import load, make_executor

SOLDIER_IDS = range(9000, 9040)


def tick(executor, seat):
    return asyncio.run(executor.tick(seat, None))


def kind(commands, name):
    return [c for c in commands if c["type"] == name]


def soldiers(seat, spec, doing="idle"):
    """The seat with its own soldiers replaced by `spec`: (id, position) pairs."""
    keep = tuple(u for u in seat.units if not u.soldier)
    new = tuple(Unit(uid, "infantry_melee", pos, doing, None, "aggressive") for uid, pos in spec)
    return replace(seat, units=keep + new)


def at(seat, ms):
    return replace(seat, time_ms=seat.time_ms + ms)


def around(point, ids):
    return [(uid, (point[0] + i % 5, point[1] + i // 5)) for i, uid in enumerate(ids)]


def setup(n_home=10, *, interface="v2", **order):
    seat, _ = load(300)
    executor, _, _ = make_executor(interface=interface)
    home = seat.base
    seat = soldiers(seat, around(home, list(SOLDIER_IDS)[:n_home]))
    executor.last_seat = seat
    result = executor.submit(Army(action="attack", target="player_2", **order))
    assert result.status == "accepted", result
    rally = executor.army.point(executor, seat, "front")
    return executor, seat, rally


def destination(commands):
    c = commands[0]
    return (c["x"], c["z"])


def near(a, b, tol=1.0):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def test_soldiers_gather_at_the_rally_point_and_the_wave_launches_when_min_size_stand_there():
    executor, seat, rally = setup(10)
    first = kind(tick(executor, seat), "attack-walk")
    assert {e for c in first for e in c["entities"]} == set(list(SOLDIER_IDS)[:10])
    assert near(destination(first), rally) and not near(rally, executor.enemy_bases(seat)[2], 50)
    gathered = at(soldiers(seat, around(rally, list(SOLDIER_IDS)[:7])), 20_000)
    assert kind(tick(executor, gathered), "attack-walk") == []  # 7 of 8: it waits, and nobody is re-sent
    full = at(soldiers(seat, around(rally, list(SOLDIER_IDS)[:8])), 25_000)
    launch = kind(tick(executor, full), "attack-walk")
    assert len(launch) == 1 and set(launch[0]["entities"]) == set(list(SOLDIER_IDS)[:8])
    assert near(destination(launch), executor.enemy_bases(seat)[2], 3)


def test_min_size_comes_from_the_order():
    executor, seat, rally = setup(12, min_size=12)
    ids = list(SOLDIER_IDS)[:12]
    assert executor.book.army.min_size == 12
    spread = soldiers(seat, around(rally, ids[:11]) + [(ids[11], seat.base)])
    sent = kind(tick(executor, spread), "attack-walk")
    assert sent and all(near(destination([c]), rally) for c in sent)  # eleven of twelve: the wave waits
    report = executor.army.report(executor, spread)
    assert (report.forming, report.gathered, report.min_size, report.waves) == (12, 11, 12, ())
    full = at(soldiers(seat, around(rally, ids)), 1_000)
    assert near(destination(kind(tick(executor, full), "attack-walk")), executor.enemy_bases(seat)[2], 3)


def test_a_group_below_min_size_goes_with_what_it_has_after_90_game_seconds():
    executor, seat, rally = setup(3)
    three = soldiers(seat, around(rally, list(SOLDIER_IDS)[:3]))
    tick(executor, three)
    assert kind(tick(executor, at(three, WAVE_WAIT_MS - 1_000)), "attack-walk") == []
    launch = kind(tick(executor, at(three, WAVE_WAIT_MS)), "attack-walk")
    assert set(launch[0]["entities"]) == set(list(SOLDIER_IDS)[:3])
    assert near(destination(launch), executor.enemy_bases(seat)[2], 3)


def test_a_wave_that_loses_most_of_its_soldiers_falls_back_to_the_rally_point_and_regroups():
    executor, seat, rally = setup(10)
    ids = list(SOLDIER_IDS)[:10]
    assert kind(tick(executor, soldiers(seat, around(rally, ids))), "attack-walk")  # ten at the rally: it goes
    enemy = executor.enemy_bases(seat)[2]
    out = at(soldiers(seat, around(enemy, ids[:5]), doing="move"), 30_000)
    assert kind(tick(executor, out), "attack-walk") == []  # 5 of 10 is above the 0.4 line: it presses on
    spent = at(soldiers(seat, around(enemy, ids[:3]), doing="fight"), 40_000)
    back = kind(tick(executor, spent), "attack-walk")
    assert set(back[0]["entities"]) == set(ids[:3]) and near(destination(back), rally)
    report = executor.army.report(executor, spent)
    assert report.waves == () and report.forming == 3


def test_a_regrouped_survivor_group_launches_again_once_it_is_gathered_or_has_waited():
    executor, seat, rally = setup(10)
    ids = list(SOLDIER_IDS)[:10]
    tick(executor, soldiers(seat, around(rally, ids)))
    enemy = executor.enemy_bases(seat)[2]
    tick(executor, at(soldiers(seat, around(enemy, ids[:3]), doing="fight"), 40_000))
    home = at(soldiers(seat, around(rally, ids[:3])), 45_000)
    assert kind(tick(executor, home), "attack-walk") == []
    again = kind(tick(executor, at(home, WAVE_WAIT_MS + 1_000)), "attack-walk")
    assert near(destination(again), enemy, 3) and set(again[0]["entities"]) == set(ids[:3])


def test_repeating_an_identical_army_order_keeps_the_waves_state_and_timer_but_a_changed_one_resets_them():
    executor, seat, rally = setup(3)
    three = soldiers(seat, around(rally, list(SOLDIER_IDS)[:3]))
    tick(executor, three)
    seq, since = executor.book.army.seq, executor.book.army.since_ms
    again = executor.submit(Army(action="attack", target="player_2"))
    same = executor.submit(Army(action="attack", target="player_2", min_size=8, regroup_below=0.4))
    assert again.status == same.status == "accepted" and "already" in again.message
    assert (executor.book.army.seq, executor.book.army.since_ms) == (seq, since)
    tick(executor, at(three, 60_000))
    executor.submit(Army(action="attack", target="player_2"))  # a turn later, the same order again
    assert kind(tick(executor, at(three, WAVE_WAIT_MS)), "attack-walk")  # the timer of the first order ran out

    changed, cseat, crally = setup(3)
    cthree = soldiers(cseat, around(crally, list(SOLDIER_IDS)[:3]))
    tick(changed, cthree)
    tick(changed, at(cthree, 60_000))
    assert changed.submit(Army(action="attack", target="player_2", min_size=10)).message.startswith("Army set")
    assert kind(tick(changed, at(cthree, WAVE_WAIT_MS)), "attack-walk") == []  # a new order, a new 90 s


def test_soldiers_trained_after_a_launch_gather_for_the_next_wave_instead_of_walking_in_alone():
    executor, seat, rally = setup(8)
    first = list(SOLDIER_IDS)[:8]
    assert kind(tick(executor, soldiers(seat, around(rally, first))), "attack-walk")
    recruits = list(SOLDIER_IDS)[8:11]
    mixed = soldiers(
        seat,
        around(
            executor.enemy_bases(seat)[2],
            first,
        )
        + around(seat.base, recruits),
    )
    sent = kind(tick(executor, at(mixed, 10_000)), "attack-walk")
    reinforcements = [c for c in sent if set(c["entities"]) & set(recruits)]
    assert reinforcements and all(near(destination([c]), rally) for c in reinforcements)
    assert not any(
        set(c["entities"]) & set(recruits) and near(destination([c]), executor.enemy_bases(seat)[2], 5) for c in sent
    )
    ready = list(SOLDIER_IDS)[8:16]
    second = soldiers(seat, around(executor.enemy_bases(seat)[2], first) + around(rally, ready))
    launch = kind(tick(executor, at(second, 20_000)), "attack-walk")
    assert any(
        set(c["entities"]) == set(ready) and near(destination([c]), executor.enemy_bases(seat)[2], 3) for c in launch
    )


def test_a_raid_gathers_and_launches_its_cavalry_the_same_way():
    seat, _ = load(300)
    executor, _, _ = make_executor(interface="v2")
    cav = tuple(Unit(8000 + i, "cavalry", seat.base, "idle", None, "") for i in range(8))
    seat = replace(seat, units=tuple(u for u in seat.units if not u.soldier) + cav)
    executor.last_seat = seat
    executor.submit(Army(action="raid", target="player_2"))
    rally = executor.army.point(executor, seat, "front")
    first = kind(tick(executor, seat), "attack-walk")
    assert near(destination(first), rally) and len(first[0]["entities"]) == 8
    gathered = at(
        replace(seat, units=tuple(replace(u, pos=rally) if u.role == "cavalry" else u for u in seat.units)), 2_000
    )
    launch = kind(tick(executor, gathered), "attack-walk")
    assert near(destination(launch), executor.enemy_bases(seat)[2], 3)


def test_other_postures_and_v1_behave_as_before():
    def commands(interface, action):
        seat, _ = load(300)
        executor, _, _ = make_executor(interface=interface)
        away = replace(seat, units=tuple(replace(u, pos=(300.0, 300.0)) if u.soldier else u for u in seat.units))
        executor.last_seat = away
        executor.submit(Army(action=action, target="home" if action != "scout" else "player_2"))
        return [c for c in tick(executor, away) if c["type"] in ("walk", "attack-walk")]

    for action in ("defend", "retreat", "scout"):
        assert commands("v2", action) == commands("v1", action) and commands("v1", action), action

    seat, snap = load(300)
    v1, _, _ = make_executor(interface="v1")
    v1.last_seat = seat
    v1.submit(Army(action="attack", target="player_2"))
    sent = kind(asyncio.run(v1.tick(seat, snap.players[1])), "attack-walk")
    assert {e for c in sent for e in c["entities"]} == {u.id for u in seat.soldiers}
    assert near(destination(sent), (268, 140), 3)  # v1 still sends every soldier straight at the enemy


def test_the_army_report_says_where_the_army_is_and_how_the_wave_is_filling():
    executor, seat, rally = setup(10)
    assert executor.army.report(executor, seat).where == "home"
    tick(executor, seat)
    gathered = at(soldiers(seat, around(rally, list(SOLDIER_IDS)[:5])), 30_000)
    tick(executor, gathered)
    report = executor.army.report(executor, gathered)
    assert report.where == "rally" and report.soldiers == {"infantry_melee": 5}
    assert (report.forming, report.gathered, report.min_size) == (5, 5, 8) and report.goes_in_s == 60
    fighting = soldiers(seat, around(rally, list(SOLDIER_IDS)[:5]), doing="fight")
    assert executor.army.report(executor, fighting).where == "engaged"
    assert GATHER_M > 0


def test_with_no_soldiers_at_the_order_the_wait_starts_with_the_first_one_not_the_order():
    seat, _ = load(300)
    executor, _, _ = make_executor(interface="v2")
    empty = soldiers(seat, [])
    executor.last_seat = empty
    executor.submit(Army(action="attack", target="player_2"))
    tick(executor, empty)
    rally = executor.army.point(executor, empty, "front")
    lone = at(soldiers(seat, around(rally, [9000])), 200_000)
    assert kind(tick(executor, lone), "attack-walk") == []  # 200 s after the order, but one soldier is no wave
    assert kind(tick(executor, at(lone, WAVE_WAIT_MS - 1_000)), "attack-walk") == []
    assert kind(tick(executor, at(lone, WAVE_WAIT_MS)), "attack-walk")
