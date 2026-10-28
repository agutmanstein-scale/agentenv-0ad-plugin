from dataclasses import replace

from agentenv_zeroad.bots import ADVANCE_AT_VILLAGERS, ATTACK_AT_ARMY, ScriptedBot
from tests.executor_fixtures import load, make_executor


def kinds(bot):
    return [o.kind for _, o, _ in bot.log]


def test_bot_booms_once_through_the_order_interface():
    seat, snap = load(60)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    bot = ScriptedBot(executor)
    bot.tick(snap.players[1], snap.time_ms)
    assert kinds(bot)[:2] == ["set_economy", "build"] and all(r.status == "accepted" for _, _, r in bot.log)
    count = len(bot.log)
    bot.tick(snap.players[1], snap.time_ms + 1000)
    assert len(bot.log) == count  # nothing repeats


def test_bot_advances_with_enough_villagers_and_targets_the_nearest_enemy():
    seat, snap = load(60)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    bot = ScriptedBot(executor)
    me = replace(snap.players[1], units={**snap.players[1].units, "villager": ADVANCE_AT_VILLAGERS})
    bot.tick(me, snap.time_ms)
    assert "advance_phase" in kinds(bot) and executor.book.advance_phase
    barracks = next(o for _, o, _ in bot.log if o.kind == "build" and o.structure == "barracks")
    assert barracks.where == "toward_2"


def test_bot_trains_an_army_in_town_phase_and_attacks_when_it_is_big_enough():
    seat, snap = load(300)
    executor, _, _ = make_executor()
    executor.last_seat = seat
    bot = ScriptedBot(executor)
    me = snap.players[1]
    assert me.phase == "town"
    bot.tick(me, snap.time_ms)
    assert {o.role for _, o, _ in bot.log if o.kind == "train"} == {"infantry_melee", "infantry_ranged", "cavalry"}
    assert executor.book.army.action == "gather"
    big = replace(me, units={**me.units, "infantry_melee": ATTACK_AT_ARMY})
    bot.tick(big, snap.time_ms + 1000)
    assert executor.book.army.action == "attack" and executor.book.army.target == "player_2"
    assert bot.attack_at_ms == snap.time_ms + 1000
