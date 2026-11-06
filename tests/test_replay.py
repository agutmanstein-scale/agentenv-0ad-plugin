"""The order-replay controller: a logged order stream read from results.json, given back to an executor at its logged
game times under either interface, and a seat kind in the match and the lobby."""

import json

import pytest
from agentenv_game import SlotRequest

from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.replay import TOOLS, ReplayController, ReplayOrder, ReplaySpec
from agentenv_zeroad.server import ZeroADEnv
from tests.executor_fixtures import load, make_executor
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

LOGGED = {
    "seats": {
        "1": {"kind": "petra"},
        "2": {
            "kind": "agent",
            "turns": [
                {
                    "began_ms": 30_000,
                    "orders": [
                        {"tool": "army", "arguments": {"action": "attack", "target": "player_2"}, "status": "partial"},
                        {"tool": "say", "arguments": {"text": "Forward."}, "status": "accepted"},
                    ],
                },
                {
                    "began_ms": 5_000,
                    "orders": [
                        {"tool": "train", "arguments": {"role": "villager", "count": 5}, "status": "accepted"},
                        {"tool": "advancement", "arguments": {}, "status": "rejected"},
                        {"tool": "set_economy", "arguments": {"food": 90, "wood": 90}, "status": "rejected"},
                    ],
                },
                {"began_ms": 60_000, "orders": []},
            ],
        },
    }
}


def spec(**kwargs) -> ReplaySpec:
    return ReplaySpec.from_results(LOGGED, "2", source="r00-t01", **kwargs)


def test_the_spec_is_the_seats_logged_orders_in_game_time_order():
    s = spec()
    assert s.source == "r00-t01" and s.slot == "2"
    assert [(o.time_ms, o.tool) for o in s.orders] == [
        (5_000, "train"),
        (5_000, "advancement"),
        (5_000, "set_economy"),
        (30_000, "army"),
        (30_000, "say"),
    ]
    assert s.orders[3].arguments == {"action": "attack", "target": "player_2"}


def test_a_seat_with_no_logged_turns_has_an_empty_spec():
    assert ReplaySpec.from_results(LOGGED, "1").orders == ()


def test_a_spec_round_trips_through_its_json_file(tmp_path):
    original = spec(delay_ms=5_000, game={"map": {"name": "mainland", "seed": 7}})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(original.to_json()))
    assert ReplaySpec.load(path) == original


def test_the_tools_a_replay_can_play_are_the_order_vocabulary():
    assert TOOLS == {"set_economy", "train", "build", "research", "advance_phase", "army", "hold", "say"}


def controller(interface: str, **kwargs):
    executor, _, _ = make_executor(interface=interface)
    seat, snap = load(300)
    executor.last_seat = seat
    return ReplayController(executor, spec(**kwargs)), snap.players[1]


def test_orders_are_submitted_once_when_their_game_time_comes_in_the_order_logged():
    replay, me = controller("v1")
    replay.tick(me, 4_999)
    assert replay.log == [] and not replay.finished
    replay.tick(me, 5_000)
    assert [(t, o.kind) for t, o, _ in replay.log] == [(5_000, "train")]
    replay.tick(me, 5_000)
    assert len(replay.log) == 1
    replay.tick(me, 31_000)
    assert [o.kind for _, o, _ in replay.log] == ["train", "army", "say"] and replay.finished
    assert [t for t, _, _ in replay.log] == [5_000, 31_000, 31_000]


def test_a_delay_shifts_every_order_later():
    replay, me = controller("v2", delay_ms=5_000)
    replay.tick(me, 9_999)
    assert replay.log == []
    replay.tick(me, 10_000)
    assert [o.kind for _, o, _ in replay.log] == ["train"]


def test_an_invented_tool_or_arguments_that_do_not_validate_are_skipped_and_counted():
    replay, me = controller("v1")
    replay.tick(me, 60_000)
    assert replay.skipped == {"unknown tool advancement": 1, "invalid arguments for set_economy": 1}
    assert [o.kind for _, o, _ in replay.log] == ["train", "army", "say"]


def test_a_logged_attack_without_wave_settings_takes_the_default_wave_under_v2_and_the_old_posture_under_v1():
    v2, me = controller("v2")
    v2.tick(me, 60_000)
    posture = v2.executor.book.army
    assert (posture.action, posture.target, posture.min_size) == ("attack", "player_2", 8)
    v1, me = controller("v1")
    v1.tick(me, 60_000)
    assert v1.executor.book.army.action == "attack" and v1.executor.book.army.target == "player_2"


def test_the_executor_judges_each_order_as_it_would_today_whatever_status_it_got_when_logged():
    wrong_at = {
        "seats": {
            "2": {
                "turns": [
                    {
                        "began_ms": 0,
                        "orders": [{"tool": "train", "arguments": {"role": "villager", "count": 2, "at": "barracks"}}],
                    }
                ]
            }
        }
    }
    for interface, status in (("v1", "rejected"), ("v2", "accepted")):
        executor, _, _ = make_executor(interface=interface)
        seat, snap = load(300)
        executor.last_seat = seat
        replay = ReplayController(executor, ReplaySpec.from_results(wrong_at, "2"))
        replay.tick(snap.players[1], 0)
        assert [r.status for _, _, r in replay.log] == [status]


def test_orders_go_through_the_submit_hook_when_one_is_given():
    seen = []
    executor, _, _ = make_executor(interface="v2")
    replay = ReplayController(executor, ReplaySpec((ReplayOrder(0, "hold", {}),)), submit=lambda o: seen.append(o))
    _, snap = load(300)
    replay.tick(snap.players[1], 0)
    assert [o.kind for o in seen] == ["hold"]


async def test_a_replay_seat_plays_through_its_executor_in_a_match(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            ReplaySpec((ReplayOrder(0, "set_economy", {"food": 50, "wood": 50, "stone": 0, "metal": 0}),)).to_json()
        )
    )
    seats = (
        SeatPlan("1", 1, "replay", "rome", "Replay 1", replay=str(path)),
        SeatPlan("2", 2, "petra", "pers", "Petra 2"),
    )
    match = LiveMatch(Handle(), MatchConfig(seats=seats, tick_s=0.05), out_dir=tmp_path, placer=placer())
    await match.setup()
    seat = match.seats["1"]
    assert isinstance(seat.bot, ReplayController) and seat.executor is not None
    assert [o.kind for _, o, _ in seat.bot.log] == ["set_economy"] and seat.executor.book.split["food"] == 50
    assert [a.by for a in seat.actions] == ["bot"] and match.results()["seats"]["1"]["kind"] == "replay"


async def test_a_replay_seat_without_a_spec_fails_to_set_up(tmp_path):
    seats = (SeatPlan("1", 1, "replay", "rome", "Replay 1"), SeatPlan("2", 2, "petra", "pers", "Petra 2"))
    match = LiveMatch(Handle(), MatchConfig(seats=seats), out_dir=tmp_path, placer=placer())
    with pytest.raises(AssertionError, match="without a spec"):
        await match.setup()


async def test_the_replay_spec_path_is_what_the_lobby_slot_carries(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec().to_json()))

    async def start():
        return Handle()

    env = ZeroADEnv(start, tmp_path, tick_s=0.05, match_options={"placer": placer()})
    env.create_app()
    env.new_lobby({"speed": 2.0})
    replay = {"controller": "replay", "replay_spec": str(path), "interface": "v2"}
    env.fill_slot(SlotRequest(player_id="1", player_kind="ai", game_settings=replay))
    env.fill_slot(SlotRequest(player_id="2", player_kind="ai"))
    await env.close_lobby()
    plan = env.live.seats["1"].plan
    await env.live.stop()
    assert (plan.kind, plan.replay, plan.interface, plan.name) == ("replay", str(path), "v2", "Replay 1")
