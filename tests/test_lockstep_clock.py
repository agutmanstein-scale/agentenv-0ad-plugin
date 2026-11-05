"""The lockstep clock on a FakeEngine: the seat tick every 5 turns, the deadline barrier at T + L whatever the reply
latency, a late reply that stops game time, a reply that never comes, overlapping seats and the raw step."""

import asyncio

import pytest

from agentenv_zeroad.clock import LockstepClock
from agentenv_zeroad.engine import TURN_MS, FakeEngine

pytestmark = pytest.mark.anyio


class Seat:
    """A scripted seat brain: its decisions begin at the game times in `at`, answer after `delay_s` wall seconds (or
    when `gate` is set, or never), and record the engine's game time when they are applied or missed."""

    def __init__(self, slot, engine, *, at=(0,), delay_s=0.0, gate=None, never=False, events=None):
        self.slot, self.engine = slot, engine
        self.at, self.delay_s, self.gate, self.never = list(at), delay_s, gate, never
        self.events = [] if events is None else events
        self.begun, self.applied, self.missed = [], [], []

    async def begin(self, game_ms):
        if not self.at or game_ms < self.at[0]:
            return None
        self.at.pop(0)
        self.begun.append(game_ms)
        self.events.append(("begin", game_ms, self.slot))
        return Decision(self)


class Decision:
    def __init__(self, seat):
        self.seat = seat

    async def answer(self):
        seat = self.seat
        if seat.never:
            await asyncio.Event().wait()
        if seat.gate is not None:
            await seat.gate.wait()
        if seat.delay_s:
            await asyncio.sleep(seat.delay_s)

    async def apply(self):
        self.seat.applied.append(self.seat.engine.time_ms)
        self.seat.events.append(("apply", self.seat.engine.time_ms, self.seat.slot))

    async def miss(self, reason):
        self.seat.missed.append((self.seat.engine.time_ms, reason))


def make(latency_ms=5000, timeout=120.0, done_at=None, observe_ok=True):
    fake = FakeEngine()
    log = {"observe": [], "act": [], "events": []}

    async def observe():
        log["observe"].append(fake.time_ms)
        log["events"].append(("observe", fake.time_ms, None))
        return observe_ok

    async def act():
        log["act"].append(fake.time_ms)
        log["events"].append(("act", fake.time_ms, None))

    clock = LockstepClock(
        fake,
        observe=observe,
        act=act,
        done=lambda: done_at is not None and fake.time_ms >= done_at,
        latency_ms=latency_ms,
        brain_timeout_s=timeout,
    )
    return fake, clock, log


@pytest.mark.parametrize("delay_s", [0.0, 0.05])
@pytest.mark.parametrize("latency_ms", [2000, 5000])
async def test_orders_land_at_exactly_t_plus_l_in_game_time_whatever_the_reply_latency(latency_ms, delay_s):
    fake, clock, _ = make(latency_ms=latency_ms)
    seat = Seat("1", fake, at=(1000,), delay_s=delay_s)
    clock.add_brain(seat)
    await clock.run(max_steps=100)
    assert seat.begun == [1000] and seat.applied == [1000 + latency_ms]
    assert clock.decisions == [
        {"slot": "1", "frozen_ms": 1000, "due_ms": 1000 + latency_ms, "applied_ms": 1000 + latency_ms, "missed": False}
    ]


async def test_a_late_reply_blocks_stepping_and_game_time_does_not_pass_t_plus_l_meanwhile():
    fake, clock, _ = make()
    gate = asyncio.Event()
    seat = Seat("1", fake, gate=gate)
    clock.add_brain(seat)
    task = asyncio.create_task(clock.run(max_steps=200))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert fake.time_ms == 5000 and not task.done() and seat.applied == []
    assert clock.counts["blocked"] == 1
    gate.set()
    await task
    assert seat.applied == [5000] and fake.time_ms == 200 * TURN_MS
    assert clock.blocked_wall_s > 0 and clock.metrics()["blocked"] == 1


async def test_the_seat_tick_runs_exactly_every_five_turns_and_never_at_turn_zero():
    fake, clock, log = make()
    await clock.run(max_steps=50)
    assert log["observe"] == log["act"] == [1000 * k for k in range(1, 10)]
    assert clock.steps == 50 and fake.time_ms == 10_000


async def test_orders_apply_between_the_observe_and_the_act_of_their_boundary():
    fake, clock, log = make()
    clock.add_brain(Seat("1", fake, events=log["events"]))
    await clock.run(max_steps=30)
    assert log["events"][0] == ("begin", 0, "1")
    at_deadline = [e for e in log["events"] if e[1] == 5000]
    assert at_deadline == [("observe", 5000, None), ("apply", 5000, "1"), ("act", 5000, None)]


async def test_overlapping_decisions_of_two_seats_each_land_at_their_own_deadline_in_deadline_order():
    fake, clock, log = make()
    clock.add_brain(Seat("1", fake, at=(0,), delay_s=0.02, events=log["events"]))
    clock.add_brain(Seat("2", fake, at=(2000,), events=log["events"]))
    await clock.run(max_steps=60)
    applies = [(slot, t) for kind, t, slot in log["events"] if kind == "apply"]
    assert applies == [("1", 5000), ("2", 7000)]
    assert clock.metrics()["decisions"] == 2 and clock.metrics()["applied"] == 2


async def test_a_reply_that_never_comes_is_missed_after_the_brain_timeout_and_the_game_goes_on():
    fake, clock, _ = make(timeout=0.05)
    seat = Seat("1", fake, never=True)
    clock.add_brain(seat)
    await clock.run(max_steps=80)
    assert seat.applied == [] and len(seat.missed) == 1
    assert seat.missed[0][0] == 5000 and "no reply within 0.05 s" in seat.missed[0][1]
    assert fake.time_ms == 80 * TURN_MS
    assert clock.metrics()["missed"] == 1
    assert clock.decisions == [{"slot": "1", "frozen_ms": 0, "due_ms": 5000, "applied_ms": None, "missed": True}]


async def test_a_seat_waits_for_its_own_pending_decision_before_beginning_another():
    fake, clock, _ = make()
    seat = Seat("1", fake, at=(0, 1000, 2000))
    clock.add_brain(seat)
    await clock.run(max_steps=100)
    assert seat.begun == [0, 5000, 10000] and seat.applied == [5000, 10000, 15000]


async def test_a_failed_observe_skips_the_act_and_starts_no_decision_at_that_boundary():
    fake, clock, log = make(observe_ok=False)
    seat = Seat("1", fake, at=(1000,))
    clock.add_brain(seat)
    await clock.run(max_steps=20)
    assert log["act"] == [] and seat.begun == []


async def test_the_run_ends_when_the_match_is_done_and_abandons_decisions_in_flight():
    fake, clock, _ = make(done_at=3000)
    seat = Seat("1", fake, never=True)
    clock.add_brain(seat)
    await asyncio.wait_for(clock.run(), 5)
    assert fake.time_ms == 3000 and seat.applied == [] and seat.missed == []


async def test_stop_releases_a_clock_blocked_on_a_reply():
    fake, clock, _ = make()
    seat = Seat("1", fake, never=True)
    clock.add_brain(seat)
    task = asyncio.create_task(clock.run())
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert fake.time_ms == 5000
    await asyncio.wait_for(clock.stop(), 5)
    await asyncio.wait_for(task, 5)
    assert seat.applied == [] and seat.missed == []


@pytest.mark.parametrize("latency_ms", [0, 500, 1500, 5100])
def test_a_latency_that_is_not_a_whole_number_of_ticks_is_refused(latency_ms):
    with pytest.raises(ValueError, match="multiple of the 1000 ms tick"):
        LockstepClock(FakeEngine(), observe=None, act=None, done=lambda: False, latency_ms=latency_ms)


async def test_steps_skip_decoding_state_unless_something_subscribed():
    fake, clock, _ = make()
    await clock.run(max_steps=10)
    assert fake.raw_steps == 10 and clock.latest_state is None
    mailbox = clock.subscribe()
    await clock.run(max_steps=5)
    assert fake.raw_steps == 10 and clock.latest_state["timeElapsed"] == 15 * TURN_MS
    assert (await mailbox.get())["timeElapsed"] == 15 * TURN_MS


async def test_every_applied_command_is_logged_with_the_step_it_went_out_with():
    fake, clock, _ = make()
    clock.queue(1, {"type": "a"})
    await clock.step_once()
    await clock.step_once()
    clock.queue(2, {"type": "b"})
    clock.queue(1, {"type": "c"})
    await clock.step_once()
    assert clock.commands == [(0, 1, {"type": "a"}), (2, 1, {"type": "c"}), (2, 2, {"type": "b"})]


async def test_a_failed_step_puts_its_commands_back_and_logs_nothing():
    fake, clock, _ = make()
    clock.queue(1, {"type": "a"})

    async def broken(commands=()):
        raise RuntimeError("engine gone")

    fake.step_raw = broken
    with pytest.raises(RuntimeError):
        await clock.step_once()
    assert clock.pending(1) == 1 and clock.commands == [] and clock.steps == 0
