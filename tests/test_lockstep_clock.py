"""The lockstep clock on a FakeEngine: the seat tick every 5 turns, the deadline barrier at T + L whatever the reply
latency, a late reply that stops game time, a reply that never comes, overlapping seats and the raw step."""

import asyncio

import pytest

from agentenv_zeroad.clock import LockstepClock
from agentenv_zeroad.engine import TURN_MS, FakeEngine
from tests.virtual_time import run_virtual

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


class TimedSeat:
    """A seat whose k-th decision answers after `delays[k]` loop-time seconds (the last delay repeats) and records the
    game time each `begin` saw."""

    def __init__(self, slot, delays):
        self.slot, self.delays = slot, list(delays)
        self.begun, self.count = [], 0
        self.clock = None

    async def begin(self, game_ms):
        self.begun.append(game_ms)
        delay = self.delays[min(self.count, len(self.delays) - 1)]
        self.count += 1
        return TimedDecision(self, delay, self.count)


class TimedDecision:
    def __init__(self, seat, delay, number):
        self.seat, self.delay, self.number = seat, delay, number

    async def answer(self):
        await asyncio.sleep(self.delay)

    async def apply(self):
        self.seat.clock.queue(int(self.seat.slot), {"type": "order", "seat": self.seat.slot, "n": self.number})

    async def miss(self, reason):
        pass


def virtual_clock(loop, *, pace_s=None, seats=(), step_starts=None):
    """A LockstepClock on `loop`'s virtual time over an instant engine; `step_starts` collects each step's start."""
    fake = FakeEngine()
    clock = LockstepClock(fake, observe=_true, act=_nothing, done=lambda: False, brains=list(seats), pace_s=pace_s)
    clock._now = loop.time
    for seat in seats:
        seat.clock = clock
    if step_starts is not None:
        real_step = clock.step_once

        async def recording():
            step_starts.append(loop.time())
            return await real_step()

        clock.step_once = recording
    return clock


async def _true():
    return True


async def _nothing():
    pass


def test_paced_clock_steps_no_faster_than_pace():
    starts = []

    async def scenario(loop):
        clock = virtual_clock(loop, pace_s=0.2, step_starts=starts)
        await clock.run(max_steps=50)
        return loop.time()

    elapsed = run_virtual(scenario)
    assert elapsed >= 9.8 - 1e-9
    assert len(starts) == 50
    assert all(b - a >= 0.2 - 1e-9 for a, b in zip(starts, starts[1:], strict=False))


def test_unpaced_clock_never_sleeps_for_pace():
    async def scenario(loop):
        clock = virtual_clock(loop, seats=[TimedSeat("1", [0.0])])
        await clock.run(max_steps=50)
        return loop.time(), clock.metrics()

    elapsed, metrics = run_virtual(scenario)
    assert elapsed == 0
    assert metrics["paced_wait_s"] == 0 and metrics["pace_s"] is None


def test_paced_clock_does_not_burst_after_hold():
    starts = []

    async def scenario(loop):
        seat = TimedSeat("1", [8.0, 0.0])  # paced game time reaches the 5 s deadline at 4.8 s, so the block lasts 3.2 s
        clock = virtual_clock(loop, pace_s=0.2, seats=[seat], step_starts=starts)
        await clock.run(max_steps=60)
        return clock

    clock = run_virtual(scenario)
    held = max(range(1, len(starts)), key=lambda i: starts[i] - starts[i - 1])
    assert starts[held] - starts[held - 1] >= 2.5
    after = starts[held : held + 11]
    assert len(after) == 11
    assert all(b - a >= 0.2 - 1e-9 for a, b in zip(after, after[1:], strict=False))
    assert clock.metrics()["paced_wait_s"] > 0


def test_holding_names_the_slot_during_a_block_and_clears_after():
    samples = []

    async def scenario(loop):
        seat = TimedSeat("1", [8.0])
        clock = virtual_clock(loop, pace_s=0.2, seats=[seat])
        task = asyncio.create_task(clock.run(max_steps=30))
        while not task.done():
            samples.append((loop.time(), clock.holding))
            await asyncio.sleep(0.5)
        return clock

    clock = run_virtual(scenario)
    held = [(t, h) for t, h in samples if h is not None]
    assert held, "the clock never reported a hold"
    assert all(h["slot"] == "1" and h["since"] == pytest.approx(4.8) for _, h in held)
    assert clock.holding is None


def test_pacing_changes_nothing_a_seat_sees():
    delays = [0.5, 4.0, 6.5, 130.0]

    async def scenario(loop, pace_s):
        seats = [TimedSeat("1", delays), TimedSeat("2", list(reversed(delays)))]
        clock = virtual_clock(loop, pace_s=pace_s, seats=seats)
        await clock.run(max_steps=600)
        return clock.commands, clock.decisions, [s.begun for s in seats]

    unpaced = run_virtual(lambda loop: scenario(loop, None))
    paced = run_virtual(lambda loop: scenario(loop, 0.2))
    assert any(d["missed"] for d in unpaced[1]), "the 130 s answer should have missed"
    assert paced == unpaced
    assert paced[0], "the seats' orders should have gone out in commands"


def test_stop_during_pace_sleep_takes_no_more_steps():
    async def scenario(loop):
        clock = virtual_clock(loop, pace_s=0.2)
        clock.start()
        await asyncio.sleep(0.5)
        clock.queue(1, {"type": "late"})
        steps, commands = clock.steps, list(clock.commands)
        await clock.stop()
        return clock, steps, commands

    clock, steps, commands = run_virtual(scenario)
    assert steps > 0 and clock.steps == steps
    assert clock.commands == commands and clock.pending(1) == 1


def test_pace_too_slow_for_the_brain_timeout_is_refused():
    with pytest.raises(ValueError, match="brain_timeout_s"):
        LockstepClock(FakeEngine(), observe=None, act=None, done=lambda: False, pace_s=5.0, brain_timeout_s=120.0)
    LockstepClock(FakeEngine(), observe=None, act=None, done=lambda: False, pace_s=0.2, brain_timeout_s=120.0)


@pytest.mark.parametrize("pace_s", [0, -1])
def test_pace_must_be_positive(pace_s):
    with pytest.raises(ValueError, match="pace_s"):
        LockstepClock(FakeEngine(), observe=None, act=None, done=lambda: False, pace_s=pace_s)
