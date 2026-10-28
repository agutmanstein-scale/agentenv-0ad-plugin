import asyncio
import logging

import pytest

from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.engine import EngineError, FakeEngine
from tests.virtual_time import VirtualTime


def make(speed=1.0, step_delay=0.0, realtime=False):
    vt = VirtualTime()
    fake = FakeEngine(step_delay=step_delay, sleep=vt.sleep)
    clock = GameClock(fake, speed=speed, realtime=realtime, now=vt.now, sleep=vt.sleep)
    return vt, fake, clock


def run(clock, fake, steps):
    async def go():
        await fake.reset("{}")
        stamps = []
        orig = fake.step

        async def stamped(commands=()):
            stamps.append(clock._now())
            return await orig(commands)

        fake.step = stamped
        await clock.run(max_steps=steps)
        return stamps

    return asyncio.run(go())


@pytest.mark.parametrize(("speed", "period"), [(1.0, 0.2), (1.5, 0.2 / 1.5), (2.0, 0.1)])
def test_steps_land_on_a_fixed_wall_clock_grid_at_200ms_over_speed(speed, period):
    vt, fake, clock = make(speed=speed, step_delay=0.03)
    stamps = run(clock, fake, 50)
    assert stamps == pytest.approx([i * period for i in range(50)])
    assert clock.steps == 50 and fake.time_ms == 50 * 200
    assert clock.overruns == 0


@pytest.mark.parametrize("speed", [0.5, 2.5])
def test_speed_outside_1_to_2_is_rejected(speed):
    with pytest.raises(ValueError):
        GameClock(FakeEngine(), speed=speed)


def test_queued_commands_go_out_with_the_next_step_per_player_in_order():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        clock.queue(2, {"type": "b1"})
        clock.queue(1, {"type": "a1"})
        clock.queue(2, {"type": "b2"})
        assert clock.pending(2) == 2
        await clock.step_once()
        await clock.step_once()
        clock.queue(1, {"type": "a2"})
        await clock.step_once()

    asyncio.run(go())
    assert fake.steps == [
        [(1, {"type": "a1"}), (2, {"type": "b1"}), (2, {"type": "b2"})],
        [],
        [(1, {"type": "a2"})],
    ]
    assert clock.pending(1) == 0 and clock.pending(2) == 0


def test_an_overrun_is_recorded_and_not_caught_up():
    # Step 3 takes 0.5 s (2.5 slots). The clock must not then fire the missed slots back to back.
    vt, fake, clock = make(step_delay=lambda i: 0.5 if i == 3 else 0.01)
    stamps = run(clock, fake, 8)

    assert clock.overruns == 1
    assert clock.worst_overrun_s == pytest.approx(0.6 + 0.5 - 0.8)  # step 3 starts at 0.6, slot 4 was due at 0.8
    assert stamps[:4] == pytest.approx([0.0, 0.2, 0.4, 0.6])
    assert stamps[4] == pytest.approx(1.1)  # straight after the slow step, once
    assert stamps[5:] == pytest.approx([1.3, 1.5, 1.7])  # then a fresh 200 ms grid
    assert clock.metrics() == {"steps": 8, "overruns": 1, "worst_overrun_ms": 300.0, "speed": 1.0, "realtime": False}


def test_worst_overrun_keeps_the_maximum():
    vt, fake, clock = make(step_delay=lambda i: {2: 0.3, 5: 0.7}.get(i, 0.0))
    run(clock, fake, 8)
    assert clock.overruns == 2
    assert clock.worst_overrun_s == pytest.approx(0.5)


def test_slow_subscribers_never_hold_the_clock_and_see_only_the_newest_state():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        mailbox = clock.subscribe()
        await clock.run(max_steps=10)  # nobody reads during the run
        newest = await mailbox.get()
        return mailbox, newest

    mailbox, newest = asyncio.run(go())
    assert clock.steps == 10 and clock.overruns == 0
    assert newest["timeElapsed"] == 2000 and clock.latest_state["timeElapsed"] == 2000
    assert mailbox.get_nowait() is None


def test_late_subscriber_gets_the_latest_state_at_once_and_can_unsubscribe():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        await clock.step_once()
        mailbox = clock.subscribe()
        first = mailbox.get_nowait()
        clock.unsubscribe(mailbox)
        await clock.step_once()
        return first, mailbox.get_nowait()

    first, after = asyncio.run(go())
    assert first["timeElapsed"] == 200 and after is None


def test_brains_queue_while_the_clock_runs_in_the_background():
    vt, fake, clock = make()

    async def brain():
        for i in range(5):
            clock.queue(1, {"type": "order", "n": i})
            await vt.sleep(0.45)

    async def go():
        await fake.reset("{}")
        task = clock.start()
        await brain()
        await clock.stop()
        assert task.done()

    asyncio.run(go())
    sent = [cmd["n"] for batch in fake.steps for _, cmd in batch]
    assert sent == [0, 1, 2, 3, 4]
    assert clock.overruns == 0


def test_start_twice_is_refused():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        clock.start()
        with pytest.raises(RuntimeError):
            clock.start()
        await clock.stop()

    asyncio.run(go())


def test_realtime_mode_runs_the_same_loop_at_a_fixed_200ms_cadence():
    vt, fake, clock = make(speed=2.0, realtime=True)
    clock.queue(1, {"type": "x"})
    stamps = run(clock, fake, 5)
    assert stamps == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert fake.steps[0] == [(1, {"type": "x"})]
    assert clock.metrics()["realtime"] is True


def test_stop_right_after_start_ends_the_run():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        task = clock.start()
        await clock.stop()
        return task

    task = asyncio.run(go())
    assert task.done() and clock.steps == 0


def test_stop_while_waiting_for_the_next_slot_sends_no_further_step():
    vt, fake, clock = make()

    async def go():
        await fake.reset("{}")
        clock.start()
        await vt.sleep(0.3)  # steps went at 0.0 and 0.2; the clock now waits for 0.4
        await clock.stop()

    asyncio.run(go())
    assert clock.steps == 2 and len(fake.steps) == 2


def test_each_overrun_is_logged_with_step_lateness_and_duration(caplog):
    vt, fake, clock = make(step_delay=lambda i: 0.5 if i == 3 else 0.0)
    with caplog.at_level(logging.WARNING, logger="agentenv_zeroad.clock"):
        run(clock, fake, 6)
    (record,) = [r for r in caplog.records if r.name == "agentenv_zeroad.clock"]
    assert "overrun" in record.getMessage()
    assert "step 4" in record.getMessage() and "300 ms late" in record.getMessage() and "500 ms" in record.getMessage()


def test_a_failed_step_puts_its_commands_back_and_stops_the_clock_with_the_error():
    vt, fake, clock = make()
    orig = fake.step
    calls = 0

    async def flaky(commands=()):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise EngineError("POST /step failed: ReadTimeout")
        return await orig(commands)

    async def go():
        await fake.reset("{}")
        fake.step = flaky
        clock.queue(1, {"type": "walk"})
        task = clock.start(max_steps=5)
        await vt.sleep(0.1)  # step 1 has sent the walk; step 2 (at 0.2) will fail
        clock.queue(1, {"type": "train"})
        clock.queue(2, {"type": "stop"})
        clock.queue(1, {"type": "later"})
        with pytest.raises(EngineError):
            await task
        assert clock.steps == 1
        fake.step = orig
        await clock.step_once()  # the next good step carries what the failed one could not send

    asyncio.run(go())
    assert fake.steps == [
        [(1, {"type": "walk"})],
        [(1, {"type": "train"}), (1, {"type": "later"}), (2, {"type": "stop"})],
    ]
