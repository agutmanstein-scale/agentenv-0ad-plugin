import asyncio

from agentenv_zeroad.engine import SIM_TIME_JS, FakeEngine
from agentenv_zeroad.scenario import Seat, build_scenario


def test_fake_engine_steps_and_evaluates_before_any_reset_like_the_real_autostart_game():
    async def go():
        fake = FakeEngine()
        state = await fake.step()
        return state, await fake.evaluate(SIM_TIME_JS)

    state, sim = asyncio.run(go())
    assert state["timeElapsed"] == 200 and sim == 200


def test_fake_engine_records_commands_and_advances_200_ms_per_step():
    async def go():
        fake = FakeEngine()
        state = await fake.reset(build_scenario([Seat("A", "rome"), Seat("B", "pers"), Seat("C", "cart")]))
        assert state["timeElapsed"] == 0 and len(state["players"]) == 4  # gaia + 3
        await fake.step({1: [{"type": "stop", "entities": [1]}]})
        state = await fake.step([(2, {"type": "walk", "entities": [2], "x": 1, "z": 1, "queued": False})])
        return fake, state

    fake, state = asyncio.run(go())
    assert state["timeElapsed"] == 400
    assert fake.steps == [
        [(1, {"type": "stop", "entities": [1]})],
        [(2, {"type": "walk", "entities": [2], "x": 1, "z": 1, "queued": False})],
    ]


def test_fake_engine_answers_sim_time_canned_evaluations_and_templates():
    async def go():
        fake = FakeEngine(evaluations={"1+1": 2}, templates={"units/x": "<x/>"})
        await fake.reset('{"settings": {}}')
        await fake.step()
        return (
            fake,
            await fake.evaluate(SIM_TIME_JS),
            await fake.evaluate("1+1"),
            await fake.templates(["units/x", "y"]),
        )

    fake, sim, two, templates = asyncio.run(go())
    assert (sim, two) == (200, 2)
    assert templates == {"units/x": "<x/>"}  # like 0.28, an unknown name is left out
    assert fake.evaluated == [SIM_TIME_JS, "1+1"]


def test_reset_clears_time_and_recorded_steps():
    async def go():
        fake = FakeEngine()
        await fake.reset("{}")
        await fake.step()
        await fake.reset("{}")
        return fake

    fake = asyncio.run(go())
    assert fake.time_ms == 0 and fake.steps == [] and fake.resets == 2
