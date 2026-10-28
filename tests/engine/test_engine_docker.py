"""Real-engine tests: the zeroad-engine:0.28 image, started with host networking, driven from host Python.

Python runs on the host, not inside the container: the spike saw python3 SIGKILLed when run via `docker exec`.
Each test module starts one container on a free port and always removes it.
"""

import asyncio
import shutil
import subprocess

import pytest

from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.engine import DEFAULT_IMAGE, EngineOptions, free_port, launch_engine
from agentenv_zeroad.scenario import Seat, build_scenario

pytestmark = pytest.mark.engine

PETRA_PAIR = [Seat("Rome", "rome", controller="petra"), Seat("Persia", "pers", controller="petra")]


@pytest.fixture(scope="module", autouse=True)
def engine_image():
    if shutil.which("docker") is None:
        pytest.fail("docker is not installed")
    found = subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True)
    if found.returncode != 0:
        pytest.fail(f"{DEFAULT_IMAGE} is missing; run scripts/build-engine.sh")


def test_clock_runs_sixty_game_seconds_with_two_petra_players():
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
            state = await engine.reset(build_scenario(PETRA_PAIR, size=128, seed=3))
            start_ms = await engine.sim_time_ms()
            clock = GameClock(engine, speed=1.0)
            await clock.run(max_steps=300)
            end_ms = await engine.sim_time_ms()
            return state, start_ms, end_ms, clock

    state, start_ms, end_ms, clock = asyncio.run(go())
    print(f"sim {start_ms}->{end_ms} ms, clock {clock.metrics()}, last step {clock.last_step_s * 1000:.1f} ms")
    assert state["entities"], "reset returned no entities"
    assert clock.steps == 300
    assert end_ms - start_ms == 60_000
    assert clock.latest_state is not None and clock.latest_state["entities"]
    assert clock.latest_state["timeElapsed"] == end_ms  # FakeEngine mirrors this field


def test_external_seat_takes_commands_and_engine_answers_evaluate_and_templates():
    seats = [Seat("Ext", "athen"), Seat("Bot", "pers", controller="petra")]

    async def go():
        async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
            state = await engine.reset(build_scenario(seats, size=128, seed=5))
            mine = [int(i) for i, e in state["entities"].items() if e.get("owner") == 1 and "units/" in e["template"]]
            target = next(e for e in state["entities"].values() if int(e["id"]) == mine[0])["position"]
            walk = {"type": "walk", "entities": mine[:1], "x": target[0] + 20, "z": target[1], "queued": False}
            state = await engine.step({1: [walk]})
            for _ in range(24):
                state = await engine.step()
            moved = state["entities"][str(mine[0])]["position"]
            two = await engine.evaluate("1 + 1")
            mine_template = state["entities"][str(mine[0])]["template"]
            templates = await engine.templates(["units/athen/no_such_unit", mine_template])
            return target, moved, two, mine_template, templates

    target, moved, two, mine_template, templates = asyncio.run(go())
    assert moved != target, "external seat's walk command had no effect"
    assert two == 2
    assert list(templates) == [mine_template], "an unknown template name must not shift the others"
    assert "Identity" in templates[mine_template]
