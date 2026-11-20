"""Real-engine test of the showcase engine (zeroad-engine:0.28-show, patches/rl-paced-render.patch).

Visual mode on an Xvfb inside the engine container, `-rl-paced`, the director mod mounted from the repo. Checks:
- a reset game is held at turn 0 (nothing runs it until the first `/step`);
- every `/step` is exactly one 200 ms turn, and `/step` answers once that turn has run;
- between steps sim time does not move, although the frame loop keeps rendering;
- the rendered session is the observer's, as in the rt engine's test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from agentenv_zeroad.engine import TURN_MS, Engine, EngineOptions, free_port, launch_engine
from agentenv_zeroad.live import GAME, engine_args
from agentenv_zeroad.scenario import Seat, build_scenario
from tests.engine.test_rt_engine import MOD, engine_mainlog, have_image

pytestmark = pytest.mark.engine

SHOW_IMAGE = "zeroad-engine:0.28-show"
needs_image = pytest.mark.skipif(
    not have_image(SHOW_IMAGE), reason=f"needs {SHOW_IMAGE} (scripts/build-engine.sh show)"
)


@asynccontextmanager
async def paced_game() -> AsyncIterator[Engine]:
    options = EngineOptions(port=free_port(), visual=True, display=":99", paced=True, extra_args=engine_args())
    docker = {"xvfb": f"{GAME[0]}x{GAME[1]}", "mounts": ((str(MOD), "/usr/share/games/0ad/mods/zeroad-director"),)}
    scenario = build_scenario(
        [Seat("Petra 1", "rome", controller="petra"), Seat("Petra 2", "pers", controller="petra")], size=128
    )
    async with launch_engine(options, mode="docker", image=SHOW_IMAGE, ready_timeout=180, docker=docker) as engine:
        await engine.reset(scenario)
        yield engine


@needs_image
def test_paced_engine_holds_until_stepped():
    async def run() -> None:
        async with paced_game() as engine:
            await asyncio.sleep(3.0)
            assert await engine.sim_time_ms() == 0

    asyncio.run(run())


@needs_image
def test_each_step_is_one_turn():
    async def run() -> None:
        async with paced_game() as engine:
            for _ in range(25):
                await engine.step()
            assert await engine.sim_time_ms() == 25 * TURN_MS == 5000

    asyncio.run(run())


@needs_image
def test_no_turns_between_steps():
    async def run() -> None:
        async with paced_game() as engine:
            for _ in range(10):
                await engine.step()
            await asyncio.sleep(2.0)
            assert await engine.sim_time_ms() == 2000

    asyncio.run(run())


@needs_image
def test_paced_view_is_the_observers():
    async def run() -> None:
        async with paced_game() as engine:
            await engine.step()
            assert "zeroad-director: viewing as player -1" in engine_mainlog(), "the view is not the observer's"

    asyncio.run(run())
