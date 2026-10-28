"""Real-engine test of the stream engine (zeroad-engine:0.28-rt, patches/rl-realtime.patch) with the director mod.

Visual mode on an Xvfb inside the engine container, `-rl-realtime`, the mod mounted from the repo. Checks:
- a reset game is held at turn 0 until the first `/step` (the env's clock decides when the match starts);
- after one step the frame loop advances the game in real time with no further steps, and steps keep answering;
- the mod's GuiInterface call is loaded and finds each player's civic centre;
- the rendered session is the observer's (player -1: no fog, no player HUD), as the mod logs it. Stock 0.28 reads
  `?playerID=` into a 1-byte buffer, so "-1" never fits and the view was player 1's.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from agentenv_zeroad.engine import LABEL_PID, EngineOptions, free_port, launch_engine
from agentenv_zeroad.live import GAME, engine_args
from agentenv_zeroad.scenario import Seat, build_scenario

pytestmark = pytest.mark.engine

RT_IMAGE = "zeroad-engine:0.28-rt"
MOD = Path(__file__).resolve().parents[2] / "mod/zeroad-director"
SCENE_JS = (
    "(function () { const gui = Engine.QueryInterface(SYSTEM_ENTITY, IID_GuiInterface);"
    " return typeof gui.ZeroadDirectorScene == 'function' ? gui.ZeroadDirectorScene(-1) : null; })()"
)


def have_image(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


@pytest.mark.skipif(not have_image(RT_IMAGE), reason=f"needs {RT_IMAGE} (scripts/build-engine.sh rt)")
def test_the_stream_engine_holds_until_the_first_step_then_runs_in_real_time_with_the_director_mod():
    asyncio.run(_run())


async def _run() -> None:
    options = EngineOptions(port=free_port(), visual=True, display=":99", realtime=True, extra_args=engine_args())
    docker = {"xvfb": f"{GAME[0]}x{GAME[1]}", "mounts": ((str(MOD), "/usr/share/games/0ad/mods/zeroad-director"),)}
    scenario = build_scenario(
        [Seat("Petra 1", "rome", controller="petra"), Seat("Petra 2", "pers", controller="petra")], size=128
    )
    async with launch_engine(options, mode="docker", image=RT_IMAGE, ready_timeout=180, docker=docker) as engine:
        await engine.reset(scenario)
        held = await engine.sim_time_ms()
        await asyncio.sleep(2.0)
        assert await engine.sim_time_ms() == held == 0, "the game ran before its first step"

        await engine.step()
        started = time.monotonic()
        begin = await engine.sim_time_ms()
        await asyncio.sleep(4.0)
        advanced = await engine.sim_time_ms() - begin
        wall = (time.monotonic() - started) * 1000
        assert advanced >= 0.6 * wall - 400, f"sim time {advanced} ms over {wall:.0f} ms of wall time, no steps"

        times = []
        for _ in range(3):
            state = await engine.step()
            times.append(state["timeElapsed"])
        assert times == sorted(times)

        scene = await engine.evaluate(SCENE_JS)
        assert scene is not None, "the director mod's GuiInterface call is not loaded"
        assert sorted(b["owner"] for b in scene["bases"]) == [1, 2]
        assert isinstance(scene["fights"], list)

        mainlog = engine_mainlog()
        assert "zeroad-director: viewing as player -1" in mainlog, "the stream's view is not the observer's"


def engine_mainlog() -> str:
    """This test's engine container's mainlog (HOME=/tmp there)."""
    names = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"label={LABEL_PID}={os.getpid()}"], capture_output=True, text=True
    ).stdout.split()
    assert len(names) == 1, f"expected this test's one engine container, found {names}"
    log = ["docker", "exec", names[0], "cat", "/tmp/.local/state/0ad/log/mainlog.html"]
    return subprocess.run(log, capture_output=True, text=True).stdout
