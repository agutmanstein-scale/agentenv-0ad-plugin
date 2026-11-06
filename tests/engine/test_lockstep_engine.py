"""A short lockstep game on the real engine image: the scripted bot against Petra, on a one-engine pool, with the
lockstep clock stepping as fast as the engine answers and the seat tick every 5 turns."""

import asyncio
import shutil
import subprocess

import pytest

from agentenv_zeroad.engine import DEFAULT_IMAGE
from agentenv_zeroad.pool import EnginePool
from agentenv_zeroad.train.game import play_game
from agentenv_zeroad.train.spec import GameSpec, SeatSpec

pytestmark = pytest.mark.engine

CAP_S = 60


@pytest.fixture(scope="module", autouse=True)
def engine_image():
    if shutil.which("docker") is None:
        pytest.fail("docker is not installed")
    found = subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True)
    if found.returncode != 0:
        pytest.fail(f"{DEFAULT_IMAGE} is missing; run scripts/build-engine.sh")


def test_a_lockstep_bot_against_petra_runs_to_the_time_cap_faster_than_real_time(tmp_path):
    spec = GameSpec(
        "lockstep-smoke",
        (SeatSpec("bot", "rome"), SeatSpec("petra", "pers", difficulty=1)),
        size=128,
        seed=3,
        time_cap_minutes=CAP_S / 60,
    )

    async def go():
        async with EnginePool(1) as pool, pool.lease() as handle:
            return await play_game(spec, handle, tmp_path / "game", tools=[])

    results = asyncio.run(go())
    print(f"{results['game_time_s']} game s in {results['wall_s']} wall s; clock {results['clock']}")
    assert results["status"] == "finished" and results["ended_by"] == "time_cap"
    assert results["game_time_s"] >= CAP_S and results["clock"]["steps"] >= CAP_S * 5
    assert results["clock"]["lockstep"] is True and results["wall_s"] < results["game_time_s"]
    assert results["commands"], "the scripted bot sent no commands"
    assert {p for rows in results["commands"].values() for p, _ in rows} == {1}
    assert results["timeline"][-1]["t_ms"] >= CAP_S * 1000 - 10_000
