"""Real-engine test of the env: an agent seat played by a fake MCP agent over HTTP, a scripted bot and Petra, on the
real 0.28 engine (docker launcher), in real time at speed 2 until a 2-minute time cap.

Checks: the engine container carries its owner labels while it runs and is gone after; each player's reveal flag is
off (fog) while the observer still sees all; the clock kept its wall-clock schedule (steps per wall second, few
overruns); the agent's orders were accepted and carried out; the match ended at the time cap with the highest engine
score winning; the results file has every seat; the replay copied out of the engine has the match's turns.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentenv_zeroad.engine import LABEL_PID
from agentenv_zeroad.server import ZeroADEnv
from tests.served import served

pytestmark = [pytest.mark.engine, pytest.mark.anyio]

CAP_MINUTES = 2
SPEED = 2.0


def engine_containers() -> list[str]:
    fmt = '{{.Names}} {{.Label "' + LABEL_PID + '"}}'
    cmd = ["docker", "ps", "-a", "--filter", "name=zeroad-engine-", "--format", fmt]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


async def play_agent(base: str, notes: list) -> None:
    """The fake agent: brief, economy, a barracks toward player 2, soldiers, then attack."""
    async with streamable_http_client(f"{base}/players/1/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            notes.append((await session.call_tool("get_brief", {})).content[0].text)
            for name, args in (
                ("set_economy", {"food": 50, "wood": 50, "stone": 0, "metal": 0}),
                ("train", {"role": "villager", "count": 6}),
                ("build", {"structure": "barracks", "where": "toward_2"}),
                ("train", {"role": "infantry_melee", "count": 5}),
            ):
                notes.append(json.loads((await session.call_tool(name, args)).content[0].text))
            await asyncio.sleep(20)
            scout = await session.call_tool("army", {"action": "scout", "target": "2"})
            notes.append(json.loads(scout.content[0].text))
            notes.append((await session.call_tool("get_brief", {})).content[0].text)


async def test_an_agent_a_bot_and_petra_play_a_real_time_match_to_the_time_cap(tmp_path):
    before = engine_containers()
    env = ZeroADEnv(out_root=tmp_path)
    notes: list = []
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=600) as http:
        lobby = "/agentenv/ext/lobby"
        settings = {"size": 128, "speed": SPEED, "time_cap_minutes": CAP_MINUTES, "start_grace_seconds": 60}
        await http.post(f"{lobby}/open", json={"game_settings": settings})
        await http.post(f"{lobby}/fill", json={"player_id": "1", "player_kind": "agent", "player_name": "fake"})
        bot = {"controller": "bot"}
        await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai", "game_settings": bot})
        await http.post(f"{lobby}/fill", json={"player_id": "3", "player_kind": "ai"})
        closed = await http.post(f"{lobby}/close", json={})
        assert closed.status_code == 200, closed.text
        running = [c for c in engine_containers() if c not in before]
        assert len(running) == 1 and running[0].endswith(f" {os.getpid()}")

        started = time.monotonic()
        await play_agent(base, notes)
        match = (await http.post("/agentenv/ext/match/finish", json={})).json()
        wall = time.monotonic() - started
        files = (await http.post("/agentenv/ext/match/files", json={})).json()["files"]
        results = (await http.get(next(f["path"] for f in files if f["kind"] == "results"))).json()
        replay = (await http.get(next(f["path"] for f in files if f["kind"] == "replay"))).text

    print(json.dumps({k: v for k, v in results.items() if k != "seats"}, indent=1))
    for slot, seat in results["seats"].items():
        print(slot, seat["kind"], seat["civ"], seat["outcome"], seat["score"], seat["orders"], seat["failures"][:2])
    assert [c for c in engine_containers() if c not in before] == []

    assert results["fog"] == {"global": False, "own": [False, False, False, False], "observer": True}
    assert notes[0].startswith("Time ")
    assert [n["status"] for n in notes[1:5]] == ["accepted"] * 4, notes[1:5]
    assert match["status"] == "finished" and results["ended_by"] == "time_cap"
    totals = {slot: seat["score"]["total"] for slot, seat in results["seats"].items()}
    for seat in results["seats"].values():
        assert seat["score"]["total"] == sum(seat["score"][k] for k in ("economy", "military", "exploration"))
        assert seat["score"]["exploration"] > 0 and seat["score"]["economy"] > 0
    winner = max(totals, key=totals.get)
    assert results["seats"][winner]["outcome"] == "won", totals
    assert match["player_states"][winner]["status"] == "won"

    clock = results["clock"]
    assert results["game_time_s"] >= CAP_MINUTES * 60
    assert clock["steps"] >= CAP_MINUTES * 60 * 5
    assert results["wall_s"] == pytest.approx(clock["steps"] * 0.2 / SPEED, rel=0.15)
    assert wall < CAP_MINUTES * 60 / SPEED + 30
    assert clock["overruns"] <= clock["steps"] * 0.02

    agent = results["seats"]["1"]
    assert agent["kind"] == "agent" and agent["tool_calls"]["n"] == 7
    assert agent["metrics"]["idle_villager_s"] >= 0 and agent["rejections"] == 0
    assert results["seats"]["2"]["kind"] == "bot" and results["seats"]["3"]["kind"] == "petra"

    lines = replay.splitlines()
    assert lines[0].startswith("start {") and '"Size":128' in lines[0]
    assert sum(1 for line in lines if line.startswith("turn ")) >= clock["steps"] * 0.9
    assert any(line.startswith("cmd 1 ") for line in lines)
