"""A fake MCP agent plays a seat through the env's real MCP endpoint over HTTP, as a brain (U5) or any MCP client
would: the lobby over HTTP, then `/players/1/mcp` with the MCP client, a brain-call report, and the match's files."""

import json

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentenv_zeroad.server import ZeroADEnv
from tests.served import served
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio


class FakeAgent:
    """Plays one seat: reads the brief, sets the economy, trains, attacks, says a line, reporting its model calls."""

    def __init__(self, base: str, player_id: str):
        self.base, self.player_id = base, player_id
        self.results: list[dict] = []
        self.brief = ""

    async def play(self) -> list[str]:
        async with streamable_http_client(f"{self.base}/players/{self.player_id}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [t.name for t in (await session.list_tools()).tools]
                self.brief = (await session.call_tool("get_brief", {})).content[0].text
                for name, args in (
                    ("set_economy", {"food": 50, "wood": 30, "stone": 10, "metal": 10}),
                    ("train", {"role": "infantry_melee", "count": 5}),
                    ("army", {"action": "attack", "target": "player_2"}),
                    ("say", {"text": "For Rome!"}),
                ):
                    result = await session.call_tool(name, args)
                    self.results.append(json.loads(result.content[0].text))
        async with httpx.AsyncClient() as http:
            await http.post(f"{self.base}/players/{self.player_id}/brain_calls", json={"seconds": 4.2})
        return tools


async def test_a_fake_mcp_agent_plays_its_seat_through_the_real_endpoint(tmp_path):
    env = ZeroADEnv(lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer()})
    async with served(env) as base, httpx.AsyncClient(base_url=base, timeout=30) as http:
        lobby = f"{base}/agentenv/ext/lobby"
        settings = {"speed": 2.0, "time_cap_minutes": 0.1}
        assert (await http.post(f"{lobby}/open", json={"game_settings": settings})).status_code == 200
        slot = await http.post(f"{lobby}/fill", json={"player_id": "1", "player_kind": "agent", "player_name": "fake"})
        assert slot.json()["environment_url"] == "/players/1"
        await http.post(f"{lobby}/fill", json={"player_id": "2", "player_kind": "ai"})
        assert (await http.post(f"{lobby}/close", json={})).json()["status"] == "closed"
        card = (await http.get("/players/1/.well-known/agent-env.json")).json()
        assert card["name"] == "zeroad/1" and card["additionalInterfaces"][0]["transport"] == "mcp"

        agent = FakeAgent(base, "1")
        tools = await agent.play()
        assert set(tools) == {
            "get_brief",
            "set_economy",
            "train",
            "build",
            "research",
            "advance_phase",
            "army",
            "hold",
            "say",
        }
        assert agent.brief.startswith("Time ") and "[" in agent.brief.splitlines()[-1]
        assert [r["status"] for r in agent.results] == ["accepted"] * 4
        assert all(r["footer"].startswith("[") for r in agent.results)

        match = (await http.post(f"{base}/agentenv/ext/match/finish", json={})).json()
        assert match["status"] == "finished" and match["player_states"]["1"]["moves"] == {"made": 1, "missed": 0}
        files = (await http.post(f"{base}/agentenv/ext/match/files", json={})).json()["files"]
        results = (await http.get(next(f["path"] for f in files if f["kind"] == "results"))).json()
    seat = results["seats"]["1"]
    assert seat["kind"] == "agent" and seat["tool_calls"]["n"] == 5
    assert seat["tool_calls"]["by_tool"] == {"get_brief": 1, "set_economy": 1, "train": 1, "army": 1, "say": 1}
    assert seat["orders"]["army"] == "attack player_2" and seat["said"][0][1] == "For Rome!"
    assert seat["brain_calls"][0]["seconds"] == 4.2 and seat["metrics"]["decisions"] == 1


async def _handle():
    return Handle()
