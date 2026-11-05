"""What a v1 seat's model sees (system prompt, tool schemas, user message, order results, briefs) on fixed snapshots.

`capture()` ran once on the commit before interface v2 and its output is `fixtures/interface_v1.json`; the tests in
`test_interface.py` run it again and compare, so v1 stays byte-identical. It only uses calls that exist in both
interfaces, with the default (v1) settings.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentenv_zeroad.brains import STRATEGIST_SYSTEM, Commander, SeatLink
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import Army, Build, Say, SetEconomy, Train
from agentenv_zeroad.pilot.games import PROMPT
from agentenv_zeroad.server import ZeroADEnv
from agentenv_zeroad.spend import SpendGuard
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

FIXTURE = Path(__file__).parent / "fixtures" / "interface_v1.json"
SNAPSHOT_SECONDS = (60, 300)
MEMORY = [
    'set_economy({"food":50,"wood":30,"stone":10,"metal":10}) -> accepted; '
    'train({"role":"villager","count":5}) -> accepted',
    'army({"action":"attack","target":"player_2"}) -> rejected',
]


class _Session:
    """The MCP client session's `list_tools`, answered by the env's own app."""

    def __init__(self, app: Any):
        self.app = app

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=await self.app.list_tools())


async def _tools() -> list[dict[str, Any]]:
    link = SeatLink(_Session(ZeroADEnv().create_app()), "http://env/players/1/mcp", None)  # type: ignore[arg-type]
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in await link.tools()]


async def _snapshot(seconds: int, tools: list[dict[str, Any]]) -> dict[str, Any]:
    seats = (SeatPlan("1", 1, "agent", "rome", "policy v3"), SeatPlan("2", 2, "petra", "athen", "Petra 2"))
    with tempfile.TemporaryDirectory() as out:
        match = LiveMatch(
            Handle(ScriptedEngine(seconds=seconds)),
            MatchConfig(seats=seats, time_cap_s=1800),
            out_dir=Path(out),
            placer=placer(),
        )
        await match.setup()
        orders = [
            SetEconomy(food=50, wood=30, stone=10, metal=10),
            Train(role="villager", count=5),
            Train(role="infantry_melee", count=3, at="stable"),
            Build(structure="barracks", where="base"),
            Army(action="attack", target="player_2"),
            Army(action="attack", target="player_9"),
            Say(text="Ave!"),
        ]
        results = [match.submit("1", order).to_json() for order in orders]
        brief = match.brief("1")
        compact = match.compact_brief("1")
    model = StrategistModel("local/qwen-base", "http://vllm", "k", SpendGuard("baseline"), extra_body={"top_p": 1.0})
    commander = Commander(model)
    commander.settings.opening = PROMPT
    commander._memory = list(MEMORY)
    user = commander._user(brief, urgent=True)
    body = model.request(STRATEGIST_SYSTEM, user, _specs(tools))[1]
    return {"results": results, "brief": brief, "compact_brief": compact, "user": user, "request": body}


def _specs(tools: list[dict[str, Any]]) -> list[ToolSpec]:
    return [ToolSpec(t["name"], t["description"], t["parameters"]) for t in tools]


def capture() -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        tools = await _tools()
        snaps = {str(s): await _snapshot(s, tools) for s in SNAPSHOT_SECONDS}
        return {"system": STRATEGIST_SYSTEM, "opening": PROMPT, "tools": tools, "snapshots": snaps}

    return json.loads(json.dumps(asyncio.run(run())))


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(capture(), indent=1) + "\n")
