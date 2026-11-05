"""What a v1 seat's model sees (system prompt, tool schemas, user message, order results, briefs) on fixed snapshots.

`capture()` ran once on the commit before interface v2 and its output is `fixtures/interface_v1.json`; the tests in
`test_interface.py` run it again and compare, so v1 stays byte-identical. It only uses calls that exist in both
interfaces, with the default (v1) settings. The system prompt, the memory lines and the user message are not written
here: they come out of `Commander.turn` over three turns against a scripted link and a fake model endpoint, so a
change anywhere on that path shows up in the comparison.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentenv_zeroad.brains import Commander, SeatLink
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import Army, Build, Say, SetEconomy, Train
from agentenv_zeroad.pilot.games import PROMPT
from agentenv_zeroad.server import ZeroADEnv
from agentenv_zeroad.spend import SpendGuard
from tests.fake_models import KEY, FakeLiteLLM
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

FIXTURE = Path(__file__).parent / "fixtures" / "interface_v1.json"
SNAPSHOT_SECONDS = (60, 300)
TURNS = [
    [
        ("set_economy", {"food": 50, "wood": 30, "stone": 10, "metal": 10}, "accepted"),
        ("train", {"role": "villager", "count": 5}, "accepted"),
    ],
    [("army", {"action": "attack", "target": "player_2"}, "rejected")],
    [("say", {"text": "Ave!"}, "accepted")],
]


class _Session:
    """The MCP client session's `list_tools`, answered by the env's own app."""

    def __init__(self, app: Any):
        self.app = app

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=await self.app.list_tools())


class _ScriptedLink:
    """The seat as `Commander.turn` sees it: a fixed brief, and each order answered with the status the script gives."""

    def __init__(self, brief: str, statuses: dict[str, str]):
        self._brief, self._statuses = brief, statuses

    async def brief(self) -> str:
        return self._brief

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"status": self._statuses[name], "message": "scripted"})

    async def report_call(self, *args: Any, **kwargs: Any) -> None:
        return None


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
    fake = FakeLiteLLM()
    model = StrategistModel(
        "local/qwen-base", "http://vllm", KEY, SpendGuard("baseline"), extra_body={"top_p": 1.0}, http=fake.http()
    )
    commander = Commander(model)
    commander.settings.opening = PROMPT
    link = _ScriptedLink(brief, {name: status for turn in TURNS for name, _, status in turn})
    specs = _specs(tools)
    for urgent, turn in zip((False, False, True), TURNS, strict=True):
        fake.tool_calls = [(name, args) for name, args, _ in turn]
        await commander.turn(link, specs, urgent=urgent)  # type: ignore[arg-type]
    request = fake.bodies("/v1/chat/completions")[-1]
    user = request["messages"][1]["content"]
    return {"results": results, "brief": brief, "compact_brief": compact, "user": user, "request": request}


def _specs(tools: list[dict[str, Any]]) -> list[ToolSpec]:
    return [ToolSpec(t["name"], t["description"], t["parameters"]) for t in tools]


def capture() -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        tools = await _tools()
        snaps = {str(s): await _snapshot(s, tools) for s in SNAPSHOT_SECONDS}
        system = Commander(StrategistModel("local/qwen-base", "http://vllm", KEY, SpendGuard("baseline"))).system
        return {"system": system, "opening": PROMPT, "tools": tools, "snapshots": snaps}

    return json.loads(json.dumps(asyncio.run(run())))


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(capture(), indent=1) + "\n")
