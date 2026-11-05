"""Whole prompts (system with tools, user message) a seat's model gets on the recorded snapshots, per interface.

The prompt-size test counts their characters; `scripts/prompt_tokens.py` counts real Qwen tokens. A scenario is one
snapshot of a match with the seat's orders of the last three turns in its memory, as the commander would keep them.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_zeroad.brains import Commander, CommanderSettings, SeatLink, _answer_of, strategist_system
from agentenv_zeroad.interface import Interface
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import Army, Build, SetEconomy, Train
from agentenv_zeroad.pilot.games import PROMPT, PROMPT_V2
from agentenv_zeroad.server import ZeroADEnv
from agentenv_zeroad.spend import SpendGuard
from tests.v1_baseline import _Session
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

SNAPSHOT_SECONDS = (60, 300)


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    tools: list[ToolSpec]


def _turns(match: LiveMatch, interface: Interface) -> list[str]:
    """Three turns of orders as their memory lines, answered by the real executor."""
    turns = [
        [SetEconomy(food=50, wood=30, stone=10, metal=10), Train(role="villager", count=5)],
        [Build(structure="barracks", where="base"), Train(role="infantry_melee", count=3, at="stable")],
        [Army(action="attack", target="player_2"), Build(structure="field", where="near_food")],
    ]
    out = []
    for orders in turns:
        lines = []
        for order in orders:
            result = json.dumps(match.submit("1", order).to_json())
            args = json.dumps(order.model_dump(exclude={"kind"}, exclude_none=True), separators=(",", ":"))
            status = json.loads(result)["status"]
            line = f"{order.kind}({args}) -> {status}"
            lines.append(f"{line}: {_answer_of(result)}" if interface == "v2" else line)
        out.append(" | ".join(lines) if interface == "v2" else "; ".join(lines))
    return out


async def prompts(interface: Interface) -> list[Prompt]:
    found = []
    for seconds in SNAPSHOT_SECONDS:
        seats = (
            SeatPlan("1", 1, "agent", "rome", "policy v3", interface=interface),
            SeatPlan("2", 2, "petra", "athen", "Petra 2", ai_difficulty=2),
        )
        with tempfile.TemporaryDirectory() as out:
            match = LiveMatch(
                Handle(ScriptedEngine(seconds=seconds)),
                MatchConfig(seats=seats, time_cap_s=1800),
                out_dir=Path(out),
                placer=placer(),
            )
            await match.setup()
            link = SeatLink(_Session(ZeroADEnv().create_app()), "http://env/players/1/mcp", None, interface)  # type: ignore[arg-type]
            tools = await link.tools()
            model = StrategistModel("local/qwen-base", "http://vllm", "k", SpendGuard("sizes"))
            settings = CommanderSettings(
                interface=interface,
                system_prompt=strategist_system(interface),
                opening=PROMPT_V2 if interface == "v2" else PROMPT,
            )
            commander = Commander(model, settings)
            commander._reference = match.reference("1")
            memory = _turns(match, interface)
            brief = match.brief("1")
        for turns in (0, 3):
            commander._memory = memory[:turns]
            found.append(Prompt(commander.system, commander._user(brief, urgent=turns == 3), tools))
    return found


def all_prompts(interface: Interface) -> list[Prompt]:
    return asyncio.run(prompts(interface))


def chat_request(prompt: Prompt) -> dict[str, Any]:
    model = StrategistModel("local/qwen-base", "http://vllm", "k", SpendGuard("sizes"))
    return model.request(prompt.system, prompt.user, prompt.tools)[1]


def request_chars(prompt: Prompt) -> int:
    """The characters of the whole chat request (system, tool schemas, user message)."""
    return len(json.dumps(chat_request(prompt), ensure_ascii=False))
