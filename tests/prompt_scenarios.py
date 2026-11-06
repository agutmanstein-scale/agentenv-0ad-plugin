"""Whole prompts (system with tools, user message) a seat's model gets on the recorded snapshots, per interface.

The prompt-size test counts their characters; `scripts/prompt_tokens.py` counts real Qwen tokens. A scenario is one
snapshot of a match with the seat's orders of the last three turns in its memory, as the commander would keep them.
The recorded snapshots are early (seconds 60 and 300 of a quiet match); `midgame` is a busy prompt on top of the
300 s one, built to be near the worst a brief gets: 14 minutes in, enemy structures and armies in sight, an attack
under way, a wave out and the next one forming, the executor's give-ups, a full events list and six orders a turn.
"""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_zeroad.brains import Commander, CommanderSettings, SeatLink, _answer_of, strategist_system
from agentenv_zeroad.brief_v2 import notices_in
from agentenv_zeroad.executor import Failure
from agentenv_zeroad.interface import Interface
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.match import LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import AdvancePhase, Army, Build, Research, Say, SetEconomy, Train
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
    name: str = ""


EARLY_TURNS = [
    [SetEconomy(food=50, wood=30, stone=10, metal=10), Train(role="villager", count=5)],
    [Build(structure="barracks", where="base"), Train(role="infantry_melee", count=3, at="stable")],
    [Army(action="attack", target="player_2"), Build(structure="field", where="near_food")],
]
MIDGAME_TURNS = [
    [
        SetEconomy(food=35, wood=30, stone=5, metal=30, villagers=60),
        Train(role="infantry_melee", count=10, at="barracks"),
        Train(role="cavalry", count=5),
        Build(structure="blacksmith", where="base"),
        Build(structure="house", where="toward_player_2"),
        Say(text="Rome marches."),
    ],
    [
        Army(action="attack", target="player_2", min_size=12),
        Train(role="infantry_ranged", count=10),
        Research(tech="soldier_attack_melee"),
        Build(structure="tower", where="front"),
        Build(structure="field", where="near_food"),
        AdvancePhase(),
    ],
    [
        Army(action="attack", target="player_2", min_size=12, regroup_below=0.5),
        Train(role="siege", count=2),
        Build(structure="temple", where="base"),
        Research(tech="gather_lumbering_ironaxes"),
        SetEconomy(food=30, wood=30, stone=0, metal=40),
        Say(text="Hold the line."),
    ],
]


def _turns(match: LiveMatch, interface: Interface, turns: list[list[Any]] = EARLY_TURNS) -> list[str]:
    """Three turns of orders as their memory lines, answered by the real executor."""
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
            name = f"{seconds}s-{turns}-turns"
            found.append(Prompt(commander.system, commander._user(brief, urgent=turns == 3), tools, name))
    found.append(await midgame(interface))
    return found


def _units(count: int, role: str, first_id: int, around: tuple[float, float], doing: str = "idle") -> list[list[Any]]:
    return [
        [first_id + i, role, around[0] + i % 6, around[1] + i // 6, doing, None, "aggressive"] for i in range(count)
    ]


def _heavy_snapshot(raw: dict[str, Any], enemy_base: tuple[float, float]) -> dict[str, Any]:
    """The 300 s snapshot as it looks in a hard fight: the seat's army larger and bloodied, a tower under attack, the
    enemy's army and ten of its structures in sight."""
    snap = copy.deepcopy(raw)
    me = snap["players"]["1"]
    me["units"] = {"cavalry": 5, "infantry_melee": 16, "infantry_ranged": 9, "villager": 48}
    me["resources"] = {"food": 410, "wood": 330, "stone": 90, "metal": 640}
    me["pop"], me["pop_cap"] = 142, 150
    me["stats"] = {**me["stats"], "units_lost": 14, "enemy_units_killed": 11, "units_trained": 70}
    me["enemies_near_base"] = 5
    tower = next(b for b in me["buildings"] if b["kind"] == "Tower")
    tower["hp"] = 55
    kinds = ["CivCentre", "House", "House", "Barracks", "Storehouse", "Farmstead", "Tower", "Temple", "Field", "Stable"]
    structures = [
        {"id": 3000 + i, "kind": k, "pos": [enemy_base[0] + 12 * i % 70, enemy_base[1] + 9 * i % 50]}
        for i, k in enumerate(kinds)
    ]
    me["enemies"] = {
        "2": {
            "units": {"infantry_melee": 13, "infantry_ranged": 7, "cavalry": 4, "villager": 11},
            "buildings": structures,
            "nearest": 38,
            "soldiers_at": [int(enemy_base[0]) - 30, int(enemy_base[1]) - 20, 24],
            "workers_at": [int(enemy_base[0]) + 15, int(enemy_base[1]) + 10, 11],
            "contact": 5,
        }
    }
    return snap


async def midgame(interface: Interface) -> Prompt:
    """The busy prompt: see the module docstring. Both interfaces get the same game state; only the text differs."""
    seats = (
        SeatPlan("1", 1, "agent", "rome", "policy v3", interface=interface),
        SeatPlan("2", 2, "petra", "athen", "Petra 2", ai_difficulty=2),
    )
    engine = ScriptedEngine(seconds=300)
    with tempfile.TemporaryDirectory() as out:
        match = LiveMatch(Handle(engine), MatchConfig(seats=seats, time_cap_s=1800), out_dir=Path(out), placer=placer())
        await match.setup()
        link = SeatLink(_Session(ZeroADEnv().create_app()), "http://env/players/1/mcp", None, interface)  # type: ignore[arg-type]
        tools = await link.tools()
        commander = _commander(interface, match)
        memory = _turns(match, interface, MIDGAME_TURNS)
        executor = match.seats["1"].executor
        assert executor is not None
        light = copy.deepcopy(engine.raw)
        engine.time_ms = 14 * 60_000
        await match._tick()
        rally = executor.army.point(executor, executor.last_seat, "front")
        enemy_base = executor.enemy_bases(executor.last_seat)[2]
        engine.raw["snapshot"] = _heavy_snapshot(light["snapshot"], enemy_base)
        view = engine.raw["view"]["players"]["1"]
        villagers = [u for u in view["units"] if u[1] == "villager"]
        view["units"] = villagers + _units(14, "infantry_melee", 9000, rally) + _units(4, "cavalry", 9100, rally)
        await match._tick()  # the wave of 18 is at the rally point: it goes
        view["units"] = (
            villagers
            + _units(11, "infantry_melee", 9000, enemy_base, "fight")
            + _units(2, "cavalry", 9100, enemy_base, "fight")
            + _units(7, "infantry_ranged", 9200, executor.last_seat.base, "idle")
        )
        engine.time_ms += 30_000
        await match._tick()
        executor.notices.extend(
            [
                Failure(
                    engine.time_ms - 25_000, 1, "barracks", "build barracks", "no free spot near wood", "near_wood"
                ),
                Failure(
                    engine.time_ms - 22_000, 1, "barracks", "build barracks", "no free spot near wood", "near_wood"
                ),
                Failure(engine.time_ms - 12_000, 1, "siege", "train siege", "no building you own trains siege units"),
                Failure(
                    engine.time_ms - 5_000, 1, "soldier_attack_melee", "research soldier_attack_melee", "no smithy"
                ),
            ]
        )
        match.trackers[1].update(match.snapshot)
        brief = match.brief("1")
    if interface == "v2":
        memory[-1] += " | executor gave up: " + "; ".join(notices_in(brief))
    commander._memory = memory
    return Prompt(commander.system, commander._user(brief, urgent=True), tools, "midgame")


def _commander(interface: Interface, match: LiveMatch) -> Commander:
    model = StrategistModel("local/qwen-base", "http://vllm", "k", SpendGuard("sizes"))
    settings = CommanderSettings(
        interface=interface,
        system_prompt=strategist_system(interface),
        opening=PROMPT_V2 if interface == "v2" else PROMPT,
    )
    commander = Commander(model, settings)
    commander._reference = match.reference("1")
    return commander


def all_prompts(interface: Interface) -> list[Prompt]:
    return asyncio.run(prompts(interface))


def chat_request(prompt: Prompt) -> dict[str, Any]:
    model = StrategistModel("local/qwen-base", "http://vllm", "k", SpendGuard("sizes"))
    return model.request(prompt.system, prompt.user, prompt.tools)[1]


def request_chars(prompt: Prompt) -> int:
    """The characters of the whole chat request (system, tool schemas, user message)."""
    return len(json.dumps(chat_request(prompt), ensure_ascii=False))
