"""The two brains that play a seat from outside the env (spec §4.6): the strategist and the Luna player.

Both play one seat through its MCP endpoint (`/players/<id>/mcp`, the tools U4 serves) and the seat's plain HTTP
routes beside it, which only the brain uses: `brain_calls` (each model call with its time, error and spend, for the
forfeit rule and the metrics), `waiting` (begin/end of each wait on the model, for `waiting_share`), `status` (polled
between calls; it drains no events) and `reflex` (the strategist hands its model endpoint to the seat's Luna reflex,
Setup A). They are wrapped as A2A agents in `agents/zeroad-commander` and `agents/zeroad-luna`.

- `Commander` (zeroad-commander): read the brief, make **one** model call with the order tools, apply the tool calls
  it returns, then wait for the next trigger: `interval_s` (25 s) after the last call, or an urgent event in the
  seat's status (under attack, building lost), at most every `min_gap_s`. Calls are sequential, so one is in flight
  at a time. The seat counts as waiting on the call (spec §5) only when its status said a decision was open as the
  turn began (`open`: an urgent event, a resource run out, an idle army or a threat, a free queue with nothing
  ordered), from the call's start until its orders are applied; a routine call with nothing open is no wait. With
  `game_clock` the interval and gap count game seconds (the status's `game_ms`) instead of wall seconds, so a match
  played faster than real time gets as many calls per game minute. With `log_turns` each reported call carries its
  turn (`turn`: when it began, the user message, the token ids the model saw and sampled, each order and its status),
  which the env keeps in results.json for training. With `interface="v2"` (docs/post-training-v2.md) it plays the v2
  interface: its prompt states the real rules and gets the seat's per-civ cost tables appended once (`/reference`), its
  tools include `hold`, its memory carries the game's answers (message and suggestion) and the executor's give-up
  notices, and it sends at most one `say` per turn. v1 is what it was before.
- `LunaPlayer` (zeroad-luna, Setup B): once the match has started, every `interval_s` (1 s) read the compact JSON brief
  and send one Decisions request with strategic and tactical questions (economy split, phase, army posture, what to
  train and build, how aggressively), each answer mapped to an order tool. An order is only sent when it changes the
  standing orders, and each question has a cooldown, so a steady answer doesn't repeat the same order every second. Like
  the strategist, the seat only waits on a tick that began with a decision open. Its phase and attack questions keep
  "not yet" all match; a "not yet" past the plan's deadline is reported as a hold (`POST .../held`), and the env's
  autopilot carries out the default, logged as the autopilot's.

Lockstep training games (`train/`) run the strategist as a library instead: `CommanderSeat` gives the lockstep clock
(`clock.LockstepClock`) one decision at a time (the brief frozen at game time T, the model call in flight while the game
steps, the orders applied at T + L), and `InProcessSeatLink` plays the seat straight against its `LiveMatch` with the
same tool schemas, brief and results as the MCP link, so the model sees byte-identical prompts. The due-time rule is
the same `decision_due` the live loop uses, counted in game seconds.

Each stops before its spend cap (`SpendGuard`) and keeps polling until the match is over, so the seat plays on with
its executor; it stops at once when the match is over or the env goes away. A defeated seat's strategist stops calling
its model and polls until the match is over. A failed model call, order call or brief costs one turn (a failed order
is recorded as rejected), never the rest of the match.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentenv_zeroad import grading
from agentenv_zeroad.autopilot import HELD_ATTACK_MIN, HELD_DEADLINES
from agentenv_zeroad.brief_v2 import notices_in
from agentenv_zeroad.decisions import DecisionsClient, Question, choice, score
from agentenv_zeroad.interface import DEFAULT_INTERFACE, Interface
from agentenv_zeroad.llm import Reply, StrategistModel, ToolSpec
from agentenv_zeroad.orders import parse_order
from agentenv_zeroad.spend import SpendCapReached

if TYPE_CHECKING:
    from agentenv_zeroad.match import LiveMatch

log = logging.getLogger(__name__)

ORDER_TOOLS = ("set_economy", "train", "build", "research", "advance_phase", "army", "say")
ORDER_TOOLS_V2 = (*ORDER_TOOLS, "hold")
V2_ONLY_PARAMS = {"army": ("min_size", "regroup_below"), "set_economy": ("villagers",)}
DROPPED_KEYS = ("title", "minimum", "maximum", "minLength", "maxLength")
MEMORY_TURNS = 3
MEMORY_MESSAGE_CHARS = 120  # a rejected or partial order's message in the v2 memory
MEMORY_ACCEPTED_CHARS = 50  # an accepted one's
MEMORY_SUGGEST_CHARS = 70

STRATEGIST_SYSTEM = """You are the strategist of one empire in a live, real-time 0 A.D. match. The game never waits \
for you: while you think, a script (the executor) keeps your empire running on your standing orders: idle gatherers \
go back to work on your economy split, houses go up, villagers train up to a target, soldiers defend themselves.
You give orders to groups through the tools; they stand until you change them:
- set_economy(food, wood, stone, metal): the gatherer split in percent (sums to 100);
- train(role, count): villager, infantry_melee, infantry_ranged, cavalry, siege or champion;
- build(structure, where): e.g. house, barracks, storehouse, farmstead, field, temple, blacksmith, tower;
  where is base, front, near_<resource> or toward_<player>;
- research(tech) and advance_phase() (village -> town -> city);
- army(action, target): gather, attack, raid, defend, retreat or scout; target player_<n>, center, front or home;
- say(text): one short in-character line for the viewers of the live stream.
You win by conquest, or with the highest score at the time cap. This is a live match that viewers watch. The \
score adds economy, military (the value of enemy units you kill and buildings you destroy) and exploration, so \
passivity loses: an empire that only gathers falls behind one that fights and wins its fights. Commit to it:
- get a barracks up early and keep soldiers in production from about 2:00 (batches of 5 or more, never an empty \
queue); soldiers gather while the army order is gather, and leave work when you send them;
- attack by about the 4:00 mark with army(attack, player_<n>) or raid their gatherers with cavalry, and keep the \
attack standing: new soldiers join it on their own. Change it only for a real reason (your base is attacked, the \
army is spent); don't flip it back to gather or defend every turn;
- advance to town phase when you can afford it; it adds score and stronger units.
Each turn: read the brief, then call the tools for every order you want now (several in one turn is normal). Don't \
ask questions; nobody answers."""

STRATEGIST_SYSTEM_V2 = """You command one empire in a live, real-time 0 A.D. match against one opponent. The game \
never waits for you: while you think, a script (the executor) runs your empire on your standing orders: idle \
gatherers work your economy split, houses go up, villagers train up to a target, soldiers defend themselves.
You win by conquest. If nobody is defeated at the time cap, the higher net worth wins: the cost value of your living \
units and buildings plus your banked resources. Spending keeps value while the units and buildings live; losses cost \
it, and your kills cost the enemy. Exploring and advancing phase add nothing by themselves.
Orders stand until you change them, and hold() keeps them all. set_economy's villagers (20-100, default 45) is how \
many villagers are kept in training. Army waves: attack and raid gather soldiers at the front rally point until \
min_size (default 8) are there, or 90 s have passed, then attack together; a wave that falls below regroup_below \
(default 0.4) of its launch size falls back to the rally point; soldiers trained after a launch form the next wave. \
Repeating an identical army order keeps the wave's progress; a different one starts over. At most one say() a turn.
Each turn: read the brief, then call the tools for the orders you want now (several is normal). Your last turns' \
orders are listed with the game's answers: when an order was rejected or the executor gave up, fix the cause instead \
of repeating it. Don't ask questions; nobody answers."""


# ---- the seat --------------------------------------------------------------------------------------------------------
class SeatLink:
    """A seat's MCP session and its plain HTTP routes. `session` is an initialized `mcp.ClientSession`."""

    def __init__(self, session: Any, seat_url: str, http: httpx.AsyncClient, interface: Interface = DEFAULT_INTERFACE):
        self.session = session
        self.base = seat_url.rstrip("/").removesuffix("/mcp")
        self.http = http
        self.interface = interface

    async def tools(self) -> list[ToolSpec]:
        """The seat's order tools as its interface offers them (`offered_tools`)."""
        listed = (await self.session.list_tools()).tools
        specs = (
            ToolSpec(t.name, t.description or "", t.inputSchema or {"type": "object", "properties": {}}) for t in listed
        )
        return offered_tools(specs, self.interface)

    async def reference(self) -> str:
        """The seat's static cost tables (interface v2), for the system prompt's prefix."""
        response = await self.http.get(f"{self.base}/reference")
        response.raise_for_status()
        return str(response.json().get("text", ""))

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self.session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return json.dumps({"status": "rejected", "message": text}) if result.isError else text

    async def brief(self, *, compact: bool = False) -> str:
        return await self.call("get_brief", {"compact": True} if compact else {})

    async def status(self) -> dict[str, Any]:
        response = await self.http.get(f"{self.base}/status")
        response.raise_for_status()
        return response.json()

    async def report_call(self, seconds: float, error: str | None, **extra: Any) -> None:
        body = {"seconds": round(seconds, 4), "error": error, **{k: v for k, v in extra.items() if v is not None}}
        await self._post("brain_calls", body)

    async def waiting(self, state: str) -> None:
        await self._post("waiting", {"state": state})

    async def hold(self, decision: str) -> None:
        """Report that the brain held `decision` (phase, attack) past the plan's deadline."""
        await self._post("held", {"decision": decision})

    async def offer_reflex(self, base_url: str, api_key: str) -> bool:
        response = await self.http.post(f"{self.base}/reflex", json={"base_url": base_url, "api_key": api_key})
        return response.status_code == 200 and response.json().get("reflex") == "luna"

    async def _post(self, route: str, body: dict[str, Any]) -> None:
        try:
            (await self.http.post(f"{self.base}/{route}", json=body)).raise_for_status()
        except httpx.HTTPError as e:  # a lost report must not stop the brain
            log.warning("POST %s failed: %r", route, e)


@asynccontextmanager
async def seat_link(server: dict[str, Any], interface: Interface = DEFAULT_INTERFACE) -> AsyncIterator[SeatLink]:
    """A link to the seat agent-env gives the agent: `{"url": .../players/<id>/mcp, "headers": {...}}`."""
    async with (
        httpx.AsyncClient(headers=dict(server.get("headers") or {}), timeout=30) as http,
        streamable_http_client(server["url"], http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield SeatLink(session, server["url"], http, interface)


class InProcessSeatLink:
    """A seat played straight against its `LiveMatch` in this process, with `SeatLink`'s interface and no MCP session
    or HTTP route between them.

    `tool_specs` are the schemas FastMCP builds from the env's tool functions (`server.tool_specs`), offered as the
    seat plan's interface offers them (`offered_tools`), so the model is shown exactly what an MCP link lists.
    `get_brief` and the order tools run through the same `LiveMatch.read_brief` and `LiveMatch.run_order` the env's
    tool functions call, so briefs and results (including the footer) are the same text; `reference` is the same
    `LiveMatch.reference` the env's route serves. The one difference is an order whose arguments break the tool's
    schema: the MCP server turns that into an MCP error before the tool runs, this link into a rejected
    `bad_arguments` result.
    """

    def __init__(self, match: LiveMatch, slot: str, tool_specs: Sequence[ToolSpec]):
        self.match = match
        self.slot = slot
        self.interface = match.seats[slot].plan.interface
        self._tools = offered_tools(tool_specs, self.interface)

    async def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    async def reference(self) -> str:
        return self.match.reference(self.slot)

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "get_brief":
            return await self.brief(compact=bool(arguments.get("compact")))
        return self.match.run_order(self.slot, name, lambda: parse_order({**arguments, "kind": name}))

    async def brief(self, *, compact: bool = False) -> str:
        return self.match.read_brief(self.slot, compact=compact)

    async def status(self) -> dict[str, Any]:
        return {**self.match.status(self.slot), "playing_out": False}

    async def report_call(self, seconds: float, error: str | None, **extra: Any) -> None:
        call = grading.BrainCall(seconds=seconds, error=error, game_ms=self.match.time_ms)
        tokens = (extra["input_tokens"], extra["output_tokens"]) if extra.get("input_tokens") is not None else None
        self.match.record_brain_call(
            self.slot, call, usd=extra.get("usd"), tokens=tokens, model=extra.get("model"), turn=extra.get("turn")
        )

    async def waiting(self, state: str) -> None:
        self.match.record_waiting(self.slot, state)


@dataclass
class BrainRecord:
    """What a brain did, for its A2A result."""

    calls: int = 0
    errors: int = 0
    orders: int = 0
    rejected: int = 0
    latencies_s: list[float] = field(default_factory=list)
    end_reason: str | None = None
    last_error: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)


def _call_failed(e: Exception) -> str:
    return json.dumps({"status": "rejected", "message": f"tool call failed: {type(e).__name__}: {str(e)[:200]}"})


async def _safe_call(link: SeatLink, name: str, arguments: dict[str, Any]) -> str:
    """The order's result text, or a rejected result when the call itself raised: one lost order call doesn't end
    the brain (a seat that is gone ends it at the next status poll)."""
    try:
        return await link.call(name, arguments)
    except Exception as e:  # noqa: BLE001 - one failed order call is a rejected order, not the end of the brain
        return _call_failed(e)


def _status_of(text: str) -> str:
    try:
        return str(json.loads(text).get("status", "accepted"))
    except (json.JSONDecodeError, AttributeError):
        return "accepted"


def _message_of(text: str) -> str:
    try:
        return str(json.loads(text).get("message", ""))[:300]
    except (json.JSONDecodeError, AttributeError):
        return text[:300]


def offered_tools(specs: Iterable[ToolSpec], interface: Interface) -> list[ToolSpec]:
    """The order tools an interface offers, from the env's tool schemas: v1 is the env's tools without v2's `hold` and
    parameters, exactly as they were before v2; v2 adds `hold`, every schema compacted (`compact_schema`)."""
    offered = ORDER_TOOLS_V2 if interface == "v2" else ORDER_TOOLS
    return [
        ToolSpec(t.name, t.description, _offered_schema(t.name, t.parameters, interface))
        for t in specs
        if t.name in offered
    ]


def _offered_schema(name: str, schema: dict[str, Any], interface: Interface) -> dict[str, Any]:
    if interface == "v2":
        return compact_schema(schema)
    if name not in V2_ONLY_PARAMS:
        return schema
    schema = copy.deepcopy(schema)
    for param in V2_ONLY_PARAMS[name]:
        schema.get("properties", {}).pop(param, None)
    return schema


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's parameters without what costs tokens and says little: pydantic's titles, the bounds (the env still
    checks them, and the descriptions of the parameters with a range name it), `default: null`, and the `null` branch
    of an optional parameter (it stays optional: it is not in `required`)."""
    out = {k: v for k, v in schema.items() if k != "title"}
    props = {}
    for name, prop in schema.get("properties", {}).items():
        branches = [b for b in prop.get("anyOf", ()) if b.get("type") != "null"]
        if len(branches) == 1:
            prop = {**{k: v for k, v in prop.items() if k != "anyOf"}, **branches[0]}
        props[name] = {k: v for k, v in prop.items() if k not in DROPPED_KEYS and not (k == "default" and v is None)}
    if props:
        out["properties"] = props
    return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _answer_of(text: str) -> str:
    """The game's answer to an order for the v2 memory: its message and its suggested next call, both clipped."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _clip(text, MEMORY_MESSAGE_CHARS)
    if not isinstance(data, dict):
        return ""
    limit = MEMORY_ACCEPTED_CHARS if data.get("status") == "accepted" else MEMORY_MESSAGE_CHARS
    answer = _clip(str(data.get("message") or ""), limit)
    if data.get("suggest"):
        answer += f" (try {_clip(str(data['suggest']), MEMORY_SUGGEST_CHARS)})"
    return answer


async def _wait_over(link: SeatLink, poll_s: float, sleep: Callable[[float], Awaitable[None]]) -> str:
    """Poll until the match is over (the brain stopped early: the executor plays on)."""
    while True:
        try:
            st = await link.status()
        except httpx.HTTPError:
            return "env_gone"
        if st.get("over") or st.get("playing_out"):
            return "match_over"
        await sleep(poll_s)


# ---- the strategist --------------------------------------------------------------------------------------------------
def decision_due(
    now: float, last: float | None, urgent: int, *, interval_s: float, min_gap_s: float, started: bool = True
) -> bool:
    """Whether a strategist call falls due: the match has started and either none was made yet, `interval_s` passed
    since the last, or an urgent event is pending and `min_gap_s` passed. `now` and `last` share one clock (wall
    seconds, or game seconds when the interval counts game time)."""
    return started and (last is None or now - last >= interval_s or (urgent > 0 and now - last >= min_gap_s))


@dataclass
class CommanderSettings:
    interval_s: float = 25.0
    min_gap_s: float = 8.0
    poll_s: float = 1.0
    system_prompt: str | None = None  # the interface's own prompt unless given
    opening: str = ""
    max_calls: int | None = None
    game_clock: bool = False
    log_turns: bool = False
    interface: Interface = DEFAULT_INTERFACE

    def __post_init__(self) -> None:
        if self.system_prompt is None:
            self.system_prompt = strategist_system(self.interface)


def strategist_system(interface: Interface) -> str:
    return STRATEGIST_SYSTEM_V2 if interface == "v2" else STRATEGIST_SYSTEM


@dataclass
class Asked:
    """One model call's outcome: the reply or the error, when it began (`perf_counter`) and how long it took."""

    reply: Reply | None
    error: Exception | None
    started: float
    seconds: float


class Commander:
    def __init__(
        self,
        model: StrategistModel,
        settings: CommanderSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.model = model
        self.settings = settings or CommanderSettings()
        self.record = BrainRecord()
        self._clock, self._sleep = clock, sleep
        self._memory: list[str] = []
        self._reference = ""
        self._notices: list[str] = []

    @property
    def system(self) -> str:
        """The system prompt: the settings' text, then (interface v2) the seat's per-civ cost tables."""
        return f"{self.settings.system_prompt}\n\n{self._reference}" if self._reference else self.settings.system_prompt

    async def offered(self, link: SeatLink | InProcessSeatLink) -> list[ToolSpec]:
        """The seat's tools for this brain's model and, on interface v2, the seat's cost tables for its system prompt
        (read once, before the first brief)."""
        tools = await link.tools()
        if self.settings.interface == "v2":
            self._reference = await link.reference()
        return tools

    async def play(self, link: SeatLink, *, reflex_key: tuple[str, str] | None = None) -> BrainRecord:
        s = self.settings
        tools = await self.offered(link)
        await link.brief()  # the first call marks the seat ready; the match starts when every agent is
        if reflex_key is not None:
            await link.offer_reflex(*reflex_key)
        last: float | None = None
        while True:
            try:
                st = await link.status()
            except httpx.HTTPError as e:
                self.record.end_reason, self.record.last_error = "env_gone", repr(e)
                return self.record
            if st.get("over") or st.get("playing_out"):
                self.record.end_reason = "match_over"
                return self.record
            if st.get("defeated"):
                self.record.end_reason = f"defeated ({await _wait_over(link, s.poll_s, self._sleep)})"
                return self.record
            now = st.get("game_ms", 0) / 1000 if s.game_clock else self._clock()
            due = decision_due(
                now,
                last,
                st.get("urgent", 0),
                interval_s=s.interval_s,
                min_gap_s=s.min_gap_s,
                started=st.get("started", True),
            )
            if not due:
                await self._sleep(s.poll_s)
                continue
            if s.max_calls is not None and self.record.calls >= s.max_calls:
                self.record.end_reason = await _wait_over(link, s.poll_s, self._sleep)
                return self.record
            last = now
            urgent, waiting = st.get("urgent", 0) > 0, bool(st.get("open"))
            if not await self.turn(link, tools, urgent=urgent, waiting=waiting, game_ms=st.get("game_ms")):
                self.record.end_reason = await _wait_over(link, s.poll_s, self._sleep)
                self.record.end_reason = f"spend_cap ({self.record.end_reason})"
                return self.record

    async def prepare(self, link: SeatLink, urgent: bool) -> str | None:
        """Freeze a turn: read the brief now and build the user message from it. None when there is no brief. A v2
        brief's executor notices are kept for this turn's memory entry."""
        try:
            brief = await link.brief()
        except Exception as e:  # noqa: BLE001 - no brief, no turn; the next trigger tries again
            self.record.last_error = f"brief failed: {type(e).__name__}: {str(e)[:200]}"
            return None
        if self.settings.interface == "v2":
            self._notices = notices_in(brief)
        return self._user(brief, urgent)

    async def turn(
        self,
        link: SeatLink,
        tools: list[ToolSpec],
        *,
        urgent: bool = False,
        waiting: bool = False,
        game_ms: int | None = None,
    ) -> bool:
        """One brief -> model -> orders turn. False when the spend cap stops the brain. `waiting`: the seat had a
        decision open when the turn began, so it waits on this call until its orders are applied. `game_ms`: the
        game time the turn began at, for its logged turn."""
        user = await self.prepare(link, urgent)
        if user is None:
            return True
        if waiting:
            await link.waiting("begin")
        try:
            asked = await self.ask(user, tools)
            return await self.settle(link, user, asked, game_ms=game_ms, urgent=urgent)
        finally:
            if waiting:
                await link.waiting("end")

    async def ask(self, user: str, tools: list[ToolSpec]) -> Asked:
        """The model call of a turn. Its failure is returned, not raised; `settle` decides what it costs."""
        started = time.perf_counter()
        try:
            reply = await self.model.call(self.system, user, tools)
        except Exception as e:  # noqa: BLE001 - a failed call is reported (missed) and the next trigger tries again
            return Asked(None, e, started, time.perf_counter() - started)
        return Asked(reply, None, started, time.perf_counter() - started)

    async def fail_call(self, link: SeatLink, error: str, seconds: float, *, usd: float | None = None) -> None:
        """Count a model call that gave no usable answer, and report it as missed."""
        self.record.calls += 1
        self.record.errors += 1
        self.record.last_error = error
        await link.report_call(seconds, error, usd=usd, model=self.model.model)

    async def settle(
        self, link: SeatLink, user: str, asked: Asked, *, game_ms: int | None = None, urgent: bool = False
    ) -> bool:
        """Apply what `ask` got: its orders go to the seat and the call is reported. False when the spend cap stops
        the brain."""
        if isinstance(asked.error, SpendCapReached):
            self.record.last_error = str(asked.error)
            return False
        if asked.error is not None:
            error = f"{type(asked.error).__name__}: {str(asked.error)[:300]}"
            await self.fail_call(link, error, asked.seconds, usd=round(self.model.guard.last_usd, 6))
            return True
        reply = asked.reply
        assert reply is not None
        applying = time.perf_counter()
        v2 = self.settings.interface == "v2"
        offered = ORDER_TOOLS_V2 if v2 else ORDER_TOOLS
        orders, applied, said = [], [], False
        for c in reply.calls:
            if c.error or c.name not in offered:
                result = json.dumps({"status": "rejected", "message": c.error or f"unknown tool {c.name}"})
            elif v2 and c.name == "say" and said:
                result = json.dumps({"status": "rejected", "message": "Only one say() per turn."})
            else:
                said = said or c.name == "say"
                result = await _safe_call(link, c.name, c.arguments)
            status = _status_of(result)
            self.record.orders += 1
            if status == "rejected":
                self.record.rejected += 1
                self.record.last_error = _message_of(result)
            line = f"{c.name}({json.dumps(c.arguments, separators=(',', ':'))}) -> {status}"
            orders.append(f"{line}: {_answer_of(result)}" if v2 else line)
            applied.append({"tool": c.name, "arguments": c.arguments, "status": status})
        seconds = asked.seconds + time.perf_counter() - applying
        self.record.calls += 1
        self.record.latencies_s.append(reply.seconds)
        turn = None
        if self.settings.log_turns:
            turn = {
                "began_ms": game_ms,
                "urgent": urgent,
                "user": user,
                "text": reply.text,
                "orders": applied,
                "prompt_token_ids": reply.prompt_token_ids,
                "token_ids": reply.token_ids,
                "logprobs": reply.logprobs,
            }
        await link.report_call(
            reply.seconds,
            None,
            usd=round(reply.usd, 6),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            model=self.model.model,
            turn=turn,
        )
        if v2:
            entry = " | ".join(orders) or "(no orders)"
            if self._notices:
                entry += " | executor gave up: " + "; ".join(self._notices)
            self._memory = (self._memory + [entry])[-MEMORY_TURNS:]
        else:
            self._memory = (self._memory + ["; ".join(orders) or "(no orders)"])[-MEMORY_TURNS:]
        self.record.turns.append({"seconds": round(seconds, 2), "orders": orders, "text": reply.text[:500]})
        return True

    def _user(self, brief: str, urgent: bool) -> str:
        parts = []
        if self.settings.opening:
            parts.append(self.settings.opening)
        parts.append(brief)
        if self._memory:
            title = (
                "Your orders in your last turns and the game's answers"
                if self.settings.interface == "v2"
                else ("Your orders in your last turns")
            )
            parts.append(f"{title} (oldest first):\n" + "\n".join(self._memory))
        parts.append(
            ("URGENT: something happened that needs an answer now. " if urgent else "")
            + "Decide your orders now and call the tools."
        )
        return "\n\n".join(parts)


class CommanderSeat:
    """A `Commander` driven by the lockstep clock instead of its own polling loop (spec: the clock calls the seat's
    strategist when a decision is due and awaits it).

    `begin` is the clock's question at each boundary: if `decision_due` says a call falls due at this game time, the
    brief is frozen now and a decision is handed back. Its `answer` is the model call, run while the game steps; its
    `apply` runs the orders against the seat at the decision's deadline; its `miss` reports a call that never answered.
    Intervals count game seconds whatever `settings.game_clock` says. The seat ends like `Commander.play` does: the
    match is over, the seat is defeated, `max_calls` reached or the spend cap hit, after which `begin` returns None.
    """

    def __init__(self, commander: Commander, link: SeatLink | InProcessSeatLink, slot: str, tools: list[ToolSpec]):
        self.commander = commander
        self.link = link
        self.slot = slot
        self.tools = tools
        self._last: float | None = None
        self._stopped = False

    async def begin(self, game_ms: int) -> _SeatDecision | None:
        if self._stopped:
            return None
        s, record = self.commander.settings, self.commander.record
        st = await self.link.status()
        if st.get("over") or st.get("playing_out"):
            return self._stop("match_over")
        if st.get("defeated"):
            return self._stop("defeated (match_over)")
        now = game_ms / 1000
        urgent = st.get("urgent", 0)
        if not decision_due(
            now, self._last, urgent, interval_s=s.interval_s, min_gap_s=s.min_gap_s, started=st.get("started", True)
        ):
            return None
        if s.max_calls is not None and record.calls >= s.max_calls:
            return self._stop("match_over")
        self._last = now
        user = await self.commander.prepare(self.link, urgent > 0)
        if user is None:
            return None
        waiting = bool(st.get("open"))
        if waiting:
            await self.link.waiting("begin")
        return _SeatDecision(self, user, game_ms, urgent > 0, waiting)

    def finish(self) -> None:
        """The match ended under this seat: say so in its record unless it ended earlier."""
        if self.commander.record.end_reason is None:
            self.commander.record.end_reason = "match_over"

    def _stop(self, reason: str) -> None:
        self._stopped = True
        self.commander.record.end_reason = reason
        return None


class _SeatDecision:
    """One `CommanderSeat` decision in flight (the clock's `Decision`)."""

    def __init__(self, seat: CommanderSeat, user: str, game_ms: int, urgent: bool, waiting: bool):
        self.seat, self.user, self.game_ms, self.urgent, self.waiting = seat, user, game_ms, urgent, waiting
        self.asked: Asked | None = None
        self._began = time.perf_counter()

    async def answer(self) -> None:
        self.asked = await self.seat.commander.ask(self.user, self.seat.tools)

    async def apply(self) -> None:
        seat, asked = self.seat, self.asked
        assert asked is not None
        try:
            more = await seat.commander.settle(seat.link, self.user, asked, game_ms=self.game_ms, urgent=self.urgent)
        finally:
            await self._end_wait()
        if not more:
            seat._stop("spend_cap (match_over)")

    async def miss(self, reason: str) -> None:
        seat = self.seat
        try:
            await seat.commander.fail_call(seat.link, reason, time.perf_counter() - self._began)
        finally:
            await self._end_wait()

    async def _end_wait(self) -> None:
        if self.waiting:
            await self.seat.link.waiting("end")


# ---- the Luna player -------------------------------------------------------------------------------------------------
SPLITS = {
    "boom": {"food": 55, "wood": 45, "stone": 0, "metal": 0},
    "balanced": {"food": 45, "wood": 35, "stone": 10, "metal": 10},
    "military": {"food": 40, "wood": 30, "stone": 5, "metal": 25},
    "fortify": {"food": 35, "wood": 35, "stone": 20, "metal": 10},
}
TRAIN_ROLES = ("villager", "infantry_melee", "infantry_ranged", "cavalry")
STRUCTURES = ("house", "barracks", "storehouse", "farmstead", "blacksmith", "temple")
AGGRESSION_BATCH = (5, 8, 10)
ATTACK_ARMY = 8  # soldiers that make an attack worth asking about before the deadline
BOOM_MS = 120_000  # no attack question before this (the plan's boom), whatever the starting army
ATTACK_DEADLINE = HELD_DEADLINES["attack"]  # share of the time cap after which the autopilot attacks if Luna holds
REGROUP_ARMY = (
    HELD_ATTACK_MIN  # an attack down to fewer soldiers than this may regroup at home; past the deadline, the least
)
# army asked to attack (so a regrouped army is not ordered straight back out)
PHASE_AFFORD = 0.8  # share of the next phase's cost in hand that opens the phase question
PHASE_DEADLINE = HELD_DEADLINES["phase"]  # share of the cap: town phase asked for whatever the resources, and the
# autopilot advances if Luna holds
CITY_MIN_LEFT_MS = 4 * 60_000  # city phase only pays off with this much of the match left
COOLDOWN_S = {"economy": 20.0, "army": 10.0, "train": 3.0, "build": 15.0, "phase": 10.0}
LUNA_SYSTEM = (
    "You play one empire alone in a short live 0 A.D. match that viewers watch, answering fixed-choice questions "
    "every second. You win by conquest or by the highest score at the time cap; the score counts the economy, "
    "enemy units killed and buildings destroyed, and exploration, so an empire that only gathers loses. The plan: "
    "boom for the first two minutes, get a barracks, advance to town phase as soon as you can afford it, keep "
    "soldiers in production, attack before the halfway mark and keep attacking; defend when enemies are at your "
    "base; advance to city phase only if the match is long enough to profit from it."
)


@dataclass
class LunaQuestion:
    question: Question
    to_orders: Callable[[Any], list[tuple[str, dict[str, Any]]]]
    held: str | None = None  # past its deadline: the decision a "wait" answer reports as held (POST .../held)


def _affordable(b: dict[str, Any], cost: dict[str, Any], share: float = 1.0) -> bool:
    res = b.get("res") or {}
    return all(res.get(r, 0) >= share * float(c) for r, c in (cost or {}).items())


def phase_question(b: dict[str, Any]) -> LunaQuestion | None:
    """Asked once the next phase is (nearly) affordable, and from `PHASE_DEADLINE` of the cap on whatever the
    resources (the executor saves for an ordered advance, holding back other spending). "Not yet" stays an answer all
    match; past the deadline a hold is reported, and the env's autopilot then advances
    to town as the plan's default, shown as the autopilot's, not Luna's. While raiders are at the base "not yet" is
    Luna's own choice and no hold (the plan defends first). City only with enough match left."""
    orders = b.get("orders") or {}
    if b.get("phase") == "city" or orders.get("advance_phase"):
        return None
    nxt = b.get("next_phase")
    t, cap = int(b.get("time_ms", 0)), int(b.get("cap_ms", 0))
    if b.get("phase") == "town" and cap - t < CITY_MIN_LEFT_MS:
        return None
    due = b.get("phase") != "town" and cap > 0 and t >= PHASE_DEADLINE * cap
    if not due and nxt is not None and not _affordable(b, nxt.get("cost") or {}, PHASE_AFFORD):
        return None
    goal = "city" if b.get("phase") == "town" else "town"
    cost = (nxt or {}).get("cost") or {}
    have = b.get("res") or {}
    text = (
        f"Advance to {goal} phase? Cost {cost or 'unknown'}, you have {have}. It raises the score and unlocks "
        "stronger soldiers and buildings; the executor saves for it and the autopilot adds the buildings it requires."
    )
    options = {"advance": "advance now"}
    if due:
        text += " The plan says town phase by now; if you hold, the autopilot advances for you."
        options["advance_boom"] = "advance now and switch to the boom split to afford it sooner"
    options["wait"] = "not yet"

    def to_orders(v: str) -> list[tuple[str, dict[str, Any]]]:
        if v == "advance":
            return [("advance_phase", {})]
        if v == "advance_boom":
            return [("set_economy", dict(SPLITS["boom"])), ("advance_phase", {})]
        return []

    under_attack = int(b.get("threat", 0) or 0) > 0  # holding to defend is Luna's choice, not a held default
    return LunaQuestion(choice("phase", text, options), to_orders, held="phase" if due and not under_attack else None)


def army_question(b: dict[str, Any], me: int) -> LunaQuestion | None:
    """Defence when enemies are at the base; a timed attack otherwise ("not yet" always an answer; past the deadline a
    hold is reported and the autopilot attacks as the plan's default); under an attack, only whether to regroup."""
    orders = b.get("orders") or {}
    standing = orders.get("army", "gather home")
    action = standing.split(" ")[0]
    soldiers = sum((b.get("army") or {}).get(r, 0) for r in ("infantry_melee", "infantry_ranged", "cavalry"))
    soldiers += sum((b.get("army") or {}).get(r, 0) for r in ("siege", "champion"))
    enemies = [str(p) for p in (b.get("players") or {}) if int(p) != me]
    threat = int(b.get("threat", 0) or 0)
    t, cap = int(b.get("time_ms", 0)), int(b.get("cap_ms", 0) or 1)

    def to_orders(v: str) -> list[tuple[str, dict[str, Any]]]:
        if v in ("press", "wait"):
            return []
        if v == "regroup":
            v = "retreat"
        act, _, target = v.partition("_")
        target = f"player_{target}" if target else "home"
        return [] if f"{act} {target}" == standing else [("army", {"action": act, "target": target})]

    if threat > 0:
        options = {"defend": "bring every soldier home and fight the raiders", "retreat": "fall back home"}
        if action in ("attack", "raid"):
            options["press"] = f"keep the {standing}: hit them harder than they hit us"
        text = f"{threat} enemy soldiers are at our base; we have {soldiers} soldiers ({standing})."
        return LunaQuestion(choice("army", text, options), to_orders)
    if action in ("attack", "raid"):
        if soldiers >= REGROUP_ARMY:
            return None
        options = {"press": f"keep the {standing}; reinforcements join it", "regroup": "regroup at home first"}
        return LunaQuestion(choice("army", f"Only {soldiers} soldiers left on the {standing}.", options), to_orders)
    deadline = t >= ATTACK_DEADLINE * cap
    if not enemies or soldiers < (REGROUP_ARMY if deadline else ATTACK_ARMY) or (not deadline and t < BOOM_MS):
        return None
    options = {}
    for q in enemies:
        seen = (b.get("enemies") or {}).get(q) or {}
        options[f"attack_{q}"] = f"attack player {q} with the whole army (seen: {seen.get('units') or 'nothing'})"
        options[f"raid_{q}"] = f"raid player {q}'s gatherers with the fast soldiers"
    options["wait"] = f"not yet: build up to more than {soldiers} soldiers first"
    when = f"before {clock(ATTACK_DEADLINE * cap)}"
    if deadline:
        when = "now: the match is past the point where waiting pays; if you hold, the autopilot attacks for you"
    text = f"Time {b.get('t')} of {b.get('cap')}. We have {soldiers} soldiers; the plan is to attack {when}."
    return LunaQuestion(choice("army", text, options), to_orders, held="attack" if deadline else None)


def clock(ms: float) -> str:
    s = int(ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def luna_questions(b: dict[str, Any]) -> list[LunaQuestion]:
    """The questions for one tick from the compact brief `b` (with its standing `orders`)."""
    orders = b.get("orders") or {}
    me = int(b.get("player", 0))
    out: list[LunaQuestion] = []
    current = orders.get("economy") or {}
    econ = {
        k: f"split {v['food']}/{v['wood']}/{v['stone']}/{v['metal']} food/wood/stone/metal" for k, v in SPLITS.items()
    }
    out.append(
        LunaQuestion(
            choice("economy", f"Economy split (now {current}).", {**econ, "keep": "keep the current split"}),
            lambda v: [] if v == "keep" or SPLITS.get(v) == current else [("set_economy", dict(SPLITS[v]))],
        )
    )
    phase = phase_question(b)
    if phase is not None:
        out.append(phase)
    army = army_question(b, me)
    if army is not None:
        out.append(army)
    pending = orders.get("train") or {}
    out.append(
        LunaQuestion(
            score(
                "aggression",
                "How hard to build up the army now?",
                {"economy": "economy first", "steady": "a steady army", "all_in": "as many soldiers as possible"},
            ),
            lambda v: [],
        )
    )
    idle = (b.get("idle") or {}).get("production") or []
    if idle or not pending:
        options = {r: f"train {r} ({pending.get(r, 0)} still ordered)" for r in TRAIN_ROLES}
        options["none"] = "train nothing more now"
        out.append(LunaQuestion(choice("train", "What to train next?", options), lambda v: []))
    building = orders.get("build") or []
    have = b.get("bld") or {}
    options = {s: f"build a {s}" for s in STRUCTURES if not any(x.startswith(s + "@") for x in building)}
    if "barracks" in options and not have.get("Barracks"):
        options["barracks"] = "build a barracks: we have none, and soldiers come from it"
    options["none"] = "build nothing now"
    pop, cap = (b.get("pop") or [0, 0])[:2]
    text = f"Build something? Population {pop}/{cap}; buildings {have or 'unknown'}."
    out.append(LunaQuestion(choice("build", text, options), lambda v: []))
    return out


def luna_orders(b: dict[str, Any], questions: list[LunaQuestion], answers: dict[str, Any]) -> dict[str, list]:
    """Orders by question name from Luna's answers (train and build use the aggression score)."""
    orders = b.get("orders") or {}
    pending = orders.get("train") or {}
    level = answers.get("aggression")
    batch = AGGRESSION_BATCH[min(2, max(0, round(level.score)))] if level and level.score is not None else 5
    out: dict[str, list] = {}
    for q in questions:
        a = answers.get(q.question.name)
        if a is None or a.choice is None:
            continue
        name = q.question.name
        if name == "train":
            role = a.choice
            size = 5 if role == "villager" else batch
            if role != "none" and pending.get(role, 0) < size:
                out[name] = [("train", {"role": role, "count": size})]
        elif name == "build":
            if a.choice != "none":
                out[name] = [("build", {"structure": a.choice, "where": "base"})]
        else:
            out[name] = q.to_orders(a.choice)
    return {k: v for k, v in out.items() if v}


@dataclass
class LunaSettings:
    interval_s: float = 1.0
    poll_s: float = 1.0


class LunaPlayer:
    def __init__(
        self,
        client: DecisionsClient,
        settings: LunaSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.client = client
        self.settings = settings or LunaSettings()
        self.record = BrainRecord()
        self._clock, self._sleep = clock, sleep
        self._last: dict[str, float] = {}

    async def play(self, link: SeatLink) -> BrainRecord:
        s = self.settings
        await link.brief(compact=True)  # the first call marks the seat ready; the match starts when every agent is
        while True:
            tick = self._clock()
            try:
                st = await link.status()
            except httpx.HTTPError as e:
                self.record.end_reason, self.record.last_error = "env_gone", repr(e)
                return self.record
            if st.get("over") or st.get("playing_out"):
                self.record.end_reason = "match_over"
                return self.record
            if not st.get("started", True):  # nothing to decide (and nothing to pay for) before the start
                await self._sleep(s.poll_s)
                continue
            try:
                b = json.loads(await link.brief(compact=True))
            except Exception as e:  # noqa: BLE001 - no brief, no tick; the next second tries again
                self.record.last_error = f"brief failed: {type(e).__name__}: {str(e)[:200]}"
                await self._sleep(max(0.0, s.interval_s - (self._clock() - tick)))
                continue
            if not await self.turn(link, b, waiting=bool(st.get("open"))):
                self.record.end_reason = "spend_cap (" + await _wait_over(link, s.poll_s, self._sleep) + ")"
                return self.record
            await self._sleep(max(0.0, s.interval_s - (self._clock() - tick)))

    async def turn(self, link: SeatLink, b: dict[str, Any], *, waiting: bool = False) -> bool:
        """One Decisions request and its orders. False when the spend cap stops the brain. `waiting`: the seat had a
        decision open when the tick began, so it waits on this request until its orders are applied."""
        if b.get("over"):
            return True
        if b.get("ready") is False or b.get("state", "active") != "active":
            return True
        if waiting:
            await link.waiting("begin")
        try:
            return await self._decide_and_apply(link, b)
        finally:
            if waiting:
                await link.waiting("end")

    async def _decide_and_apply(self, link: SeatLink, b: dict[str, Any]) -> bool:
        questions = luna_questions(b)
        context = LUNA_SYSTEM + "\nEmpire now: " + json.dumps(b, separators=(",", ":"))
        started = time.perf_counter()
        try:
            decision = await self.client.decide(context, [q.question for q in questions])
        except SpendCapReached as e:
            self.record.last_error = str(e)
            return False
        except Exception as e:  # noqa: BLE001 - a failed tick is reported as missed; the next second tries again
            self.record.calls += 1
            self.record.errors += 1
            self.record.last_error = f"{type(e).__name__}: {str(e)[:300]}"
            usd = round(self.client.guard.last_usd, 6)
            await link.report_call(
                time.perf_counter() - started, self.record.last_error, usd=usd, model=f"openai/{self.client.model}"
            )
            return True
        now = self._clock()
        sent = []
        for q in questions:
            a = decision.answers.get(q.question.name)
            if q.held and a is not None and a.choice == "wait" and now - self._last.get(f"held:{q.held}", -1e9) >= 10:
                self._last[f"held:{q.held}"] = now
                await link.hold(q.held)
                sent.append(f"held {q.held}")
        for name, calls in luna_orders(b, questions, decision.answers).items():
            if now - self._last.get(name, -1e9) < COOLDOWN_S.get(name, 0.0):
                continue
            self._last[name] = now
            for tool, args in calls:
                result = await _safe_call(link, tool, args)
                status = _status_of(result)
                self.record.orders += 1
                if status == "rejected":
                    self.record.rejected += 1
                    self.record.last_error = _message_of(result)
                sent.append(f"{tool}({json.dumps(args, separators=(',', ':'))}) -> {status}")
        self.record.calls += 1
        self.record.latencies_s.append(decision.seconds)
        await link.report_call(decision.seconds, None, usd=round(decision.usd, 6), model=f"openai/{self.client.model}")
        if sent:
            self.record.turns.append({"t": b.get("t"), "orders": sent})
        return True


def summary(record: BrainRecord, agent: str, model: str, spend: dict[str, Any]) -> dict[str, Any]:
    lat = sorted(record.latencies_s)
    return {
        "agent": agent,
        "model": model,
        "end_reason": record.end_reason,
        "calls": record.calls,
        "errors": record.errors,
        "orders": record.orders,
        "rejected": record.rejected,
        "latency_p50_s": round(lat[len(lat) // 2], 3) if lat else None,
        "last_error": record.last_error,
        "spend": spend,
    }
