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
  ordered), from the call's start until its orders are applied; a routine call with nothing open is no wait.
- `LunaPlayer` (zeroad-luna, Setup B): once the match has started, every `interval_s` (1 s) read the compact JSON brief
  and send one Decisions request with strategic and tactical questions (economy split, phase, army posture, what to
  train and build, how aggressively), each answer mapped to an order tool. An order is only sent when it changes the
  standing orders, and each question has a cooldown, so a steady answer doesn't repeat the same order every second. Like
  the strategist, the seat only waits on a tick that began with a decision open. Its phase and attack questions keep
  "not yet" all match; a "not yet" past the plan's deadline is reported as a hold (`POST .../held`), and the env's
  autopilot carries out the default, logged as the autopilot's.

Each stops before its spend cap (`SpendGuard`) and keeps polling until the match is over, so the seat plays on with
its executor; it stops at once when the match is over or the env goes away. A failed model call, order call or brief
costs one turn (a failed order is recorded as rejected), never the rest of the match.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentenv_zeroad.autopilot import HELD_ATTACK_MIN, HELD_DEADLINES
from agentenv_zeroad.decisions import DecisionsClient, Question, choice, score
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendCapReached

log = logging.getLogger(__name__)

ORDER_TOOLS = ("set_economy", "train", "build", "research", "advance_phase", "army", "say")
MEMORY_TURNS = 3

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


# ---- the seat --------------------------------------------------------------------------------------------------------
class SeatLink:
    """A seat's MCP session and its plain HTTP routes. `session` is an initialized `mcp.ClientSession`."""

    def __init__(self, session: Any, seat_url: str, http: httpx.AsyncClient):
        self.session = session
        self.base = seat_url.rstrip("/").removesuffix("/mcp")
        self.http = http

    async def tools(self) -> list[ToolSpec]:
        listed = (await self.session.list_tools()).tools
        return [
            ToolSpec(t.name, t.description or "", t.inputSchema or {"type": "object", "properties": {}})
            for t in listed
            if t.name in ORDER_TOOLS
        ]

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
async def seat_link(server: dict[str, Any]) -> AsyncIterator[SeatLink]:
    """A link to the seat agent-env gives the agent: `{"url": .../players/<id>/mcp, "headers": {...}}`."""
    async with (
        httpx.AsyncClient(headers=dict(server.get("headers") or {}), timeout=30) as http,
        streamable_http_client(server["url"], http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield SeatLink(session, server["url"], http)


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
@dataclass
class CommanderSettings:
    interval_s: float = 25.0
    min_gap_s: float = 8.0
    poll_s: float = 1.0
    system_prompt: str = STRATEGIST_SYSTEM
    opening: str = ""
    max_calls: int | None = None


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

    async def play(self, link: SeatLink, *, reflex_key: tuple[str, str] | None = None) -> BrainRecord:
        s = self.settings
        tools = await link.tools()
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
            now = self._clock()
            due = st.get("started", True) and (
                last is None or now - last >= s.interval_s or (st.get("urgent", 0) > 0 and now - last >= s.min_gap_s)
            )
            if not due:
                await self._sleep(s.poll_s)
                continue
            if s.max_calls is not None and self.record.calls >= s.max_calls:
                self.record.end_reason = await _wait_over(link, s.poll_s, self._sleep)
                return self.record
            last = now
            if not await self.turn(link, tools, urgent=st.get("urgent", 0) > 0, waiting=bool(st.get("open"))):
                self.record.end_reason = await _wait_over(link, s.poll_s, self._sleep)
                self.record.end_reason = f"spend_cap ({self.record.end_reason})"
                return self.record

    async def turn(self, link: SeatLink, tools: list[ToolSpec], *, urgent: bool = False, waiting: bool = False) -> bool:
        """One brief -> model -> orders turn. False when the spend cap stops the brain. `waiting`: the seat had a
        decision open when the turn began, so it waits on this call until its orders are applied."""
        try:
            brief = await link.brief()
        except Exception as e:  # noqa: BLE001 - no brief, no turn; the next trigger tries again
            self.record.last_error = f"brief failed: {type(e).__name__}: {str(e)[:200]}"
            return True
        user = self._user(brief, urgent)
        if waiting:
            await link.waiting("begin")
        try:
            return await self._call_and_apply(link, user, tools)
        finally:
            if waiting:
                await link.waiting("end")

    async def _call_and_apply(self, link: SeatLink, user: str, tools: list[ToolSpec]) -> bool:
        started = time.perf_counter()
        try:
            reply = await self.model.call(self.settings.system_prompt, user, tools)
        except SpendCapReached as e:
            self.record.last_error = str(e)
            return False
        except Exception as e:  # noqa: BLE001 - a failed call is reported (missed) and the next trigger tries again
            seconds = time.perf_counter() - started
            self.record.calls += 1
            self.record.errors += 1
            self.record.last_error = f"{type(e).__name__}: {str(e)[:300]}"
            usd = round(self.model.guard.last_usd, 6)
            await link.report_call(seconds, self.record.last_error, usd=usd, model=self.model.model)
            return True
        orders = []
        for c in reply.calls:
            if c.error or c.name not in ORDER_TOOLS:
                result = json.dumps({"status": "rejected", "message": c.error or f"unknown tool {c.name}"})
            else:
                result = await _safe_call(link, c.name, c.arguments)
            status = _status_of(result)
            self.record.orders += 1
            if status == "rejected":
                self.record.rejected += 1
                self.record.last_error = _message_of(result)
            orders.append(f"{c.name}({json.dumps(c.arguments, separators=(',', ':'))}) -> {status}")
        seconds = time.perf_counter() - started
        self.record.calls += 1
        self.record.latencies_s.append(reply.seconds)
        await link.report_call(
            reply.seconds,
            None,
            usd=round(reply.usd, 6),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            model=self.model.model,
        )
        self._memory = (self._memory + ["; ".join(orders) or "(no orders)"])[-MEMORY_TURNS:]
        self.record.turns.append({"seconds": round(seconds, 2), "orders": orders, "text": reply.text[:500]})
        return True

    def _user(self, brief: str, urgent: bool) -> str:
        parts = []
        if self.settings.opening:
            parts.append(self.settings.opening)
        parts.append(brief)
        if self._memory:
            parts.append("Your orders in your last turns (oldest first):\n" + "\n".join(self._memory))
        parts.append(
            ("URGENT: something happened that needs an answer now. " if urgent else "")
            + "Decide your orders now and call the tools."
        )
        return "\n\n".join(parts)


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
