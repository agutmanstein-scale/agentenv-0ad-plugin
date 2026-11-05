"""0 A.D. as an AgentEnv game env (spec §4.6, §4.7, §4.9): a lobby of seats, a live match, and MCP tools per seat.

The env is built on agentenv-game-env: its lobby (`urn:game:lobby/v1`) takes 2-8 player slots, `"1"` to `"8"`, the
engine's player numbers. A slot is one of
- an `agent`: a brain (model, Luna, a test's fake agent) that plays at the slot's own MCP address,
  `/players/<id>/mcp`, through `get_brief` and one tool per §4.5 order; the executor and autopilot carry out its
  orders every tick, so the empire keeps running however slowly the brain answers;
- an `ai` slot whose `controller` is `petra` (0 A.D.'s own AI, in the engine) or `bot` (U3's scripted bot, which
  gives orders through the same executor as a brain, at $0). The lobby protocol's player kinds are agent, human and
  ai; a scripted bot is the game's own AI here, so it is an `ai` slot.

Closing the lobby starts the engine and resets it to the match; the match (`urn:game:match/v1`) starts once every
agent has made its first call, or after `start_grace_seconds`, and from then on the game clock runs on the wall clock
and never waits for anyone. It ends by conquest or at the time cap (`grading`), and `finish_match` plays it out with
no more brain orders (the executor and autopilot go on). Its files are the results (per-seat outcome, score, grade,
§5 metrics and call timings) and the replay (`commands.txt`, which 0 A.D. plays back).

Tool results follow the OpenCiv3 plugin's protocol (its docs/protocol.md): a JSON object with `status` (accepted,
rejected or partial), `message`, and for a rejection `code`, `alternatives` and `suggest`; every result ends with the
brief footer. Each call's wall time is recorded per seat. A brain reports each model call it makes, with its wall time
and error, at `POST /players/<id>/brain_calls`, which is how the forfeit rule (grading) sees a model that never answers.

Brains (U5) also use plain HTTP routes beside the seat's MCP endpoint, which a model can't call for itself:
`POST /players/<id>/waiting` (`{"state": "begin" | "end"}`, the seat's wait on its brain, for `waiting_share`),
`GET /players/<id>/status` (match running or over, game time, pending and urgent events: what a brain polls between
calls without draining its events), and `POST /players/<id>/reflex` (`{"base_url", "api_key"}`: the seat's brain
hands its model endpoint to the seat's Luna reflex, when the slot sets `reflex: luna`; held in memory, never echoed
or written). And `POST /players/<id>/held` (`{"decision": "phase" | "attack"}`, U5c): the Luna player held a decision
it was asked about; past the plan's deadline the autopilot carries out the default, logged as the autopilot's.

Live view (U6, spec §4.8): a lobby with `render: true` plays its match on the stream engine (visual, `-rl-realtime`,
the observer's view, the director mod) on an X display with the overlay beside it, and records it (`live.py`); the
recording is a third match file, `recording` (an MP4), with the engine's log (`engine_log`: the director's cuts and
the frame rate). The overlay page is `GET /live` (`?stream`: the 480x1080
panel the broadcast shows) and its data `GET /live/state.json`, which shows every empire without fog. It goes only to
the stream client: the env's own Chrome, inside the env container, on loopback, holding the match's stream token (made
per rendered match and put only in the URL the env opens itself). An agent never gets the token, and any request at
an agent's or a spectator's address (`/players/<id>/...`, `/spectators/...`) is refused even with it, so no agent can
read through its fog.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from agentenv_game import (
    AgentEnvGameEnv,
    Counter,
    GameError,
    Lobby,
    MatchFile,
    MatchFiles,
    MatchReport,
    MatchStatus,
    Moves,
    PlayerKind,
    PlayerSlot,
    PlayerSlotLimits,
    PlayerStatus,
    Score,
    begin_game,
    check_player_slot,
    create_game,
    match_files,
    match_report,
    play_out,
    player_slot_limits,
)
from agentenv_game.routing import current_player, current_spectator
from agentenv_protocol import environment_card, tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from agentenv_zeroad import grading
from agentenv_zeroad.engine import DEFAULT_IMAGE, EngineOptions, free_port, start_engine
from agentenv_zeroad.live import Broadcast, BroadcastSettings, overlay_html, overlay_state
from agentenv_zeroad.match import TICK_S, EngineHandle, LiveMatch, MatchConfig, SeatPlan
from agentenv_zeroad.orders import (
    MAX_TRAIN,
    AdvancePhase,
    Army,
    ArmyAction,
    Build,
    OrderResult,
    Research,
    Role,
    Say,
    SetEconomy,
    Train,
    _Order,
    rejected,
)
from agentenv_zeroad.roles import CIVS
from agentenv_zeroad.scenario import MAP_SIZES, MAX_PLAYERS, PLAYER_PLACEMENTS

log = logging.getLogger(__name__)

DEFAULT_CIVS = ("rome", "athen", "pers", "cart", "brit", "iber", "mace", "maur")
FILE_KINDS = {
    "results": ("results.json", "application/json"),
    "replay": ("commands.txt", "text/plain"),
    "recording": ("recording.mp4", "video/mp4"),
    "engine_log": ("engine-log.html", "text/html"),
}
RT_IMAGE = "zeroad-engine:0.28-rt"
ROLES = list(Role.__args__)  # type: ignore[attr-defined]
ACTIONS = list(ArmyAction.__args__)  # type: ignore[attr-defined]
WHERE_HELP = "base, front, near_<food|wood|stone|metal> or toward_<player> (e.g. toward_2)"
TARGET_HELP = "a player (player_2 or 2), center, front or home"


class MatchSettings(BaseModel):
    """A lobby's game_settings: the map, the pace and how the match ends."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    map: str = "mainland"
    """A random map script under maps/random (mainland: a small land map)."""
    size: int = 192
    """Map size in tiles: 128, 192, 256, 320, 384, 448 or 512."""
    seed: int = Field(7, ge=0)
    biome: str = "generic/temperate"
    placement: str = "circle"
    """Player placement: circle, river, groupedLines, randomGroup or stronghold."""
    speed: float = Field(1.0, ge=1.0, le=8.0)
    """Game speed: 1.0 steps the engine every 200 ms of wall time (real time), 2.0 every 100 ms, up to 8.0 (25 ms) for
    an unrendered match. The executors tick once a game second at any speed."""
    time_cap_minutes: float = Field(40.0, gt=0, le=720)
    """Game minutes until the time cap, where the highest engine score wins."""
    draw_margin: float = Field(0.0, ge=0)
    """At the time cap, players within this many score points of the top draw."""
    fog: bool = True
    """Each player sees only what its units see (the observer always sees all)."""
    start_grace_seconds: float = Field(30.0, ge=0, le=600)
    """With agents, the match starts when every agent has made its first call, or after this many seconds."""
    render: bool = False
    """Render the match for the live stream and record it (the zeroad-live env): the observer's view, moved by the
    director, beside the overlay, recorded to an MP4. Needs speed 1.0."""
    rtmp: bool = False
    """With render: also push the stream over RTMP to the URL in the env's ZEROAD_RTMP_URL. Off unless asked for."""

    @model_validator(mode="after")
    def _render(self) -> MatchSettings:
        if self.render and self.speed != 1.0:
            raise ValueError("a rendered match runs at speed 1.0 (the engine advances it in real time)")
        if self.rtmp and not self.render:
            raise ValueError("rtmp needs render: true")
        if self.rtmp and not os.environ.get("ZEROAD_RTMP_URL"):
            raise ValueError("rtmp needs ZEROAD_RTMP_URL in the env's environment; the URL is never a lobby setting")
        return self

    @field_validator("size")
    @classmethod
    def _size(cls, value: int) -> int:
        if value not in MAP_SIZES:
            raise ValueError(f"size must be one of {MAP_SIZES}")
        return value

    @field_validator("placement")
    @classmethod
    def _placement(cls, value: str) -> str:
        if value not in PLAYER_PLACEMENTS:
            raise ValueError(f"placement must be one of {PLAYER_PLACEMENTS}")
        return value


class SlotSettings(BaseModel):
    """A player slot's game_settings."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    civ: str | None = None
    """The civilization (rome, athen, pers, cart, ...); by default rome, athen, pers, cart, ... by player number."""
    controller: Literal["petra", "bot"] | None = None
    """For an ai slot: petra (0 A.D.'s AI, the default) or bot (the scripted bot). Not for an agent slot."""
    ai_difficulty: int = Field(3, ge=0, le=5)
    """Petra's level, 0 (sandbox) to 5 (very hard)."""
    label: str | None = Field(None, max_length=40)
    """The seat's display name; by default the agent's name, or Petra/Bot and the player number."""
    reflex: Literal["luna"] | None = None
    """luna: GPT-6 Luna reflexes inside the env for this seat (Setup A); for agent and bot seats, not Petra."""
    reflex_max_usd: float | None = Field(None, gt=0, le=100)
    """The reflex's spend cap for the match, in estimated USD; it stops asking before a call could pass it."""

    @field_validator("civ")
    @classmethod
    def _civ(cls, value: str | None) -> str | None:
        if value is not None and value not in CIVS:
            raise ValueError(f"civ must be one of {', '.join(CIVS)}")
        return value


def engine_mode() -> Literal["local", "docker"]:
    """`local` runs pyrogenesis from PATH (the env image); `docker` runs the engine image (a dev machine)."""
    mode = os.environ.get("ZEROAD_ENGINE")
    if mode in ("local", "docker"):
        return mode  # type: ignore[return-value]
    return "local" if shutil.which("pyrogenesis") or Path("/usr/games/pyrogenesis").exists() else "docker"


async def default_engine() -> EngineHandle:
    options = EngineOptions(port=free_port())
    return await start_engine(options, mode=engine_mode(), image=os.environ.get("ZEROAD_ENGINE_IMAGE", DEFAULT_IMAGE))


async def render_engine(broadcast: Broadcast) -> EngineHandle:
    """The stream engine on the broadcast's display: visual, real time, the director mod, the observer's view."""
    options = EngineOptions(
        port=free_port(), visual=True, display=broadcast.display, realtime=True, extra_args=broadcast.engine_args()
    )
    image = os.environ.get("ZEROAD_ENGINE_IMAGE", RT_IMAGE)
    return await start_engine(options, mode=engine_mode(), image=image, ready_timeout=180.0)


LOOPBACK = frozenset({"127.0.0.1", "::1"})


def overlay_url(token: str) -> str:
    """The overlay as the env's own Chrome opens it: on loopback, with the match's stream token."""
    return f"http://127.0.0.1:{os.environ.get('MCP_PORT', '18765')}/live?stream&token={token}"


def _result(result: OrderResult, footer: str) -> str:
    return json.dumps({**result.to_json(), "footer": footer})


@environment_card(name="zeroad")
class ZeroADEnv(AgentEnvGameEnv):
    """One 0 A.D. match at a time: the lobby's close starts an engine for it; the next lobby replaces it."""

    GameSettings = MatchSettings
    PlayerSlotSettings = SlotSettings

    def __init__(
        self,
        start: Callable[[], Any] = default_engine,
        out_root: Path | None = None,
        *,
        tick_s: float = TICK_S,
        match_options: dict[str, Any] | None = None,
        start_rendered: Callable[[Broadcast], Any] = render_engine,
        broadcast_for: Callable[[MatchSettings | dict, Path], Broadcast] | None = None,
    ) -> None:
        self.start_engine = start
        self.start_rendered = start_rendered
        self.broadcast_for = broadcast_for or _broadcast
        # Only the live env image (streamer/Dockerfile sets ZEROAD_LIVE=1) has the stream engine, Chrome and the
        # director mod; the plain env would start its stock engine with flags and a mod it lacks.
        self.can_render = start_rendered is not render_engine or os.environ.get("ZEROAD_LIVE") == "1"
        self.tick_s = tick_s
        self.match_options = dict(match_options or {})
        self.out_root = out_root or Path(os.environ.get("ZEROAD_OUT") or tempfile.mkdtemp(prefix="zeroad-"))
        self.live: LiveMatch | None = None
        self.stream_token: str | None = None  # the rendered match's; only in the overlay URL the env opens itself
        self.playing_out = False
        self._grace: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    # ---- the lobby -------------------------------------------------------------------------------------------------
    @player_slot_limits
    def slot_limits(self, game_settings: MatchSettings, requested: PlayerSlotLimits) -> PlayerSlotLimits:
        """Two to eight players: agents and the game's own AI (Petra or the scripted bot)."""
        if requested.max is not None and requested.max > MAX_PLAYERS:
            raise ValueError(f"a match has at most {MAX_PLAYERS} players, not {requested.max}")
        kinds = [PlayerKind.AGENT, PlayerKind.AI]
        return PlayerSlotLimits(min=max(requested.min or 2, 2), max=requested.max or MAX_PLAYERS, player_kinds=kinds)

    @check_player_slot
    def slot_rules(self, slot: PlayerSlot, lobby: Lobby) -> None:
        """Player slots are the engine's player numbers, "1" to "8"; only an ai slot has a controller."""
        if not (slot.player_id.isdigit() and 1 <= int(slot.player_id) <= MAX_PLAYERS):
            message = f"player slots are the player numbers 1 to {MAX_PLAYERS}, not {slot.player_id!r}"
            raise GameError("bad_slot", message)
        if slot.player_kind is PlayerKind.AGENT and slot.game_settings.get("controller"):
            raise ValueError("controller is for ai slots (petra or bot); an agent slot is played by its agent")
        petra = slot.player_kind is PlayerKind.AI and (slot.game_settings.get("controller") or "petra") == "petra"
        if petra and slot.game_settings.get("reflex"):
            raise ValueError("reflex is for seats the executor drives (agent or bot); Petra plays on its own")

    @create_game
    async def new_match(self, lobby: Lobby) -> None:
        """Start an engine and reset it to the lobby's match: one seat per slot, by player number."""
        await self._drop_match()
        s = lobby.game_settings
        slots = sorted(lobby.player_slots, key=lambda p: int(p.player_id))
        numbers = [int(p.player_id) for p in slots]
        if numbers != list(range(1, len(slots) + 1)):
            raise ValueError(f"player slots must be 1 to {len(slots)} without gaps, not {numbers}")
        seats = tuple(self._plan(p) for p in slots)
        config = MatchConfig(
            seats=seats,
            map_name=s["map"],
            biome=s["biome"],
            placement=s["placement"],
            size=s["size"],
            seed=s["seed"],
            speed=s["speed"],
            time_cap_s=s["time_cap_minutes"] * 60,
            draw_margin=s["draw_margin"],
            fog=s["fog"],
            tick_s=self.tick_s / s["speed"],
        )
        out_dir = self.out_root / lobby.lobby_id
        broadcast = self.broadcast_for(s, out_dir) if s.get("render") else None
        try:
            if broadcast is not None:
                await broadcast.open_display()
                handle = await self.start_rendered(broadcast)
            else:
                handle = await self.start_engine()
        except BaseException:
            if broadcast is not None:
                await broadcast.stop()
            raise
        live = LiveMatch(handle, config, out_dir=out_dir, broadcast=broadcast, **self.match_options)
        token = secrets.token_urlsafe(24) if broadcast is not None else None
        try:
            await live.setup()
            if token is not None:
                await live.start_broadcast(overlay_url(token))
        except BaseException:
            if broadcast is not None:
                await broadcast.stop()
            await handle.close()
            raise
        self.live, self.playing_out, self.stream_token = live, False, token
        log.info("NEW MATCH %s: %s", lobby.lobby_id, ", ".join(f"{p.slot}={p.kind}:{p.civ}" for p in seats))
        if any(p.kind == "agent" for p in seats):
            self._grace = asyncio.create_task(self._start_after(s["start_grace_seconds"]))

    @staticmethod
    def _plan(slot: PlayerSlot) -> SeatPlan:
        g, n = slot.game_settings, int(slot.player_id)
        if slot.player_kind is PlayerKind.AGENT:
            kind, default_name = "agent", slot.player_name or f"Agent {n}"
        else:
            kind = g.get("controller") or "petra"
            default_name = f"{'Bot' if kind == 'bot' else 'Petra'} {n}"
        return SeatPlan(
            slot=slot.player_id,
            player=n,
            kind=kind,
            civ=g.get("civ") or DEFAULT_CIVS[n - 1],
            name=g.get("label") or default_name,
            ai_difficulty=g.get("ai_difficulty", 3),
            reflex=g.get("reflex"),
            reflex_max_usd=g.get("reflex_max_usd"),
        )

    async def _start_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        match = self.match
        if match is not None and match.status is MatchStatus.NOT_STARTED:
            missing = [p for p, st in match.player_states.items() if st.status is PlayerStatus.NOT_READY]
            await self.begin_match(f"started after {seconds:.0f} s without {', '.join(missing)}")

    async def _drop_match(self) -> None:
        if self._grace is not None:
            self._grace.cancel()
            self._grace = None
        if self.live is not None:
            live, self.live = self.live, None
            await live.stop("replaced by a new lobby")

    def new_lobby(self, game_settings: dict | None = None, player_slot_limits: Any = None) -> Lobby:
        """A new lobby drops the last match; its engine is stopped in the background."""
        settings = self._settings(self.GameSettings, {} if game_settings is None else game_settings)
        if settings.render and not self.can_render:
            raise GameError(
                "render_unavailable",
                "render: true needs the live env (zeroad-live, from `agent-env zeroad setup --live`); "
                "this env has no stream engine, overlay browser or director mod",
            )
        lobby = super().new_lobby(game_settings, player_slot_limits)
        if self.live is not None:
            live, self.live = self.live, None
            self._background(live.stop("replaced by a new lobby"))
        return lobby

    def _background(self, work: Any) -> None:
        """Run `work` (a coroutine) on the running loop, keeping a reference so it isn't collected mid-way."""
        task = asyncio.get_running_loop().create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- the match -------------------------------------------------------------------------------------------------
    @begin_game
    async def run_clock(self) -> None:
        """The clock starts: from now on the game runs on the wall clock and waits for no one."""
        if self._grace is not None and self._grace is not asyncio.current_task():
            self._grace.cancel()
        assert self.live is not None
        self.live.start()

    @play_out
    async def run_out(self) -> None:
        """finish: no more brain orders; the executors and autopilot play on until conquest or the time cap."""
        self.playing_out = True
        assert self.live is not None
        await self.live.wait_end()

    def cancel_match(self, lobby_id: str | None = None):
        match = super().cancel_match(lobby_id)
        if self.live is not None and not self.live.over:
            self._background(self.live.stop("cancelled"))
        return match

    @match_report
    def report(self) -> MatchReport:
        """Game time against the cap; outcomes from conquest or the time cap (a defeated player has lost at once);
        each player's engine score (total, economy, military, exploration); and each agent's brain calls as moves."""
        live = self.live
        if live is None:
            return MatchReport()
        status = "failed" if live.failed else "finished" if live.outcomes is not None else "started"
        running = live.started and not live.over
        outcomes = live.outcomes or {slot: "lost" for slot, state in live.states.items() if state == "defeated"}
        scores = {
            slot: [
                Score(name="score", value=sc["total"], better="higher", unit="points"),
                Score(name="economy", value=sc["economy"], better="higher", unit="points"),
                Score(name="military", value=sc["military"], better="higher", unit="points"),
                Score(name="exploration", value=sc["exploration"], better="higher", unit="points"),
            ]
            for slot, sc in live.scores.items()
        }
        moves = {
            slot: Moves(made=s.moves()[0], missed=s.moves()[1]) for slot, s in live.seats.items() if s.graded_by_calls
        }
        return MatchReport(
            status=status,
            status_detail=live.detail(),
            progress=[
                Counter(
                    name="game",
                    unit="seconds",
                    value=round(live.time_ms / 1000, 1),
                    limit=live.config.time_cap_s,
                    rate=live.config.speed if running else 0,
                )
            ],
            outcomes=outcomes,
            scores=scores,
            moves=moves,
        )

    @match_files
    async def kept(self, kinds: list[str] | None) -> MatchFiles:
        """`results` (results.json: outcome, score, grade, metrics and timings per seat) and `replay` (the engine's
        commands.txt, which 0 A.D. plays back); both by default."""
        rendered = ["recording", "engine_log"] if self.live and self.live.broadcast else []
        wanted = kinds or ["results", "replay", *rendered]
        if unknown := sorted(set(wanted) - set(FILE_KINDS)):
            raise ValueError(f"0 A.D. keeps a match's {', '.join(FILE_KINDS)}, not {', '.join(unknown)}")
        if self.live is None:
            return MatchFiles(notes=["no match was played"])
        await self.live.finalize()
        files, notes = [], []
        lobby_id = self.lobby.lobby_id or "match"
        for kind in wanted:
            path = self.live.files.get(kind)
            name, content_type = FILE_KINDS[kind]
            if path is None or not path.exists():
                notes.append(f"no {kind}: the engine did not write one")
                continue
            files.append(MatchFile(name=f"zeroad-{lobby_id}-{name}", kind=kind, content_type=content_type, file=path))
        return MatchFiles(files=files, notes=notes)

    # ---- tools ---------------------------------------------------------------------------------------------------
    # Tools have no return annotation: `-> str` makes FastMCP send every result twice (text and structuredContent).

    async def _seat(self, tool_name: str) -> tuple[str | None, str | None]:
        """The slot this call plays, or the rejection to answer instead."""
        try:
            slot = self.player()
        except GameError as e:
            return None, json.dumps(rejected(e.code, e.message, suggest="connect at /players/<id>/mcp").to_json())
        if slot is None:
            ids = [p.player_id for p in self.lobby.player_slots if p.player_kind is PlayerKind.AGENT]
            return None, json.dumps(
                rejected(
                    "no_seat",
                    "This address plays no seat: each agent plays at its own player slot's address.",
                    [f"/players/{i}/mcp" for i in ids],
                ).to_json()
            )
        if self.live is None:
            refusal = rejected("no_match", "No match is being played: the lobby has not closed.")
            return None, json.dumps(refusal.to_json())
        if self.match is not None and self.match.status is MatchStatus.NOT_STARTED:
            await self.player_ready(slot.player_id)
        return slot.player_id, None

    async def _order(self, tool_name: str, build: Callable[[], _Order]):
        started = time.perf_counter()
        slot, refusal = await self._seat(tool_name)
        if refusal is not None:
            return refusal
        assert slot is not None and self.live is not None
        live = self.live
        if live.over or self.playing_out:
            result = rejected("match_over", f"The match is over: {live.detail() or 'it is being played out'}.")
        else:
            try:
                order = build()
                result = live.submit(slot, order)
            except ValidationError as e:
                result = rejected(
                    "bad_arguments", "; ".join(err["msg"] for err in e.errors()), suggest="get_brief() for the state"
                )
        live.record_tool_call(slot, tool_name, time.perf_counter() - started, result.status)
        return _result(result, live.footer(slot))

    @tool()
    async def get_brief(
        self,
        compact: Annotated[bool, Field(description="true: the compact JSON form (numbers and ids, no prose).")] = False,
    ):
        """Your empire now: time, phase, population, resources and gatherers, army by role, idle units, buildings,
        research, the enemies you can see, the events since your last brief (urgent ones first), your standing orders,
        and how the match ends (conquest, or the highest score at the time cap). Call it first, and whenever a
        footer says there are new events."""
        started = time.perf_counter()
        slot, refusal = await self._seat("get_brief")
        if refusal is not None:
            return refusal
        assert slot is not None and self.live is not None
        text = self.live.compact_brief(slot) if compact else self.live.brief(slot)
        self.live.record_tool_call(slot, "get_brief", time.perf_counter() - started, "accepted")
        return text

    @tool()
    async def set_economy(
        self,
        food: Annotated[int, Field(ge=0, le=100, description="Percent of gatherers on food.")],
        wood: Annotated[int, Field(ge=0, le=100, description="Percent on wood.")],
        stone: Annotated[int, Field(ge=0, le=100, description="Percent on stone.")],
        metal: Annotated[int, Field(ge=0, le=100, description="Percent on metal.")],
    ):
        """Split your gatherers between the four resources, in percent; the four must add up to 100. Idle and surplus
        gatherers move over the next seconds."""
        return await self._order("set_economy", lambda: SetEconomy(food=food, wood=wood, stone=stone, metal=metal))

    @tool()
    async def train(
        self,
        role: Annotated[Role, Field(description="villager, infantry_melee, infantry_ranged, cavalry, siege, champion")],
        count: Annotated[int, Field(ge=1, le=MAX_TRAIN, description="How many more units of the role to train.")],
        at: Annotated[str | None, Field(description='Only at this kind of building, e.g. "barracks".')] = None,
    ):
        """Train `count` more units of a role, mapped to your civilization's units; batches go to free buildings as
        population room and resources allow."""
        return await self._order("train", lambda: Train(role=role, count=count, at=at))

    @tool()
    async def build(
        self,
        structure: Annotated[str, Field(min_length=1, max_length=40, description='e.g. "house", "barracks".')],
        where: Annotated[str, Field(description=f"Where to place it: {WHERE_HELP}.")] = "base",
    ):
        """Place one building; villagers build it. The spot is found near the anchor `where` names."""
        return await self._order("build", lambda: Build(structure=structure, where=where))

    @tool()
    async def research(
        self, tech: Annotated[str, Field(min_length=1, max_length=80, description='e.g. "gather_lumbering_ironaxes".')]
    ):
        """Research one technology by its engine name, at a building that offers it."""
        return await self._order("research", lambda: Research(tech=tech))

    @tool()
    async def advance_phase(self):
        """Advance to the next phase (village, town, city) as soon as its requirements and resources allow."""
        return await self._order("advance_phase", AdvancePhase)

    @tool()
    async def army(
        self,
        action: Annotated[ArmyAction, Field(description="gather, attack, raid, defend, retreat or scout.")],
        target: Annotated[str, Field(description=f"The target: {TARGET_HELP}.")] = "home",
    ):
        """Set your army's posture: gather (soldiers work), attack or raid a player, defend a place, retreat home, or
        scout."""
        return await self._order("army", lambda: Army(action=action, target=target))

    @tool()
    async def say(self, text: Annotated[str, Field(min_length=1, max_length=200, description="One short line.")]):
        """Say a short line in character, shown on the live overlay. Optional."""
        return await self._order("say", lambda: Say(text=text))

    # ---- brain calls (U5) and routes ------------------------------------------------------------------------------
    def create_app(self) -> Any:
        app = super().create_app()
        app.custom_route("/brain_calls", methods=["POST"])(self._brain_calls)
        app.custom_route("/waiting", methods=["POST"])(self._waiting)
        app.custom_route("/status", methods=["GET"])(self._status)
        app.custom_route("/reflex", methods=["POST"])(self._reflex)
        app.custom_route("/held", methods=["POST"])(self._held)
        app.custom_route("/live", methods=["GET"])(self._live_page)
        app.custom_route("/live/state.json", methods=["GET"])(self._live_state)
        return app

    async def _live_page(self, request: Request) -> HTMLResponse:
        """The overlay page; `?stream` lays it out as the broadcast's 480x1080 panel."""
        return HTMLResponse(overlay_html())

    async def _live_state(self, request: Request) -> JSONResponse:
        """The overlay's data: match clock, each empire's panel, scoreboard. No keys or prompts, but every empire
        without fog, so only for the stream client (spec §4.3: agents play under fog; the stream has none)."""
        if current_player() is not None or current_spectator():
            return self._forbidden("the overlay's data is not served at a player's or spectator's address")
        if self.live is None or self.live.broadcast is None or self.stream_token is None:
            return JSONResponse({"status": "no_match", "seats": [], "scoreboard": []})
        host = request.client.host if request.client is not None else ""
        token = request.query_params.get("token", "")
        if host not in LOOPBACK or not hmac.compare_digest(token.encode(), self.stream_token.encode()):
            return self._forbidden("the overlay's data goes only to the stream's own browser")
        return JSONResponse(overlay_state(self.live), headers={"Cache-Control": "no-store"})

    @staticmethod
    def _forbidden(message: str) -> JSONResponse:
        return JSONResponse({"ok": False, "error": {"code": "forbidden", "message": message}}, status_code=403)

    def _route_seat(self, route: str) -> tuple[str | None, JSONResponse | None]:
        """The seat a /players/<id>/<route> request is for, or the error response."""
        try:
            slot = self.player()
        except GameError as e:
            return None, JSONResponse({"ok": False, "error": {"code": e.code, "message": e.message}}, status_code=404)
        if slot is None or self.live is None:
            message = f"use /players/<id>/{route} while a match is played"
            return None, JSONResponse({"ok": False, "error": {"code": "no_seat", "message": message}}, status_code=400)
        return slot.player_id, None

    @staticmethod
    def _bad(message: str) -> JSONResponse:
        return JSONResponse({"ok": False, "error": {"code": "bad_request", "message": message}}, status_code=400)

    async def _waiting(self, request: Request) -> JSONResponse:
        """A brain begins or ends a wait of its seat: `{"state": "begin"}` / `{"state": "end"}`."""
        slot, error = self._route_seat("waiting")
        if error is not None:
            return error
        try:
            body = WaitingBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            return self._bad('the body is {"state": "begin" | "end"}')
        assert slot is not None and self.live is not None
        self.live.record_waiting(slot, body.state)
        return JSONResponse({"ok": True})

    async def _status(self, request: Request) -> JSONResponse:
        """The match as a brain polls it between calls; reading it drains no events."""
        slot, error = self._route_seat("status")
        if error is not None:
            return error
        assert slot is not None and self.live is not None
        return JSONResponse({"ok": True, **self.live.status(slot), "playing_out": self.playing_out})

    async def _reflex(self, request: Request) -> JSONResponse:
        """The seat's brain hands its model endpoint to the seat's Luna reflex. The key is never echoed or logged."""
        slot, error = self._route_seat("reflex")
        if error is not None:
            return error
        try:
            body = ReflexBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            return self._bad('the body is {"base_url": <url>, "api_key": <key>}')
        assert slot is not None and self.live is not None
        enabled = self.live.set_reflex_credentials(slot, body.base_url, body.api_key)
        return JSONResponse({"ok": True, "reflex": "luna" if enabled else None})

    async def _held(self, request: Request) -> JSONResponse:
        """The seat's brain held a decision it was asked about: `{"decision": "phase" | "attack"}`. Past the plan's
        deadline the autopilot carries out the default, logged as "autopilot (<brain> held)"."""
        slot, error = self._route_seat("held")
        if error is not None:
            return error
        try:
            body = HeldBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            return self._bad('the body is {"decision": "phase" | "attack"}')
        assert slot is not None and self.live is not None
        if self.playing_out:
            return JSONResponse({"ok": True, "applied": False, "reason": "the match is being played out"})
        return JSONResponse({"ok": True, **self.live.record_hold(slot, body.decision)})

    async def _brain_calls(self, request: Request) -> JSONResponse:
        """A brain reports one model call of its seat: `{"seconds": 12.5, "error": null}`, optionally with its
        `usd` estimate, `input_tokens`, `output_tokens`, `model` and `turn` (the call's logged turn, kept in
        results.json for training). At /players/<id>/."""
        slot_id, error = self._route_seat("brain_calls")
        if error is not None:
            return error
        try:
            body = BrainCallBody.model_validate(await request.json())
        except (ValueError, ValidationError) as e:
            message = f'the body is {{"seconds": <finite number >= 0>, "error": <string or null>}}: {e!r}'
            return JSONResponse({"ok": False, "error": {"code": "bad_request", "message": message}}, status_code=400)
        assert slot_id is not None and self.live is not None
        call = grading.BrainCall(seconds=body.seconds, error=body.error, game_ms=self.live.time_ms)
        tokens = (body.input_tokens, body.output_tokens) if body.input_tokens is not None else None
        self.live.record_brain_call(slot_id, call, usd=body.usd, tokens=tokens, model=body.model, turn=body.turn)
        made, missed = self.live.seats[slot_id].moves()
        return JSONResponse({"ok": True, "missed": call.missed, "made": made, "missed_total": missed})


class BrainCallBody(BaseModel):
    """One model call a brain reports; NaN, infinite or negative times are refused so they can't pass as on time."""

    model_config = ConfigDict(extra="ignore")

    seconds: float = Field(ge=0, allow_inf_nan=False)
    error: str | None = None
    usd: float | None = Field(None, ge=0, allow_inf_nan=False)
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    model: str | None = Field(None, max_length=100)
    turn: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _tokens_come_in_pairs(self) -> BrainCallBody:
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input_tokens and output_tokens come together or not at all")
        return self


class WaitingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["begin", "end"]


class HeldBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["phase", "attack"]


class ReflexBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(min_length=1, max_length=500, repr=False)


def _broadcast(settings: MatchSettings | dict, out_dir: Path) -> Broadcast:
    rtmp = settings["rtmp"] if isinstance(settings, dict) else settings.rtmp
    return Broadcast(BroadcastSettings.from_env(rtmp=bool(rtmp)), out_dir)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per engine step otherwise
    ZeroADEnv().serve()


if __name__ == "__main__":
    main()
