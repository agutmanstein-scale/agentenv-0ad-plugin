"""One lockstep training game: `LockstepMatch` is a `LiveMatch` whose clock is the `LockstepClock`, and `play_game`
plays a `GameSpec` on an engine handle from start to results.json.

What changes against a live match: no seat loop beside the clock (the clock runs the seat tick at its boundaries, in
game time), no Luna reflex (it counts wall seconds), and LLM seats are `CommanderSeat`s the clock asks for decisions.
Everything else is the live match's, so results.json has the schema the pilot reads (`seats`, with `turns` carrying
`prompt_token_ids`, `token_ids` and now `logprobs`; `timeline`; `outcome`; `metrics`). It adds:
- `commands`: every command applied to the engine, by the index of the step it went out with
  (`{"<step>": [[player, command], ...]}`), enough to replay the game on a fresh engine;
- `lockstep`: the latency, and for each decision when its brief was frozen, when its orders landed, and whether it
  was missed;
- `sampler`: per LLM seat, what its model did (`StrategistModel.sampler_stats()`: calls, failed calls, and how many
  replies had log-probs aligned with their token ids, misaligned or missing);
- `spec`: the game spec (without keys).

Each LLM seat's commander plays the seat's interface (`SeatSpec.interface`): its system prompt
(`brains.strategist_system`), with the seat's cost tables appended on v2, its opening (the pilot's, per interface)
and the tools the interface offers; the seat's plan gets the same interface, so its briefs and executor match.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from agentenv_zeroad.brains import Commander, CommanderSeat, CommanderSettings, InProcessSeatLink
from agentenv_zeroad.clock import DEFAULT_LATENCY_MS, LockstepClock
from agentenv_zeroad.grading import BRAIN_TIMEOUT_S
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.match import EngineHandle, LiveMatch, MatchConfig
from agentenv_zeroad.pilot.games import PROMPT, PROMPT_V2
from agentenv_zeroad.spend import LOCAL_PREFIX, SpendGuard
from agentenv_zeroad.train.spec import GameSpec, SeatSpec
from agentenv_zeroad.view import ExecutorView

log = logging.getLogger(__name__)

ModelFactory = Callable[[SeatSpec, str], StrategistModel]


class LockstepMatch(LiveMatch):
    """A match on a `LockstepClock`: `setup()` resets the engine, `add_brain` registers the seats the clock asks for
    decisions, `start()` runs the game; it ends on its own by conquest or the time cap, or by `stop()`."""

    def __init__(
        self,
        handle: EngineHandle,
        config: MatchConfig,
        *,
        latency_ms: int = DEFAULT_LATENCY_MS,
        brain_timeout_s: float = BRAIN_TIMEOUT_S,
        **kwargs: Any,
    ):
        if any(seat.reflex for seat in config.seats):
            raise ValueError("a lockstep match has no Luna reflex: it counts wall seconds")
        self.lockstep = LockstepClock(
            handle.engine,
            observe=self._lockstep_observe,
            act=self._lockstep_act,
            done=lambda: self.over,
            latency_ms=latency_ms,
            brain_timeout_s=brain_timeout_s,
        )
        super().__init__(handle, config, clock=self.lockstep, **kwargs)
        self.extra: dict[str, Any] = {}
        self.models: dict[str, StrategistModel] = {}
        self._observed: ExecutorView | None = None

    def start(self) -> None:
        """Run the clock; the match ends through its ticks (conquest, the time cap, or too many failed ticks)."""
        if self._clock_task is not None:
            return
        self.started_wall = self._now()
        self._clock_task = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        try:
            await self.clock.run()
        except Exception as e:  # noqa: BLE001 - a dead engine ends the match as failed, with its results
            log.exception("the lockstep clock stopped")
            self._fail(f"the engine stopped stepping: {e!r}")

    async def _lockstep_observe(self) -> bool:
        try:
            self._observed = await self._observe()
        except Exception as e:  # noqa: BLE001 - a bad read costs this tick, like the live seat loop
            self._tick_failed(e)
            return False
        return True

    async def _lockstep_act(self) -> None:
        try:
            await self._act(self._observed)
            self.failed_ticks = 0
        except Exception as e:  # noqa: BLE001
            self._tick_failed(e)

    def results(self) -> dict[str, Any]:
        commands: dict[str, list[list[Any]]] = {}
        for step, player, command in self.lockstep.commands:
            commands.setdefault(str(step), []).append([player, command])
        return {
            **super().results(),
            "commands": commands,
            "lockstep": {"latency_ms": self.lockstep.latency_ms, "decisions": self.lockstep.decisions},
            "sampler": {slot: model.sampler_stats() for slot, model in self.models.items()},
            **self.extra,
        }


def build_model(
    seat: SeatSpec,
    slot: str,
    *,
    environ: Mapping[str, str] | None = None,
    http: httpx.AsyncClient | None = None,
    cap_usd: float | None = None,
) -> StrategistModel:
    """The strategist model of an LLM seat: its key from the named variable, its sampling in the request body, and
    `return_token_ids` and `logprobs` so each turn keeps the token ids the sampler saw and chose and their
    log-probabilities (a `local/` model only: a frontier model's API has neither and may refuse the request).
    `cap_usd` caps what this seat's model may spend in the game (the guard stops it before a call that could pass
    the cap); a `local/` model costs nothing, so it never binds."""
    environ = os.environ if environ is None else environ
    assert seat.model and seat.base_url and seat.api_key_env
    key = environ.get(seat.api_key_env)
    if not key:
        raise ValueError(f"seat {slot}: the environment variable {seat.api_key_env} is not set")
    sampler = (
        {"return_token_ids": True, "logprobs": True, "top_logprobs": 0} if seat.model.startswith(LOCAL_PREFIX) else {}
    )
    body = {**sampler, **seat.sampling, **seat.extra_body}
    guard = SpendGuard(f"seat {slot} {seat.model}", cap_usd=cap_usd)
    return StrategistModel(
        seat.model,
        seat.base_url,
        key,
        guard,
        effort=seat.effort,
        max_tokens=seat.max_tokens,
        http=http,
        extra_body=body,
    )


async def play_game(
    spec: GameSpec,
    handle: EngineHandle,
    out_dir: Path,
    *,
    tools: Sequence[ToolSpec],
    model_for: ModelFactory | None = None,
    brain_timeout_s: float = BRAIN_TIMEOUT_S,
    match_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Play `spec` on `handle` in lockstep and return its results (also written to `out_dir/results.json`).

    `tools` are the tool schemas every LLM seat shows its model (`server.tool_specs()`); `model_for(seat, slot)` builds
    a seat's model (by default `build_model` with the process environment). An error before or during the game stops
    the match, which still writes what it has, and is re-raised.
    """
    model_for = model_for or build_model
    match = LockstepMatch(
        handle,
        spec.match_config(),
        out_dir=out_dir,
        latency_ms=spec.latency_ms,
        brain_timeout_s=brain_timeout_s,
        **dict(match_options or {}),
    )
    match.extra["spec"] = spec.to_json()
    brains: list[CommanderSeat] = []
    try:
        await match.setup()
        for slot, seat in enumerate(spec.seats, 1):
            if seat.kind == "llm":
                match.models[str(slot)] = model_for(seat, str(slot))
                brains.append(await _commander_seat(match, str(slot), seat, match.models[str(slot)], tools))
                match.lockstep.add_brain(brains[-1])
        match.start()
        await match.wait_end()
    except BaseException:
        await match.stop("failed")
        raise
    for brain in brains:
        brain.finish()
    return match.results()


async def _commander_seat(
    match: LockstepMatch, slot: str, seat: SeatSpec, model: StrategistModel, tools: Sequence[ToolSpec]
) -> CommanderSeat:
    assert seat.interface is not None
    settings = CommanderSettings(
        interval_s=seat.interval_s,
        min_gap_s=seat.min_gap_s,
        opening=seat.opening if seat.opening is not None else PROMPT if seat.interface == "v1" else PROMPT_V2,
        game_clock=True,
        log_turns=True,
        interface=seat.interface,
    )
    commander = Commander(model, settings)
    link = InProcessSeatLink(match, slot, tools)
    return CommanderSeat(commander, link, slot, await commander.offered(link))
