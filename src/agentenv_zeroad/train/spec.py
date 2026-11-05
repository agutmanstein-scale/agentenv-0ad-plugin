"""What one training game is: the map, the seats and how long a decision takes, as plain JSON (one spec per line).

A seat is an LLM brain (`kind: "llm"`: the endpoint, the name of the environment variable holding its key, the model,
sampling parameters and `extra_body`), a controller (`petra` with a difficulty and a behaviour, `random` included, or
the scripted `bot`) or an order replay (`replay`, with `replay` the path of its replay spec, `replay.py`). The key
itself never appears in a spec or in the results. Each seat plays an `interface` (`interface.py`): an LLM seat v2 by
default (the frozen interface training runs on; v1 is allowed), the others v1, so the bot plays as it always did. The
game's `score_rule` decides a game that reaches its cap; unset, it is the rule the LLM seats' prompt names (net worth
on v2, the engine's score on v1, `grading.resolve_score_rule`), and a rule that prompt contradicts is refused. A game
without a time cap (`time_cap_minutes: null`) has `time_cap_s` 0 in its match config: it ends by conquest or at
`match.SAFETY_CAP_S` (`ended_by` "safety_cap"), which its briefs count down to, so a stalemate can't run forever.
`ai_seed` and `ceasefire_minutes` are the lobby's game knobs, and the `experimental_` keys its unverified handicaps
(starting resources for everyone, or per seat, and a seat's population limit). A seat with `trainable: true` is one
whose decisions the learner trains on (credit reads the flag from the spec in results.json).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from agentenv_zeroad import grading
from agentenv_zeroad.clock import DEFAULT_LATENCY_MS
from agentenv_zeroad.interface import INTERFACES, Interface
from agentenv_zeroad.match import MatchConfig, SeatPlan
from agentenv_zeroad.roles import CIVS
from agentenv_zeroad.scenario import AI_BEHAVIORS, MAP_SIZES, MAX_PLAYERS, PLAYER_PLACEMENTS, AiBehavior

DEFAULT_MAX_TOKENS = 512
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}  # the plain softmax, sent explicitly, never defaulted
SeatKind = Literal["llm", "petra", "bot", "replay"]
SEAT_KINDS: tuple[SeatKind, ...] = get_args(SeatKind)
LLM_INTERFACE: Interface = "v2"  # what an LLM seat plays unless its spec says
NAMES = {"llm": "Agent", "petra": "Petra", "bot": "Bot", "replay": "Replay"}
SEAT_KEYS = {
    "kind", "civ", "name", "model", "base_url", "api_key_env", "sampling", "extra_body", "max_tokens", "effort",
    "difficulty", "behaviour", "opening", "interval_s", "min_gap_s", "trainable", "interface", "replay",
    "experimental_resources", "experimental_population_limit",
}  # fmt: skip
GAME_KEYS = {
    "game_id", "seats", "map", "seed", "ai_seed", "size", "biome", "placement", "time_cap_minutes", "latency_ms",
    "score_rule", "ceasefire_minutes", "experimental_starting_resources",
}  # fmt: skip


@dataclass(frozen=True)
class SeatSpec:
    kind: SeatKind
    civ: str
    name: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None
    difficulty: int = 3
    behaviour: AiBehavior = "balanced"
    opening: str | None = None
    interval_s: float = 25.0
    min_gap_s: float = 8.0
    trainable: bool = False
    interface: Interface | None = None  # None: v2 for an LLM seat, v1 for the others
    replay: str | None = None  # a replay seat's spec file
    experimental_resources: int | None = None
    experimental_population_limit: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in SEAT_KINDS:
            raise ValueError(f"seat kind must be {', '.join(SEAT_KINDS)}, not {self.kind!r}")
        if self.civ not in CIVS:
            raise ValueError(f"civ must be one of {', '.join(CIVS)}, not {self.civ!r}")
        if self.kind == "llm":
            missing = [k for k in ("model", "base_url", "api_key_env") if not getattr(self, k)]
            if missing:
                raise ValueError(f"an llm seat needs {', '.join(missing)}")
        if self.kind == "petra" and not 0 <= self.difficulty <= 5:
            raise ValueError("petra difficulty is 0 to 5")
        if self.behaviour not in AI_BEHAVIORS:
            raise ValueError(f"behaviour must be one of {', '.join(AI_BEHAVIORS)}")
        if (self.kind == "replay") != (self.replay is not None):
            raise ValueError("a replay seat, and only a replay seat, names its replay spec")
        if self.interface is None:
            object.__setattr__(self, "interface", LLM_INTERFACE if self.kind == "llm" else "v1")
        elif self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {', '.join(INTERFACES)}, not {self.interface!r}")

    def to_json(self) -> dict[str, Any]:
        """The seat as a spec line has it (no secrets: only the key's variable name)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def plan(self, n: int) -> SeatPlan:
        """The seat as player `n` of a match."""
        assert self.interface is not None
        return SeatPlan(
            slot=str(n),
            player=n,
            kind="agent" if self.kind == "llm" else self.kind,
            civ=self.civ,
            name=self.name or f"{NAMES[self.kind]} {n}",
            ai_difficulty=self.difficulty,
            ai_behavior=self.behaviour,
            interface=self.interface,
            starting_resources=self.experimental_resources,
            population_limit=self.experimental_population_limit,
            replay=self.replay,
        )


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    seats: tuple[SeatSpec, ...]
    map: str = "mainland"
    seed: int = 7
    ai_seed: int | None = None
    size: int = 128
    biome: str = "generic/temperate"
    placement: str = "circle"
    time_cap_minutes: float | None = 30.0
    latency_ms: int = DEFAULT_LATENCY_MS
    score_rule: grading.ScoreRule | None = None  # None: the rule the LLM seats' prompt names
    ceasefire_minutes: int = 0
    experimental_starting_resources: int | None = None

    def __post_init__(self) -> None:
        if not self.game_id or "/" in self.game_id or self.game_id.startswith("."):
            raise ValueError(f"game_id must be a plain folder name, not {self.game_id!r}")
        if not 2 <= len(self.seats) <= MAX_PLAYERS:
            raise ValueError(f"a game has 2 to {MAX_PLAYERS} seats, not {len(self.seats)}")
        if self.size not in MAP_SIZES:
            raise ValueError(f"size must be one of {MAP_SIZES}")
        if self.placement not in PLAYER_PLACEMENTS:
            raise ValueError(f"placement must be one of {PLAYER_PLACEMENTS}")
        if self.time_cap_minutes is not None and self.time_cap_minutes <= 0:
            raise ValueError("time_cap_minutes must be positive, or null for the safety cap")
        if self.ceasefire_minutes < 0:
            raise ValueError("ceasefire_minutes must be >= 0")
        if self.score_rule is not None and self.score_rule not in grading.SCORE_KEYS:
            raise ValueError(f"score_rule must be one of {', '.join(grading.SCORE_KEYS)}, not {self.score_rule!r}")
        grading.resolve_score_rule(self.score_rule, (s.interface for s in self.seats if s.kind == "llm"))

    @property
    def time_cap_s(self) -> float:
        """The match config's time cap: 0 for a game with none."""
        return 0 if self.time_cap_minutes is None else self.time_cap_minutes * 60

    def match_config(self) -> MatchConfig:
        return MatchConfig(
            seats=tuple(seat.plan(n) for n, seat in enumerate(self.seats, 1)),
            map_name=self.map,
            biome=self.biome,
            placement=self.placement,
            size=self.size,
            seed=self.seed,
            ai_seed=self.ai_seed,
            time_cap_s=self.time_cap_s,
            score_rule=self.score_rule,
            ceasefire_minutes=self.ceasefire_minutes,
            starting_resources=self.experimental_starting_resources,
        )

    def to_json(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "seats"}
        return {**out, "seats": [s.to_json() for s in self.seats]}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> GameSpec:
        unknown = set(raw) - GAME_KEYS
        if unknown:
            raise ValueError(f"unknown game spec keys: {', '.join(sorted(unknown))}")
        seats = []
        for item in raw.get("seats") or []:
            extra = set(item) - SEAT_KEYS
            if extra:
                raise ValueError(f"unknown seat keys: {', '.join(sorted(extra))}")
            seats.append(SeatSpec(**item))
        return cls(**{**raw, "seats": tuple(seats)})


def load_specs(path: Path) -> list[GameSpec]:
    """One `GameSpec` per non-empty line of a JSON-lines file; game ids must be unique."""
    specs = [GameSpec.from_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    ids = [s.game_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("game ids must be unique")
    return specs
