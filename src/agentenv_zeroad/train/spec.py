"""What one training game is: the map, the seats and how long a decision takes, as plain JSON (one spec per line).

A seat is an LLM brain (`kind: "llm"`: the endpoint, the name of the environment variable holding its key, the model,
sampling parameters and `extra_body`) or a controller (`petra` with a difficulty and a behaviour, or the scripted
`bot`). The key itself never appears in a spec or in the results. A game without a time cap (`time_cap_minutes: null`)
has `time_cap_s` 0 in its match config: it ends by conquest or at `match.SAFETY_CAP_S` (`ended_by` "safety_cap"), so a
stalemate can't run forever, and its briefs never mention a cap, so the policy can't tell it from a game it must win.
A seat with `trainable: true` is one whose decisions the learner trains on (credit reads the flag from the spec in
results.json). `interface` is the seat-facing interface the game is played on and recorded in results.json; only `v1`
exists until the interface switch is merged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentenv_zeroad.clock import DEFAULT_LATENCY_MS
from agentenv_zeroad.match import MatchConfig, SeatPlan
from agentenv_zeroad.roles import CIVS
from agentenv_zeroad.scenario import MAP_SIZES, MAX_PLAYERS, PLAYER_PLACEMENTS

AI_BEHAVIORS = ("balanced", "aggressive", "defensive")
DEFAULT_MAX_TOKENS = 512
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}  # the plain softmax, sent explicitly, never defaulted
SeatKind = Literal["llm", "petra", "bot"]
SEAT_KEYS = {
    "kind", "civ", "name", "model", "base_url", "api_key_env", "sampling", "extra_body", "max_tokens", "effort",
    "difficulty", "behaviour", "opening", "interval_s", "min_gap_s", "trainable",
}  # fmt: skip
GAME_KEYS = {
    "game_id", "seats", "map", "seed", "ai_seed", "size", "biome", "placement", "time_cap_minutes", "latency_ms",
    "interface",
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
    behaviour: str = "balanced"
    opening: str | None = None
    interval_s: float = 25.0
    min_gap_s: float = 8.0
    trainable: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("llm", "petra", "bot"):
            raise ValueError(f"seat kind must be llm, petra or bot, not {self.kind!r}")
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

    def to_json(self) -> dict[str, Any]:
        """The seat as a spec line has it (no secrets: only the key's variable name)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


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
    interface: str = "v1"

    def __post_init__(self) -> None:
        if self.interface != "v1":
            raise ValueError(f"interface must be v1, not {self.interface!r}: v2 arrives with the interface switch")
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

    @property
    def time_cap_s(self) -> float:
        """The match config's time cap: 0 for a game with none."""
        return 0 if self.time_cap_minutes is None else self.time_cap_minutes * 60

    def match_config(self) -> MatchConfig:
        plans = tuple(
            SeatPlan(
                slot=str(n),
                player=n,
                kind="agent" if seat.kind == "llm" else seat.kind,
                civ=seat.civ,
                name=seat.name or f"{'Agent' if seat.kind == 'llm' else 'Bot' if seat.kind == 'bot' else 'Petra'} {n}",
                ai_difficulty=seat.difficulty,
                ai_behavior=seat.behaviour,
            )
            for n, seat in enumerate(self.seats, 1)
        )
        return MatchConfig(
            seats=plans,
            map_name=self.map,
            biome=self.biome,
            placement=self.placement,
            size=self.size,
            seed=self.seed,
            ai_seed=self.ai_seed,
            time_cap_s=self.time_cap_s,
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
