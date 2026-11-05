"""The order vocabulary every brain uses (spec §4.5), the per-seat book of standing orders, and order results.

Orders are about groups, never single units. They are pydantic models because U4 turns each one into an MCP tool
schema as is (`model_json_schema()`), and Luna's choices are drawn from the same fields.

The `OrderBook` holds what a seat currently wants, not a log: the economy split, production still to queue, buildings
still to place, research still to start, whether to advance phase, and one army posture. The executor reads it every
tick and turns it into commands; a newer order of the same kind replaces or adds to the old one as its docstring says.

Every order gets an `OrderResult`: accepted, rejected with a reason, or partial, in the error style of the OpenCiv3
plugin's protocol (docs/protocol.md in agentenv-openciv-plugin): `message` is a sentence the player can act on,
`alternatives` lists valid choices, `suggest` is one concrete next call written as the tool call. `footer` is the brief
footer U4 adds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from agentenv_zeroad.state import RESOURCES

Role = Literal["villager", "infantry_melee", "infantry_ranged", "cavalry", "siege", "champion"]
ArmyAction = Literal["gather", "attack", "raid", "defend", "retreat", "scout"]
Status = Literal["accepted", "rejected", "partial"]

WHERE_RE = re.compile(r"^(base|front|near_(food|wood|stone|metal)|toward_(player_)?\d+)$")
TARGET_RE = re.compile(r"^(home|center|front|(player_)?\d+)$")
MAX_TRAIN = 50
DEFAULT_SPLIT = {"food": 55, "wood": 40, "stone": 0, "metal": 5}
DEFAULT_VILLAGER_TARGET = 45
MIN_VILLAGER_TARGET, MAX_VILLAGER_TARGET = 20, 100
DEFAULT_MIN_SIZE, MIN_WAVE_SIZE, MAX_WAVE_SIZE = 8, 4, 30
DEFAULT_REGROUP_BELOW = 0.4


class _Order(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SetEconomy(_Order):
    """Split gatherers between the four resources, in percent (they must add up to 100).

    Interface v2 only: `villagers` sets how many villagers the autopilot trains (the default is 45)."""

    kind: Literal["set_economy"] = "set_economy"
    food: int = Field(ge=0, le=100)
    wood: int = Field(ge=0, le=100)
    stone: int = Field(ge=0, le=100)
    metal: int = Field(ge=0, le=100)
    villagers: int | None = Field(None, ge=MIN_VILLAGER_TARGET, le=MAX_VILLAGER_TARGET)

    @model_validator(mode="after")
    def _sums_to_100(self) -> SetEconomy:
        total = self.food + self.wood + self.stone + self.metal
        if total != 100:
            raise ValueError(f"food+wood+stone+metal must be 100, got {total}")
        return self

    def split(self) -> dict[str, int]:
        return {r: getattr(self, r) for r in RESOURCES}


class Train(_Order):
    """Train `count` more units of `role`, optionally only at one kind of building (e.g. "barracks")."""

    kind: Literal["train"] = "train"
    role: Role
    count: int = Field(ge=1, le=MAX_TRAIN)
    at: str | None = None


class Build(_Order):
    """Place one `structure` (e.g. "house", "barracks", "storehouse") at `where`."""

    kind: Literal["build"] = "build"
    structure: str = Field(min_length=1, max_length=40)
    where: str = "base"

    @field_validator("where")
    @classmethod
    def _where(cls, value: str) -> str:
        if not WHERE_RE.match(value):
            raise ValueError("where must be base, front, near_<food|wood|stone|metal> or toward_<player>")
        return value


class Research(_Order):
    """Research one technology by its engine name (e.g. "gather_lumbering_ironaxes")."""

    kind: Literal["research"] = "research"
    tech: str = Field(min_length=1, max_length=80)


class AdvancePhase(_Order):
    """Advance to the next phase (village -> town -> city) as soon as requirements and resources allow."""

    kind: Literal["advance_phase"] = "advance_phase"


class Army(_Order):
    """Set the army's posture. `target` is a player ("player_2" or "2"), a named area ("center", "front") or "home".

    Interface v2 only: an attack or raid goes in waves of `min_size` soldiers (`DEFAULT_MIN_SIZE` when unset), and a
    wave below `regroup_below` of its launch strength falls back (`DEFAULT_REGROUP_BELOW`). Both stay `None` unless
    given, so a v1 order reads and records exactly as before."""

    kind: Literal["army"] = "army"
    action: ArmyAction
    target: str = "home"
    min_size: int | None = Field(None, ge=MIN_WAVE_SIZE, le=MAX_WAVE_SIZE)
    regroup_below: float | None = Field(None, ge=0.0, le=0.9)

    @field_validator("target")
    @classmethod
    def _target(cls, value: str) -> str:
        if not TARGET_RE.match(value):
            raise ValueError("target must be home, center, front or a player (player_<n>)")
        return value

    @property
    def target_player(self) -> int | None:
        digits = self.target.removeprefix("player_")
        return int(digits) if digits.isdigit() else None

    @property
    def wave_min_size(self) -> int:
        return DEFAULT_MIN_SIZE if self.min_size is None else self.min_size

    @property
    def wave_regroup_below(self) -> float:
        return DEFAULT_REGROUP_BELOW if self.regroup_below is None else self.regroup_below


class Hold(_Order):
    """Interface v2: keep every standing order as it is (an accepted no-op, so a quiet turn is on record)."""

    kind: Literal["hold"] = "hold"


class Say(_Order):
    """A short line in character for the overlay."""

    kind: Literal["say"] = "say"
    text: str = Field(min_length=1, max_length=200)


Order = Annotated[SetEconomy | Train | Build | Research | AdvancePhase | Army | Hold | Say, Field(discriminator="kind")]
ORDER_TYPES: tuple[type[_Order], ...] = (SetEconomy, Train, Build, Research, AdvancePhase, Army, Hold, Say)
_ORDER_ADAPTER: TypeAdapter[Any] = TypeAdapter(Order)


def parse_order(raw: dict[str, Any]) -> _Order:
    """Validate a raw `{"kind": ..., ...}` dict into its order model (raises pydantic.ValidationError)."""
    return _ORDER_ADAPTER.validate_python(raw)


@dataclass(frozen=True)
class OrderResult:
    status: Status
    message: str
    code: str | None = None
    alternatives: tuple[str, ...] = ()
    suggest: str | None = None
    footer: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "rejected"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "message": self.message}
        if self.code:
            out["code"] = self.code
        if self.alternatives:
            out["alternatives"] = list(self.alternatives)
        if self.suggest:
            out["suggest"] = self.suggest
        if self.footer:
            out["footer"] = self.footer
        return out


def accepted(message: str) -> OrderResult:
    return OrderResult("accepted", message)


def partial(message: str, *, suggest: str | None = None) -> OrderResult:
    return OrderResult("partial", message, suggest=suggest)


def rejected(code: str, message: str, alternatives: list[str] | tuple[str, ...] = (), suggest: str | None = None):
    return OrderResult("rejected", message, code=code, alternatives=tuple(alternatives), suggest=suggest)


@dataclass
class TrainRequest:
    """Units still to train; `attempts` counts batches the engine refused."""

    role: str
    remaining: int
    at: str | None = None
    attempts: int = 0
    failure: str | None = None


@dataclass
class BuildRequest:
    """One building still to place; `attempts` counts placement searches or engine rejections that failed."""

    structure: str
    where: str
    template: str
    attempts: int = 0
    placed_id: int | None = None
    failure: str | None = None


@dataclass
class ArmyPosture:
    """`seq` counts the army orders given: two orders in one executor tick share `since_ms`, but not `seq`.

    `focus` is what an attack or raid hits first: `any` (the nearest known structure, fighting what it meets), `army`
    (the enemy's visible soldiers), `workers` (its visible gatherers) or `buildings`. It is not part of the order
    vocabulary: every army order resets it to `any`, and only the Luna reflex changes it (Setup A).

    `min_size` and `regroup_below` are the waves' settings (interface v2, `army.py`)."""

    action: str = "gather"
    target: str = "home"
    since_ms: int = 0
    seq: int = 0
    focus: str = "any"
    min_size: int = DEFAULT_MIN_SIZE
    regroup_below: float = DEFAULT_REGROUP_BELOW

    def waves_text(self) -> str:
        return f"waves of {self.min_size}, regrouping below {round(self.regroup_below * 100)}%"

    def describe(self) -> str:
        """The posture as a brain reads it: `attack player_2, waves of 8, regrouping below 40%`."""
        text = f"{self.action} {self.target}"
        return f"{text}, {self.waves_text()}" if self.action in ("attack", "raid") else text


@dataclass
class OrderBook:
    """The standing orders of one seat. The executor consumes `train`, `build` and `research` as it issues them."""

    player: int
    split: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPLIT))
    villager_target: int = DEFAULT_VILLAGER_TARGET
    train: list[TrainRequest] = field(default_factory=list)
    build: list[BuildRequest] = field(default_factory=list)
    research: list[str] = field(default_factory=list)
    advance_phase: bool = False
    army: ArmyPosture = field(default_factory=ArmyPosture)
    said: list[tuple[int, str]] = field(default_factory=list)

    def set_split(self, split: dict[str, int]) -> None:
        self.split = {r: int(split.get(r, 0)) for r in RESOURCES}

    def pending_count(self, role: str) -> int:
        return sum(t.remaining for t in self.train if t.role == role)

    def summary(self) -> dict[str, Any]:
        """Compact view of the standing orders (for briefs and the overlay)."""
        return {
            "economy": dict(self.split),
            "villager_target": self.villager_target,
            "train": {t.role: t.remaining for t in self.train if t.remaining},
            "build": [f"{b.structure}@{b.where}" for b in self.build if b.placed_id is None and b.failure is None],
            "research": list(self.research),
            "advance_phase": self.advance_phase,
            "army": f"{self.army.action} {self.army.target}",
        }


def call_text(order: _Order) -> str:
    """The order written as the tool call a brain would make, for `suggest` fields."""
    args = order.model_dump(exclude={"kind"}, exclude_none=True)
    inner = ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in args.items())
    return f"{order.kind}({inner})"
