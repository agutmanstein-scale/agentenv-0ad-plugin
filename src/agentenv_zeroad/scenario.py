"""Build the 0 A.D. 0.28 `/reset` scenario JSON for random maps.

The RL interface's `/reset` takes the same JSON as line one of a replay's `commands.txt`. 0.28's `generateMap`
throws unless the top level carries `mapType`, `map` and `script` (relative to `maps/random/`) and `settings` carries
`mapName`, `Biome`, `PlayerPlacement` and a `Color` per player (spike-results.md, engine quirk 2). Every field the
spike proved working is set here so callers can't forget one.

A seat is either driven from outside (our brains and executor send its commands through `/step`; `AI` is "") or
played by the engine's built-in Petra AI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Controller = Literal["external", "petra"]

PLAYER_PLACEMENTS = ("circle", "river", "groupedLines", "randomGroup", "stronghold")
MAP_SIZES = (128, 192, 256, 320, 384, 448, 512)
CONQUEST_TRIGGER_SCRIPTS = ("scripts/TriggerHelper.js", "scripts/ConquestCommon.js", "scripts/Conquest.js")
DEFAULT_COLORS: tuple[tuple[int, int, int], ...] = (
    (150, 20, 20),
    (20, 60, 180),
    (40, 150, 40),
    (200, 170, 20),
    (120, 30, 160),
    (20, 150, 150),
    (210, 100, 20),
    (90, 90, 90),
)
MAX_PLAYERS = 8


@dataclass(frozen=True)
class Seat:
    """One player slot. `team` -1 means no team (free for all)."""

    name: str
    civ: str
    controller: Controller = "external"
    ai_difficulty: int = 3
    ai_behavior: str = "balanced"
    team: int = -1
    color: tuple[int, int, int] | None = None


def build_scenario(
    seats: list[Seat] | tuple[Seat, ...],
    *,
    map_name: str = "mainland",
    biome: str = "generic/temperate",
    placement: str = "circle",
    size: int = 192,
    seed: int = 7,
    ai_seed: int | None = None,
    ceasefire_minutes: int = 0,
    conquest: bool = True,
    circular: bool = True,
) -> dict[str, Any]:
    """Return the `/reset` JSON object for a random-map game with these seats (player 1 is `seats[0]`)."""
    if not 1 <= len(seats) <= MAX_PLAYERS:
        raise ValueError(f"need 1-{MAX_PLAYERS} seats, got {len(seats)}")
    if placement not in PLAYER_PLACEMENTS:
        raise ValueError(f"placement must be one of {PLAYER_PLACEMENTS}, got {placement!r}")
    if size not in MAP_SIZES:
        raise ValueError(f"size must be one of {MAP_SIZES}, got {size}")
    if ceasefire_minutes < 0:
        raise ValueError("ceasefire_minutes must be >= 0")
    script = f"{map_name}.js"
    settings: dict[str, Any] = {
        "mapName": map_name,
        "mapType": "random",
        "Script": script,
        "Biome": biome,
        "PlayerPlacement": placement,
        "CircularMap": circular,
        "Size": size,
        "Seed": seed,
        "AISeed": seed if ai_seed is None else ai_seed,
        "Ceasefire": ceasefire_minutes,
        "CheatsEnabled": False,
        "VictoryConditions": ["conquest"] if conquest else [],
        "TriggerScripts": list(CONQUEST_TRIGGER_SCRIPTS) if conquest else [],
        "PlayerData": [_player_data(i, seat) for i, seat in enumerate(seats)],
    }
    return {
        "mapType": "random",
        "map": f"maps/random/{map_name}",
        "script": script,
        "gameSpeed": 1,
        "settings": settings,
    }


def scenario_json(seats: list[Seat] | tuple[Seat, ...], **kwargs: Any) -> str:
    """`build_scenario` serialised, ready to POST to `/reset`."""
    return json.dumps(build_scenario(seats, **kwargs))


def _player_data(index: int, seat: Seat) -> dict[str, Any]:
    if not seat.civ:
        raise ValueError(f"seat {index + 1} has no civ")
    if seat.controller not in ("external", "petra"):
        raise ValueError(f"seat {index + 1}: controller must be 'external' or 'petra', got {seat.controller!r}")
    r, g, b = seat.color or DEFAULT_COLORS[index]
    petra = seat.controller == "petra"
    return {
        "Name": seat.name,
        "Civ": seat.civ,
        "Color": {"r": r, "g": g, "b": b},
        "AI": "petra" if petra else "",
        "AIDiff": seat.ai_difficulty if petra else 0,
        "AIBehavior": seat.ai_behavior if petra else "balanced",
        "Team": seat.team,
    }
