"""0 A.D. 0.28 command JSON, as the public mod's `simulation/helpers/Commands.js` reads it.

Plain functions returning dicts, so tests compare exact commands. Group commands leave `formation` out, so each unit
gets its own order (no formation entity is created).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Pos = tuple[float, float]
ATTACK_CLASSES = {"attack": ["Unit", "Structure"]}
RAID_CLASSES = {"attack": ["Support", "Worker"]}
STRUCTURE_CLASSES = {"attack": ["Structure"]}


def cmd_train(building: int, template: str, count: int) -> dict[str, Any]:
    return {"type": "train", "entities": [building], "template": template, "count": count}


def cmd_research(building: int, tech: str) -> dict[str, Any]:
    return {"type": "research", "entity": building, "template": tech}


def cmd_construct(template: str, x: float, z: float, angle: float, builders: Sequence[int]) -> dict[str, Any]:
    return {
        "type": "construct",
        "template": template,
        "x": x,
        "z": z,
        "angle": angle,
        "actorSeed": 1,
        "entities": list(builders),
        "autorepair": True,
        "autocontinue": True,
        "queued": False,
    }


def cmd_repair(units: Sequence[int], target: int) -> dict[str, Any]:
    return {"type": "repair", "entities": list(units), "target": target, "autocontinue": True, "queued": False}


def cmd_gather(units: Sequence[int], target: int) -> dict[str, Any]:
    return {"type": "gather", "entities": list(units), "target": target, "queued": False}


def cmd_attack_walk(units: Sequence[int], pos: Pos, classes: Mapping[str, Any] = ATTACK_CLASSES) -> dict[str, Any]:
    return {
        "type": "attack-walk",
        "entities": list(units),
        "x": round(pos[0], 1),
        "z": round(pos[1], 1),
        "targetClasses": dict(classes),
        "allowCapture": False,
        "queued": False,
    }


def cmd_walk(units: Sequence[int], pos: Pos) -> dict[str, Any]:
    return {"type": "walk", "entities": list(units), "x": round(pos[0], 1), "z": round(pos[1], 1), "queued": False}


def cmd_stance(units: Sequence[int], name: str) -> dict[str, Any]:
    return {"type": "stance", "entities": list(units), "name": name}
