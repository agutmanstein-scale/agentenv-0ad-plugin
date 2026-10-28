"""An engine for env and match tests without 0 A.D.: FakeEngine stepping, recorded views and snapshots, a scripted
outcome query, and a handle that records how the env used it."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from agentenv_zeroad.engine import FakeEngine
from tests.executor_fixtures import FakePlacer, load_raw, rome_templates


class ScriptedEngine(FakeEngine):
    """Steps like FakeEngine; `/evaluate` answers by the query: the recorded snapshot and executor view at the
    engine's own time, the fog query, and `outcome` (each player's state and score), which a test changes."""

    def __init__(self, *, seconds: int = 60, **kwargs: Any):
        super().__init__(templates=rome_templates(), **kwargs)
        self.raw = load_raw(seconds)
        self.outcome: dict[str, dict[str, Any]] = {
            "1": {"state": "active", "total": 120, "economy": 100, "military": 0, "exploration": 20},
            "2": {"state": "active", "total": 90, "economy": 70, "military": 0, "exploration": 20},
        }
        self.fog_calls = 0
        self.saved_replay: bool | None = None

    async def reset(self, scenario, *, save_replay: bool = False):
        self.saved_replay = save_replay
        return await super().reset(scenario, save_replay=save_replay)

    async def evaluate(self, code: str) -> Any:
        self.evaluated.append(code)
        if "GetPercentMapExplored" in code:
            return copy.deepcopy(self.outcome)
        if "SetLosRevealAll(p, false));\n  return" in code:
            self.fog_calls += 1
            return {"global": False, "own": [False] * (self.players + 1), "observer": True}
        if '"baseRadius"' in code:
            return {**copy.deepcopy(self.raw["snapshot"]), "time_ms": self.time_ms}
        if '"resourceRadius"' in code:
            return {**copy.deepcopy(self.raw["view"]), "time_ms": self.time_ms}
        return await super().evaluate(code)


class Handle:
    """What `start_engine` returns, around a ScriptedEngine: records close and writes a replay when asked."""

    def __init__(self, engine: ScriptedEngine | None = None, *, replay: bool = True):
        self.engine = engine or ScriptedEngine()
        self.replay = replay
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1

    async def copy_replay(self, dest: Path) -> Path | None:
        if not self.replay:
            return None
        dest.write_text('{"settings": {}}\nturn 1 200\nend\n')
        return dest


def placer() -> FakePlacer:
    return FakePlacer()
