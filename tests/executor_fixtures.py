"""Shared offline fixtures for the executor tests: real views, snapshots and Rome's templates (no engine)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agentenv_zeroad.engine import FakeEngine
from agentenv_zeroad.executor import Executor
from agentenv_zeroad.placement import Placement, Search
from agentenv_zeroad.roles import RoleCatalog, RoleTable
from agentenv_zeroad.state import Snapshot
from agentenv_zeroad.view import ExecutorView, SeatView

FIXTURES = Path(__file__).parent / "fixtures"


def load_raw(seconds: int) -> dict[str, Any]:
    return json.loads((FIXTURES / f"bot_view_{seconds:03d}s.json").read_text())


def load(seconds: int, player: int = 1) -> tuple[SeatView, Snapshot]:
    raw = load_raw(seconds)
    return ExecutorView.from_json(raw["view"]).seats[player], Snapshot.from_json(raw["snapshot"])


def rome_templates() -> dict[str, str]:
    return json.loads((FIXTURES / "templates_rome.json").read_text())


def rome_table() -> RoleTable:
    return asyncio.run(RoleCatalog(FakeEngine(templates=rome_templates())).table("rome"))


class FakePlacer:
    """Accepts every search at the first anchor (or refuses all), recording each request."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.requests: list[tuple[int, str, Search]] = []

    async def find(self, player: int, template: str, search: Search) -> Placement:
        self.requests.append((player, template, search))
        if not self.ok:
            return Placement(False, tried=10, reasons={"it cannot be built on another building or resource": 10})
        x, z = search.anchors[0]
        return Placement(True, x=x + 20.0, z=z, tried=1)


def make_executor(*, ok: bool = True, players=(1, 2)) -> tuple[Executor, list[tuple[int, dict]], FakePlacer]:
    queued: list[tuple[int, dict]] = []
    placer = FakePlacer(ok)
    executor = Executor(1, rome_table(), queue=lambda p, c: queued.append((p, c)), placer=placer, players=players)
    return executor, queued, placer
