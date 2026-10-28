"""Record the executor's unit-test fixtures from a real engine (U3).

Starts the zeroad-engine:0.28 image, plays the scripted bot (player 1, Romans) against Petra (player 2, Persians) on
Mainland 192 seed 3, and writes:
- tests/fixtures/bot_view_{060,300}s.json: the executor view and the U2 snapshot of both seats at that game time;
- tests/fixtures/templates_rome.json: every template the role catalog reads for Rome, trimmed to the elements
  `roles.parse_template` uses (Identity, Cost, Trainer, Builder, ResourceDropsite).

Run: `uv run python scripts/record_executor_fixtures.py` (needs Docker and the image from scripts/build-engine.sh).
"""

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "engine"))

import test_bot_engine as bot_test  # noqa: E402

from agentenv_zeroad.engine import EngineOptions, free_port, launch_engine  # noqa: E402
from agentenv_zeroad.roles import RoleCatalog  # noqa: E402
from agentenv_zeroad.scenario import build_scenario  # noqa: E402
from agentenv_zeroad.state import snapshot_js  # noqa: E402
from agentenv_zeroad.view import executor_view_js  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
KEEP = ("Identity", "Cost", "Trainer", "Builder", "ResourceDropsite")
TIMES_S = (60, 300)


class Recording:
    """Wraps an engine and keeps every template it returns."""

    def __init__(self, engine):
        self.engine = engine
        self.seen: dict[str, str] = {}

    async def templates(self, names):
        found = await self.engine.templates(names)
        self.seen.update(found)
        return found

    async def existing_templates(self, names):
        return await self.engine.existing_templates(names)


def trim(xml: str) -> str:
    return "".join(m.group(0) for tag in KEEP for m in re.finditer(rf"<{tag}>.*?</{tag}>", xml, re.S))


async def main() -> None:
    async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
        recording = Recording(engine)
        await RoleCatalog(recording).table("rome")
        (FIXTURES / "templates_rome.json").write_text(
            json.dumps({k: trim(v) for k, v in sorted(recording.seen.items())}, indent=0) + "\n"
        )
        await engine.reset(build_scenario(bot_test.SEATS, size=192, seed=3))
        done: set[int] = set()

        async def on_tick(time_ms: int) -> None:
            second = time_ms // 1000
            if second in TIMES_S and second not in done:
                done.add(second)
                view = await engine.evaluate(executor_view_js([1, 2]))
                snap = await engine.evaluate(snapshot_js([1, 2]))
                path = FIXTURES / f"bot_view_{second:03d}s.json"
                path.write_text(json.dumps({"view": view, "snapshot": snap}, separators=(",", ":"), sort_keys=True))
                print("wrote", path)

        await bot_test.play(engine, max(TIMES_S) * 5 + 10, stop_when_near=False, on_tick=on_tick)


if __name__ == "__main__":
    asyncio.run(main())
