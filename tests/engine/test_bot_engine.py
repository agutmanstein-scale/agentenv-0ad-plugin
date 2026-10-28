"""Real-engine test: the scripted bot (player 1, through its executor) against Petra (player 2), 2 players.

The engine is stepped as fast as it answers (no wall-clock wait); every 5 steps (1 game-second) the snapshot and
executor view are queried, the bot reads its snapshot and submits orders, and the executor turns the book into
commands for the next step. Checks, within 20 game-minutes: the bot reaches town phase, launches an attack, and its
attacking units get near player 2's civic centre (engine positions); at least one building was placed from each
`where` anchor (base, near_<resource>, toward_<player>, front); the engine refused none of the effect-checked commands
(construct, train, research); and every resource site's distance in the executor view is to a drop-site that takes
that resource.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import pytest

from agentenv_zeroad.bots import ScriptedBot
from agentenv_zeroad.clock import GameClock
from agentenv_zeroad.engine import DEFAULT_IMAGE, LABEL_PID, EngineOptions, GameEngine, free_port, launch_engine
from agentenv_zeroad.executor import build_executors
from agentenv_zeroad.roles import CIVS, RoleCatalog
from agentenv_zeroad.scenario import Seat, build_scenario
from agentenv_zeroad.state import ROLES, SnapshotQuery
from agentenv_zeroad.view import distance

pytestmark = pytest.mark.engine

SEATS = [Seat("Bot", "rome"), Seat("Petra", "pers", controller="petra")]
GAME_MINUTES = 20
STEPS = GAME_MINUTES * 60 * 5
TICK_STEPS = 5
NEAR_M = 80.0
SOLDIERS_NEAR_ENEMY_JS = """(function () {
  const cmpRange = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
  const at = e => {
    const p = Engine.QueryInterface(e, IID_Position); return p && p.IsInWorld() ? p.GetPosition2D() : null; };
  const ccs = cmpRange.GetEntitiesByPlayer(2).filter(e => {
    const id = Engine.QueryInterface(e, IID_Identity); return id && id.GetClassesList().indexOf("CivCentre") !== -1; });
  let best = null, near = 0;
  for (const e of cmpRange.GetEntitiesByPlayer(1)) {
    const id = Engine.QueryInterface(e, IID_Identity);
    if (!id || id.GetClassesList().indexOf("Soldier") === -1) continue;
    const p = at(e); if (!p) continue;
    for (const c of ccs) { const q = at(c); if (!q) continue; const d = Math.hypot(p.x - q.x, p.y - q.y);
      if (best === null || d < best) best = d; if (d <= __NEAR__) near += 1; } }
  return [best, near]; })()"""


@dataclass
class Timeline:
    events: list[tuple[int, str]] = field(default_factory=list)
    town_ms: int | None = None
    attack_ms: int | None = None
    near_ms: int | None = None
    closest_m: float | None = None
    soldiers_near: int = 0
    wall_s: float = 0.0
    steps: int = 0

    def note(self, time_ms: int, text: str) -> None:
        self.events.append((time_ms, text))


async def play(engine: GameEngine, steps: int, *, stop_when_near: bool = True, on_tick=None):
    """Run the bot match for up to `steps` steps; returns (timeline, bot, executor)."""
    clock = GameClock(engine)
    snapshots = SnapshotQuery(engine, [1, 2])
    executors, view = await build_executors(engine, clock, [1], players=[1, 2])
    executor = executors[1]
    bot = ScriptedBot(executor)
    line = Timeline()
    started = time.perf_counter()
    seen_phase = "village"
    for step in range(steps):
        await clock.step_once()
        line.steps = step + 1
        if step % TICK_STEPS:
            continue
        snap = await snapshots()
        seat = (await view()).seats[1]
        me = snap.players[1]
        await executor.tick(seat, me)
        bot.tick(me, snap.time_ms)
        if on_tick is not None:
            await on_tick(snap.time_ms)
        if me.phase != seen_phase:
            seen_phase = me.phase
            line.note(snap.time_ms, f"phase {me.phase}")
            if me.phase == "town" and line.town_ms is None:
                line.town_ms = snap.time_ms
        if bot.attack_at_ms is not None and line.attack_ms is None:
            line.attack_ms = bot.attack_at_ms
            line.note(snap.time_ms, f"attack ordered with army {me.army}")
        if step % 150 == 0 and line.attack_ms is not None:
            line.note(snap.time_ms, f"closest soldier to the enemy civic centre: {line.closest_m}")
        if step % 150 == 0:
            line.note(
                snap.time_ms,
                f"pop {me.pop}/{me.pop_cap} vill {me.units.get('villager', 0)} army {me.army} "
                f"res {me.resources} buildings {sorted(me.buildings_by_kind())}",
            )
        if line.attack_ms is not None:
            closest, near = await engine.evaluate(SOLDIERS_NEAR_ENEMY_JS.replace("__NEAR__", str(NEAR_M)))
            if closest is not None and (line.closest_m is None or closest < line.closest_m):
                line.closest_m = closest
            if near and line.near_ms is None:
                line.near_ms = snap.time_ms
                line.soldiers_near = near
                line.note(snap.time_ms, f"{near} soldiers within {NEAR_M:.0f} m of the enemy civic centre")
                if stop_when_near:
                    break
    line.wall_s = time.perf_counter() - started
    return line, bot, executor


@pytest.fixture(scope="module", autouse=True)
def engine_image():
    if shutil.which("docker") is None:
        pytest.fail("docker is not installed")
    if subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True).returncode != 0:
        pytest.fail(f"{DEFAULT_IMAGE} is missing; run scripts/build-engine.sh")


def test_scripted_bot_reaches_town_and_attacks_petra_within_twenty_minutes():
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
            await engine.reset(build_scenario(SEATS, size=192, seed=3))
            return await play(engine, STEPS)

    line, bot, executor = asyncio.run(go())
    for t, text in line.events:
        print(f"{t // 60000:02d}:{t // 1000 % 60:02d} {text}")
    for r in executor.rejections:
        print(f"REJECTED {r.time_ms // 1000}s {r.what}: {r.command}")
    print(
        f"steps {line.steps} in {line.wall_s:.0f} s wall; commands {dict(executor.issued)}; "
        f"rejected {len(executor.rejections)}; placed {executor.placed}; failures {executor.failures}"
    )
    print("orders: " + "; ".join(f"{t // 1000}s {o.kind}->{r.status}" for t, o, r in bot.log))

    assert line.town_ms is not None and line.town_ms <= GAME_MINUTES * 60_000, "bot never reached town phase"
    assert line.attack_ms is not None, "bot never launched its attack"
    assert line.near_ms is not None, f"attack never got within {NEAR_M} m (closest {line.closest_m})"
    anchors = {where.split("_")[0] for _, where, _ in executor.placed}
    assert {"base", "near", "toward", "front"} <= anchors, f"placed from {sorted(anchors)}"
    assert all(r.status != "rejected" for _, _, r in bot.log), [(o, r) for _, o, r in bot.log if not r.ok]
    assert not executor.rejections, [(r.what, r.command) for r in executor.rejections]
    seat = executor.last_seat
    for r, sites in seat.sites.items():
        drops = [d.pos for d in seat.dropsites(r)]
        for site in sites:
            nearest = min(distance(site.pos, d) for d in drops)
            assert abs(site.distance - nearest) <= 1, (r, site, nearest)


def test_role_tables_from_engine_templates_cover_every_selectable_civ_and_log_no_engine_errors():
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
            catalog = RoleCatalog(engine)
            tables = {civ: await catalog.table(civ) for civ in CIVS}
            return tables, engine_mainlog()

    tables, mainlog = asyncio.run(go())
    # A rendered match shows the engine's errors on screen: building every civ's table must log none.
    errors = [line for line in mainlog.splitlines() if "ERROR" in line]
    assert mainlog and not errors, errors[:5]
    for civ, table in tables.items():
        print(civ, {r: len(table.templates_for(r)) for r in table.roles()}, len(table.structures), "structures")
        assert set(table.roles()) == set(ROLES), civ
        assert all(n.startswith(f"units/{civ}/") for r in ROLES for n in table.templates_for(r)), civ
        for key in ("civil_centre", "house", "storehouse", "farmstead", "field", "barracks"):
            assert table.structure(key) is not None, (civ, key)
        assert table.trained_at("villager") and table.trained_at("infantry_melee"), civ


def engine_mainlog() -> str:
    """This test's engine container's mainlog (HOME=/tmp there)."""
    names = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"label={LABEL_PID}={os.getpid()}"], capture_output=True, text=True
    ).stdout.split()
    assert len(names) == 1, f"expected this test's one engine container, found {names}"
    cmd = ["docker", "exec", names[0], "cat", "/tmp/.local/state/0ad/log/mainlog.html"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout
