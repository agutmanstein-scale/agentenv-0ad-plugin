"""Real-engine test of the snapshot query and briefs: 2 Petra players, ~2 game-minutes, Mainland 192.

The snapshot query runs after every 200 ms step (what the env does every tick), and both seats get a text and a
JSON brief every 5 game-seconds. Checks: the snapshot matches the engine's own numbers, fog hides the far enemy base
at minute 1 although the engine itself has "reveal all" on, briefs fit the 3k-token budget, and the query's round
trip (host Python -> engine and back) stays under 30 ms at p95. After the two minutes the game is switched to real
fog: the query must leave the reveal flags exactly as found (global and per player), and an enemy cavalryman moved
next to player 1's civic centre must show up as a visible enemy near base and raise an urgent under_attack.
"""

import asyncio
import shutil
import subprocess

import pytest

from agentenv_zeroad.brief import TOKEN_BUDGET, approx_tokens, json_brief_text, text_brief
from agentenv_zeroad.engine import DEFAULT_IMAGE, EngineOptions, free_port, launch_engine
from agentenv_zeroad.metrics import MatchMetrics, percentile
from agentenv_zeroad.scenario import Seat, build_scenario
from agentenv_zeroad.state import BASE_RADIUS, ROLES, EventTracker, SnapshotQuery

pytestmark = pytest.mark.engine

SEATS = [Seat("Rome", "rome", controller="petra"), Seat("Persia", "pers", controller="petra")]
STEPS = 600  # 2 game-minutes
BRIEF_EVERY = 25  # 5 game-seconds
RANGE = "Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager)"
# Exact reveal flags: [global, gaia, p1, p2]. Slot 17 holds the global flag; own flags read true while it is on.
READ_REVEAL = f"""(function () {{ const r = {RANGE}; const g = r.GetLosRevealAll(17); r.SetLosRevealAll(-1, false);
  const own = [0, 1, 2].map(p => r.GetLosRevealAll(p)); r.SetLosRevealAll(-1, g); return [g].concat(own); }})()"""
ENGINE_RESOURCES = "[1, 2].map(p => QueryPlayerIDInterface(p).GetResourceCounts())"


def set_reveal(global_on: bool, players_on: bool) -> str:
    on = "true" if players_on else "false"
    return f"""(function () {{ const r = {RANGE}; r.SetLosRevealAll(-1, {str(global_on).lower()});
  [0, 1, 2].forEach(p => r.SetLosRevealAll(p, {on})); return true; }})()"""


def jump_enemy_cavalry_next_to(pos: tuple[int, int]) -> str:
    return f"""(function () {{ const r = {RANGE};
  for (const e of r.GetEntitiesByPlayer(2)) {{
    const id = Engine.QueryInterface(e, IID_Identity);
    if (!id || id.GetClassesList().indexOf("Cavalry") === -1) continue;
    Engine.QueryInterface(e, IID_Position).JumpTo({pos[0]} + 25, {pos[1]}); return e; }}
  return null; }})()"""


@pytest.fixture(scope="module", autouse=True)
def engine_image():
    if shutil.which("docker") is None:
        pytest.fail("docker is not installed")
    if subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True).returncode != 0:
        pytest.fail(f"{DEFAULT_IMAGE} is missing; run scripts/build-engine.sh")


def test_two_petra_minutes_briefs_fog_and_query_cost():
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), mode="docker") as engine:
            await engine.reset(build_scenario(SEATS, size=192, seed=3))
            query = SnapshotQuery(engine, [1, 2])
            trackers = {p: EventTracker(p) for p in (1, 2)}
            metrics = MatchMetrics([1, 2])
            briefs: list[tuple[int, int, str, str]] = []
            minute_one = None
            snaps = []
            for step in range(1, STEPS + 1):
                await engine.step()
                snap = await query()
                snaps.append(snap)
                if step == 1:
                    engine_resources = await engine.evaluate(ENGINE_RESOURCES)
                metrics.observe(snap)
                for tracker in trackers.values():
                    tracker.update(snap)
                if step % BRIEF_EVERY == 0:
                    for p, tracker in trackers.items():
                        events = tracker.drain()
                        briefs.append((snap.time_ms, p, text_brief(snap, p, events), json_brief_text(snap, p, events)))
                if step == 300:
                    minute_one = snap
                    cc2 = next(b.id for b in snap.players[2].buildings if b.kind == "CivCentre")
                    cc1 = next(b.id for b in snap.players[1].buildings if b.kind == "CivCentre")
                    engine_view = await engine.evaluate(
                        f"[{RANGE}.GetLosVisibility({cc2}, 1), {RANGE}.GetLosVisibility({cc1}, 2), "
                        f"{RANGE}.GetLosRevealAll(-1), {RANGE}.GetLosRevealAll(1)]"
                    )
            costs = list(query.costs_s)
            # Real fog from here on. Every query must leave the flags exactly as found.
            flags = {
                "rl_default": (await engine.evaluate(READ_REVEAL), await query(), await engine.evaluate(READ_REVEAL))
            }
            for name, global_on in (("global_on_players_off", True), ("fog", False)):
                await engine.evaluate(set_reveal(global_on, players_on=False))
                flags[name] = (await engine.evaluate(READ_REVEAL), await query(), await engine.evaluate(READ_REVEAL))
            base1 = snaps[-1].players[1].base
            moved = await engine.evaluate(jump_enemy_cavalry_next_to(base1))
            await engine.step()
            contact = await query()
            contact_events = trackers[1].update(contact)
            urgent = trackers[1].has_urgent()
            result = costs, briefs, minute_one, (cc1, cc2), engine_view, metrics.summary(), snaps, engine_resources
            return result, flags, moved, contact, contact_events, urgent

    result, flags, moved, contact, contact_events, urgent = asyncio.run(go())
    costs, briefs, minute_one, (cc1, cc2), engine_view, summary, snaps, engine_resources = result
    p50, p95, worst = percentile(costs, 50) * 1000, percentile(costs, 95) * 1000, max(costs) * 1000
    tokens = max(approx_tokens(text) for _, _, text, _ in briefs)
    json_tokens = max(approx_tokens(js) for _, _, _, js in briefs)
    print(f"snapshot query n={len(costs)} p50 {p50:.1f} ms p95 {p95:.1f} ms max {worst:.1f} ms")
    print(f"briefs {len(briefs)}, max text ~{tokens} tokens, max json ~{json_tokens} tokens; metrics {summary}")
    print("last brief (player 1):\n" + next(t for _, p, t, _ in reversed(briefs) if p == 1))

    assert len(costs) == STEPS
    assert p95 < 30, f"snapshot query p95 {p95:.1f} ms"
    # Fog: neither base sees the other's civic centre at minute 1, while the engine's own (revealed) view does.
    assert engine_view == ["visible", "visible", True, True], "control: engine reveal-all should be on and unchanged"
    seen_by_1 = {b.id for v in minute_one.players[1].enemies.values() for b in v.buildings}
    seen_by_2 = {b.id for v in minute_one.players[2].enemies.values() for b in v.buildings}
    assert cc2 not in seen_by_1 and cc1 not in seen_by_2
    # Briefs: 2 seats x every 5 s for 2 minutes, clock in each, within budget.
    assert len(briefs) == 2 * STEPS // BRIEF_EVERY
    assert all(text.startswith(f"Time {t // 60000:02d}:{t // 1000 % 60:02d}.") for t, _, text, _ in briefs)
    assert tokens <= TOKEN_BUDGET and json_tokens <= TOKEN_BUDGET
    assert any("trained" in text for _, _, text, _ in briefs), "Petra trains units within 2 minutes"
    assert summary[1]["idle_villager_s"] > 0 and summary[1]["match_s"] == pytest.approx(119.8)

    # Content: the first snapshot says what the engine says, and the economy moves over two minutes.
    first = snaps[0]
    for p, counts in zip((1, 2), engine_resources, strict=True):
        me = first.players[p]
        assert me.resources == {r: round(counts[r]) for r in me.resources}
        assert me.phase == "village" and me.base is not None and me.pop_cap > 0
        assert me.pop >= sum(n for r, n in me.units.items() if r != "other") > 0
        assert me.nearby_resources["stone"].sources >= 1 and me.nearby_resources["metal"].sources >= 1
    every = [s.players[p] for s in snaps for p in (1, 2)]
    assert all(set(me.units) <= set(ROLES) | {"other"} for me in every)
    assert any(sum(me.gatherers.values()) > 0 for me in every)
    assert any(me.idle_production for me in every)
    assert all(snaps[-1].players[p].stats["units_trained"] > 0 for p in (1, 2))
    assert {r: snaps[-1].players[1].resources[r] for r in ("food", "wood")} != {
        r: first.players[1].resources[r] for r in ("food", "wood")
    }

    # Flags: [global, gaia, p1, p2], read exactly, unchanged by a query in every starting state.
    assert flags["rl_default"][0] == [False, True, True, True]
    assert flags["global_on_players_off"][0] == [True, False, False, False]
    assert flags["fog"][0] == [False, False, False, False]
    for name, (before, _, after) in flags.items():
        assert after == before, f"{name}: query changed reveal flags {before} -> {after}"

    # Positive fog case: the enemy cavalryman next to the CC is seen, the far enemy CC still is not.
    assert moved is not None, "player 2 has a cavalryman to move"
    seen = contact.players[1].enemies.get(2)
    assert seen is not None and seen.units.get("cavalry", 0) >= 1
    assert seen.nearest is not None and seen.nearest <= BASE_RADIUS
    assert contact.players[1].enemies_near_base >= 1
    assert cc2 not in {b.id for b in seen.buildings}
    assert "under_attack" in [e.kind for e in contact_events] and urgent
