"""The economy floor (interface v2): a field or storehouse that finds no spot falls back to wider ring searches around
the farmstead, civic centre or resource site before the executor gives up, and every give-up reaches the notices."""

import asyncio
import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from agentenv_zeroad.executor import Executor
from agentenv_zeroad.orders import Build
from agentenv_zeroad.placement import (
    FALLBACK_ANY_WITHIN,
    FALLBACK_RINGS,
    Placement,
    Search,
    fallback_searches,
    placement_js,
    resource_anchor,
)
from agentenv_zeroad.view import distance
from tests.executor_fixtures import FakePlacer, load, make_executor

RICH = {"food": 2000, "wood": 2000, "stone": 2000, "metal": 2000}
NO_SPOT = Placement(False, tried=10, reasons={"it cannot be built on another building or resource": 10})


class RefusesNarrowSearches(FakePlacer):
    """Refuses the normal search; accepts one of the wider ones (those with exact ring radii, or the any-tile one)."""

    def __init__(self, accept_from: int = 1):
        super().__init__(True)
        self.accept_from = accept_from

    async def find(self, player: int, template: str, search: Search) -> Placement:
        self.requests.append((player, template, search))
        wide = sum(1 for _, t, s in self.requests if t == template and (s.radii is not None or s.spacing == 0.0))
        if wide >= self.accept_from:
            return Placement(True, x=search.anchors[0][0] + 1.0, z=search.anchors[0][1], tried=3)
        return NO_SPOT


def tick(executor: Executor, seat):
    return asyncio.run(executor.tick(seat, None))


def field_seat():
    seat, _ = load(300)
    return replace(seat, resources=RICH, structures=tuple(s for s in seat.structures if s.id != 2013))


def place_field(executor: Executor, seat, ticks: int = 1):
    executor.submit(Build(structure="field", where="near_food"))
    for i in range(ticks):
        tick(executor, replace(seat, time_ms=seat.time_ms + 1000 * i))


def test_a_field_falls_back_to_rings_around_the_farmstead_then_the_civic_centre():
    seat, _ = load(300)
    rings, anywhere = fallback_searches(seat, "near_food", "Field")
    farmstead = next(s.pos for s in seat.structures if s.kind == "Farmstead")
    assert rings.anchors == (farmstead, seat.base) and rings.radii == FALLBACK_RINGS == (15.0, 25.0, 35.0)
    assert anywhere.anchors == rings.anchors and anywhere.max_r == FALLBACK_ANY_WITHIN == 60.0
    assert anywhere.min_r == 0.0 and anywhere.spacing == 0.0 and rings.spacing > 0


def test_a_field_without_a_farmstead_rings_around_the_civic_centre_alone():
    seat, _ = load(300)
    bare = replace(seat, structures=tuple(s for s in seat.structures if s.kind != "Farmstead"))
    assert fallback_searches(bare, "near_food", "Field")[0].anchors == (seat.base,)


def test_a_storehouse_falls_back_to_rings_around_the_resource_site_then_the_civic_centre():
    seat, _ = load(300)
    for resource in ("wood", "metal"):
        rings, anywhere = fallback_searches(seat, f"near_{resource}", "Storehouse")
        site = resource_anchor(seat, resource)
        assert (rings.anchors[0], rings.anchors[-1]) == (site, seat.base) and anywhere.max_r == 60.0


@pytest.mark.parametrize(
    ("where", "kind"), [("base", "Field"), ("near_food", "House"), ("front", "Storehouse"), ("near_wood", "Barracks")]
)
def test_other_buildings_and_places_get_no_fallback(where, kind):
    seat, _ = load(300)
    assert fallback_searches(seat, where, kind) == ()


def test_no_base_means_no_fallback():
    seat, _ = load(300)
    assert fallback_searches(replace(seat, base=None), "near_food", "Field") == ()


def test_in_v2_a_field_the_normal_search_cannot_place_is_placed_by_a_wider_ring():
    executor, queued, _ = make_executor(interface="v2")
    executor.placer = RefusesNarrowSearches()
    seat = field_seat()
    place_field(executor, seat)
    construct = [c for _, c in queued if c["type"] == "construct" and c["template"].endswith("/field")]
    assert len(construct) == 1 and executor.failures == [] and executor.drain_notices() == []
    searches = [s for _, t, s in executor.placer.requests if t.endswith("/field")]
    assert searches[0].radii is None and searches[1].radii == FALLBACK_RINGS


def test_the_any_tile_search_runs_only_when_the_rings_found_nothing():
    executor, queued, _ = make_executor(interface="v2")
    executor.placer = RefusesNarrowSearches(accept_from=2)
    place_field(executor, field_seat())
    searches = [s for _, t, s in executor.placer.requests if t.endswith("/field")]
    assert len(searches) == 3 and searches[2].spacing == 0.0
    assert any(c["type"] == "construct" for _, c in queued)


def test_in_v1_the_fallback_is_off_and_the_field_gives_up_as_before():
    executor, queued, placer = make_executor(ok=False, interface="v1")
    place_field(executor, field_seat(), ticks=3)
    assert len(placer.requests) == 3 and not any(c["type"] == "construct" for _, c in queued)
    assert [f.reason for f in executor.failures if f.key == "field"] == [
        "no valid spot at near_food: it cannot be built on another building or resource"
    ]


def test_in_v2_a_field_with_no_spot_anywhere_gives_up_into_the_notices_after_trying_every_search():
    executor, _, placer = make_executor(ok=False, interface="v2")
    place_field(executor, field_seat(), ticks=3)
    assert len(placer.requests) == 9  # three attempts, each the normal search, the rings and the any-tile search
    [failure] = [f for f in executor.failures if f.key == "field"]
    assert failure.where == "near_food" and "rings out to 60 m tried" in failure.reason
    assert executor.drain_notices() == [failure] and executor.drain_notices() == []


def test_a_storehouse_in_v2_is_placed_by_the_fallback_next_to_its_resource():
    seat, _ = load(300)
    far = tuple(replace(u, pos=(5.0, 5.0)) if u.gathering == "wood" and u.doing == "gather" else u for u in seat.units)
    executor, queued, _ = make_executor(interface="v2")
    executor.placer = RefusesNarrowSearches()
    executor.submit(Build(structure="storehouse", where="near_wood"))
    tick(executor, replace(seat, resources=RICH, units=far))
    construct = [c for _, c in queued if c["type"] == "construct" and c["template"].endswith("/storehouse")]
    assert construct and distance((construct[0]["x"], construct[0]["z"]), resource_anchor(seat, "wood")) < 5


NODE = shutil.which("node")

PLACEMENT_HARNESS = """
const SYSTEM_ENTITY = 1;
for (const n of ["RangeManager", "Identity", "Position", "Ownership", "BuildRestrictions", "Obstruction"])
  globalThis["IID_" + n] = n;
globalThis.INVALID_ENTITY = -1;
const at = { x: 0, z: 0 };
const probes = [];
const comps = {
  Position: { JumpTo: (x, z) => { at.x = x; at.z = z; }, SetYRotation: () => {}, MoveOutOfWorld: () => {} },
  Ownership: { SetOwner: () => {} },
  Obstruction: { GetSize: () => 0 },
  BuildRestrictions: { CheckPlacement: () => { probes.push([at.x, at.z]); return { success: __ACCEPT__ }; } },
};
globalThis.Engine = {
  AddLocalEntity: () => 7,
  DestroyEntity: () => {},
  QueryInterface: (id, iid) => id === SYSTEM_ENTITY ? { GetEntitiesByPlayer: () => [] } : comps[iid],
};
const result = __SEARCH__;
console.log(JSON.stringify({ result, probes }));
"""


def run_placement(search: Search, accept: str = "false") -> dict:
    code = placement_js(1, "structures/rome/field", search).strip().rstrip(";")
    script = PLACEMENT_HARNESS.replace("__ACCEPT__", accept).replace("__SEARCH__", code)
    done = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


@pytest.mark.skipif(NODE is None, reason="needs node to run the engine's JavaScript")
def test_the_js_walks_only_the_given_radii_when_a_search_lists_them():
    out = run_placement(Search(((100.0, 100.0),), step=4.0, spacing=0.0, radii=FALLBACK_RINGS))
    radii = {round(distance(p, (100.0, 100.0))) for p in out["probes"]}
    assert radii == {15, 25, 35} and out["result"]["ok"] is False


@pytest.mark.skipif(NODE is None, reason="needs node to run the engine's JavaScript")
def test_the_js_without_radii_walks_min_to_max_in_steps_and_stops_at_the_first_valid_spot():
    out = run_placement(
        Search(((0.0, 0.0),), min_r=10.0, max_r=22.0, step=6.0, spacing=0.0), accept="Math.hypot(at.x, at.z) > 15"
    )
    radii = sorted({round(distance(p, (0.0, 0.0))) for p in out["probes"]})
    assert radii == [10, 16] and out["result"]["ok"] is True
