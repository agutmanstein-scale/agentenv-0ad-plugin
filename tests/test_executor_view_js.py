"""Runs `js/executor_view.js` under Node against a small mocked simulation (no engine), so its arithmetic is tested
offline. The world mirrors the recorded 300 s view: a farmstead next to a wood line and a storehouse next to fruit."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agentenv_zeroad.view import ExecutorView, executor_view_js

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

HARNESS = """
const WORLD = __WORLD__;
const SYSTEM_ENTITY = 1;
const IIDS = ["RangeManager", "Timer", "TemplateManager", "TerritoryManager", "Terrain", "Identity", "Position",
  "Foundation", "ProductionQueue", "Trainer", "Researcher", "ResourceSupply", "ResourceDropsite", "UnitAI",
  "TechnologyManager"];
for (const n of IIDS) globalThis["IID_" + n] = n;
const ents = {};
for (const e of WORLD.entities) ents[e.id] = e;
const near = (a, b, r) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= r;
const system = {
  RangeManager: {
    GetEntitiesByPlayer: p => WORLD.entities.filter(e => e.owner === p).map(e => e.id),
    ExecuteQuery: (src, minR, maxR, players, iid) => WORLD.entities.filter(e => players.indexOf(e.owner) !== -1 &&
      (iid !== "ResourceSupply" || e.resource) && near(e.pos, ents[src].pos, maxR)).map(e => e.id),
  },
  Timer: { GetTime: () => 300000 },
  TemplateManager: { GetCurrentTemplateName: id => ents[id].tpl },
  TerritoryManager: { GetOwner: () => 0 },
  Terrain: { GetMapSize: () => 512 },
};
function component(e, iid) {
  switch (iid) {
    case "Identity": return e.classes ? { GetClassesList: () => e.classes } : null;
    case "Position": return { IsInWorld: () => true, GetPosition2D: () => ({ x: e.pos[0], y: e.pos[1] }) };
    case "ResourceSupply": return e.resource ? {
      GetType: () => ({ generic: e.resource, specific: e.resource }),
      GetCurrentAmount: () => e.infinite ? Infinity : 100, IsInfinite: () => !!e.infinite,
      GetNumGatherers: () => 0, GetMaxGatherers: () => 8 } : null;
    case "ResourceDropsite": return e.drop ? { GetTypes: () => e.drop } : null;
    case "ProductionQueue": return e.queue ? { GetQueue: () => e.queue } : null;
    default: return null;
  }
}
globalThis.Engine = { QueryInterface: (id, iid) => id === SYSTEM_ENTITY ? system[iid] : component(ents[id], iid) };
globalThis.QueryPlayerIDInterface = (p, iid) => {
  if (iid === "Identity") return { GetCiv: () => "rome" };
  if (iid === "TechnologyManager") return null;
  return { GetResourceCounts: () => ({}), GetState: () => "active", GetPopulationCount: () => 0,
    GetPopulationLimit: () => 20, GetMaxPopulation: () => 300 };
};
console.log(JSON.stringify(__VIEW__));
"""

STRUCTURE = ["Structure"]
WORLD = {
    "entities": [
        {"id": 1, "owner": -1, "pos": [0, 0]},
        {"id": 10, "owner": 1, "classes": STRUCTURE + ["CivCentre"], "tpl": "structures/rome/civil_centre",
         "pos": [500, 628], "drop": ["food", "wood", "stone", "metal"],
         "queue": [{"unitTemplate": "units/rome/support_female_citizen", "count": 2}, {"technologyTemplate": "x"}]},
        {"id": 11, "owner": 1, "classes": STRUCTURE + ["Farmstead"], "tpl": "structures/rome/farmstead",
         "pos": [458, 630], "drop": ["food"]},
        {"id": 12, "owner": 1, "classes": STRUCTURE + ["Storehouse"], "tpl": "structures/rome/storehouse",
         "pos": [526, 568], "drop": ["wood", "stone", "metal"]},
        {"id": 20, "owner": 0, "pos": [423, 616], "resource": "wood"},
        {"id": 21, "owner": 0, "pos": [509, 562], "resource": "food"},
    ]
}  # fmt: skip


def run_view(world: dict = WORLD) -> ExecutorView:
    view_js = executor_view_js([1]).strip().rstrip(";")
    script = HARNESS.replace("__WORLD__", json.dumps(world)).replace("__VIEW__", view_js)
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout
    return ExecutorView.from_json(json.loads(out))


def test_site_distance_counts_only_drop_sites_that_take_that_resource():
    seat = run_view().seats[1]
    tree = seat.sites["wood"][0]
    fruit = seat.sites["food"][0]
    # The farmstead (food only) is 38 m from the tree; the nearest wood drop-site is the storehouse, 78 m away.
    assert tree.distance == 78
    # The storehouse (no food) is 18 m from the fruit; the nearest food drop-site is the civic centre, 67 m away.
    assert fruit.distance == 67


def test_production_queue_lists_queued_units_per_batch():
    cc = run_view().seats[1].structure(10)
    assert cc.queue == 2
    assert cc.queued == (("units/rome/support_female_citizen", 2),)


def test_a_field_left_to_gaia_by_a_fallen_empire_is_not_a_resource_site():
    """A field holds infinite food, and a fallen empire's buildings decay to gaia: an infinite amount would reach the
    view as null and fail every tick of the match."""
    field = {"id": 22, "owner": 0, "pos": [470, 640], "resource": "food", "infinite": True}
    seat = run_view({"entities": [*WORLD["entities"], field]}).seats[1]
    assert [site.id for site in seat.sites["food"]] == [21]
