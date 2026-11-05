"""Runs `js/snapshot.js` under Node against a small mocked simulation (no engine), so what the reflex reads from it is
tested offline: where an enemy's visible soldiers and workers stand, how many of its soldiers are in contact with
ours, the combat counts, which 0.28 keeps per class (no `total`) for units lost and enemy units killed, and the net
worth: what a player's living units and finished structures cost, plus its banked resources."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agentenv_zeroad.state import Snapshot, snapshot_js

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

HARNESS = """
const WORLD = __WORLD__;
const SYSTEM_ENTITY = 1;
const IIDS = ["RangeManager", "Timer", "PlayerManager", "Identity", "Position", "Health", "Foundation",
  "ProductionQueue", "Trainer", "UnitAI", "ResourceSupply", "StatisticsTracker", "TechnologyManager", "Diplomacy",
  "Cost"];
for (const n of IIDS) globalThis["IID_" + n] = n;
const ents = {};
for (const e of WORLD.entities) ents[e.id] = e;
const system = {
  RangeManager: {
    GetLosRevealAll: () => false, SetLosRevealAll: () => {}, GetLosVisibilityPosition: () => "visible",
    GetEntitiesByPlayer: p => WORLD.entities.filter(e => e.owner === p).map(e => e.id),
    ExecuteQuery: () => [],
  },
  Timer: { GetTime: () => 300000 },
  PlayerManager: { GetNumPlayers: () => 3 },
};
function component(e, iid) {
  if (!e) return null;
  switch (iid) {
    case "Identity": return { GetClassesList: () => e.classes };
    case "Position": return { IsInWorld: () => true, GetPosition2D: () => ({ x: e.pos[0], y: e.pos[1] }) };
    case "Foundation": return e.foundation ? {} : null;
    case "Cost": return e.cost ? { GetResourceCosts: owner => {
      if (e.cost_throws) throw new Error("GetResourceCosts: no such cost");
      return owner === e.owner ? e.cost : { food: 99999 };
    } } : null;
    default: return null;
  }
}
globalThis.Engine = { QueryInterface: (id, iid) => id === SYSTEM_ENTITY ? system[iid] : component(ents[id], iid) };
globalThis.QueryPlayerIDInterface = (p, iid) => {
  if (iid === "Identity") return { GetName: () => "P" + p, GetCiv: () => "rome" };
  if (iid === "TechnologyManager") return null;
  if (iid === "Diplomacy") return { IsEnemy: q => q !== p };
  if (iid === "StatisticsTracker") return WORLD.stats[p] || null;
  return { GetResourceCounts: () => WORLD.banked[p] || {}, GetResourceGatherers: () => ({}), GetState: () => "active",
    GetPopulationCount: () => 0, GetPopulationLimit: () => 20, GetMaxPopulation: () => 300 };
};
console.log(JSON.stringify(__SNAPSHOT__));
"""

UNIT = ["Unit"]
WORLD = {
    "entities": [
        {"id": 10, "owner": 1, "classes": ["Structure", "CivCentre"], "pos": [100, 100],
         "cost": {"wood": 500, "stone": 500, "metal": 500}},
        {"id": 11, "owner": 1, "classes": UNIT + ["Infantry"], "pos": [200, 200], "cost": {"food": 50, "wood": 50}},
        {"id": 12, "owner": 1, "classes": UNIT + ["FemaleCitizen"], "pos": [110, 100], "cost": {"food": 50}},
        {"id": 13, "owner": 1, "classes": ["Structure", "House"], "pos": [120, 120], "cost": {"wood": 150}},
        {"id": 14, "owner": 1, "classes": ["Structure", "Barracks"], "pos": [140, 120],
         "cost": {"wood": 200, "stone": 100}, "foundation": True},
        {"id": 15, "owner": 1, "classes": ["Gaia"], "pos": [150, 150], "cost": {"food": 7777}},
        {"id": 20, "owner": 2, "classes": UNIT + ["Cavalry"], "pos": [230, 200]},  # 30 m from our spearman
        {"id": 21, "owner": 2, "classes": UNIT + ["Infantry", "Ranged"], "pos": [500, 500]},
        {"id": 22, "owner": 2, "classes": UNIT + ["FemaleCitizen"], "pos": [600, 600]},
    ],
    "banked": {"1": {"food": 120, "wood": 80, "stone": 0, "metal": 5}},
    "stats": {
        "1": {
            "unitsTrained": {"total": 9, "Unit": 9},
            "unitsLost": {"Unit": 3, "Infantry": 3},
            "enemyUnitsKilled": {"Unit": 5, "Cavalry": 5},
            "buildingsLost": {"Structure": 1, "House": 1},
            "buildingsConstructed": {"total": 2},
        }
    },
}  # fmt: skip


def run_snapshot(world: dict | None = None) -> Snapshot:
    code = snapshot_js([1]).strip().rstrip(";")
    script = HARNESS.replace("__WORLD__", json.dumps(world or WORLD)).replace("__SNAPSHOT__", code)
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout
    return Snapshot.from_json(json.loads(out))


def test_an_enemy_seen_reports_where_its_soldiers_and_workers_stand_and_how_many_are_in_contact():
    enemy = run_snapshot().players[1].enemies[2]
    assert enemy.soldiers_at == (365, 350, 2)
    assert enemy.workers_at == (600, 600, 1)
    assert enemy.contact == 1  # the cavalryman next to our spearman; the far archer is no contact


def test_combat_counts_come_from_the_per_class_counters_when_there_is_no_total():
    stats = run_snapshot().players[1].stats
    assert stats["units_lost"] == 3 and stats["enemy_units_killed"] == 5 and stats["buildings_lost"] == 1
    assert stats["units_trained"] == 9 and stats["buildings_constructed"] == 2


def test_net_worth_is_the_cost_of_living_units_and_finished_structures_plus_the_bank():
    me = run_snapshot().players[1]
    units = 100 + 50  # a spearman and a villager
    structures = 1500 + 150  # the civic centre and a house; the barracks foundation and the Gaia entity add nothing
    bank = 120 + 80 + 0 + 5
    assert me.net_worth == units + structures + bank == 2005


def test_a_cost_the_engine_cannot_answer_drops_the_net_worth_and_keeps_the_snapshot():
    world = json.loads(json.dumps(WORLD))
    world["entities"][1]["cost_throws"] = True
    me = run_snapshot(world).players[1]
    assert me.net_worth is None
    assert len(me.buildings) == 3 and me.units["villager"] == 1
