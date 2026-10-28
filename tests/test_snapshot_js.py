"""Runs `js/snapshot.js` under Node against a small mocked simulation (no engine), so what the reflex reads from it is
tested offline: where an enemy's visible soldiers and workers stand, how many of its soldiers are in contact with
ours, and the combat counts, which 0.28 keeps per class (no `total`) for units lost and enemy units killed."""

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
  "ProductionQueue", "Trainer", "UnitAI", "ResourceSupply", "StatisticsTracker", "TechnologyManager", "Diplomacy"];
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
    default: return null;
  }
}
globalThis.Engine = { QueryInterface: (id, iid) => id === SYSTEM_ENTITY ? system[iid] : component(ents[id], iid) };
globalThis.QueryPlayerIDInterface = (p, iid) => {
  if (iid === "Identity") return { GetName: () => "P" + p, GetCiv: () => "rome" };
  if (iid === "TechnologyManager") return null;
  if (iid === "Diplomacy") return { IsEnemy: q => q !== p };
  if (iid === "StatisticsTracker") return WORLD.stats[p] || null;
  return { GetResourceCounts: () => ({}), GetResourceGatherers: () => ({}), GetState: () => "active",
    GetPopulationCount: () => 0, GetPopulationLimit: () => 20, GetMaxPopulation: () => 300 };
};
console.log(JSON.stringify(__SNAPSHOT__));
"""

UNIT = ["Unit"]
WORLD = {
    "entities": [
        {"id": 10, "owner": 1, "classes": ["Structure", "CivCentre"], "pos": [100, 100]},
        {"id": 11, "owner": 1, "classes": UNIT + ["Infantry"], "pos": [200, 200]},
        {"id": 20, "owner": 2, "classes": UNIT + ["Cavalry"], "pos": [230, 200]},  # 30 m from our spearman
        {"id": 21, "owner": 2, "classes": UNIT + ["Infantry", "Ranged"], "pos": [500, 500]},
        {"id": 22, "owner": 2, "classes": UNIT + ["FemaleCitizen"], "pos": [600, 600]},
    ],
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


def run_snapshot() -> Snapshot:
    code = snapshot_js([1]).strip().rstrip(";")
    script = HARNESS.replace("__WORLD__", json.dumps(WORLD)).replace("__SNAPSHOT__", code)
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
