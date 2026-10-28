"""outcome.js, run under node against a stubbed engine: each player's state and the summary screen's score parts.

The time cap is decided on `total`, so the formula is checked here against 0.28's gui/summary/counters.js
(calculateEconomyScore, calculateMilitaryScore, calculateExplorationScore, calculateScoreTotal) with numbers where
every term counts.
"""

import json
import shutil
import subprocess

import pytest

from agentenv_zeroad.match import js

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node to run the engine's JavaScript")

STUB = """
const SYSTEM_ENTITY = 1, IID_PlayerManager = "pm", IID_StatisticsTracker = "st";
const players = %s;
const Engine = { QueryInterface: () => ({ GetNumPlayers: () => players.length + 1 }) };
function QueryPlayerIDInterface(p, iid) {
  const row = players[p - 1];
  if (iid === undefined) return { GetState: () => row.state };
  return { ...row.stats, GetPercentMapExplored: () => row.explored };
}
console.log(JSON.stringify(
%s
));
"""


def run_outcome(players: list[dict]) -> dict:
    code = STUB % (json.dumps(players), js("outcome.js"))
    done = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def test_the_score_is_the_summary_screens_economy_military_and_exploration():
    stats = {
        "resourcesGathered": {"food": 1000, "wood": 500, "stone": 40, "metal": 60, "vegetarianFood": 800},
        "tradeIncome": 100,
        "enemyUnitsKilledValue": 300,
        "unitsCapturedValue": 100,
        "enemyBuildingsDestroyedValue": 500,
        "buildingsCapturedValue": 100,
    }
    idle = {**stats, "resourcesGathered": {"food": 0}, "tradeIncome": 0, "enemyUnitsKilledValue": 0}
    idle |= {"unitsCapturedValue": 0, "enemyBuildingsDestroyedValue": 0, "buildingsCapturedValue": 0}
    out = run_outcome(
        [{"state": "active", "stats": stats, "explored": 12}, {"state": "defeated", "stats": idle, "explored": 3}]
    )
    # economy: (1000 + 500 + 40 + 60 + trade 100) / 10, vegetarianFood skipped; military: (300+100+500+100) / 10
    assert out["1"] == {"state": "active", "economy": 170, "military": 100, "exploration": 120, "total": 390}
    assert out["2"] == {"state": "defeated", "economy": 0, "military": 0, "exploration": 30, "total": 30}
