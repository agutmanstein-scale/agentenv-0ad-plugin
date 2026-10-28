// Each player's engine state ("active", "won", "defeated") and score, as 0.28's summary screen computes it
// (gui/summary/counters.js calculateScoreTotal): economy = resources gathered (+ trade income) / 10, military = value
// of enemy units killed, units captured, enemy buildings destroyed and buildings captured / 10, exploration = percent
// of the map explored * 10. Read from the StatisticsTracker's running totals, not its sequences.
(function () {
  const numPlayers = Engine.QueryInterface(SYSTEM_ENTITY, IID_PlayerManager).GetNumPlayers();
  const out = {};
  for (let p = 1; p < numPlayers; ++p) {
    const player = QueryPlayerIDInterface(p);
    const st = QueryPlayerIDInterface(p, IID_StatisticsTracker);
    if (!player || !st) continue;
    let gathered = 0;
    for (const [type, amount] of Object.entries(st.resourcesGathered))
      if (type !== "vegetarianFood") gathered += amount;
    const economy = Math.round((gathered + st.tradeIncome) / 10);
    const military = Math.round((st.enemyUnitsKilledValue + st.unitsCapturedValue + st.enemyBuildingsDestroyedValue +
      st.buildingsCapturedValue) / 10);
    const exploration = st.GetPercentMapExplored() * 10;
    out[p] = { state: player.GetState(), economy: economy, military: military, exploration: exploration,
      total: economy + military + exploration };
  }
  return out;
})()
