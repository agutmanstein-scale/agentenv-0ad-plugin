// One /evaluate call that returns every player's view of the game as compact JSON (spec §4.3).
// Python replaces __CONFIG__ with {"players": [viewer ids], "baseRadius": metres}; the other players (enemy
// candidates, gaia 0 included) come from the engine's PlayerManager, so no caller can leave an enemy out.
// Fog of war: a game reset through the RL interface (observer, playerID -1) switches LOS "reveal all" on for every
// player, and GetLosVisibility(ent, p) answers from a per-turn cache that already includes that reveal. So the query
// switches reveal-all off for the duration of this call, asks the LOS grid directly with
// GetLosVisibilityPosition(x, z, p), and restores every flag before returning (also on error).
// Restoring needs the flags as they really were, and GetLosRevealAll lies about them: GetLosRevealAll(-1) is always
// true, and GetLosRevealAll(p) is true whenever the global flag is on. The global flag that SetLosRevealAll(-1, _)
// writes lives in slot MAX_LOS_PLAYER_ID + 1 = 17 and reads back exactly there; each player's own flag reads back
// exactly once the global flag is off.
(function (cfg) {
  const RES = ["food", "wood", "stone", "metal"];
  const KINDS = ["CivCentre", "House", "Barracks", "Stable", "Range", "ArcheryRange", "Arsenal", "Workshop",
    "Fortress", "Temple", "Market", "Storehouse", "Farmstead", "Field", "Corral", "Dock", "Blacksmith", "Forge",
    "Tower", "Wall", "Gate", "Wonder", "Gymnasium", "ElephantStable", "Theater", "Palisade"];
  const cmpRange = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
  const cmpTimer = Engine.QueryInterface(SYSTEM_ENTITY, IID_Timer);
  const r1 = v => Math.round(v);

  function role(cls) {
    const has = c => cls.indexOf(c) !== -1;
    if (has("Siege")) return "siege";
    if (has("Hero")) return "other";
    if (has("Champion")) return "champion";
    if (has("Ship") || has("Trader") || has("Healer") || has("Animal")) return "other";
    if (has("Cavalry")) return "cavalry";
    if (has("Infantry")) return has("Ranged") ? "infantry_ranged" : "infantry_melee";
    if (has("FemaleCitizen") || has("Worker")) return "villager";
    return "other";
  }
  function kind(cls) {
    for (const k of KINDS) if (cls.indexOf(k) !== -1) return k;
    return "Other";
  }
  function pos(ent) {
    const p = Engine.QueryInterface(ent, IID_Position);
    if (!p || !p.IsInWorld()) return null;
    const v = p.GetPosition2D();
    return [r1(v.x), r1(v.y)];
  }
  function info(ent) {
    const id = Engine.QueryInterface(ent, IID_Identity);
    if (!id) return null;
    return id.GetClassesList();
  }
  function hpFrac(ent) {
    const h = Engine.QueryInterface(ent, IID_Health);
    return h && h.GetMaxHitpoints() ? Math.round(100 * h.GetHitpoints() / h.GetMaxHitpoints()) : 100;
  }
  function phaseOf(tm) {
    if (!tm) return "";
    if (tm.IsTechnologyResearched("phase_city")) return "city";
    if (tm.IsTechnologyResearched("phase_town")) return "town";
    if (tm.IsTechnologyResearched("phase_village")) return "village";
    return "";
  }
  // What an entity cost to make (food + wood + stone + metal) at the owner's current tech modifiers. null when the
  // engine's Cost interface does not answer as expected: the net worth is then left out instead of failing the snapshot.
  function worth(ent, p) {
    try {
      const cost = Engine.QueryInterface(ent, IID_Cost);
      if (!cost) return 0;
      const c = cost.GetResourceCosts(p);
      return RES.reduce((sum, r) => sum + (c[r] || 0), 0);
    } catch (e) {
      return null;
    }
  }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

  const GLOBAL_REVEAL_SLOT = 17;
  const numPlayers = Engine.QueryInterface(SYSTEM_ENTITY, IID_PlayerManager).GetNumPlayers();
  const allPlayers = [];
  for (let p = 0; p < numPlayers; ++p) allPlayers.push(p);
  const globalWas = cmpRange.GetLosRevealAll(GLOBAL_REVEAL_SLOT);
  let ownWas = null;
  function visibleTo(at, p) { return cmpRange.GetLosVisibilityPosition(at[0], at[1], p) === "visible"; }
  function stats(p) {
    const st = QueryPlayerIDInterface(p, IID_StatisticsTracker);
    if (!st) return {};
    // 0.28 keeps `total` only for some counters (trained, constructed); the killed and lost ones count per class,
    // and every unit has the class Unit, every building Structure.
    const tot = o => (o ? Math.max(o.total || 0, o.Unit || 0, o.Structure || 0) : 0);
    return { units_trained: tot(st.unitsTrained), units_lost: tot(st.unitsLost), buildings_lost: tot(st.buildingsLost),
      buildings_constructed: tot(st.buildingsConstructed), enemy_units_killed: tot(st.enemyUnitsKilled) };
  }
  try {
  cmpRange.SetLosRevealAll(-1, false);
  ownWas = allPlayers.map(p => cmpRange.GetLosRevealAll(p));
  allPlayers.forEach(p => cmpRange.SetLosRevealAll(p, false));
  const out = { time_ms: cmpTimer.GetTime(), players: {} };
  const owned = {};
  for (const p of allPlayers) owned[p] = cmpRange.GetEntitiesByPlayer(p);

  for (const p of cfg.players) {
    const cmpPlayer = QueryPlayerIDInterface(p);
    if (!cmpPlayer) continue;
    const ident = QueryPlayerIDInterface(p, IID_Identity);
    const tm = QueryPlayerIDInterface(p, IID_TechnologyManager);
    const dip = QueryPlayerIDInterface(p, IID_Diplomacy);
    const res = cmpPlayer.GetResourceCounts();
    const gat = cmpPlayer.GetResourceGatherers();
    const me = {
      name: ident ? ident.GetName() : "", civ: ident ? ident.GetCiv() : "", state: cmpPlayer.GetState(),
      resources: {}, gatherers: {}, pop: cmpPlayer.GetPopulationCount(), pop_cap: cmpPlayer.GetPopulationLimit(),
      pop_max: cmpPlayer.GetMaxPopulation(), phase: phaseOf(tm),
      techs: tm ? Array.from(tm.GetResearchedTechs()).sort() : [],
      researching: tm ? Object.keys(tm.GetBasicInfoOfStartedTechs()).sort() : [],
      buildings: [], units: {}, idle_units: {}, idle_villagers: [], idle_production: [], nearby_resources: {},
      enemies: {}, enemies_near_base: 0, stats: stats(p)
    };
    for (const r of RES) { me.resources[r] = r1(res[r] || 0); me.gatherers[r] = gat[r] || 0; }
    const bases = [];
    const mySoldiers = [];
    let netWorth = 0;
    let worthKnown = true;
    function addWorth(ent) {
      const w = worth(ent, p);
      if (w === null) worthKnown = false; else netWorth += w;
    }
    for (const ent of owned[p]) {
      const cls = info(ent);
      if (!cls) continue;
      const at = pos(ent);
      if (cls.indexOf("Structure") !== -1) {
        const found = !!Engine.QueryInterface(ent, IID_Foundation);
        const b = { id: ent, kind: kind(cls), pos: at, hp: hpFrac(ent) };
        if (found) b.foundation = true;
        const pq = Engine.QueryInterface(ent, IID_ProductionQueue);
        const trains = !!Engine.QueryInterface(ent, IID_Trainer);
        if (pq && !found) {
          b.queue = pq.GetQueue().length;
          if (trains && b.queue === 0 && b.kind !== "House") me.idle_production.push(ent);
        }
        me.buildings.push(b);
        if (!found) addWorth(ent);
        if (!found && at && (b.kind === "CivCentre" || b.kind === "Fortress")) bases.push(at);
      } else if (cls.indexOf("Unit") !== -1) {
        const rl = role(cls);
        addWorth(ent);
        me.units[rl] = (me.units[rl] || 0) + 1;
        if (at && rl !== "villager" && rl !== "other") mySoldiers.push(at);
        const ai = Engine.QueryInterface(ent, IID_UnitAI);
        if (ai && ai.IsIdle() && !ai.IsFormationController()) {
          me.idle_units[rl] = (me.idle_units[rl] || 0) + 1;
          if (rl === "villager") me.idle_villagers.push(ent);
        }
      }
    }
    // Net worth decides a time-capped match under the "net_worth" score rule: living units and finished structures at
    // their cost, plus what is banked. Foundations are not counted until they are built.
    me.net_worth = worthKnown ? r1(netWorth + RES.reduce((sum, r) => sum + me.resources[r], 0)) : null;
    if (!bases.length) for (const b of me.buildings) if (b.pos && !b.foundation) { bases.push(b.pos); break; }
    me.base = bases.length ? bases[0] : null;
    // Resources left near each base: what "resource exhausted near base" events read.
    const seen = {};
    for (const r of RES) me.nearby_resources[r] = { amount: 0, sources: 0 };
    for (const b of me.buildings) {
      if (b.foundation || !(b.kind === "CivCentre")) continue;
      for (const ent of cmpRange.ExecuteQuery(b.id, 0, cfg.baseRadius, [0], IID_ResourceSupply, true)) {
        if (seen[ent]) continue;
        seen[ent] = 1;
        const rs = Engine.QueryInterface(ent, IID_ResourceSupply);
        const t = rs.GetType().generic;
        if (!me.nearby_resources[t] || rs.IsInfinite()) continue;
        me.nearby_resources[t].amount += rs.GetCurrentAmount();
        me.nearby_resources[t].sources += 1;
      }
    }
    for (const r of RES) me.nearby_resources[r].amount = r1(me.nearby_resources[r].amount);
    // Enemies this player can see right now.
    for (const q of allPlayers) {
      if (q === p || q === 0 || !dip || !dip.IsEnemy(q)) continue;
      // Where its visible soldiers and workers stand (centroids) and how many of its soldiers are within
      // cfg.contactRadius of one of ours: what the reflex's engage and target questions read.
      const seenBy = { units: {}, buildings: [], nearest: null, soldiers_at: null, workers_at: null, contact: 0 };
      const sums = { soldiers: [0, 0, 0], workers: [0, 0, 0] };
      let any = false;
      for (const ent of owned[q]) {
        const at = pos(ent);
        if (!at || !visibleTo(at, p)) continue;
        const cls = info(ent);
        if (!cls) continue;
        any = true;
        let d = null;
        for (const b of bases) d = d === null ? dist(at, b) : Math.min(d, dist(at, b));
        if (d !== null && (seenBy.nearest === null || d < seenBy.nearest)) seenBy.nearest = r1(d);
        if (cls.indexOf("Structure") !== -1) {
          seenBy.buildings.push({ id: ent, kind: kind(cls), pos: at });
        } else if (cls.indexOf("Unit") !== -1) {
          const rl = role(cls);
          seenBy.units[rl] = (seenBy.units[rl] || 0) + 1;
          if (d !== null && d <= cfg.baseRadius && rl !== "villager" && rl !== "other") me.enemies_near_base += 1;
          const group = rl === "villager" ? sums.workers : rl === "other" ? null : sums.soldiers;
          if (group) { group[0] += at[0]; group[1] += at[1]; group[2] += 1; }
          if (group === sums.soldiers && mySoldiers.some(s => dist(s, at) <= (cfg.contactRadius || 60)))
            seenBy.contact += 1;
        }
      }
      for (const k of ["soldiers", "workers"]) {
        const g = sums[k];
        if (g[2]) seenBy[k + "_at"] = [r1(g[0] / g[2]), r1(g[1] / g[2]), g[2]];
      }
      if (any) me.enemies[q] = seenBy;
    }
    out.players[p] = me;
  }
  return out;
  } finally {
    if (ownWas) allPlayers.forEach((p, i) => cmpRange.SetLosRevealAll(p, ownWas[i]));
    cmpRange.SetLosRevealAll(-1, globalWas);
  }
})(__CONFIG__)
