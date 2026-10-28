// One /evaluate call returning what the executor needs to act for each seat (spec §4.4): its own units with ids,
// roles, positions and what they are doing; its structures with production queues, what each can train and research
// now (the engine's own Trainer/Researcher lists filtered by CanProduce/CanResearch, so civ and phase rules come
// from the game), foundations and their
// builders, drop-sites and what they accept; and the gaia resources around its territory, nearest first.
// Python replaces __CONFIG__ with {"players": [seat ids], "resourceRadius": metres, "perType": n}.
// Resources are the map's static features; only those in the seat's own or neutral territory are listed, so the
// executor never sends gatherers into an enemy base. Enemy units and structures are not here: those are fog-limited
// and come from the U2 snapshot.
(function (cfg) {
  const RES = ["food", "wood", "stone", "metal"];
  const KINDS = ["CivCentre", "House", "Barracks", "Stable", "Range", "ArcheryRange", "Arsenal", "Workshop",
    "Fortress", "Temple", "Market", "Storehouse", "Farmstead", "Field", "Corral", "Dock", "Blacksmith", "Forge",
    "Tower", "Wall", "Gate", "Wonder", "Gymnasium", "ElephantStable", "Theater", "Palisade"];
  const cmpRange = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
  const cmpTimer = Engine.QueryInterface(SYSTEM_ENTITY, IID_Timer);
  const cmpTemplates = Engine.QueryInterface(SYSTEM_ENTITY, IID_TemplateManager);
  const cmpTerritory = Engine.QueryInterface(SYSTEM_ENTITY, IID_TerritoryManager);
  const cmpTerrain = Engine.QueryInterface(SYSTEM_ENTITY, IID_Terrain);
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
  function phaseOf(tm) {
    if (!tm) return "";
    if (tm.IsTechnologyResearched("phase_city")) return "city";
    if (tm.IsTechnologyResearched("phase_town")) return "town";
    return "village";
  }
  function doing(ai) {
    if (ai.IsIdle()) return "idle";
    // The order says what the unit is for (a villager slaughtering a chicken is in COMBAT but gathering food).
    const order = ai.order ? ai.order.type : "";
    if (order === "Gather" || order === "GatherNearPosition" || order === "ReturnResource") return "gather";
    if (order === "Repair") return "build";
    const s = ai.GetCurrentState() || "";
    if (s.indexOf("GARRISON") !== -1) return "garrison";
    if (s.indexOf("GATHER") !== -1 || s.indexOf("RETURNRESOURCE") !== -1) return "gather";
    if (s.indexOf("REPAIR") !== -1) return "build";
    if (s.indexOf("COMBAT") !== -1) return "fight";
    if (s.indexOf("WALKING") !== -1 || s.indexOf("PATROL") !== -1) return "move";
    return "other";
  }
  function gatherType(ent, ai) {
    const o = ai.order;
    if (o && o.data && o.data.type && o.data.type.generic) return o.data.type.generic;
    const rg = Engine.QueryInterface(ent, IID_ResourceGatherer);
    const last = rg && rg.GetLastCarriedType ? rg.GetLastCarriedType() : null;
    return last && last.generic ? last.generic : null;
  }
  function techCost(name) {
    try {
      const t = TechnologyTemplates.Get(name);
      return t && t.cost ? t.cost : {};
    } catch (e) { return {}; }
  }

  const out = { time_ms: cmpTimer.GetTime(), map_size: cmpTerrain.GetMapSize(), players: {} };
  for (const p of cfg.players) {
    const cmpPlayer = QueryPlayerIDInterface(p);
    if (!cmpPlayer) continue;
    const ident = QueryPlayerIDInterface(p, IID_Identity);
    const tm = QueryPlayerIDInterface(p, IID_TechnologyManager);
    const res = cmpPlayer.GetResourceCounts();
    const me = {
      civ: ident ? ident.GetCiv() : "", state: cmpPlayer.GetState(), resources: {},
      pop: cmpPlayer.GetPopulationCount(), pop_cap: cmpPlayer.GetPopulationLimit(), pop_max: cmpPlayer.GetMaxPopulation(),
      phase: phaseOf(tm), researching: tm ? Object.keys(tm.GetBasicInfoOfStartedTechs()).sort() : [],
      units: [], structures: [], resource_sites: {}, base: null
    };
    for (const r of RES) me.resources[r] = r1(res[r] || 0);
    for (const ent of cmpRange.GetEntitiesByPlayer(p)) {
      const id = Engine.QueryInterface(ent, IID_Identity);
      if (!id) continue;
      const cls = id.GetClassesList();
      const at = pos(ent);
      if (cls.indexOf("Structure") !== -1) {
        const fnd = Engine.QueryInterface(ent, IID_Foundation);
        const s = { id: ent, tpl: cmpTemplates.GetCurrentTemplateName(ent).replace(/^foundation\|/, ""),
          kind: kind(cls), pos: at };
        if (fnd) {
          s.foundation = true;
          s.progress = Math.round(100 * fnd.GetBuildProgress());
          s.builders = fnd.GetNumBuilders();
        } else {
          const pq = Engine.QueryInterface(ent, IID_ProductionQueue);
          if (pq) {
            const q = pq.GetQueue();
            s.queue = q.length;
            // Units waiting in the queue (count per batch), so villagers in production count toward the target.
            s.queued = q.filter(i => i.unitTemplate).map(i => [i.unitTemplate, i.count || 1]);
          }
          const tr = Engine.QueryInterface(ent, IID_Trainer);
          if (tr) s.trains = tr.GetEntitiesList().filter(t => !tm || tm.CanProduce(t));
          const rs = Engine.QueryInterface(ent, IID_Researcher);
          if (rs && tm) {
            s.techs = rs.GetTechnologiesList().filter(t => typeof t === "string" && !tm.IsTechnologyResearched(t) &&
              !tm.IsInProgress(t)).map(t => [t, tm.CanResearch(t), techCost(t)]);
          }
          const supply = Engine.QueryInterface(ent, IID_ResourceSupply);
          if (supply) s.gatherers = [supply.GetNumGatherers(), supply.GetMaxGatherers()];
          const ds = Engine.QueryInterface(ent, IID_ResourceDropsite);
          if (ds) s.drop = ds.GetTypes();
          if (!me.base && s.kind === "CivCentre" && at) me.base = at;
        }
        me.structures.push(s);
      } else if (cls.indexOf("Unit") !== -1) {
        const ai = Engine.QueryInterface(ent, IID_UnitAI);
        if (!ai || ai.IsFormationController()) continue;
        const d = doing(ai);
        me.units.push([ent, role(cls), at ? at[0] : null, at ? at[1] : null, d,
          d === "gather" || d === "idle" ? gatherType(ent, ai) : null, ai.GetStanceName()]);
      }
    }
    if (!me.base) for (const s of me.structures) if (s.pos && !s.foundation) { me.base = s.pos; break; }
    // Resources around the territory: searched around the base and every drop-site, nearest to a drop-site for them first.
    const anchors = me.structures.filter(s => !s.foundation && s.pos && (s.drop || s.kind === "CivCentre"));
    for (const r of RES) me.resource_sites[r] = [];
    if (anchors.length) {
      const seen = {};
      const found = [];
      for (const a of anchors) {
        for (const ent of cmpRange.ExecuteQuery(a.id, 0, cfg.resourceRadius, [0], IID_ResourceSupply, true)) {
          if (seen[ent]) continue;
          seen[ent] = 1;
          const rs = Engine.QueryInterface(ent, IID_ResourceSupply);
          const type = rs.GetType();
          if (RES.indexOf(type.generic) === -1 || rs.GetCurrentAmount() <= 0) continue;
          const at = pos(ent);
          if (!at) continue;
          const owner = cmpTerritory.GetOwner(at[0], at[1]);
          if (owner !== 0 && owner !== p) continue;
          // Distance to the nearest drop-site that takes this resource (a farmstead does not serve wood); with none,
          // the nearest anchor plus the search radius, so the site counts as far from any drop-site.
          let d = null;
          let any = null;
          for (const b of anchors) {
            const dd = Math.hypot(at[0] - b.pos[0], at[1] - b.pos[1]);
            if (any === null || dd < any) any = dd;
            if ((b.drop || []).indexOf(type.generic) !== -1 && (d === null || dd < d)) d = dd;
          }
          if (d === null) d = any + cfg.resourceRadius;
          found.push([type.generic, ent, at[0], at[1], r1(rs.GetCurrentAmount()), type.specific, r1(d),
            rs.GetNumGatherers ? rs.GetNumGatherers() : 0, rs.GetMaxGatherers ? rs.GetMaxGatherers() : 0]);
        }
      }
      found.sort((a, b) => a[6] - b[6]);
      for (const f of found) {
        const list = me.resource_sites[f[0]];
        if (list.length < cfg.perType) list.push(f.slice(1));
      }
    }
    out.players[p] = me;
  }
  return out;
})(__CONFIG__)
