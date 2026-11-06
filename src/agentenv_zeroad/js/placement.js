// Find a valid spot for one building with the engine's own placement rules (spec §9: "Use engine placement checks
// via /evaluate, searching outward from anchor points"). A local preview entity, exactly what the GUI uses while a
// player drags a building (GuiInterface.SetBuildingPlacementPreview), is moved over rings around each anchor and
// asked BuildRestrictions.CheckPlacement(): territory, terrain passability, obstructions, and distance rules. The
// first ring position that passes wins, so the result is the closest valid spot to the first anchor that has one.
// Python replaces __CONFIG__ with {"player", "template", "anchors": [[x, z]], "minR", "maxR", "step", "angle",
// "spacing", "phase", "radii"}. A non-null "radii" lists the ring radii to walk instead of minR..maxR in `step`s;
// `step` still sets the spacing of the spots along a ring. `spacing` keeps a corridor free: a spot closer than that to
// an own structure's footprint centre than both half-diagonals (Obstruction.GetSize) plus `spacing` is skipped, so
// houses don't wall villagers in. Local entities never enter the synchronised simulation, so the search can't change
// the game.
(function (cfg) {
  const cmpRange = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
  const ent = Engine.AddLocalEntity("preview|" + cfg.template);
  if (ent === INVALID_ENTITY) return { ok: false, tried: 0, reasons: { "unknown template": 1 } };
  const own = [];
  for (const e of cmpRange.GetEntitiesByPlayer(cfg.player)) {
    const id = Engine.QueryInterface(e, IID_Identity);
    if (!id || id.GetClassesList().indexOf("Structure") === -1) continue;
    const p = Engine.QueryInterface(e, IID_Position);
    if (!p || !p.IsInWorld()) continue;
    const v = p.GetPosition2D();
    const ob = Engine.QueryInterface(e, IID_Obstruction);
    own.push([v.x, v.y, ob ? ob.GetSize() : 0]);
  }
  const reasons = {};
  let tried = 0;
  try {
    const pos = Engine.QueryInterface(ent, IID_Position);
    Engine.QueryInterface(ent, IID_Ownership).SetOwner(cfg.player);
    const br = Engine.QueryInterface(ent, IID_BuildRestrictions);
    const ob = Engine.QueryInterface(ent, IID_Obstruction);
    const half = ob ? ob.GetSize() : 0;
    const radii = [];
    if (cfg.radii) radii.push(...cfg.radii);
    else for (let r = cfg.minR; r <= cfg.maxR; r += cfg.step) radii.push(r);
    for (const a of cfg.anchors) {
      for (const r of radii) {
        const n = Math.max(8, Math.round(2 * Math.PI * r / cfg.step));
        for (let i = 0; i < n; ++i) {
          const t = cfg.phase + 2 * Math.PI * i / n;
          const x = a[0] + r * Math.cos(t);
          const z = a[1] + r * Math.sin(t);
          if (cfg.spacing > 0 && own.some(o => Math.hypot(o[0] - x, o[1] - z) < o[2] + half + cfg.spacing)) continue;
          ++tried;
          pos.JumpTo(x, z);
          pos.SetYRotation(cfg.angle);
          const res = br.CheckPlacement();
          if (res.success) return { ok: true, x: Math.round(x * 10) / 10, z: Math.round(z * 10) / 10, angle: cfg.angle, tried };
          const why = res.message || "invalid";
          reasons[why] = (reasons[why] || 0) + 1;
        }
      }
    }
    return { ok: false, tried, reasons };
  } finally {
    Engine.QueryInterface(ent, IID_Position).MoveOutOfWorld();
    Engine.DestroyEntity(ent);
  }
})(__CONFIG__)
