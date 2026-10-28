// Fog of war for a match (spec §4.3): a game reset through the RL interface turns LOS "reveal all" on for each
// player's own flag (gaia and 1..N), so every seat, Petra included, would see the whole map. This turns each player's
// own flag off. It never calls SetLosRevealAll(-1, true): that is the global flag (slot MAX_LOS_PLAYER_ID + 1 = 17),
// which reveals everything to everyone. The observer (-1) sees everything regardless, because GetLosRevealAll(-1) is
// hard-wired true. Returns the flags as read back afterwards: the global one, then each player's own.
(function () {
  const GLOBAL_REVEAL_SLOT = 17;
  const cmpRange = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
  const numPlayers = Engine.QueryInterface(SYSTEM_ENTITY, IID_PlayerManager).GetNumPlayers();
  const players = [];
  for (let p = 0; p < numPlayers; ++p) players.push(p);
  players.forEach(p => cmpRange.SetLosRevealAll(p, false));
  return { global: cmpRange.GetLosRevealAll(GLOBAL_REVEAL_SLOT), own: players.map(p => cmpRange.GetLosRevealAll(p)),
    observer: cmpRange.GetLosRevealAll(-1) };
})()
