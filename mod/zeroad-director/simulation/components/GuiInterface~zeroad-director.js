// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * zeroad-director: one GUI call for the director (gui/session/zeroad_director.js), so it can find the action without
 * reading every entity's full state. It returns the map position of every unit in combat (UnitAI state with COMBAT),
 * of every civic centre, and of every damaged entity with its id and hitpoints (foundations, which gain hitpoints as
 * they are built, are left out). The GUI's player is the observer, who may see everything, so nothing is filtered.
 * Loaded after GuiInterface.js (the file sorts after it); if it isn't there, the director falls back to the
 * stock GetNonGaiaEntities and GetMultipleEntityStates calls.
 */
if (typeof GuiInterface !== "undefined")
{
	GuiInterface.prototype.ZeroadDirectorScene = function(player)
	{
		const fights = [];
		const bases = [];
		const hurt = [];
		const cmpRangeManager = Engine.QueryInterface(SYSTEM_ENTITY, IID_RangeManager);
		for (const ent of cmpRangeManager.GetNonGaiaEntities())
		{
			const cmpPosition = Engine.QueryInterface(ent, IID_Position);
			if (!cmpPosition || !cmpPosition.IsInWorld())
				continue;
			const pos = cmpPosition.GetPosition2D();
			const cmpOwnership = Engine.QueryInterface(ent, IID_Ownership);
			const owner = cmpOwnership ? cmpOwnership.GetOwner() : -1;
			const cmpIdentity = Engine.QueryInterface(ent, IID_Identity);
			if (cmpIdentity && cmpIdentity.HasClass("CivCentre"))
				bases.push({ "x": pos.x, "z": pos.y, "owner": owner });
			const cmpUnitAI = Engine.QueryInterface(ent, IID_UnitAI);
			const state = cmpUnitAI && cmpUnitAI.GetCurrentState();
			if (state && state.indexOf("COMBAT") != -1)
				fights.push({ "x": pos.x, "z": pos.y, "owner": owner });
			const cmpHealth = Engine.QueryInterface(ent, IID_Health);
			if (cmpHealth && !Engine.QueryInterface(ent, IID_Foundation) &&
				cmpHealth.GetHitpoints() < cmpHealth.GetMaxHitpoints())
				hurt.push({ "id": ent, "x": pos.x, "z": pos.y, "owner": owner, "hp": cmpHealth.GetHitpoints() });
		}
		return { "fights": fights, "bases": bases, "hurt": hurt };
	};
	GuiInterface.prototype.exposedFunctions.ZeroadDirectorScene = 1;
}
