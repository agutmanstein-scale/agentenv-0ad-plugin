// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * zeroad-director: the live stream's camera, so nobody has to steer the observer's view.
 *
 * About once a second it reads the scene (where units are fighting, where the civic centres stand) and picks a shot:
 * the largest fight (the most units in combat within a few grid cells, of two players or more, so a hunt or a lone
 * raid on a building isn't one unless it is big), else a slow tour of the bases. With the mod's call the scene also
 * lists every damaged entity's hitpoints, and a fight counts only where something lost hitpoints in the last
 * hitMemoryMs: units stuck in a combat state that hit nothing (an army that can't reach its target) hold no shot,
 * and a base being razed by one player is a fight of two (the damaged buildings' owner counts). A new shot is
 * a "cut": the camera glides there in ZEROAD_DIRECTOR.glideMs with an ease-in-out curve, and the shot is held at least
 * minShotMs before the director cuts again. While a fight shot is held the camera follows the fight's centre slowly.
 * The camera is moved every frame with Engine.CameraMoveTo, which the engine smooths further.
 *
 * The scene comes from the mod's GuiInterface call (simulation/components/GuiInterface~zeroad-director.js), or from the
 * stock entity-state calls if that call is missing. Cuts are written to the engine's log (mainlog.html).
 * The class takes its engine and clock as arguments so tests can run it under node.
 */

var ZEROAD_DIRECTOR = {
	"queryMs": 1000,
	"minShotMs": 8000,
	"tourShotMs": 14000,
	"glideMs": 3000,
	"followRate": 0.05,
	"cell": 40,
	"minFight": 3,
	"bigFight": 8,
	"hitMemoryMs": 5000,
	"fpsLogMs": 10000
};

class ZeroadDirector
{
	constructor(engine, now, settings)
	{
		this.engine = engine;
		this.now = now;
		this.settings = Object.assign({}, ZEROAD_DIRECTOR, settings || {});
		this.shot = null;
		this.glide = null;
		this.lastQuery = -Infinity;
		this.tour = -1;
		this.cuts = [];
		this.useStock = false;
		this.hp = null;
		this.hits = [];
	}

	tick()
	{
		const now = this.now();
		this.countFrame(now);
		if (now - this.lastQuery >= this.settings.queryMs)
		{
			this.lastQuery = now;
			this.decide(this.scene(), now);
		}
		this.move(now);
	}

	/** The session ticks once a rendered frame: log the frame rate every fpsLogMs, for the stream's health checks. */
	countFrame(now)
	{
		if (this.fpsSince === undefined)
			this.fpsSince = now;
		this.frames = (this.frames || 0) + 1;
		if (now - this.fpsSince >= this.settings.fpsLogMs)
		{
			this.fps = this.frames * 1000 / (now - this.fpsSince);
			if (this.engine.log)
				this.engine.log("zeroad-director: " + this.fps.toFixed(1) + " fps");
			this.fpsSince = now;
			this.frames = 0;
		}
	}

	scene()
	{
		if (!this.useStock)
		{
			let scene;
			try
			{
				scene = this.engine.GuiInterfaceCall("ZeroadDirectorScene");
			}
			catch (e)
			{
				scene = null;
			}
			if (scene)
				return scene;
			// Without the mod's call, don't ask for it again: every failed call logs an error.
			this.useStock = true;
		}
		return this.stockScene();
	}

	/** The same scene from the stock calls: slower (every entity's full state), used only without the mod's call. */
	stockScene()
	{
		const fights = [];
		const bases = [];
		const ents = this.engine.GuiInterfaceCall("GetNonGaiaEntities") || [];
		for (const { state } of this.engine.GuiInterfaceCall("GetMultipleEntityStates", ents) || [])
		{
			if (!state || !state.position)
				continue;
			const at = { "x": state.position.x, "z": state.position.z, "owner": state.player };
			if (state.identity && state.identity.classes.indexOf("CivCentre") != -1)
				bases.push(at);
			if (state.unitAI && state.unitAI.state && state.unitAI.state.indexOf("COMBAT") != -1)
				fights.push(at);
		}
		return { "fights": fights, "bases": bases };
	}

	/**
	 * The entities that lost hitpoints since the last scene, kept for hitMemoryMs; null when the scene lists no
	 * hitpoints (the stock calls).
	 */
	recentHits(scene, now)
	{
		if (!scene.hurt)
			return null;
		const prev = this.hp;
		this.hp = new Map(scene.hurt.map(h => [h.id, h.hp]));
		if (prev)
			for (const h of scene.hurt)
				if (!prev.has(h.id) || h.hp < prev.get(h.id))
					this.hits.push({ "x": h.x, "z": h.z, "owner": h.owner, "hit": true, "at": now });
		this.hits = this.hits.filter(h => now - h.at < this.settings.hitMemoryMs);
		return this.hits;
	}

	/**
	 * The largest fight: the grid cell whose 3x3 neighbourhood holds the most points (units in combat and recent
	 * hits), and its centre. With needHits a neighbourhood with no hit is skipped.
	 */
	hottest(points, needHits)
	{
		const cell = this.settings.cell;
		const counts = new Map();
		const hitCounts = new Map();
		const key = (i, j) => i + "," + j;
		for (const p of points)
		{
			const k = key(Math.floor(p.x / cell), Math.floor(p.z / cell));
			counts.set(k, (counts.get(k) || 0) + 1);
			if (p.hit)
				hitCounts.set(k, (hitCounts.get(k) || 0) + 1);
		}
		let best = null;
		for (const k of counts.keys())
		{
			const [i, j] = k.split(",").map(Number);
			let n = 0;
			let hits = 0;
			for (let di = -1; di <= 1; ++di)
				for (let dj = -1; dj <= 1; ++dj)
				{
					n += counts.get(key(i + di, j + dj)) || 0;
					hits += hitCounts.get(key(i + di, j + dj)) || 0;
				}
			if (needHits && !hits)
				continue;
			if (!best || n > best.n)
				best = { "i": i, "j": j, "n": n };
		}
		if (!best)
			return null;
		const near = points.filter(f =>
			Math.abs(Math.floor(f.x / cell) - best.i) <= 1 && Math.abs(Math.floor(f.z / cell) - best.j) <= 1);
		const owners = new Set(near.map(f => f.owner));
		return {
			"x": near.reduce((s, f) => s + f.x, 0) / near.length,
			"z": near.reduce((s, f) => s + f.z, 0) / near.length,
			"n": near.length,
			"owners": owners.size
		};
	}

	decide(scene, now)
	{
		const s = this.settings;
		const hits = this.recentHits(scene, now);
		const hot = this.hottest((scene.fights || []).concat(hits || []), hits !== null);
		const fight = hot && hot.n >= s.minFight && (hot.owners >= 2 || hot.n >= s.bigFight) ? hot : null;
		const held = this.shot ? now - this.shot.start : Infinity;
		if (this.shot && this.shot.kind == "fight" && fight &&
			Math.hypot(fight.x - this.shot.x, fight.z - this.shot.z) < 3 * s.cell)
		{
			// The same fight: follow its centre without a cut.
			this.shot.x = fight.x;
			this.shot.z = fight.z;
			this.shot.n = fight.n;
			return;
		}
		if (held < s.minShotMs)
			return;
		if (fight)
		{
			this.cut({ "kind": "fight", "x": fight.x, "z": fight.z, "n": fight.n }, now);
			return;
		}
		const bases = (scene.bases || []).slice().sort((a, b) => a.owner - b.owner || a.x - b.x || a.z - b.z);
		if (!bases.length || (this.shot && this.shot.kind == "base" && held < s.tourShotMs))
			return;
		this.tour = (this.tour + 1) % bases.length;
		const base = bases[this.tour];
		this.cut({ "kind": "base", "x": base.x, "z": base.z, "owner": base.owner }, now);
	}

	cut(shot, now)
	{
		const pivot = this.engine.GetCameraPivot();
		const from = pivot ? { "x": pivot.x, "z": pivot.z } : { "x": shot.x, "z": shot.z };
		this.shot = Object.assign({ "start": now }, shot);
		this.glide = { "from": from, "start": now };
		this.cuts.push({ "kind": shot.kind, "x": Math.round(shot.x), "z": Math.round(shot.z), "at": now });
		if (this.engine.log)
			this.engine.log("zeroad-director: cut to " + shot.kind + " at " + Math.round(shot.x) + "," +
				Math.round(shot.z) + (shot.n ? " (" + shot.n + " units in combat)" : ""));
	}

	move(now)
	{
		if (!this.shot)
			return;
		const target = { "x": this.shot.x, "z": this.shot.z };
		let at;
		if (this.glide)
		{
			const t = Math.min(1, (now - this.glide.start) / this.settings.glideMs);
			const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
			const from = this.glide.from;
			at = { "x": from.x + (target.x - from.x) * e, "z": from.z + (target.z - from.z) * e };
			if (t >= 1)
			{
				this.glide = null;
				this.at = at;
			}
		}
		else
		{
			const cur = this.at || target;
			const r = this.settings.followRate;
			at = { "x": cur.x + (target.x - cur.x) * r, "z": cur.z + (target.z - cur.z) * r };
		}
		this.at = at;
		this.engine.CameraMoveTo(at.x, at.z);
	}
}

var g_ZeroadDirector = null;
var g_ZeroadDirectorFailed = false;

if (typeof onTick == "function")
{
	const zeroadSessionTick = onTick;
	onTick = function()
	{
		zeroadSessionTick();
		if (g_ZeroadDirectorFailed)
			return;
		try
		{
			if (!g_ZeroadDirector)
			{
				// The stream's view: -1 is the observer (no fog, no player HUD); the engine test checks this line.
				if (typeof log == "function")
					log("zeroad-director: viewing as player " + Engine.GetPlayerID());
				g_ZeroadDirector = new ZeroadDirector({
					"GuiInterfaceCall": (name, args) => Engine.GuiInterfaceCall(name, args),
					"GetCameraPivot": () => Engine.GetCameraPivot(),
					"CameraMoveTo": (x, z) => Engine.CameraMoveTo(x, z),
					"log": typeof log == "function" ? log : null
				}, () => Date.now());
			}
			g_ZeroadDirector.tick();
		}
		catch (e)
		{
			g_ZeroadDirectorFailed = true;
			if (typeof log == "function")
				log("zeroad-director stopped: " + e + "\n" + e.stack);
		}
	};
}
