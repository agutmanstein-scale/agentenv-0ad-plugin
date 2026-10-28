"""Throwaway spike: time /step on a random map with N Petra players driven through the RL interface."""
import json, os, statistics, sys, time, urllib.request

URL = os.environ.get("RL_URL", "http://127.0.0.1:6000")
PLAYERS = int(sys.argv[1]); STEPS = int(sys.argv[2]); MAP = sys.argv[3] if len(sys.argv) > 3 else "mainland"
CIVS = ["rome", "athen", "pers", "cart"]

def post(route, data):
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(f"{URL}/{route}", data.encode()), timeout=600) as r:
        body = r.read()
    return time.time() - t, body

COLORS = [(150, 20, 20), (20, 60, 180), (40, 150, 40), (200, 170, 20)]
cfg = {"settings": {"mapName": MAP, "mapType": "random", "Script": f"{MAP}.js", "Ceasefire": 0, "Biome": "generic/temperate", "PlayerPlacement": "circle", "CircularMap": True, "Size": 192, "Seed": 7, "AISeed": 7, "CheatsEnabled": False,
                    "VictoryConditions": ["conquest"], "TriggerScripts": ["scripts/TriggerHelper.js", "scripts/ConquestCommon.js", "scripts/Conquest.js"],
                    "PlayerData": [{"Name": f"P{i+1}", "Civ": CIVS[i], "Color": dict(zip("rgb", COLORS[i])), "AI": "petra", "AIDiff": 3, "AIBehavior": "balanced", "Team": -1} for i in range(PLAYERS)]},
       "mapType": "random", "map": f"maps/random/{MAP}", "script": f"{MAP}.js", "gameSpeed": 1}
for _ in range(120):
    try:
        urllib.request.urlopen(URL, timeout=2)
    except urllib.error.HTTPError:
        break
    except OSError:
        time.sleep(1)
dt, body = post("reset?playerID=-1", json.dumps(cfg))
s = json.loads(body); print(f"reset {dt:.1f}s state {len(body)/1e6:.2f} MB entities {len(s.get('entities', {}))}", flush=True)
times, sizes = [], []
for i in range(STEPS):
    dt, body = post("step", ""); times.append(dt); sizes.append(len(body))
    if i % 600 == 0:
        _, ev = post("evaluate", "Engine.QueryInterface(SYSTEM_ENTITY, IID_Timer).GetTime()")
        print(f"step {i}: sim_ms={ev.decode()} last={dt*1000:.0f}ms size={len(body)/1e6:.2f}MB", flush=True)
ts = sorted(times)
q = lambda p: ts[int(p * (len(ts) - 1))] * 1000
print(f"== players={PLAYERS} steps={STEPS}: p50={q(.5):.1f}ms p95={q(.95):.1f}ms p99={q(.99):.1f}ms max={ts[-1]*1000:.0f}ms "
      f"state p50={statistics.median(sizes)/1e6:.2f}MB max={max(sizes)/1e6:.2f}MB", flush=True)
