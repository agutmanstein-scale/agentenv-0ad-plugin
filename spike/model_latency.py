"""Throwaway spike: how long does each strategist model take to answer one brief with the order tools?"""
import json, os, statistics, sys, time, urllib.request

BASE = os.environ["LITELLM_BASE_URL"].rstrip("/")
KEY = os.environ["LITELLM_API_KEY"]
MODELS = sys.argv[1].split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SPEND = sys.argv[3] if len(sys.argv) > 3 else None

BRIEF = """You command the Romans in a live 0 A.D. match (free-for-all, 4 empires). The game does not wait for you.
Game time 09:12. Phase: village (town phase costs 500 food 500 wood, needs 5 village structures; you have 6).
Resources: food 620, wood 480, stone 150, metal 90. Population 42/50 (houses: 5, one under construction).
Economy: 30 villagers (food 14, wood 11, stone 2, metal 3, idle 0). Gatherer split target: food 45 wood 35 stone 10 metal 10.
Buildings: civic centre, 2 barracks, farmstead, storehouse, 6 fields, 5 houses, 1 sentry tower.
Army (home): 8 hastati (infantry_melee), 6 velites (infantry_ranged), 2 equites (cavalry).
Production: barracks#1 idle, barracks#2 training 3 hastati (20 s left), civic centre training villager.
Visible enemies: Persians (orange) 7 cavalry archers raiding your eastern woodline (400 m from civic centre); 2 of your woodcutters killed in the last minute.
Known rivals: Persians (east, 34 pop seen), Athenians (north, unscouted), Carthaginians (west, 28 pop seen).
Events since last brief: under_attack(woodline east), unit_lost(woodcutter x2), house_complete.
Your last orders: set_economy(45,35,10,10); train(infantry_melee,5,barracks); scout(north).
Decide your orders now. Use the tools. Also give one short in-character line with say()."""

ROLES = ["villager", "infantry_melee", "infantry_ranged", "cavalry", "siege", "champion"]
def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
TOOLS = [
    tool("set_economy", "Set gatherer split in percent.", {k: {"type": "integer"} for k in ("food", "wood", "stone", "metal")}, ["food", "wood", "stone", "metal"]),
    tool("train", "Train units.", {"role": {"type": "string", "enum": ROLES}, "count": {"type": "integer"}, "at": {"type": "string"}}, ["role", "count"]),
    tool("build", "Build a structure.", {"structure": {"type": "string"}, "where": {"type": "string"}}, ["structure", "where"]),
    tool("research", "Research a tech.", {"tech": {"type": "string"}}, ["tech"]),
    tool("advance_phase", "Advance to the next phase.", {}, []),
    tool("army", "Command the army.", {"action": {"type": "string", "enum": ["gather", "attack", "raid", "defend", "retreat", "scout"]}, "target": {"type": "string"}}, ["action", "target"]),
    tool("say", "One short in-character line for viewers.", {"text": {"type": "string"}}, ["text"]),
]

def call(model):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": BRIEF}], "tools": TOOLS,
                       "tool_choice": "auto", "max_tokens": 2000}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    dt = time.time() - t
    msg = d["choices"][0]["message"]
    calls = [c["function"]["name"] for c in msg.get("tool_calls") or []]
    return dt, calls, d.get("usage", {})

for m in MODELS:
    ts, toks = [], [0, 0]
    for i in range(N):
        try:
            dt, calls, u = call(m)
        except Exception as e:  # noqa: BLE001 - spike
            print(f"{m}: error {str(e)[:200]}"); continue
        ts.append(dt); toks[0] += u.get("prompt_tokens", 0); toks[1] += u.get("completion_tokens", 0)
        print(f"{m} #{i}: {dt:.1f}s tools={calls}", flush=True)
    if ts:
        ts.sort()
        print(f"== {m}: n={len(ts)} p50={statistics.median(ts):.1f}s max={ts[-1]:.1f}s in={toks[0]} out={toks[1]}", flush=True)
        if SPEND:
            with open(SPEND, "a") as f:  # rough upper bound: $15/M in, $75/M out
                f.write(json.dumps({"what": f"spike latency {m}", "usd_est": round(toks[0] * 15e-6 + toks[1] * 75e-6, 4)}) + "\n")
