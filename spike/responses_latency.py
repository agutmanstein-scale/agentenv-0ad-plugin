"""Throwaway spike: GPT-6 strategists through /v1/responses (chat completions refuses tools with reasoning)."""
import json, os, statistics, sys, time, urllib.request
src = open(os.path.join(os.path.dirname(__file__), "model_latency.py")).read().split("def call(model):")[0]
ns = {"__name__": "x"}; argv = sys.argv; sys.argv = ["x", "x"]; exec(src, ns); sys.argv = argv
TOOLS = [{"type": "function", **t["function"]} for t in ns["TOOLS"]]
BASE, KEY = os.environ["LITELLM_BASE_URL"].rstrip("/"), os.environ["LITELLM_API_KEY"]
for m in sys.argv[1].split(","):
    for effort in sys.argv[2].split(","):
        ts, tin, tout = [], 0, 0
        for i in range(3):
            body = json.dumps({"model": m, "input": ns["BRIEF"], "tools": TOOLS, "reasoning": {"effort": effort}, "max_output_tokens": 4000}).encode()
            req = urllib.request.Request(f"{BASE}/v1/responses", body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            t = time.time()
            try:
                d = json.load(urllib.request.urlopen(req, timeout=180))
            except urllib.error.HTTPError as e:
                print(m, effort, "error", e.read()[:300]); break
            dt = time.time() - t; ts.append(dt)
            u = d.get("usage", {}); tin += u.get("input_tokens", 0); tout += u.get("output_tokens", 0)
            print(f"{m} {effort} #{i}: {dt:.1f}s tools={[o.get('name') for o in d.get('output', []) if o.get('type') == 'function_call']}", flush=True)
        if ts:
            ts.sort(); print(f"== {m} effort={effort}: p50={statistics.median(ts):.1f}s max={ts[-1]:.1f}s in={tin} out={tout}", flush=True)
            with open(sys.argv[3], "a") as f:
                f.write(json.dumps({"what": f"spike latency {m} {effort}", "usd_est": round(tin * 15e-6 + tout * 75e-6, 4)}) + "\n")
