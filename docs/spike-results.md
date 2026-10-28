# Feasibility spike results (2026-10-24/25)

These are the spike's numbers (spec §6) and the settings they feed. The spike code in `spike/` is throwaway.

## Verdict
Real-time play works for both setups, with headroom. Nothing here blocks the build.

| Question | Answer |
|---|---|
| Can we get 0 A.D. 0.28 with the RL interface? | Yes: `ppa:wfg/0ad` on `ubuntu:24.04` (0.28.0-1ubuntu0.24.04~wfg2). releases.wildfiregames.com fails TLS from this machine, so the AppImage is out. |
| Does the engine keep 1x at 4 players? | Yes. Nonvisual p95 is 17.5 ms per 200 ms turn over 30 game-minutes (gate: < 100 ms). |
| Is Luna fast enough for a reflex layer? | Yes. p50 0.23 s, max 0.32 s. |
| Can every big model plan inside a strategist interval? | Yes. The slowest, Grok 4.7, has p50 20 s and max 25 s. |
| Does stock visual mode look smooth? | No. It advances one 200 ms turn per `/step`, so units jump 5 times a second. Fixed with the `-rl-realtime` patch. |

## Engine: `/step` time (Petra AI players, Mainland, size 192)
Measured with `spike/step_timing.py` and `spike/start_engine.sh` in the `zeroad-spike:0.28` image, on 8 CPUs with no
GPU.

| Mode | Players | Turns (game time) | p50 | p95 | p99 | max | State JSON |
|---|---|---|---|---|---|---|---|
| nonvisual | 2 | 3000 (10 min) | 5.6 ms | 8.4 ms | 11.6 ms | 47 ms | 0.22 MB p50, 0.26 MB max |
| nonvisual | 4 | 9000 (30 min) | 10.6 ms | 17.5 ms | 21.9 ms | 63 ms | 0.31 MB p50, 0.39 MB max |
| visual, Xvfb + llvmpipe (software GL) | 2 | 30 | 85 ms | 94 ms | 99 ms | 239 ms | 0.20 MB |
| visual, Xvfb + llvmpipe (software GL) | 4 | 1500 (5 min) | 96 ms | 102 ms | 108 ms | 433 ms | 0.25 MB max |

- In visual mode a step is answered from the frame loop, so step time is about one rendered frame. With software
  GL that's about 10 fps. The GPU box will be far faster. Visual mode still holds 1x here.
- Sim time per step is exactly 200 ms (`IID_Timer.GetTime()` advances 200 per step).
- `/reset` takes 1.1–1.8 s.
- The screenshot check in visual mode showed a correctly rendered session: HUD, minimap and four Petra empires.

### Engine quirks found (now in the spec)
1. **Nonvisual startup:** 0.28 quits nonvisual mode unless `-autostart=<map>` is given. `InitNonVisual` is just
   `Autostart`. It then hangs on shutdown instead of exiting, so the RL port never opens. Start with
   `-autostart-nonvisual -autostart=random/mainland -autostart-size=128`; `/reset` replaces that map.
2. **Random-map scenario JSON (0.28):** these are needed, or `generateMap` throws.
   - Top level: `"mapType": "random"`, `"map": "maps/random/mainland"`, `"script": "mainland.js"` (relative to
     `maps/random/`).
   - Under `settings`: `mapName`, `Biome` (e.g. `generic/temperate`), `PlayerPlacement` (`circle | river |
     groupedLines | randomGroup | stronghold`) and per-player `Color {r,g,b}`.
3. **Visual smoothness:** with the RL interface on, `main.cpp` never calls `g_Game->Update` from the frame loop. Only
   `/step` advances the game, once per call. The patch (now `patches/rl-realtime.patch`, 3 files)
   adds `-rl-realtime`.
   - In that mode the frame loop advances the game in real time, so motion is interpolated.
   - `/step` then only posts commands and returns state.
   - It's built from the PPA source package (now `engine/Dockerfile.rt`, `scripts/build-engine.sh rt`, U6).
   - **Verified 2026-10-25:** the package rebuild takes about 15 min on 8 CPUs. `spike/engine/Dockerfile.runtime` layers the
     patched `.deb` onto `zeroad-engine:0.28`, giving `zeroad-engine:0.28-rt`. Visual under Xvfb with `-rl-realtime`: sim
     time went 200 → 10000 ms over 10 s of wall time with no `/step` calls. Five back-to-back steps (≈75 ms each)
     added only the matching real time (+400 ms), so steps just post commands and read state.
4. **Unexplained:** `python3 /spike/step_timing.py` run via `docker exec` in the visual container was SIGKILLed at
   once, every time. The same code run through `exec(compile(...))`, or with curl, works. The cause is unknown. The
   real env uses its own client, so this doesn't block anything, but U4/U6 engine tests should watch for it.

## Models (via a LiteLLM proxy)
These use the game-sized brief from `spike/model_latency.py`: a Roman empire at 09:12 under a Persian cavalry raid,
with the seven order tools.

| Model | Route | p50 | max | n |
|---|---|---|---|---|
| GPT-6 Luna, Decisions API, 4 choice questions | `/openai/v1/decisions` | 0.23 s | 0.32 s | 12 |
| GPT-6 Astra, effort low | `/v1/responses` | 2.4 s | 3.1 s | 5 |
| GPT-6 Astra, effort medium | `/v1/responses` | 3.6 s | 4.4 s | 5 |
| GPT-6 Sol, effort low | `/v1/responses` | 3.8 s | 5.7 s | 5 |
| GPT-6 Sol, effort medium | `/v1/responses` | 5.7 s | 6.9 s | 5 |
| Claude Opus 5.5 | `/v1/chat/completions` | 7.2 s | 8.0 s | 5 |
| Gemini 3.1 Pro | `/v1/chat/completions` | 10.8 s | 15.2 s | 5 |
| Grok 4.7 | `/v1/chat/completions` | 20.3 s | 24.7 s | 5 |

Route notes:
- The proxy refuses `/v1/decisions`; use its OpenAI pass-through, `/openai/v1/decisions`. Choices are `{value, description}` and
  score levels are `{label, description}`.
- GPT-6 models reject tools with `reasoning_effort` on chat completions, so they go through `/v1/responses` with
  flat function tools.

## Settings chosen from the numbers
| Setting | Rule (spec §6) | Value |
|---|---|---|
| Luna reflex interval (Setup A) | `max(1 s, p95)` | **1 s** |
| Luna player tick (Setup B) | same | **1 s** |
| Strategist interval | `max(25 s, p95)` per model | **25 s** for all; Grok 4.7 is closest at about 25 s, so it may run 25–30 s |
| Players | 4 unless step p95 > 100 ms | **4** (nonvisual p95 17.5 ms) |
| GPT-6 reasoning effort | my call: speed matters more live | **low** default, task-configurable |
| Engine for the stream | stock vs patched | **patched `-rl-realtime`** |

## Per-setup verdict
- **Setup A (big strategist + Luna reflexes):** works. Luna answers about 4x faster than its 1 s tick needs. The
  strategist has 25 s to plan and every model answers inside that.
- **Setup B (Luna-only vs strategist-only):** works. Luna seats act every second, while strategist-only seats act
  every 25 s with the autopilot covering in between. That difference is what the demo shows.

## Spend
About $0.70 of the $25 cap (logged in the loop's `spend.jsonl`).
