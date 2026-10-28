# 0 A.D. Live Arena: design

Date: 2026-10-24

## 1. What this is and why

This is a live-streamed match of **0 A.D.**, the open-source Age of Empires-style real-time strategy game, where
**each empire is played by a different AI model**. It's an **agentenv plugin**, the real-time counterpart of the
OpenCiv3 plugin (`github.com/earakely-scale/agentenv-openciv-plugin`). The audience is people watching a stream, as
with the OpenCiv3 livestream.

This is **a fun demo, not a benchmark**. Success means it's genuinely fun to watch live, and it shows that models
can play a real-time game at full speed.

The difference from Civ: Civ waits for every player to end their turn, but 0 A.D. keeps running. Models take
anywhere from hundreds of milliseconds to tens of seconds to answer, so this design centers on keeping every empire
busy while its model thinks.

### Two setups (the user asked for both to be specced and built)
- **Setup A, shared reflexes.** Every empire is a big model plus **GPT-6 Luna on OpenAI's Decisions API**. The big
  model is the *strategist* and sets orders every ~20–30 s. Luna is the *reflex* layer and makes quick choices every
  ~1–2 s within those orders. Every empire has the same reflexes, so what's being compared is the big models.
- **Setup B, Luna as a player.** At least one empire is run **by Luna alone**, which answers both strategic and
  tactical questions as fixed-choice decisions every tick. It plays against empires run **by a big model alone**,
  which have no reflex layer and only the executor (§4.4). This answers whether being fast beats thinking deeply.

## 2. Goals, non-goals, anti-goals

**Goals**
1. The game runs at **1x speed or faster**, live, with every empire controlled by a model.
2. Both setups can be run as agentenv tasks.
3. A **feasibility spike** answers with numbers whether this works in real time (§6), before the full build.
4. A stream picture: the real 3D 0 A.D. game seen as an observer, plus a panel per empire showing the model's name,
   its latest reasoning and orders, and whether it's "thinking…".
5. A **$0 smoke task** (scripted bots, no model calls) that plays a full short match and grades it, as the
   OpenCiv3 plugin's `smoke` does.

**Non-goals**
- Training models on the game, or a rigorous benchmark.
- Humans playing, and multiplayer lobbies with remote clients.
- Naval play, and map types beyond one or two land maps.
- AI commentators. The OpenCiv3 plugin has these (`streamer/caster.py`), and they're a good add-on later, but not in v1.

**Anti-goals: these would make it a failure even if it technically works**
- **Slowing the game down.** The user ruled this out because "slowed down would not be fun to watch." The game is
  never below 1x, and freezing the game while models think is also out.
- **Visibly frozen empires.** Villagers standing idle for long stretches and armies ignoring attacks look broken on
  stream. When a model is slow, the executor keeps the empire running, and the game is never slowed.
- **Models clicking individual units.** Brains issue orders to groups of units, and a script carries them out.
- Leaking a secret (API key or stream key) into logs, the overlay, a commit or chat.

## 3. Decisions already made (by the user)
- Game: **0 A.D.** (not openage, which needs the original AoE files).
- Different models play different empires, and the match runs **live**.
- **Both** setups, A and B, are specced and built.
- **Luna via the OpenAI Decisions API** (`POST /v1/decisions`, model `gpt-6-luna`) is the fast model.
- A new repo, `agentenv-0ad-plugin`.

### Model access (verified 2026-10-24)
- A LiteLLM proxy serves the big models through OpenAI-compatible chat/tools. Confirmed ids include `anthropic/claude-opus-5-5`, `openai/gpt-6-astra`, `openai/gpt-6-sol`,
  `gemini/gemini-3.1-pro-preview` and `xai/grok-4.7`.
- **Decisions API:** the proxy refuses `/v1/decisions`; its OpenAI pass-through **`/openai/v1/decisions` works**.
  - Choice items are `{value, description}` objects, and score levels are `{label, description}`.
  - **Measured with a game-sized brief and 4 choice questions: p50 0.23 s, max 0.32 s (n=12).**
  - So the reflex interval defaults to **1 s**.
- The key lives outside the repo, in `~/.config/zeroad-arena/litellm.env` (mode 600), and goes into env and agent
  containers as `LITELLM_BASE_URL` / `LITELLM_API_KEY`. It's never committed, logged or shown on the overlay.
- **Test spend (my call): capped at $25 total** across the spike and test matches unless the user changes it. The
  runner tracks the cumulative estimate and stops before passing the cap.

### Livestream format (my call; the user hasn't specified one)
- The `livestream` task plays **one match** with the full overlay and recording. Back-to-back matches with
  intermissions are a later add-on.
- The RTMP push to X happens only on the user's say-so.

## 4. How it works

```
             ┌──────────── agentenv env container ("zeroad") ─────────────┐
 strategist  │  per-seat MCP endpoint ──► Order book (per seat)            │
 agents ─────┼─►  /players/<id>/mcp          │                             │
 (A2A, LLM)  │                               ▼                             │
             │  Luna reflex ◄─ brief ── Executor (per seat, every tick) ───┼─► commands
 Decisions ◄─┼─ (Setup A/B)                  │                             │
 API         │                               ▼                             │
             │  Game clock: POST /step every 200 ms wall clock ──► pyrogenesis --rl-interface
             │  State: /step result + /evaluate JS queries                 │   (visual on GPU box,
             │  Live server: /state.json, /live overlay page               │    nonvisual in tests)
             └─────────────────────────────────────────────────────────────┘
```

### 4.1 Game engine
- **0 A.D. Release 28 (0.28.0)**, from the Wildfire Games PPA (`ppa:wfg/0ad`) on an `ubuntu:24.04` image. (Updated
  after the spike: releases.wildfiregames.com refuses TLS from the dev machine, so the AppImage is out. Ubuntu
  22.04's apt package is 0.0.25b, which is too old.) The stream image rebuilds the PPA source package with a small
  patch, `-rl-realtime` (§6.5).
- It's driven through the engine's built-in **RL interface** (`pyrogenesis --rl-interface=127.0.0.1:6000`). From the
  source (`source/rlinterface/RLInterface.cpp`, `source/tools/rlclient/python/zero_ad`):
  - `POST /reset` takes a scenario config, the JSON from line one of a replay's `commands.txt`, plus `playerID`.
  - `POST /step` takes newline-separated lines of the form `<player>;<command json>`, for **any player**, so one
    connection drives every empire. It advances **one sim turn (`DEFAULT_TURN_LENGTH`, 200 ms)** and returns the
    full game state as JSON.
  - `POST /evaluate` runs JS in the simulation for arbitrary state queries.
  - `POST /templates` returns entity templates.
  - **The game only advances when `/step` is called.** It works in both nonvisual mode (`-autostart-nonvisual`, for
    tests and CI) and visual mode (rendered, for the stream).
- **Verified in the spike** (`docs/spike-results.md`): the PPA build has the RL interface. Nonvisual mode needs a
  throwaway `-autostart=random/mainland` map or 0.28 quits at startup; `/reset` replaces it. Random-map scenarios
  need top-level `script` plus `Biome` and `PlayerPlacement` settings. Stock visual mode advances once per `/step`,
  so units visibly jump 5 times a second; the `-rl-realtime` patch fixes that.

### 4.2 Game clock: real time without waiting on anyone
- The env owns a clock loop that calls `/step` on a **fixed wall-clock schedule, every 200 ms, which is 1x**. Every
  command queued since the previous step goes out with it.
- **Nothing ever blocks the clock.** Brains run on their own loops and drop orders into a queue.
- Speed is set per task (`speed: 1.0` default, allowed range 1.0–2.0). Above 1x, the clock steps more often. If a
  step overruns its slot, the clock doesn't try to catch up. It logs the overrun, and the metric goes into results.
- **Real-time engine mode (after the spike).** On the patched visual engine started with `-rl-realtime`, the
  engine advances the game from its own frame loop, smoothly and in real time, and `/step` only posts the queued
  commands and returns the state. The env clock code is the same in both modes; only who advances sim time
  differs. Speed above 1x is then set through the sim rate instead of the step rate.

### 4.3 What each empire is told (the brief)
- Built from the `/step` state plus `/evaluate` queries: resources, population and cap, phase, research, buildings,
  army by unit class, idle counts, events since the last brief (under attack, building lost, phase reached, unit
  trained), and enemies currently visible.
- **Fog of war is honored** (my call), using the sim's line-of-sight query, so models only learn what their empire
  can see. The stream view itself has no fog.
- There are two forms: a text brief for strategists, and a compact JSON brief for Luna's decision questions.
  Briefs are kept small (target ≤ 3k tokens) for latency.

### 4.4 Executor: keeps every empire running
A deterministic per-seat script that runs every tick (~1 s). It does two jobs.
1. **Carries out orders** from the order book (§4.5) as real 0 A.D. commands: picking units, choosing building
   placement, issuing train/gather/construct/attack/research commands.
2. **Autopilot defaults**, so a slow brain never leaves the empire frozen:
   - idle gatherers go back to work, following the current economy split;
   - a house goes up before population is capped;
   - villagers keep training up to the target;
   - units use the engine's stances, so they defend when attacked;
   - drop-sites get built near resources being worked far from one.

The autopilot is deliberately mediocre. Strategy, timing, army composition, when to advance phase, and where and
when to attack all come from the brains.

### 4.5 Order vocabulary (shared by every brain)
Everything is about groups, never single units. The same small, typed set is used in all setups, and it's exposed
as MCP tools for strategists and as choice options for Luna.
- `set_economy(food, wood, stone, metal)`: the gatherer split, as percentages.
- `train(role, count, at?)`: role is one of `villager | infantry_melee | infantry_ranged | cavalry | siege | champion`,
  mapped to the civ's real templates.
- `build(structure, where)`: `where` is one of `base | near_<resource> | toward_<player> | front`.
- `research(tech)` and `advance_phase()`.
- `army(action, target)`: action is one of `gather | attack | raid | defend | retreat | scout`. The target is a
  player, a named area or `home`.
- `say(text)`: a short line in character for the overlay (strategists only, and optional).
- Every tool result returns accepted, rejected with the reason, or partially done, plus the current brief footer,
  following the error style of the OpenCiv3 plugin's protocol (`message`, `alternatives`, `suggest`).

### 4.6 Brains
- **Strategist (big models).** A thin agentenv **A2A agent**, `zeroad-commander`, that runs a loop: read the brief,
  call one model through LiteLLM with the order tools, apply its orders, and wait for the next trigger. The trigger
  is every N seconds (default 25) or an urgent event (under attack, building lost), whichever comes first. One
  model call is in flight per seat at a time.
  - **Routes (verified in the spike):** OpenAI GPT-6 models go through LiteLLM `/v1/responses`, because chat
    completions rejects function tools combined with `reasoning_effort`. Responses tools are flat
    `{"type": "function", name, description, parameters}` with `reasoning: {effort}`. Every other model uses
    `/v1/chat/completions` with tools.
  - **My call:** not Claude Code, Codex or Gemini CLI as players, unlike the OpenCiv3 plugin. Each tool call through a
    coding CLI adds seconds, which a real-time game can't absorb. Because the env exposes a standard per-seat MCP
    endpoint, any MCP agent *can* still play.
- **Luna reflex (Setup A).** Inside the env, per seat, every ~1–2 s: **one** Decisions API request carrying a
  `choice` question for each open decision. Examples: an idle army group chooses `attack / hold / retreat /
  regroup`. A production building with a free queue chooses among the roles. Gatherers whose resource ran out
  choose a resource. The choices are limited to what the current strategist orders allow, and each answer becomes
  executor commands.
- **Luna player (Setup B).** The same loop as a reflex seat, plus strategic `choice`/`score` questions answered every
  tick: advance phase now, which economy split, whom to attack, what to train. There is no free-text planning.
  - **My call:** it's a thin **A2A agent**, `zeroad-luna`, that plays through its seat's MCP endpoint, so it is an
    `agent` slot and graded like any other seat. The Setup A reflex layer stays inside the env.
- **Strategist-only seats (Setup B):** strategist and executor, with no reflex layer.
- **Petra seats.** 0 A.D.'s built-in AI, configured as an AI player slot. It costs nothing, so it's the control and
  can fill empty seats.

### 4.7 Match format (my calls, adjustable per task)
- 2–4 players, free-for-all, on a small land random map such as **Mainland**. Each seat gets a different civ chosen
  for variety: Romans, Athenians, Persians, Carthaginians, and so on.
- Victory: conquest. There's also a **time cap (default 40 min of game time)**, after which the highest score wins.
- Model lineup: set per task through `deploy_agent` / `prompt_agent` as the OpenCiv3 plugin's tasks do. Defaults are Opus 5.5 and
  GPT-6 Astra, plus one or two more models available through LiteLLM.
- Grading follows agentenv-game-env's convention: a win scores 1, a draw 0.5 and a loss 0.
  - **My call on forfeits:** a brain call that errors or runs over 120 s counts as *missed*. A seat with more than
    50% of its calls missed scores 0, because that means its model was effectively unreachable. Slow but answering
    models are never zeroed.

### 4.8 Live view and stream
- **Rendering:** 0 A.D. in visual mode as an **observer** (player −1, no fog, and the observer setting needs
  verifying in the spike), on an Xorg display with a GPU.
- **Auto-director:** a small 0 A.D. GUI mod (JS) that moves the camera to where the action is. The default is the
  largest fight, falling back to a slow tour of the bases, and a cut lasts at least ~8 s.
- **Overlay panel:** a page served by the env's live server (`/live?stream`) shown beside the game: per empire, the
  model name, civ, color, resources, population, army size, the latest orders and `say` line, and a live
  "thinking… 12 s" timer. It also shows a match clock and a scoreboard.
- **Capture and push:** ffmpeg captures the display to a recorded file (always) and to RTMP (X) on request,
  following the stack in `agentenv-airops-plugin/live/`: one GPU box (g6.4xlarge, L4), Xorg, Chrome, ffmpeg NVENC
  and secrets in SSM.
- This dev machine has **no GPU**. Everything except the real rendered stream is built and tested here. The visual
  path gets a low-fps check under Xvfb, and the GPU box is a separate step that needs approval.

### 4.9 agentenv packaging (following the OpenCiv3 plugin)
- Python package `agentenv_zeroad`, with entry points `agent_env.bundles` (bundle `zeroad`), `agent_env.cli_plugins`
  (`agent-env zeroad setup|recordings|…`) and task steps as needed.
- Uses `agentenv-game-env` (lobby `urn:game:lobby/v1`, match `urn:game:match/v1`), pinned to the same commit as
  the OpenCiv3 plugin. Env image: Docker on `ubuntu:24.04` with 0 A.D. from `ppa:wfg/0ad`; the stream image swaps in
  the `-rl-realtime` rebuild.
- Tasks: `smoke` (scripted bots, $0), `petra` (Petra vs Petra, $0), `spike` (§6), `setup-a`, `setup-b`, and
  `livestream`.
- Results: match outcome plus metrics per seat (§5), the replay (`commands.txt`, which 0 A.D. can play back),
  recordings and decision logs.

## 5. Metrics (also the spike's output)
For each seat, `waiting_share` is the fraction of match time the empire has something it can't do until its brain
answers. It's the one number that says whether real time works. It comes with:
- decision latency p50/p95/max per model;
- decisions per minute;
- executor idle-villager-seconds and idle-production-seconds;
- clock overruns.

## 6. Feasibility spike (phase 0, throwaway)
Runs before the full build, headless here:
1. 0 A.D. 0.28 AppImage: confirm `--rl-interface` works, nonvisual at first, then visual under Xvfb.
2. Clock: 4 players, 20 game-minutes of scripted play. Measure `/step` time and state size. **Gate: p95 step time
   < 100 ms** at 4 players, so 1x has headroom.
3. Models: send real briefs taken from that game. Measure latency for **Luna Decisions** (target p95 < 1.5 s), and
   for each big model with the order tools (target p95 < 30 s).
4. Report: a table of numbers per setup. **No gate stops the build (my call).** The build runs straight through, and
   each measurement feeds a setting:
   - **Luna:** the reflex interval becomes `max(1 s, measured p95)`. An independent test put Luna's median at about
     1.46 s, so expect a 2–4 s cadence. That's still about 10x faster than a strategist's 25 s, so Setup A stays
     different from a strategist-only seat, and Setup B is always built. If Luna is slow, that's part of the result.
   - **Big models:** a model's strategist interval becomes `max(25 s, its p95)`.
   - **Steps:** if p95 step time is over 100 ms at 4 players, drop to 3 players, then 2. The game never runs below 1x.
   - **Non-costly fallbacks (source build, engine patch, fewer players) are taken automatically.** Spending money and
     anything public still need the user's go-ahead (§7).
5. **Expected engine patch:** `source/main.cpp` skips `g_Game->Update` while the RL interface is on, so the visual
   game probably won't interpolate motion between external steps. Plan for a small patch, plus a source build of
   0.28, that lets the engine's own frame loop advance and interpolate. The env keeps the authority over pacing.
   - **Confirmed (2026-10-25):** the patch is `-rl-realtime` (`engine/rl-realtime.patch`, 4 files). It is built
     from the PPA source package. In that mode the engine owns sim time and the env clock posts commands and reads
     state. Results are in `docs/spike-results.md`.

## 7. Build order (milestones)
1. **Spike** (§6).
2. **Env core:** game process, clock, state and briefs, order vocabulary, executor and autopilot, Petra slots, MCP
   tools, scripted bot, and the `smoke` and `petra` tasks. Tests use the nonvisual engine plus a fake engine for
   unit tests.
3. **Brains:** `zeroad-commander` agent, Luna reflex and Luna player, the `setup-a` and `setup-b` tasks, and
   metrics.
4. **Live view:** observer rendering, the director mod, overlay page, recording. Checked under Xvfb here.
5. **Stream:** the GPU box and RTMP push, reusing the AirOps approach. **Each costly or public step needs the
   user's go-ahead.**

## 8. Costs (rough)
- Luna: about 3k input tokens × 1 request/s × 4 seats × 2,400 s comes to about 29M tokens, roughly **$3 per 40-min
  match**. Output is free.
- Strategists: about 100 calls × ~8k tokens per seat per match, which is a few dollars per model per match at
  frontier prices.
- GPU box: about $1.32/h while streaming.

## 9. Risks
| Risk | Mitigation |
|---|---|
| The engine download is unreachable | Resolved: `ppa:wfg/0ad` on `ubuntu:24.04` (0.28.0, RL interface present) |
| Visual mode looks jerky when stepped externally | Confirmed jerky. `-rl-realtime` patch, built from the PPA source package |
| Building placement is hard to get right | Use engine placement checks via `/evaluate`, searching outward from anchor points |
| The Decisions API isn't reachable through the LiteLLM proxy | Direct OpenAI key in the env (secret) |
| Luna's speed claim doesn't hold up | The spike measures it, and Setup A falls back to executor-only reflexes |
| Observer camera control from outside | The GUI mod does the director work inside the game |

## 10. Left to the builder
Module layout, the brief's wording and format, executor heuristics, Luna's question set, the director's heuristics,
overlay styling, test structure, and how agentenv-game-env is wired in. These are within the constraints above.
