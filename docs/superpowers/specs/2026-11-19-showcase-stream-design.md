# Showcase stream: the trained Qwen plays Opus, Astra and Petra-3 live

Status: design, 19 Nov 2026. Approach A of the livestream options (showcase, no training during the stream). Air date:
Friday 20 Nov 2026, if every stage passes in full (see "Order of work and the air date").

## Why

Run 1 turned base Qwen3-8B into a model that wins 72.4% of held-out games against Petra (base 37.8%), and then beat
Opus 13–3 and played Astra about even (9–7) (`docs/s2-run1-results.md`). The stream is meant to make that real for people
in the org who will never read the results doc. Someone who watches for ten minutes should come away believing: *a small
open model learned this game in 8 hours on one box, it now beats Opus, and here it is doing it.* The user asked for one or
two games per round and a live tracker of scores, wins and a few training numbers.

The audience and that takeaway are Claude's proposal (the user didn't answer when asked and then picked the approach built
on them).

## Goals

- The trained model (Run 1's final adapter, version 50) plays one game at a time, live and rendered, against Opus,
  Astra and Petra-3 in turn.
- Every game is played under the same rules as the 13–3, so the record on screen means the same thing.
- A side panel shows:
  - the running record for the session against each opponent;
  - the current game's state;
  - each side's latest orders;
  - the backstory: Run 1's gains and the training numbers.
- Between games, a short intermission card: the last result, the next matchup and the session table.
- A session runs unattended for its set length (6 hours by default, about 3–4 rounds and 11 games), recorded from start
  to end, and the box powers itself off after.

## Non-goals

- Training during the stream (that's option B, with Run 2).
- More than one game on screen at a time.
- Commentary or voice.
- A public page.
- An edited highlight reel.
- Going public. The push to X exists (`live/push.sh`) and is **not approved**. Nothing in this build uses it. Where the
  stream airs is decided after the rehearsal.

## Anti-goals

These would make the stream a failure even if it ran:

- **Different rules from the 13–3.** Real-time play, a different decision delay, or other seat settings, with the
  record presented as comparable.
- **Cherry-picking.** Every game played in a session counts in the on-screen record. A game lost to a technical failure
  (an engine crash, a failed box) shows as "no result" and stays on the table.
- **Anything secret on screen or in logs:** the LiteLLM key, endpoints, prompts or the bucket name. The overlay already
  carries none of these (`live.py`), and that must hold.
- **Spend beyond the cap.** The box's budget guard refuses a game that would pass it.

## Constraints

- **The rules the 13–3 was played under (training's lockstep match, `clock.py` `LockstepClock`):**
  - A decision's brief is frozen at game time T and its orders land at exactly T + 5 s, game time (`latency_ms` 5000).
  - The game doesn't pass T + 5 s until that seat's reply is in, up to `brain_timeout_s`; a later reply is recorded as
    missed.
  - Decisions come every 25 s of game time per seat, plus urgent ones.
  - Interface v2, 30-minute cap, net-worth score rule.
  - A win is a conquest, or ahead in net worth at 30:00.
  - Seat settings are the frontier panel's (`frontier_specs` / `teacher_specs` in `train/s2.py`): the teachers'
    effort and max tokens, and the policy's sampling (`TRAIN_SAMPLING`).
- **How the existing renderer differs.** It plays in real time (`-rl-realtime`: the engine's frame loop advances the
  game, and model thinking time costs game time). That's a different contest, so it can't be used as is.
- **The stock rendered engine can't be used either.** Stepped one turn per `/step`, it advances in 200 ms jumps, so
  units move five times a second (`docs/spike-results.md`). Smooth live play under the lockstep rules needs a new
  engine patch (decided at planning, see "The box").
- **How fast each model replies, measured on base's frontier games** (`live/fetched/run1/base-frontier`):

  | Model | Median | 90th percentile | 99th percentile | Calls over 5 s |
  |---|---|---|---|---|
  | Opus | 3.8 s | 5.9 s | 7.9 s | 24% |
  | Qwen, on the shared training servers | 2.8 s | 5.4 s | 6.9 s | 13% |
  | Astra (those games had failed calls) | 1.7 s | 2.6 s | 5.6 s | 2% |

- **Pieces that already exist:**
  - `LiveMatch` takes a `Broadcast`: Chrome overlay, window layout, ffmpeg with NVENC, an RTMP leg to the box's relay,
    and an MP4.
  - `LockstepMatch` is a `LiveMatch`.
  - `overlay_state(match)` is a pure function of a `LiveMatch`.
  - The director mod moves the camera.
  - The box-side relay (mediamtx) records everything in one-hour fMP4 segments.
- **Serving Qwen:** the pilot's vLLM setup (`live/box/pilot-bootstrap.sh`) serves Qwen with runtime LoRA. It requires a
  GPU with at least 40 GB, so the box needs an L40S.
- **Standing rules** (CLAUDE.md, memory, this session):
  - **Scripts:** every script that launches, changes or spends prints its plan without `--yes`. Run it once without
    `--yes`, read the plan, then run it with `--yes`; never chain the two.
  - **On the box:** no ad-hoc commands outside `live/` scripts.
  - **Secrets:** the LiteLLM key is never printed or pasted (sha256 prefixes and status codes only). The bucket name
    is redacted.
  - **Pushing:** `gh stack push` only, with the leak grep before every push.
  - **Implementing:** Sonnet subagents implement; the main session reviews.
  - **Tests:** only changed test files run locally; CI runs the rest.
  - **Approval:** AWS writes (launching the box) need the user's go-ahead at the time.

## Approaches considered

| | What viewers see | Cost | Verdict |
|---|---|---|---|
| **A. Showcase** | The finished model plays live against Opus, Astra and Petra-3 | ~$5–6/h | **Picked by the user.** Leads with the headline and doesn't wait on Run 2. |
| B. Ride along Run 2 | Training runs; each round the newest version plays one rendered game | ~$12/h + ~$75/day frontier | Later. The curve only climbs if Run 2 restarts from base. |
| C. Highlights only | Clips of the best games rendered afterwards from saved replays | a few dollars on top of Run 2 | Not live. |

Within A, how a game is played and drawn:

| | How | Trade-off | Verdict |
|---|---|---|---|
| **A1. Live, paced lockstep** | Training's lockstep match on the rendered engine, stepped at no more than real-time speed. When a model is late past T + 5 s, the game holds until it replies. | The same rules as the 13–3, and live. Brief holds when Opus is slow: about 1 in 4 of its calls, mostly 1–3 s. | **Claude's call.** |
| A2. Headless lockstep, shown as a replay | Play headless with replays saved (`--replays`), then render the game's `commands.txt` ~15 min later | The same rules and smooth video, but delayed. Needs a replay renderer and an overlay synced to the game's timeline: more new code, and replay fidelity is unproven. | Fallback if the rendered engine can't be stepped by the lockstep clock. Built alongside the spike because there's an air date. |
| A3. Real time on the existing stack | `zeroad-commander` agents, including Qwen through local vLLM, on the `-rl-realtime` engine | Least new code, but a different contest: Qwen never trained in real time, and slow models lose game time | Rejected (anti-goal 1) |

## The design

### What's on screen

The existing 1920×1080 layout (`live.py`): the game in 1440×1080 with the director camera and no fog, and the overlay in
480×1080 beside it.

```
+------------------------------------------+-------------------------------+
|                                          | QWEN3-8B + 8 h RL  vs  OPUS   |
|                                          | game 4 · round 2 · 14:32      |
|   0 A.D., observer view, director cam    |-------------------------------|
|                                          | SESSION   W-L                 |
|                                          |  vs Opus     1-0              |
|                                          |  vs Astra    0-1              |
|                                          |  vs Petra-3  1-0              |
|                                          |-------------------------------|
|                                          | [existing per-empire panels:  |
|                                          |  civ, resources, pop, army,   |
|                                          |  latest orders, thinking 4 s] |
|                                          | ⏸ waiting on Opus · 6.2 s     |
|                                          |-------------------------------|
|                                          | RUN 1: 8 h on one box         |
|                                          |  vs Petra  38% → 72% won      |
|                                          |  vs Opus 13-3 · Astra 9-7     |
|                                          |  43.7M of 8.2B weights (0.5%) |
+------------------------------------------+-------------------------------+
```

- **Series header:** the model names, game and round number, and the game clock.
- **Session table:** wins and losses this session by opponent, plus a draw if net worth is exactly tied at 30:00. A failed
  game shows "no result".
- **Per-empire panels:** as today.
- **The hold line:** while the clock is holding at T + 5 s for a seat, the panel says which model it's waiting on and
  for how long. This is the "thinking…" timer already in the overlay, labelled as a hold.
- **Backstory card:** fixed text from `docs/s2-run1-results.md`:
  - Run 1's won rate against Petra, before and after;
  - the Opus and Astra records;
  - weights trained, 43.7M of 8.2B (0.5%);
  - optionally 50 updates and 1,462 games.
- **Intermission card,** in the overlay's place for about 2 minutes between games: who won the last game and how
  (conquest or the 30:00 net-worth lead), the next matchup and the session table. The game view shows the final frame
  or the next game's map.

The overlay is the existing page plus these blocks. Its look follows the current overlay's.

### The session

- **The rotation:** Opus, Astra, Petra-3, repeated. One rotation is a round, about 1.6 h at 30-minute games (conquests
  end sooner).
- **Session length:** set at start, 6 h by default. The session ends after the game in progress at that point. A
  maximum game count is a second stop.
- **Seats:** they alternate between games, as in the frontier panel.
- **Starts:** fresh held-out starts. Civ pairs come from `[held_out] civ_pairs` and seeds from `[held_out] seed_ranges`,
  picking seeds no earlier panel or game used, so none is a start the model has seen. Map: mainland, the only one.
- **Petra-3:** Petra at the difficulty the panels call petra-3, with the same settings as in training.
- **Results:** each game writes its `results.json` as training's games do. A session file collects them, and the
  overlay's table is read from it.

### Which games count (Claude's call)

The rule is the same for both sides, so it can't favour Qwen.

- **A game counts** unless a technical failure voids it.
  - Slow replies are part of the rules: a reply later than `brain_timeout_s` is a missed decision, as in training.
  - `grading.forfeited` (more than half the calls missed scores 0) applies as in training.
- **"No result"** is shown when:
  - either side's failed calls exceed 10% of its decisions. Failed calls are errors from the proxy or from vLLM, not
    slow replies.
  - or the engine or the box fails.

  Qwen's vLLM failing on the shared GPU falls under the same 10% rule as the proxy failing Opus or Astra. This is
  deliberately stricter than the frontier panel, which counted games with failed teacher calls. On a stream, a win over
  a crippled opponent shouldn't go on the board.
- **Ending early:** a game ends as "no result" at once when a side's calls fail 5 times in a row.
- **No replays:** a no-result game stays on the table with its reason (for example "no result: Astra, 41 failed
  calls"), and the rotation moves on to the next start. A start is never played again, so no result can be re-rolled.
- **Dropping an opponent:** after two no-results in a row caused by the same opponent's failures, that opponent is
  skipped for the rest of the session. The overlay says so.

### Rules

- **Exactly the lockstep rules above.** The only change is pacing: the clock steps the engine no faster than one turn
  per 200 ms of wall time (1×), so the picture moves at normal speed.
- **When a reply is late past T + 5 s:** the clock holds as it does in training. The engine isn't stepped and the
  picture freezes, then the game resumes when the reply is in. Holds don't change the result, only the viewing.
- **Pacing stays off for training:** it's a setting, default off, so training games don't change.

### The box

- **Instance:** one g6e.2xlarge, about $2.24/h: 1 L40S with 48 GB, 8 vCPU, 64 GB RAM, in us-east-2 where the training
  box lives.
- **What runs on it:**
  - Xorg on the GPU;
  - the relay;
  - vLLM serving Qwen3-8B with the v50 adapter (as the pilot does);
  - the rendered 0 A.D. engine;
  - Chrome and ffmpeg (NVENC);
  - the session runner.
- **The adapter:** copied from Run 1's S3 folder to the box.
- **Frontier models:** through the LiteLLM proxy, with the key from SSM as the live stack does.
- **The engine (decided at planning):** a new patch, `rl-paced-render`, adds `-rl-paced`. The frame loop draws and
  interpolates, but the game runs a turn only when `/step` allows one. It goes on its own image
  (`zeroad-engine:0.28-show`), so training's engine is unchanged.
- **vLLM:** the same image and flags as Run 1's serving (bf16, max context 8192), with a smaller GPU memory share so the
  renderer fits beside it. Neither the share nor the adapter-slot count changes what the model sees.
- **Fallback if Xorg or NVENC won't run on the L40S:** the g6.4xlarge (L4) render box the runbook uses. That needs
  vLLM to fit on 24 GB (bf16 weights about 16.4 GB, a shorter context) and the pilot's 40 GB check relaxed for this mode.
  It's allowed only if every prompt in Run 1's frontier games fits the shorter context, so no brief is ever cut.
  Otherwise the date moves.
- **Shutdown:** the box powers itself off after the session, and in any case `LIVE_MAX_HOURS` after launch.
- **One box for every stage (Claude's call):** spike, test game, rehearsal and session all run on the same box.
  - It's launched once with `--shutdown-behavior stop`, as `s2.env` does, and stays stopped between stages.
  - The budget guard counts every boot's hours on it, so the $60 cap covers all the stages together.
  - Each restart is an AWS write and needs the user's go-ahead at the time.
  - After a restart, `extend-poweroff.sh` moves the power-off deadline.
  - Teardown (`teardown.sh`) happens when the user says the showcase is done.

### Watching and recording

- **Recording:** one broadcast runs for the whole session, intermissions included. The relay records it in one-hour
  segments, and ffmpeg keeps its own MP4 of the session as a second copy.
- **Getting it off the box (Claude's call):** before the box powers off, the session runner uploads to
  `s3://$LIVE_BUCKET/zeroad-train/run/showcase-<stamp>/`. Fetching is then an S3 read from the laptop, with no
  restart.
  - **What's uploaded:** the session file, every game's `results.json` and the relay's continuous segments, which
    include the intermissions. A 6-hour session is about 16 GB, and `aws s3 cp` multipart has no 5 GB limit.
  - **What stays on the box:** ffmpeg's MP4 of the session, which duplicates the segments.
  - **Permission:** the box uses the instance profile the training box already has (S3 write under
    `zeroad-train/run/`, `train-iam.sh`), so no new IAM is needed. If that profile doesn't fit the live stack's SSM
    reads, a scoped role is an IAM write for the user's go-ahead at launch.
- **Watching during the rehearsal:** an SSM port-forward from the laptop to the relay. HLS or RTMP is read on loopback
  only, and the security group keeps no inbound rule. Screen-sharing it into a call is the user's call later.
- **Going public:** a separate decision. The X push is not used.

### Cost

| Item | Cost |
|---|---|
| Box | about $2.24/h |
| Opus game (measured) | about $2.45 |
| Astra game (measured) | about $2.41 |
| Petra-3 game | $0 |
| One round (about 1.6 h) | about $4.90 in games + $3.60 box ≈ **$8.50** |
| 6-hour session | about **$32** |
| Setup, spike and one Opus test game | about 3 h of box + $2.50 ≈ $10 |

Proposed budget cap for the box's guard (`LIVE_BUDGET_USD`): **$60**, covering the spike, a rehearsal and one 6-hour
session. The user confirms the number at launch.

### Order of work and the air date

**Air date: Friday 20 Nov 2026, if every stage passes in full.** The user asked for tomorrow "if possible" and for no
shortcuts to get there. So no stage is skipped or shortened, and no check is loosened, to make the date. If a stage fails
and can't be fixed in time, the date moves and the user picks the new one.

| Stage | When (ET) | User's go-ahead at the time |
|---|---|---|
| 1. Spike | Thursday night | launching the box (an AWS write) |
| 2. Overlay blocks, session runner and A2 | Thursday night, alongside 1 | none |
| 3. Test game against Opus | Friday, about 09:00 | restarting the box |
| 4. Rehearsal: 2 hours, one full round | Friday, about 10:00–12:00 | none |
| 5. Session, 6 hours | Friday from about 13:00, ending about 19:00–19:30 | restarting the box, and the go |

What each stage must show before the next one starts:

1. **Spike, at $0 model spend:** one rendered, paced-lockstep game, v50 against Petra-3, on the new box. It's recorded,
   fetched and watched. It must show all of these:
   - Qwen fits on the GPU beside the renderer;
   - the engine renders while the lockstep clock steps it;
   - the director camera works;
   - the overlay updates;
   - 30 fps;
   - units move smoothly, not in five jumps a second;
   - the game's result matches its `results.json`.

   The spike game also saves its replay, so A2 can be tested on the same game. If the stepped renderer doesn't work,
   stages 3–5 run on A2.
2. **The overlay blocks, the session runner and A2:** each reviewed, with its tests passing in CI.
3. **Test game,** one game against Opus (about $2.50). It must show all of these:
   - the hold line appears while Opus is late and clears when its reply lands;
   - every order in the turn log lands at exactly T + 5 s;
   - the result on screen matches `results.json`;
   - the spend is close to the measured $2.45.
4. **Rehearsal session:** 2 hours, so one full round (Opus, Astra, Petra-3), recorded and watched from the laptop. It
   must show all of these:
   - it runs to the end with nothing done by hand;
   - the session table matches the games' `results.json`;
   - the end of the session works: the upload to S3, then the power-off;
   - the recording is fetched from S3 and plays.
5. **A full session** when the user says go.

### A2 as insurance

Because there's a date, A2 is built alongside the spike rather than only after a failed one. It's a whole path, not a
shortcut, so it has to pass the same checks:

- **Proving it:** A2 renders the spike game's replay and must pass stage 1's checks on it. Stages 3 and 4 then run
  unchanged on A2.
- **On screen:** the header says the game is a replay and when it was played (for example "replay · played 14:05 ET").
  It's never shown as live. There's no hold line, since a replay has no holds. The overlay follows the replay's game
  time from the game's turn log.
- **No choosing:** every game played is shown, in the order played. A2 never picks which games to show.
- **If neither A1 nor A2 passes,** the date moves.

## Left to the builder

- **The paced clock:**
  - where the wall-time floor lives (a `LockstepClock` option is the obvious place);
  - how a hold shows in the overlay's data.
- **The paced engine patch's C++:** how `-rl-paced` gates the turn manager, and whether the director mod needs
  anything in that mode.
- **The overlay server:** where it runs (the session runner serving `overlay_state` for its match, or the env server),
  and how the intermission card replaces the game panels.
- **A2:** how the replay is rendered (the engine's replay mode and its flags), and how the overlay follows the replay's
  game time.
- **Fresh starts:** how seeds are picked and checked against every seed earlier runs used.
- **On the box:**
  - Xorg configuration on the L40S;
  - vLLM's memory share beside the renderer;
  - how the adapter is fetched.
- **Scripts:** their shape (for example `live/showcase.sh ID start|status|stop|fetch`, plan without `--yes`), and
  reusing `launch.sh`, `bootstrap.sh`, `budget.sh`, `health.sh`, `repair.sh` and `teardown.sh`.
- **Tests:** pacing on a fake engine, the session table from a set of results, the overlay's new blocks, and the
  scripts' plans.
