# Post-training v2: Lockstep Climb

Design for the second post-training run, 2026-11-05. It replaces the 2026-11-04 pilot design (docs/post-training-pilot.md). It is based on five research passes over the code and the pilot's data, five competing designs scored by three judges, and four critics who attacked the merged draft. The research, drafts and critiques are in `live/fetched/design-2026-11-05/`, which git ignores.

## Why the pilot didn't learn

All of these are measured on the pilot's 809 games and 28k logged decisions.

| Cause | Evidence |
|---|---|
| The interface capped play for any model | The army(attack) posture sends each new soldier at the enemy on its own. The policy's median army was 0–1 at minutes 5, 10, 15, 20 and 25, while its opponents had 10, 18, 29, 39 and 53. Its population peaked at about 42 at minute 10 and then shrank. Field placement failed about 3,800 times. The villager target is fixed at 45. The env writes a reason for each rejected order, and the policy never sees it. |
| The per-decision scores held no decision signal | The ridge residual predicted the decision's content with cross-validated R² of 0.001–0.006. Its lag-1 autocorrelation was 0.85. The "to game end" part had R² 0.57 in sample and 0.02 out of sample, so it was leaking each batch's own outcomes. |
| The loss favoured short replies | `pg = -advantage * logp.mean()` (trainer/train.py:118), combined with all-positive weights under the "best" objective, shrank replies from 106 to 50 tokens and from 3.3 to 1.75 orders per turn, on all three boxes. Pooling amplified it. |
| Too little training | About 8 optimizer steps and 260 trained decisions per box-hour. vLLM was stopped for about 40% of the time, and only 20–25% of what was played was trained on. The engine sat idle about 93% of each game, because it steps in 5.6 ms but was paced to a 100 ms wall-clock turn. |
| The evaluation couldn't see a change | Per-version 95% CIs were ±250–400 Elo. Detecting +10 percentage points of win rate against Petra-1 takes about 390 games per arm (about 155 using score margin). The fixed-seed Petra-1 probe sat at a 7.5% floor. |

A correction to what I said during the pilot: the frontier models were not handicapped by latency at 2x. Their median call latency was Astra 1.9 s, Sonnet 3.6 s, Sol 3.6 s and Opus 5.3 s, against Qwen's 5.5 s. Base Qwen beat Sonnet (8 of 15) and Sol (11 of 15) despite being the slowest.

## The design in one paragraph

First fix the interface so skill can show, freeze it, and re-baseline the base model on it. That way interface gains are never counted as learning. Then play training games in a lockstep mode: the simulation pauses for each decision, and orders land a fixed 5 game-seconds after the brief. This uses only verified engine behaviour. A 30-minute game should take 6–8 minutes instead of 15, with many games in flight. Separate GPUs serve and train, so the box is never idle. One persistent learner trains on a signed, zero-mean, token-level policy-gradient loss. Credit comes from each game's result (conquest dominates; at the time cap, net worth decides) minus a baseline, with importance ratios against the sampler's stored log-probs. Progress shows as milestone steps, such as "Petra-2 at 30%", each passed on a lower confidence bound, plus a Bradley-Terry Elo curve with confidence intervals. A pre-registered held-out panel decides whether training worked. Same-state forking, the principled answer to "every game's state differs", stays off the critical path. It gets a properly powered test once the platform exists.

## What changes against the pilot

| Pilot choice | Instead | Why |
|---|---|---|
| 30-min games at 2x wall clock, 15 min each, orders 4–31 game-s late | Lockstep: the sim pauses for each decision and orders land exactly 5 game-s after the brief, for both seats | The engine only advances on `/step` (5.6 ms per 200 ms turn). Each game takes about 6–8 min, latency noise disappears, and frontier comparisons become fair. |
| ~17 games per 35-min round; vLLM stopped to train on the same GPU | 2 L40S GPUs serve continuously and 2 train; about 285–375 games and 3.5–5k trained decisions per box-hour (estimates, measured on box day 1) | The pilot's binding constraint was data and optimizer steps. |
| Per-decision ridge scoring, fit on the batch itself | Game return minus a baseline: a per-opponent running mean, replaced by a state critic only when it beats that mean on later games. Advantages mean-centred within (opponent, game-time bin) strata | Honest, leak-free credit. A non-centred advantage would bring back the self-imitation pull. |
| Score = total (≈99% resources gathered) at the cap | Conquest win = +1 and conquest loss = −1. At the cap, R = 0.5·sign + 0.5·tanh(net-worth margin / s). 15% of games have no time cap (45-min safety cap) | Economy is about 99% of the econ+mil margin's variance, which rewards booming over fighting. Final margins also fell for a policy that learns to survive. Exploration flipped only 8 of 450 cap outcomes; it is excluded as a guard anyway. |
| `-adv · mean(logp)` per decision; "best" positive-only imitation | Token-sum PPO with a constant divisor, clipped against the start-of-update policy, with a truncated importance weight against the stored vLLM log-probs; say() tokens masked | This removes the length bias at its root and corrects the FP8/bf16 and staleness mismatch. |
| AdamW rebuilt every round, adapter saved in bf16, KL 0.02 | One persistent learner: fp32 master weights, kept optimizer state, fp32 LM head, a rollback ring, no KL term (KL to base is a soft alarm) | The KL term was about 5% of the loss and did not stop the collapse. |
| Adaptive ladder over Petra 1–5 plus drifting snapshots (44% of games) | Run 1: a fixed mix (bot 10%, Petra-1 35%, Petra-2 35%, Petra-3 10%, frozen v0 10%) with random AI behaviour, seed, civs and maps. Adaptive PFSP plus snapshots from Run 2 | A fixed mix keeps the trend test clean and avoids running out of LoRA slots. Ratings use fixed anchors only. |
| Progress = per-version probes at T=0.7 (35–64% of games) | Every game is a rating game, played at training temperature. Graded milestones such as "Petra-2 ≥ 30%", passed on a 97.5% lower bound. A 384-game held-out panel decides. | Retried per-version SPRTs would promote a 55% policy 50–90% of the time. Milestones checked on a fixed schedule with Bonferroni correction don't. |
| Frontier models as 2x probes, prompts not logged | Lockstep at equal latency, turn-logged. Sonnet and Sol are calibration rows (already beaten); Opus, then Astra, are the targets | Logged frontier turns become exact-format training data. |
| FedAvg of LoRA factors across 3 boxes | One learner. Extra boxes are rollout actors once the learner has headroom, or independent replicate runs rated on one board | Averaging amplified a shared bias. Two boxes feeding a 2-GPU learner add games per update, not updates. |

## Components

### 1. Interface v2 (frozen before any baseline)

- **Waves.** `army(action, target, min_size=N, regroup_below=0.4)`. Soldiers rally at the front until N have gathered, with a 90 game-s cap on the wait, then attack together, and regroup below 40% strength. Re-issuing the same order keeps the wave's state and timer, with a unit test for that. The "keep the attack standing" line goes.
- **The prompt tells the truth.** STRATEGIST_SYSTEM and the opening prompt still say exploration and town phase add score and that the model should attack by 4:00. They are rewritten to the real rules: conquest, the net-worth time cap, and how waves work.
- **Feedback.** Each order's result message, suggestion and the executor's give-up reasons go into the next turn's memory. Today only accepted/rejected survives (brains.py:317, :342).
- **An invalid `at` is ignored and auto-routed**, with a message saying where the units will train. `at` is already optional; only the rejection path changes (executor.py:208–214).
- **Villager lever.** `set_economy(..., villagers=N)` with N from 20 to 100. Today the target is fixed at 45 (orders.py:34).
- **`hold()`** is an explicit no-change action. Today 35–44% of turns repeat the previous orders.
- **Brief.** It gains own score, army state (location, engaged, losses in the last interval, wave size against min_size) and what is affordable now. Static cost and tech tables go in a cached system prefix per civ, keeping the prompt at or under 2.6k tokens p90. This is checked offline with the tokenizer.
- **Anonymity.** The "policy vN" label and the opponent's name and level leave the brief.
- **`say()`** is capped at one per turn and masked from the loss.
- **No guided decoding.** An invented tool name is played as a no-op and counted.
- **Economy floor**, time-boxed to 1 day. Fields and storehouses fall back to a ring search.
- **Interface switch.** One switch keeps the v1 path, so the pilot's base model stays on the board as "base-v1".

### 2. Reward and scoring

- **Conquest decides first:** R = +1 or −1.
- **At the time cap:** the winner has more net worth, meaning units and buildings alive valued at cost plus banked resources. R = 0.5·sign(m) + 0.5·tanh(m/s), where m is the net-worth margin and s is its within-opponent SD. Ordering: conquest win > any cap win > any cap loss > conquest loss.
- **15% of training games have no time cap** (45-min safety cap), so finishing the game stays rewarded.
- **Validate before freezing**, on the pilot timelines (an afternoon, $0). In games that ended by conquest, the net-worth leader at minute 20 should predict the winner better than total score does.
- **Exploration is excluded** from the reward and the critic. The official score stays on screen for viewers.

### 3. Lockstep mode and the in-process runner

- **LockstepClock.** It runs beside the live GameClock. It steps as fast as the engine allows and ticks each seat's executor every 5 turns, i.e. every game-second. Decisions fall due in game time: every 25 s, or after an 8 s gap on urgent events. The brief is frozen at T, orders apply at T+5 s, and stepping blocks only if the reply is late.
- **In-process runner.** An InProcessSeatLink runs `Commander.turn` against LiveMatch as a library, with tool schemas generated from the same FastMCP functions. A test checks that the base-v1 prompt is byte-identical to the pilot's. Engines are a pool of long-lived docker containers, reset per game (1.1–1.8 s).
- **No state-JSON parsing** on training runs. Nothing outside clock.py reads it.
- **Calibration measured on the CPU box:** step time by game minute over a full 30 minutes, and core-seconds and RSS at 20, 40 and 60 engines.

### 4. Learner and objective

- **Credit.** A = R − b. b starts as a running mean per (opponent, slot). A LightGBM critic on observer features switches in only when it beats that mean on the next 200 games. The critic uses army and military features and no exploration, is grouped by game in cross-validation, is weighted toward recent games, and is blended in over 15 min after each refit. λ = 1 (Monte Carlo), the honest default. GAE(0.95) is an ITERATE arm once the critic's calibration is shown.
- **Normalization.** Subtract the batch mean of A within (opponent, 5-min game-time bin), then divide by an EMA SD and clip at ±5. A pre-centring |mean/sd| above 0.1 for 3 updates is an alarm.
- **Loss.** Decoupled PPO. ratio = π_θ/π_prox, where π_prox is the learner at the start of the update, clipped to [0.8, 1.28]. The behaviour weight is min(π_prox/μ, 2), where μ is vLLM's stored per-token log-probs. The token sum is divided by the constant 128 × batch size, and say() tokens are masked.
- **Trust region.** Each update stops early above KL(π_after‖π_before) ≈ 0.005 nats/token on choice tokens. KL to base is logged as a soft alarm.
- **Hyperparameters.**
  - LoRA: r32 with alpha 64 on all linear layers.
  - Learning rate: 1e-5, constant after a 10-update warmup. It rises to 2e-5 only if the per-update KL stays tiny and the curve is flat after 50 updates, and halves on every rollback.
  - Optimizer: AdamW, betas (0.9, 0.99), weight decay 0, gradient clipping at 1.0.
  - Batches: 256 decisions per update (8–12 per seat-game, stratified by game-time bin), in 2 minibatches.
  - Staleness: data at most 6 versions old.
- **Sampling.** T 1.0, top_p 1, top_k −1, `--generation-config vllm`, bf16 on the 48 GB L40S, max_tokens 512. Per-token log-probs are stored, and every row asserts `len(logprobs) == len(token_ids)`.

### 5. Opponents and progression

- **Run 1 mix (fixed).**
  - Opponents: scripted bot 10%, Petra-1 35%, Petra-2 35%, Petra-3 10%, frozen v0 10%.
  - Variety: AI behaviour random over balanced, aggressive and defensive. The AI seed is separate from the map seed. Civs are random over the 4 the executor covers, with 2 civ pairs held out. Maps are mainland plus up to 2 that pass a bot smoke test, with 1 held out.
- **Milestones (the progression graph).**
  - The rungs are graded steps inside each Petra level: Petra-1 ≥ 50%, ≥ 65%; Petra-2 ≥ 15%, ≥ 30%, ≥ 45%, ≥ 50%; Petra-3 ≥ 15%; and so on.
  - Ceasefire sub-rungs can be added; the scenario builder already accepts a ceasefire.
  - A milestone passes when the one-sided 97.5% lower bound of the binned win rate clears it, pooled over the last 2 hourly bins. Checks run every 3 h, Bonferroni-corrected across milestones.
  - v0's baseline sets the starting milestone.
- **From Run 2.**
  - Opponent mix: PFSP over the rungs, weighted toward opponents near 50% with a 3% floor per rung. Frozen snapshots get at most 20%, and v0 keeps 10%.
  - Frontier sparring: a fixed ~30 games a day, not a share.
  - Handicap rungs: starting resources, population limit and an honest Petra, once the 0.28 settings are verified.
  - Note: the gather multipliers 0.42–1.56 behind Petra-4/5's strength may live in InitGame.js, not petra/config.js. Check that before overriding them.
- **Frontier models.**
  - Played in lockstep at equal latency.
  - The S2 baseline panel is 8 Opus, 8 Sonnet and 4 Astra games, about $26, repeated at the end of Run 1.
  - Frontier rungs (Opus, then Astra) are tested at champion promotions in later stages.

### 6. Evaluation and graphs

- **Ratings.** One Bradley-Terry model over every version-pinned game, Petra-3 = 1000, hourly bins (about ±40–50 Elo), bootstrapped at report cadence by a vectorized fit.
- **Primary endpoint (pre-registered).**
  - Panel: 384 held-out configs (held-out map, seeds and civ pairs; Petra-1/2/3 plus a non-Petra slice of v0 seats and ~8 Sonnet/Opus games), played by v0 in S2 and by the final Run 1 version.
  - Test: an opponent-stratified difference in margin, one-sided α = 0.05.
  - Only one confirmatory look. The SE comes from S2's own SD.
- **Trend test.** A logistic regression of result on training hours with opponent fixed effects, on fixed-anchor games only.
- **Health tripwires.**
  - What they track: completion length, calls per turn, exact-completion share, distinct fraction and entropy, all on non-hold turns and relative to v0's own distribution (±4 SD per 10 updates). Also the hold share (alarm above 60%, trip above 80%), invented tool names above 0.5%, the current-update clipped share, gradient norm above 5× its median, and rejected share above max(2× v0, 3%).
  - What a trip does: it rolls back to the last green checkpoint (ring of 20) and halves the learning rate. Two trips in 3 h pause the learner, upload everything and stop the box.
- **Behaviour panel.** It usually moves before Elo: army at 5/10/15 min, time of the first wave, waves launched, kills at home vs away, kill:loss, villagers at 15 min, field failures, conquest share, hold share and tool mix.
- **Decision after Run 1 (pre-registered).**
  - GO: the panel passes, the trend test passes, and no tripwire fired in the last 3 h.
  - ITERATE: health is fine but a test fails. Add the next signal: distilled init, a DAgger auxiliary term, GAE or the fork test.
  - NO-GO: a tripwire fired. Debug before spending more.
- **Report template, written in advance.** Rows that exist whatever happens: the interface delta, the E2 control curve, the distillation step, behaviour metrics, the starting milestone, the frontier baseline and the RL curve.

### 7. Several boxes and the livestream

- **Several boxes.** One learner. Extra boxes become rollout actors only behind a 4-GPU learner (3 boxes), because a 2-GPU learner is already the bottleneck. Otherwise the cheapest honest use of a second box is an independent replicate run on the same rating board. "2 of 2 runs climbed" is a strong claim for a stream.
- **Livestream (S5).** A $1.32/h render box plays the latest pinned version (champion line overlaid) at 1x against frontier models or Petra rungs, calling the training box's vLLM. If the learner pauses, the stream keeps playing the champion and alerts.

## Stages, gates and cost

| Stage | What | Gate to leave | Time | Cost |
|---|---|---|---|---|
| S0 Build | Items below | CI green on the new unit tests; interface v2 frozen; offline learner dry run on pilot decisions; prompt ≤ 2.6k p90 | ~15 engineering days, ~8–10 calendar days in 3 agent streams | $0 |
| S1 Ceiling check | On a CPU box: bot v1 vs v2 (field failures −80%, villagers at 15 min +30%, less idle production); Qwen's logged order streams replayed through the v2 executor (median army at 10 min ≥ 8); engine core-s, RSS and step time by minute | All pass, or fix the executor first | ~2 h | ~$5 |
| S2 Box day 1 | GPU gates (≥ 3 decisions/s per GPU with the real adapter mix, ≤ 2 s per trained decision, log-prob agreement < 0.05); positive control through the full scoring path; v0 384-game panel, base-v1 subset, frontier baseline; optional distillation probe | Gates met and the control learns | ~6.5 h on a g6e.12xlarge | ~$68 box + ~$26 frontier + ~$63 distillation labels ≈ $157 (≈ $94 without the probe) |
| S3 Run 1 | 10 h of continuous RL from v0 or the distilled adapter | Pre-registered GO / ITERATE / NO-GO | 10 h | ~$105 box + ~$26 frontier ≈ $131 |
| S4 Run 2 | 24 h: one box, or two replicate runs, or a 3-box actor-learner | Slope still positive; a milestone passed; health clean | 24 h | ~$252–760 + ≤ $50/day frontier |
| S5 Livestream | Continual learning on stream | Ongoing, under a daily cap | Days | ~$335/day in one-box mode |

The total to the first GO decision is about $295 with the distillation probe and about $230 without it, against about $150 for the pilot day. Each of S2 through S5 needs your approval with a dollar figure, and `LIVE_BUDGET_USD` is set per stage.

### The distillation probe (optional, recommended)

- **Teacher choice:** whichever of Opus or Astra does better at equal latency in the S2 frontier panel.
- **Labels:** the teacher relabels about 3,000 stratified v2 briefs from the v0 baseline games, at about $57, plus $6 for a label-noise check.
- **Training:** SFT a separate adapter, about 40 min on 2 GPUs, then play it on the panel.
- **Rule, written in advance:** if it beats v0 at p < 0.05, Run 1 starts from it with its own baseline row.
- **Honest story either way:** base → distilled → RL. If outcome-only RL comes out flat, which is about as likely as not, this is still a labelled step.

## Engineering (S0), about 15 focused days

| Item | Days | Stream |
|---|---|---|
| Interface v2, prompt rewrite, idempotent waves, interface switch | 2.5 | env |
| Economy floor (time-boxed) | 1 | env |
| Net-worth score rule, no-cap option, validation on pilot rows | 0.75 | env |
| Knobs: AI seed, AI behaviour, ceasefire, civ and map lists | 0.25 | env |
| Replay of Qwen order streams for S1 | 0.5 | env |
| LockstepClock, in-process runner, engine pool | 3.5 | platform |
| Sampling hygiene: bf16, 2 vLLM instances at TP=1, stored log-probs | 0.5 | platform |
| Orchestrator: mix, panels, budget check every 10 min, S3 sync, vLLM watchdog that re-registers adapters, auto-stop | 1.25 | platform |
| Box ops: training env file (g6e.12xlarge, then g6e.24xlarge), max-hours = run + 2 h with stop-not-terminate, correct $/h, baked AMI, test launch | 0.25 | platform |
| Persistent learner: decoupled PPO, TIS, token-sum, tripwires, rollback ring, hot-load | 2.5 | learning |
| Baseline and centring; critic with gated switch-in | 1.25 | learning |
| Ratings, trend and panel tests, milestones, dashboard (reusing ratings.py, report.py, combine.py, pilot-dashboard.html) | 1.25 | learning |
| Distillation probe pipeline | 1 | learning |

The serial floor is about 7–8 days, because interface v2 has to freeze before the runner is integrated. Integration tests that need a real engine run on a cloud box, not in CI.

## Ops fixes required before any long run

- **Box lifetime:** `live/pilot.env` sets `LIVE_MAX_HOURS=11`, and launch.sh terminates the box and deletes its volume at that point. A 12 h run would be destroyed before its final panel. Use run length + 2 h, with stop rather than terminate (or terminate only after the final upload).
- **Off-box copy:** results only leave the box through laptop-signed URLs, which die with SSO after about 1 h. Use an instance role, as with `live/pilot-iam.sh`, which you ran last time, to sync to S3 every 15 min.
- **Budget guard:** it prices the box at $4.53/h, against $10.49 for a g6e.12xlarge, and checks only per round. Check every 10 min, pricing box time and frontier spend through one ledger.
- **vLLM restarts:** a restart forgets runtime-loaded adapters. Add a watchdog that re-registers them, write adapters atomically, and keep them on local NVMe.
- **Capacity:** the G and P on-demand quotas are 768 vCPUs each (checked), so quota is fine. A g6e.12xlarge has never been launched from this setup, so do a test launch before S2.

## Expected outcome

These are projections, not measurements.

- **Interface delta, not counted as learning:** a guess of +50–150 Elo for v0 on v2 over base-v1, shown as its own row.
- **Run 1, 10 h:** about 3.5k training games and about 150 updates. Chance of GO with outcome-only credit is about 30–45%, and there is about a 1-in-3 chance of no detectable gain. Behaviour metrics should move first.
- **24 h:** a central expectation of +50–150 Elo, and milestones such as "Petra-2 ≥ 30% passed". "Petra-2 at 50%" is a stretch.
- **72 h:** likely a plateau near Petra-2 or Petra-3. The next levers are same-state fork credit, a plan field or a short thinking budget, Petra-macro behaviour cloning, or a bigger model.
- **Closest published analogue:** Diplomacy GRPO with Qwen3-14B and LoRA went from 72% to 80% against a scripted bot after about 46 steps of 100+ games (about +77 Elo). No published RL fine-tune of an open LLM beats a built-in RTS AI.

## Open bets (off the critical path)

| Bet | Test | If it fails |
|---|---|---|
| Same-state fork credit carries action signal | On ITERATE: a determinism harness plus a fork manager (~3.5 days; a `/save` and `/load` engine patch is optional). 80 branch points × 8 null branches, a per-point log-variance-ratio estimator, a check that each rebuilt state matches the mainline game's hash, and a gate of lower bound ≥ 0.05. (A 40-point version has about 21% power and was dropped.) | Stay with outcome credit |
| Handicap rungs on 0.28 | One `/reset` per setting on box day 1, then 20 bot games per rung | Use a coarser ladder from difficulty × behaviour × ceasefire |
| Lockstep play transfers to wall-clock play | 16 live games at each champion promotion; within ±150 Elo | Train with the measured live lag and add 10% wall-clock games |
| Petra-macro behaviour cloning (Petra's commands are not in replays) | A simulation-side command hook mapped to the 7 tools (2–4 days) | Treat the executor as the cap; try a plan field |
