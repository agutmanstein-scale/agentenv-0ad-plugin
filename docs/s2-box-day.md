# Box day 1 and Run 1: runbook

Stages S2 and S3 of `docs/post-training-v2.md`, run back to back and unattended on one g6e.12xlarge: the GPU gates, the
v0 baselines and the teacher calibration, v0's tripwire baseline, the positive control, the distillation probe, then ten
hours of RL. Every step
is a script in `live/`; each prints its plan when run without `--yes` and acts only with `--yes`. Nothing is run by hand
on the box. Once `live/s2.sh ID start --yes` has returned, the box decides for itself whether to go on: each step starts
only if the gates of the one before passed, and the day ends with the report, the final upload to S3 and the box
stopping itself (never terminated from the box).

## What runs

The sequence is `src/agentenv_zeroad/train/s2.py`, started by `live/box/s2.sh start` as the systemd unit `zeroad-s2`
(it survives an SSM disconnect). Each step runs through `live/box/train-run.sh`, the same script a `train.sh` run uses (the
learner container on GPUs 2 and 3, the orchestrator on the engine pool, vLLM on GPUs 0 and 1), so a step is a training run
config and nothing else. The configs are in `live/s2/`, one per step, with `sequence.toml` for the caps and the gates.

| Step | What it does | Gate (all must pass) | Expected | Cap |
|---|---|---|---|---|
| e1 | `smoke.py` on both vLLM servers; `probe.py` holds 16, 32, 48 and 64 calls in flight per server for 60 s each on the recorded v2 prompts (60 s, 300 s and the busy midgame brief) and keeps 48 sampled decisions; `learner.py --probe` times one update on 32 of them and compares the learner's log-probs with vLLM's on 5 | log-probs aligned on every server; best decisions per second per GPU at least 3; learner at most 2 s per trained decision; mean absolute log-prob gap under 0.05 nats per token | 0.5 h | 0.75 h |
| baselines | Play-only v0 (the base model on interface v2): teacher calibration (16 games), frontier baseline (8 Opus, 8 Sonnet, 4 Astra against v0 at equal latency), the 384-game held-out panel (Petra-1 120, Petra-2 120, Petra-3 96, the bot 24, a frozen-v0 mirror 24), a 48-game base-v1 subset (interface v1 on the first held-out positions); then ordinary training-mix games (bot 5%, Petra-1 10%, Petra-2 50%, Petra-3 35%) until the games hold enough briefs for 3,000 labels, 60% of them losing positions | every panel at least 90% finished; briefs for 3,000 labels available | 3.9 h | 6 h |
| calibration | `stats.pick_teacher` over the calibration panel: the best win rate against Petra-3, ties broken by Petra-4 | a teacher played Petra-3 | minutes | |
| tripwire | v0's tripwire baseline, without training. `train/baseline.py` credits the ordinary (non-panel) baseline games through the same `Credit` Run 1 uses and keeps v0's decision rows; `learner.py --stats-only` (`train-run.sh --stats-only`, the learner alone on GPUs 2 and 3, the base model) writes one health record per 256 decisions with the same row statistics, entropy and KL fields an update logs, and changes no weights; `baseline_from` of the records is `s2/baseline.json` | at least 10 records (2,560 decisions of v0) and the file written | 0.3 h | 0.5 h |
| e2 | The positive control: an RL run with `credit.positive_control` (the reward is the army at minute 10, z-scored per opponent), 12-minute games against Petra-1 and Petra-2, 2 hours, started with `--baseline s2/baseline-e2.json` and `--lr 3e-5`. That baseline is v0's own on E2's kind of games: first a 0.75 h play-only v0 run (`e2-baseline.toml`), its decisions credited with E2's credit and measured by the stats-only learner (9 Nov: on the baselines' band, v0 itself tripped the distinct-completion check at E2's first records). The second attempt runs in `s2/e2b`; the first one's files stay in `s2/e2`. A frozen-v0 arm (60 games, the `control-v0` panel, same kind of games) plays beside it. Its adapters stay in `s2/e2` and are never loaded into anything later | at least 5 updates; the policy's last 40 games' median army at minute 10 beats the control's (Mann-Whitney, one-sided, p under 0.01, at least 30 games in each arm) and is at least 6; no tripwire trip (with the banded checks on); the run ended by its duration | 3.3 h | 3.8 h |
| labels | `distill.py` under `with-model.sh`: the teacher relabels 3,000 briefs with the policy's exact prompt, reference tables and tools; a second teacher labels 200 for the noise check; SpendGuard caps $60 and $8 | at least 2,400 labels written | 0.4 h | 1.0 h |
| sft | `learner.py --sft`, 2 epochs, lr 1e-4, a checkpoint per epoch | both epochs published, `sft-done.json` written | 0.8 h | 1.25 h |
| sft_panels | Each epoch's adapter plays the first 192 games of the held-out panel; `stats.select_sft_checkpoint` picks one by held-out win rate with the entropy floor 0.7 and at least 20 games. The chosen epoch then plays the other 192 configs of the held-out panel (the `sft-rest` panel, `start = 192`), so it has the same 384 configs v0 has | every panel at least 90% finished, the epoch panels and the rest | 2.1 h | 3.0 h |
| init | Run 1's start: the chosen epoch if `stats.panel_test` (opponent-stratified margin, on all 384 configs: the epoch's 192 and its 192 more against v0's games on the same configs) says it beats v0 at one-sided p under 0.05, else the base model. This is a decision, not a gate: it always passes and writes its reason | none | minutes | |
| run1 | The RL run, 10 hours, Run 1's fixed mix, `--baseline s2/baseline.json`, `--init` and `--anchor` the chosen epoch (or neither), the critic on, hard states off unless `run1.toml` enables them. Panels at hour 8: the 384-game held-out panel and the frontier panel. Then the report, the final upload and `shutdown -h now` | the run ended by its duration (not budget, tripwire pause or crash); a missing or short tripwire baseline stops the step before it starts | 10.3 h | 11.0 h |

A failed gate, a step that reaches its cap, a budget refusal, two tripwire trips (the orchestrator's own rule: the
second within 3 hours pauses and stops the run) or an error stops the sequence. The reason goes into
`gates.json` and `status.json`; the report and the final upload still run, and then the box stops itself, unless the
upload failed (then it stays up so the files are not lost with the disk). A stop by hand (`s2.sh stop`) leaves the box up.

Why a step that fails the gate is not worked around: E1 and E2 exist to say "do not spend the rest". A box that
fails E1 has cost about $16 (bootstrap and E1). E2 comes after the baselines because it needs v0's tripwire baseline
(next section), so a box that fails E2 has cost about $110: the baselines' $27 of box and $41 of frontier games are
spent before the control runs. What it saves is the labels ($68), the SFT and its panels, and Run 1's $140.

Going on past a failed step is a person's call, never the sequencer's, and it is written down in `sequence.toml`'s
`[sequence]` table, which every `--resume` reads again. `accept_failed = ["e2"]` skips a step whose recorded failure was
accepted (its gates stay in `gates.json`, its status becomes `accepted`; a step that never failed still runs), and
`stop_after = "sft_panels"` stops the sequence after that step, with the usual report, upload and power-off. Only the
steps up to `stop_after` count in the expected hours that `start` checks. `rerun = ["e2"]` runs a step again although
`gates.json` records it (its old entry is kept under `earlier`); once rerun, the step follows the usual rules again, and
`accept_failed` does not skip it while it is listed. `[e2] drift = "alarm"` starts E2's learner with `--drift alarm`, so
behaviour moving outside v0's bands is an alarm to look at, not a trip that rolls back. 9 Nov: E2 failed twice (on the second attempt
its trained policy's median army was 4 against the control's 2, p 0.36, with three trips; against 442 games of v0 on
E2's kind of games both attempts were only marginally better, p 0.08 and 0.06), so E2 was accepted and the sequence
runs the distillation steps only; Run 1 is decided on the SFT panels' results.

### Why the tripwire baseline sits between the baselines and E2

The banded collapse tripwires (completion length, tool calls per decision, exact-completion share, distinct fraction,
entropy) compare a window of updates with base Qwen's own distribution of the same numbers. They catch the pilot's
length and echo collapse, and they need a baseline file; without `--baseline` the learner runs only the checks that need
none (hold share, invented tools, rejected share, clipped share, gradient-norm spikes). So E2 and Run 1 both get it, and
`s2.py` refuses to start either without `s2/baseline.json` and at least 10 records in `s2/baselines/baseline-records.jsonl`.

The two ways to get v0's numbers before E2 were to play v0 first (reordering) or to take them from E2's own first
updates, before its adapter has moved. The second was rejected: E2's first updates are a different game (12-minute
games, a positive-control reward, Petra-1 and Petra-2 only) with sampling that has already begun to shift, and the file
would be the control's own. The baselines already play v0 on the ordinary training mix, so the sequence is
e1, baselines, calibration, tripwire, e2, labels, sft, sft_panels, init, run1, and E2's banded checks judge it against
v0 on the mix Run 1 trains on. One check gets a looser reference than ideal: E2's games are shorter, so its briefs are
earlier-game on average than the baseline games'; a flat threshold band has absorbed that in the one comparison
available. That is the price of E2 using the same file as Run 1 and not a baseline of its own.

The records are made by the learner because entropy needs the GPU model: `learner.py --stats-only` runs the same
`row_stats`, entropy and KL-to-base code as an update, in 256-decision batches, with no optimizer step, version,
publish or checkpoint (a short last batch is dropped). The rows are v0's decisions from `s2/baselines/games/` (not the
panels), credited with Run 1's `[credit]` settings and no critic.

### How the gates were chosen

- Throughput: the design's 3 decisions per second per GPU, taken as the best of the four levels (the knee, the smallest
  level within 10% of the best, is recorded as the concurrency to run at). The probe samples T=1, 512 tokens, as training does.
- Learner: 2 s per trained decision is a 256-decision update in under 9 minutes, which keeps the learner ahead of 2 GPUs
  of sampling. The timed update includes the anchor pass, the proximal pass, the checkpoint and the publish, after one
  warm-up update.
- Agreement: the mean absolute difference between the learner's fp32-head log-prob of each sampled token and vLLM's
  stored one, before any update (the adapter is still the base model). This is the importance weight's error.
- Positive control: the policy must beat a frozen v0 arm, not its own early games, so a drift in the game or the box
  cannot pass it. Its tripwire rule is strict: any trip fails, because a control that trips has not shown the path
  learns cleanly.

## Setup (once)

1. `cp live/s2.env.example live/s2.env` (git-ignored) and fill in the AWS profile, account, subnets, security group,
   instance profile (needs SSM), bucket and owner tag, and the LiteLLM URL, as for `live/train.env`. The rest is set:
   `g6e.12xlarge` then `g6e.24xlarge` at $10.49 and $15.07 an hour, `LIVE_REQUIRE_RATE=1`, stop-not-terminate,
   `LIVE_MAX_HOURS=22`, `LIVE_BUDGET_USD=600`. If the launch falls back to the g6e.24xlarge, change `box_usd_per_hour` in
   every `live/s2/*.toml` to 15.07 first (`s2.sh start` refuses a mismatch) and expect about a third more box cost.
2. Copy `live/s2` somewhere to edit, or edit it in place: nothing in it is a secret. Check the three frontier model names
   in `baselines.toml` and `run1.toml` (`anthropic/claude-opus-5-5`, `anthropic/claude-sonnet-5-5`, `openai/gpt-6-astra`)
   against what your LiteLLM serves; the Sonnet name is the one not already used elsewhere in this repo. The `s3.uri`s say
   `s3://BUCKET/zeroad-train/run/s2-...`; `s2.sh` fills in `LIVE_BUCKET` when it sends them.
3. On the laptop: Scale VPN, `aws sso login`, `uv`, `zstd` (`brew install zstd`).
4. `export LIVE_ENV=live/s2.env`.

## Commands

Run each once without `--yes`, read the plan, then with `--yes`.

```bash
export LIVE_ENV=live/s2.env
aws sso login

live/launch.sh                              # plan: a g6e.12xlarge, $10.49/h, stop after 22 h (at most about $231)
ID=$(live/launch.sh --yes)                  # instance id on stdout

live/bootstrap.sh $ID --train               # plan
live/bootstrap.sh $ID --train --yes         # about 30 min: Docker, uv, two bf16 vLLM instances, the trainer image, the smoke check

# THE IAM STEP, after the bootstrap and before s2.sh start, with the ID that launch.sh printed. It is the one thing
# you do by hand: the sequence copies every step to S3 with the box's instance role and reads the LiteLLM key from SSM
# with it, and an instance without the role has neither. It puts the role zeroad-train-box (SSM core, read/write on
# s3://$LIVE_BUCKET/zeroad-train/run/*, read on /zeroad-live/* parameters) on the box in place of its profile:
live/train-iam.sh create $ID                # plan
live/train-iam.sh create $ID --yes

# The LiteLLM key, once per account (not per box), into SSM; the script reads it without echoing and never prints it.
# A key piped in without a trailing newline works too: printf %s "$KEY" | live/put-secret.sh litellm-key --yes
live/put-secret.sh litellm-key              # plan
live/put-secret.sh litellm-key --yes

live/s2.sh $ID start live/s2                # plan: the local check of every step config, the send, the unit and its budget check
live/s2.sh $ID start live/s2 --yes          # from here the box runs by itself, about 19 hours

live/s2.sh $ID status                       # whenever you like
live/s2.sh $ID status 60                    # more of the progress log
live/s2.sh $ID fetch --yes                  # gates.json, progress.log, s2-report.json, the labels, the logs, into live/fetched/s2-<time>/
live/s2.sh $ID stop --yes                   # only to end it early; the step stops and uploads, the box stays up
```

`start` refuses: a config directory that does not pass the check (every step config's panels and one sample game, the
roots all under `s2/`, `shutdown = false`, one box rate, the positive-control credit, the budget at or under
`LIVE_BUDGET_USD`, a panel name the sequence refers to that is missing); a box rate that is not the configs'; a
power-off deadline that leaves less than the expected hours plus half an hour; a vLLM instance that is down; a budget
the expected hours would pass; and an earlier sequence's `gates.json` unless `--resume` (continue it, skipping the steps
that passed, which is what to use after an SSM hiccup or a box restart) or `--fresh` (move it aside) is given.

## Reading the status

`status` prints, in order: the gates so far (`PASS`/`FAIL name: value op threshold`), the step now running, the last 20
lines of the progress log, that step's orchestrator (`run_hours`, spend, games started, finished and failed per kind,
panels, `briefs` for the baseline run, alarms, the stop reason) and the learner's last three health records, the
containers, GPUs, vLLM, the spend (box hours at the rate plus the ledger) and when the box powers off.

`gates.json` on the box is the record: per step its status, start, end and seconds, each gate with its value, operator,
threshold and verdict, a `detail` (the throughput levels, the control's medians and p, the calibration table, the
selection of the SFT checkpoint and the init decision) and the reason it failed, if it did. `verdict` is `running`,
`complete` or `stopped` with its reason. `progress.log` has the same in lines. Logs of every command are in
`s2/logs/`.

What lands where: `s2/e1`, `s2/e2`, `s2/baselines`, `s2/sft`, `s2/run1` are runs' roots (games, panels, health,
adapters); each orchestrator syncs its root to `s3://BUCKET/zeroad-train/run/s2-<step>/` every 15 minutes and at its end
(games, panels, health, manifests, hourly and champion adapters). The sequence uploads its own files and the SFT epochs
(`gates.json`, `progress.log`, `s2-report.json`, `distill/`, `logs/`, `generated/`) to `.../run/s2/s2/` at the end,
three tries. `s2-report.json` is `report.py`'s report over all of it: the board, trend test, milestones, the panel
test between v0's held-out panel and Run 1's end panel (`v0-heldout` against `run1-heldout`), the teacher calibration,
the SFT selection, behaviour, health, throughput and spend, with the panels linked into `s2/report-panels/`.

## The decision after Run 1

Unchanged from the design: GO if the panel test passes, the trend test passes and no tripwire fired in the last 3
hours; ITERATE if health is fine but a test fails; NO-GO if a tripwire fired. The sequence only runs the day; the decision
is read from `s2-report.json`. The base-v1 subset and the v0 panel give the interface delta, which is not learning.

## Teardown

When `gates.json` says `complete` or `stopped` and the box has stopped itself (or you have stopped it), fetch, check the
S3 prefixes have `final.json` in each step's `orchestrator/` folder and `s2/gates.json`, then:

```bash
live/teardown.sh $ID                        # plan; its by-hand list is for the stream, ignore the SSM/X lines
live/teardown.sh $ID --yes                  # terminates: everything on the box's disk is lost
```

A stopped box bills its disk (about $16 a month) until it is terminated; `live/stop-box.sh $ID --yes` stops a box that is
still up after `s2.sh stop`. Remove the LiteLLM key from SSM when no more runs need it.

## Time and cost

| | Hours | Box at $10.49/h | Frontier |
|---|---|---|---|
| Bootstrap and the IAM step | 1.0 | $10.5 | |
| e1 | 0.5 | $5.2 | |
| baselines | 2.6 | $27.3 | frontier panel about $26, calibration about $15 |
| tripwire | 0.3 | $3.1 | |
| e2 | 2.3 | $24.1 | |
| labels | 0.4 | $4.2 | up to $68 (the caps; about $63 expected) |
| sft | 0.8 | $8.4 | |
| sft_panels | 2.1 | $22.0 | |
| run1 | 10.3 | $108.0 | end frontier panel about $26 |
| **Total** | **20.3** | **about $213** | **about $130** |

About $343 if every step takes its expected time, at the planning prices. On the first box day (6 Nov) frontier games cost
about $2.5-3.5 each and an Opus label about $0.036, so the frontier spend is nearer $275 than $130 and the day about
$490 at the on-demand box rate; `LIVE_BUDGET_USD=600`. The design's S2 + S3 line was about $290
($157 + $131). This sequence costs more because it adds the E2 positive control's two hours (the design counted E2 inside
the 6.5 h), the SFT panels (two epochs of 192 games, then the chosen epoch's other 192: 0.7 h, $7 more than the 1.4 h
that two epochs alone took), the tripwire baseline (0.3 h), the 6 extra calibration games and the bootstrap hour. The
planned wall clock is 20.3 h from launch against `LIVE_MAX_HOURS=22`. Every step's cap is a ceiling the budget check
counts before the step starts: the sum of the caps is 23.6 h, which the box could not afford, so a sequence whose earlier
steps ran long is stopped by that check (with the guard's words in `gates.json`) rather than by the power-off.

The check before Run 1 is the tight one. It adds what the box and the ledger have spent (10.0 h of box and the bootstrap
at $10.49 is $105, plus about $104 of frontier spend: the calibration, the frontier panel and the labels) to Run 1's
11 h cap ($115) and its $30 of spend caps: about $354 at the expected pace. At $350 the sequence would stop at Run 1's
start with "the budget guard refuses it", so the budget is $600 (`LIVE_BUDGET_USD` and every config's `budget_usd`,
which `s2.sh start` requires to be at or under it): room for steps that run long, and still a stop for a runaway box.

## Uncertain until the first run

- E1's recorded prompts are the repo's three snapshot briefs (60 s, 300 s and a built-to-be-busy midgame prompt); a
  real 25-minute game's briefs are longer than the first two and shorter than the third. The throughput number is
  therefore a middle estimate. The probe runs from the repo checkout on the box (`tests/prompt_scenarios.py`).
- Whether the engine pool of 24 keeps both vLLM servers busy is E1's sibling question and not tested by E1: the
  platform stream is changing the runner's worker split. The configs pass `engines` and `concurrency` through unchanged
  (24, 24; 4 for E1's probe config, which plays no game).
- The tripwire baseline is 10 or more records of 256 v0 decisions: its sd is the per-record SD over the square root of
  10 when there are fewer than two windows of 10, so a band built from 10 to 19 records is the loosest the learner
  makes. A baseline of more records narrows it: the baseline run plays at least 375 ordinary games (3,000 briefs at 8
  a game) and credit takes 8 to 12 decisions from each seat, so an estimated 11 to 17 records. Whether E2's window sits inside the bands of a baseline made on the 30-minute training mix is
  unknown until it runs; a trip in E2 fails the control, and the record says which band.
- No held-out map: `held_out.maps` is empty until a second map passes the bot smoke test, so the held-out panels
  use held-out seeds (900000 to 900499) and civ pairs on the training map.
- The init decision is made on all 384 paired configs: the 192 of each epoch choose between the epochs, and the chosen
  epoch then plays the other 192. A gain of about 100 Elo is still only borderline at p under 0.05 on 384 games, so
  "the base model" is a likely result of the rule written in advance, not an error.
- The labels' prompts are rebuilt from the recorded `reference` of v0's own games; a v0 game whose seat recorded none
  is skipped, and the baselines gate counts only usable briefs.
