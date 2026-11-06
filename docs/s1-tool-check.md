# S1 tool check: runbook

Stage S1 of `docs/post-training-v2.md`: on a CPU box, with no model seat in any game, show that interface v2's
executor fixes what v1 got wrong, and measure what a 30-minute game costs in CPU and memory. Every step is a script in
`live/`; each one prints its plan when run without `--yes` and acts only with `--yes`. Nothing is run by hand on the box.

## What it plays

All games go through the lockstep runner (`python -m agentenv_zeroad.train.runner`) on a pool of engines (docker, host
network, one container each; the box's vCPUs minus 4), mainland map, 30-minute cap, civs from the executor's civs.

| Batch | Games | What |
|---|---|---|
| Bot | 600 (N=100 per cell) | The scripted bot against Petra 1, 2 and 3, bot seat on v1 and on v2. Petra seats are always v1. Game `i` has the same seed and civs under both interfaces and at every level; the bot starts first on even `i` and second on odd. |
| Replay | 80 (K=40 pilot games, two each) | Qwen's logged orders from `live/fetched/pilot-final` (policy v0, at least 15 turns, against Petra at a known level, spread over the levels), replayed under v1 and v2 in the setup they came from (map, seed, civs, Petra's level, the policy's start), 5 s late as lockstep play is. Only those pilot `results.json` files go to the box, as the `agent/v1` column. |
| Calibration | 12 + 24 + 44, per worker 4 and one loop | 12, 24 and 44 30-minute games at once on as many engines (the v2 bot against Petra 3), each level in workers of 4 engines and in one loop of up to 24; idle engines (1 and 20) on the PPA image and the patched one; a same-seed parity game on both. Step time by game minute, core-seconds per game, Python CPU and loop lag per worker, engine RSS. |

Order on the box: replays, calibration, bots. The bot specs are ordered game by game, so a batch cut by its wall-clock
limit still has equal numbers in every cell; the summary says how many games each column has.

## Gates

`scripts/s1_summary.py` writes `s1-summary.txt` and `s1-summary.json` (the table of metrics per `<seat kind>/<interface>`,
the gates, the calibration). A game that did not finish is left out and counted.

| Gate | PASS when | Meaning |
|---|---|---|
| Field placement failures per game | bot v2 mean at least 80% below bot v1 | The "no valid spot" failures that made Qwen's farms fail are gone. NO DATA if v1 had none. |
| Villagers at 15 min | bot v2 median at least 30% above v1 | The economy actually grows under v2. |
| Army at 10 min | median of the v2 replays at least 8 | Qwen's own orders, run through the v2 executor, make an army. |
| Bot win rate | at every Petra level where the v1 bot wins at least 10%, v2 is not significantly lower (Fisher, one-sided, p < 0.05) | v2 did not break the bot. A draw is not a win. |

Overall PASS only if all four PASS; one FAIL is FAIL; a gate with no data makes it NO DATA, which is not a pass. The
design's rule is "all pass, or fix the executor first". The replays' own field failures and villagers are in the report
and gate nothing: the bot is the clean pair.

## Setup

1. `cp live/s1.env.example live/s1.env` (git-ignored) and fill in the AWS profile, account, subnets, security group,
   instance profile (needs SSM; the box needs no S3 rights), bucket and owner tag, as for `live/train.env`.
   `LIVE_GPU_TYPES` and `LIVE_GPU_RATES` hold CPU types here (launch.sh's names, not renamed): c7i.12xlarge $2.142,
   c6a.12xlarge $1.836, c7a.12xlarge $2.463, each with its own rate; `LIVE_AMI_SSM_PARAM` is Canonical's Ubuntu 24.04
   AMI parameter. The box stops (does not terminate) at `LIVE_MAX_HOURS=4`, so results survive a late fetch.
2. On the laptop: Scale VPN, `aws sso login`, `uv`, `zstd` (`brew install zstd`), and `live/fetched/pilot-final`.
3. `export LIVE_ENV=live/s1.env`.

## Commands

Run each once without `--yes`, read the plan, then with `--yes`.

```bash
export LIVE_ENV=live/s1.env
aws sso login

live/launch.sh                       # plan: one of 3 CPU types, ~$2.14/h, stop after 4 h (at most ~$8.57)
ID=$(live/launch.sh --yes)           # instance id on stdout (its "Next:" hint omits --cpu: use the next line)

live/bootstrap.sh $ID --cpu          # plan: Docker, uv, repo, engine image, box-time timer; no GPU steps
live/bootstrap.sh $ID --cpu --yes    # ~25 min, mostly the engine image build (--images pull skips it)

live/s1.sh $ID start                 # plan: pick + pack 40 replay specs, upload, start the unit, budget check
live/s1.sh $ID start --yes           # starts zeroad-s1; survives SSM disconnects

live/s1.sh $ID status                # phase, games done, calibration levels, engines, load, memory, spend
live/s1.sh $ID fetch                 # plan
live/s1.sh $ID fetch --yes           # into live/fetched/s1-<time>/s1/out/, and prints s1-summary.txt

live/teardown.sh $ID                 # plan
live/teardown.sh $ID --yes           # terminates; its by-hand list is for the stream, ignore the SSM/X lines
aws s3 rm --recursive s3://$LIVE_BUCKET/zeroad-s1/    # the shipped code, inputs and archives
```

The batch plays on the patched engine, `zeroad-engine:0.28-rt` (labelled `zeroad.patches="rl-realtime rl-idle-wait"`, built by
`scripts/build-engine.sh rt`, which `cpu-bootstrap.sh` runs); `s1.sh start` and `s1-run.sh` refuse an image without that label
(`S1_ALLOW_UNPATCHED=1` for a deliberate comparison). To re-measure the CPU alone: `live/s1.sh $ID start --calibration-only --yes`
(its default flags are `--levels 12,24,44 --per-worker 4,24 --idle-engines 1,20`; `--cal-args "..."` overrides them) and read
`calibration.txt`: idle cores per engine on both images, whether the parity game matched, and per cell the busiest worker's
Python cores and its loop-lag p95 (near 1.0 cores or tens of ms means that loop is the bottleneck, not the box).

Options: `live/s1.sh $ID start --n 100 --replays 40 --wall-hours 2.5 --engines 44` (defaults from `LIVE_S1_*`).
`--wall-hours` must leave `LIVE_MAX_HOURS` an hour for bootstrap and fetch. `start` is refused when the wall-clock
limit at the box's rate would pass `LIVE_BUDGET_USD`. To end early: `live/s1.sh $ID stop --yes`
(games in flight are lost, the finished ones stay in `s1/out`); `fetch` then brings them back, but no summary is written,
so run `scripts/s1_summary.py` on the fetched `bot/` and `replay/` folders and the pilot inputs by hand.

## What lands in `live/fetched/s1-<time>/s1/out/`

`s1-summary.txt`, `s1-summary.json`, `progress.log` (phases and their limits), `specs/` (the exact game specs),
`replay/`, `bot/` (`<game id>/results.json` and `games.jsonl` per phase), `calibration/calibration.json` (the numbers,
with a per-sample series) and `calibration/level-N/`.

## Duration and cost

About 25 min bootstrap, up to 2.5 h batch (replays at most 25 min, calibration at most 50 min, bots the rest), 5 min
fetch: about 3 to 3.5 h of box time, **$6 to $8** at $2.142/h, never more than $8.57 (the 4 h power-off) plus a few cents of
disk until teardown. The design estimated about 2 h and $5; the batch ends on its wall-clock limit whatever the games
cost, so what moves is how many bot games finish. There is no model spend.

## Uncertain until the first run

- Wall time of a 30-minute game on one engine is unknown; it decides how many of the 600 bot games finish in the
  time the replays and the calibration leave. The calibration's first level measures it.
- Engine RSS: unknown. 60 engines must fit in 96 GB (c7i.12xlarge); the calibration reports the maximum.
- 60 engines on 48 vCPUs are deliberately oversubscribed: the 60-level's step times show the knee.
- The engine image build (`scripts/build-engine.sh`, ppa:wfg/0ad) takes about 10 min and needs the PPA to be reachable.
  `--images pull` needs `zeroad-engine:<tag>` in `LIVE_IMAGE_REGISTRY`; nothing pushes one.
- The calibration reads cgroup v2 through `/proc/<pid>/cgroup` (Ubuntu 24.04's default). If it can't, CPU and memory are
  null and the step times still stand.
