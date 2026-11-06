# Several boxes: rollout actors for one learner

Stage S4 of `docs/post-training-v2.md` §7 when it is run as a 3-box actor-learner instead of one box: one learner box
trains exactly as in S3, and extra boxes only play games and feed it. Adapters are never averaged; there is one learner,
one set of per-opponent baselines and one critic. Single-box mode is the default and is untouched: nothing here happens
unless the run config has `[transport] enabled = true`.

## How the boxes meet

The boxes are SSM-only with no inbound ports and may be in different subnets or regions, so they do not call each other.
They share one S3 prefix (`[transport] uri`) that each reaches with its instance role. The code is
`src/agentenv_zeroad/train/transport.py`.

| Object under the prefix | Written by | Read by | What it is |
|---|---|---|---|
| `adapters/v{N}/...`, then `adapters/v{N}/READY` | learner | actors | each new current adapter; `READY` lists every file with its size and sha256 and is written last |
| `current.json` | learner, every poll | actors | `{version, published_at, learner_alive_at}`: the version to serve and the learner's pulse |
| `games/{actor}/{game}.tgz`, then `.json` | actor | learner | a finished game's folder, then its manifest (version pinned, opponent, size, sha256) |
| `actors/{actor}.json` | actor, every poll | learner | heartbeat: version served, games in flight and done, box hours and rate, frontier spend, last error |
| `stop.json` | learner at its stop, or you | actors | drain, upload and power off |

An item counts only once its last object (`READY`, the game's `.json`) exists, and every file is checked against its
sha256, so a box that dies mid-upload leaves nothing the other side uses. Nothing needs a VPC route.

What each side does:

- **Learner** (`role = "learner"`, the `train.sh` box). Publishes each new current version off the main loop, then
  updates `current.json`. Ingests actors' games: downloads each manifested bundle, checks the sha256, unpacks it to
  `ROOT/remote/{actor}/{game}/` and runs it through the same credit path as a local game, so rows land in `ROOT/inbox`
  the usual way. A game pinned to a version more than `max_version_lag` behind the current one is dropped and counted
  (`status.json` counts `remote_stale`). What it has ingested is in `ROOT/orchestrator/ingested.json`, so a restart
  does not credit a game twice. Panels run on this box only. At any stop it writes `stop.json`.
- **Actor** (`role = "actor"`, `live/actor.sh`). The same game loop, vLLM instances and hot-loading as a single box,
  with no learner, credit, inbox or panels. It pulls each new version into `ROOT/adapters` (downloaded to a hidden
  folder, then renamed), pins every new game to the newest version all its vLLM instances hold, and ships each finished
  game, retrying with backoff. Its game stream is the learner's config seeded by `actor_id`, and its game ids are
  `{run_id}-{actor_id}-gNNNNNN`. It starts games only after it has read the learner's `current.json`.
- **Membership is elastic.** An actor that starts later joins when its first heartbeat shows; one that dies shows as
  stale in the learner's `status.json` after `actor_stale_minutes`.
- **Budget.** The learner's `budget_usd` covers every box: each actor's box hours at its own rate plus its frontier
  spend are added to the learner's. An actor that goes quiet without a final heartbeat is counted for at most
  `actor_stale_minutes` more.

## Setup (once)

1. Learner box: as for S3 (`docs/s2-box-day.md`, or `live/launch.sh`, `bootstrap.sh ID --train`).
2. Each actor box: `LIVE_ENV=live/train.env live/launch.sh --yes` for a box of any size with 40 GB GPUs (a g6e.12xlarge
   is four of them), then `live/bootstrap.sh ACTOR_ID --actor --yes`. Every GPU serves one vLLM instance (ports 8000
   upward), the rt engine image is built, and the power-off is armed; there is no trainer image. Different subnet or
   region is fine; the box needs S3 and SSM, nothing else.
3. IAM, for the learner and every actor in one command (the role reads and writes only the `run/` prefix):
   `live/train-iam.sh create LEARNER_ID ACTOR_ID... --yes`.
4. Configs. Add the `[transport]` block from `live/train-run.example.toml` to the learner's config, with `uri` under
   `s3://BUCKET/zeroad-train/run/RUN_ID/` and `role = "learner"`. Copy `live/actors/actor.example.toml` once per actor:
   a unique `actor_id` and `root`, `box_usd_per_hour` that box's own rate, the same `uri`, and the learner's `run_id`,
   `seed`, `[games]`, `[mix]` and `[held_out]` copied exactly. An actor config with panels or hard states is refused.

## Commands, in order

```
live/train.sh LEARNER_ID start learner.toml --init PATH --yes        # the learner first: actors wait for its pulse
live/actor.sh ACTOR_ID start actor-a1.toml --yes                     # then each actor, in any order, any time
live/actor.sh ACTOR_ID status                                        # an actor's unit, status, GPUs, vLLM, spend
live/train.sh LEARNER_ID status                                      # the learner; status.json has the actors block
```

Every command that changes anything prints its plan without `--yes`. An actor started before the learner waits for
`current.json`, and gives up after `learner_stale_minutes` if no learner appears.

## What to watch

In the learner's `status.json` (`train.sh ID status`):

- `actors.members.ID`: `state` (alive, stale or stopped), `version` and `lag` against the learner's current,
  `games_per_h`, `box_hours`, `box_usd`, `frontier_usd` and `last_error`.
- `actors.ingested` and `actors.dropped` (by reason: `stale` is a game too far behind, `rejected` failed its sha256 three
  times or had no usable `results.json`). Many stale drops mean the actors are slow to take new versions or
  `max_version_lag` is small for the learner's update rate.
- `counts`: `published`, `remote_credited`, `remote_failed`, `transport_error_*`.
- `spend` includes every actor; `alarms` has `actor:ID` while one is stale.

On an actor (`actor.sh ID status`): `transport.ready`, `transport.unshipped` (finished games waiting to upload; it
should hover near zero), `last_error`, and `counts` (`pulled`, `shipped`, `ship_failed`). The transport can also be read
from anywhere with S3 access: `python -m agentenv_zeroad.train.transport status --uri URI`.

## Stopping

- **All of it.** The learner's stop (`train.sh LEARNER_ID stop --yes`, its budget, duration or a tripwire pause) writes
  `stop.json`. Each actor then finishes its games in flight (up to `drain_timeout_s`), ships them, sends a final
  heartbeat and powers off if its config says `shutdown = true`. A learner that stops hard or dies stops the actors too:
  its pulse goes `learner_stale_minutes` old.
- **Everything, by hand, from anywhere:** `python -m agentenv_zeroad.train.transport stop --uri URI --reason "..."`.
- **One actor, now:** `live/actor.sh ACTOR_ID stop --yes` cancels its games in flight, ships the finished ones, sends
  its final heartbeat and ends (the box stays up unless `shutdown = true`; stop it with `live/stop-box.sh`).
- An actor also stops on its own `budget_usd` or `duration_hours`, and on SIGTERM.
- An actor that could not ship its last games does not power off, even with `shutdown = true`: it raises `unshipped` and
  stays up so the games are not lost with the disk. Fix the network or the role, or `actor.sh ID fetch --yes` for its
  files, then stop it by hand.
- `live/actor.sh ID fetch --yes` copies an actor's orchestrator folder and log to `live/fetched/actor-<time>/`. The games
  themselves are in the transport prefix.

## What fails and what happens

| Failure | Result |
|---|---|
| An actor dies mid-game | The game is lost; the learner never sees a manifest. After `actor_stale_minutes` the learner shows it stale. |
| An upload dies halfway | No marker, so it is ignored; the actor retries the whole game with backoff (5 s, doubling to 5 min). |
| A corrupt or truncated bundle or adapter | Refused by its sha256 or size. A game is retried three times, then rejected and counted. An actor retries an adapter at the next poll. |
| S3 or the network is down for a box | Reads and uploads fail and are retried every poll. The learner keeps training on its own games. An actor that can reach nothing sees the learner's pulse go stale and stops. |
| The learner restarts | It continues from the published version and its `ingested.json`, and clears the old `stop.json`. Actors keep running through it if it is back within `learner_stale_minutes`. |
| An actor restarts | It keeps its queue of unshipped games (`ROOT/orchestrator/ship.json`) and its game stream; a `stop.json` from before it started is ignored. |

## Cost

A g6e.12xlarge actor is $10.49/h on demand in us-west-2, four vLLM servers' worth of games. The learner's
`LIVE_BUDGET_USD` check at `train.sh start` counts only the learner's box; the run's `budget_usd` is what stops a
multi-box run, and it counts all boxes. Set each actor's own `budget_usd` and `duration_hours` too, and keep the
`LIVE_MAX_HOURS` power-off on every box.
