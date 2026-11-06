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
| `current.json` | learner, every poll | actors | `{version, published_at, learner_alive_at, epoch, fingerprint}`: the version to serve, the learner's pulse, a random id per learner process, and a hash of the games config |
| `games/{actor}/{game}.tgz`, then `.json` | actor | learner | a finished game's folder, then its manifest (version pinned, opponent, size, sha256) |
| `actors/{actor}.json` | actor, every poll | learner | heartbeat: version served, games in flight, done and unshipped, box hours and rate, frontier spend, instance id, last error |
| `stop.json` | learner at a final stop, or you | actors | `{reason, epoch, drain}`: stop, for the learner epoch in `current.json` |

An item counts only once its last object (`READY`, the game's `.json`) exists, and every file is checked against its
sha256, so a box that dies mid-upload leaves nothing the other side uses. Nothing needs a VPC route. Whether a key exists
is decided by listing it, because the run's IAM role may list only its own prefix and S3 then answers 403, not 404, for
a key that is not there.

What each side does:

- **Learner** (`role = "learner"`, the `train.sh` box). Publishes each new current version off the main loop, then
  updates `current.json`, and prunes published versions older than `max_version_lag + 2` behind (hourly and champion
  ones stay). Ingests actors' games: downloads each manifested bundle, checks the sha256, unpacks it and runs it through
  the same credit path as a local game, so rows land in `ROOT/inbox` the usual way. It then keeps only a slim
  `ROOT/remote/{actor}/{game}/results.json` (no token ids), which `report.py` counts with the local games; when the run
  syncs to S3 the folder is synced first and slimmed after a successful sync. A game is dropped and counted when it was
  pinned more than `max_version_lag` behind (`stale`), when its spec breaks `[held_out]` (`config`), when it is
  unusable (`rejected`: bad sha256 three times, no `results.json`, an empty or unsafe archive) or when the store
  cannot give it five times (`unfetchable`). What it has ingested, with each game's sha256, is in
  `ROOT/orchestrator/ingested.json`, and an intent record resolves a crash between writing rows and recording them (rows
  in `inbox/` or `inbox/consumed/` mean credited). A game shipped again with a different sha256 raises an alarm and is
  not credited twice. Ingest pauses while the disk is short. Panels run on this box only.
- **Actor** (`role = "actor"`, `live/actor.sh`). The same game loop, vLLM instances and hot-loading as a single box,
  with no learner, credit, inbox or panels. It reads `current.json`, refuses to start when its `[games]`, `[mix]`,
  `[held_out]` or seed hash differs from the learner's (the reason is in its heartbeat and log), pulls the learner's
  current version into `ROOT/adapters` (downloaded to a hidden folder, then renamed) and starts no game until it holds
  it, so its first wave never pins a stale version. Only an adapter that fails to download three times lets it play on
  an older one, and then only within `max_version_lag`. Each new game pins the newest version all its vLLM instances
  hold. Finished games are shipped from a persistent queue with retries and backoff. Its game ids are
  `{run_id}-{actor_id}-gNNNNNN` and its stream is the learner's config seeded by `actor_id`; a replacement box that
  reuses an `actor_id` starts past the highest id already shipped and reseeds, so its games are neither skipped nor
  replays.
- **Epochs.** Each learner process writes a random `epoch` in `current.json` and in `stop.json`. An actor acts only on a
  `stop.json` of the epoch it follows, so an old stop is never taken for a new run's, and an actor started after the
  learner stopped still obeys that stop. A restarted learner is a new epoch that running actors follow.
- **Membership is elastic.** An actor that starts later joins when its first heartbeat shows; one that dies shows as
  stale in the learner's `status.json` after `actor_stale_minutes`. Two live boxes heartbeating as one `actor_id` raise
  a `duplicate:ID` alarm (each heartbeat carries the EC2 instance id and process start).
- **Budget.** The learner's `budget_usd` covers every box: each actor's box hours at its own rate plus its frontier
  spend are added to the learner's. An actor that goes quiet without a final heartbeat is counted for at most
  `actor_stale_minutes` more.

## Setup (once)

1. Learner box: as for S3 (`docs/s2-box-day.md`, or `live/launch.sh`, `bootstrap.sh ID --train`).
2. Each actor box: `LIVE_ENV=live/train.env live/launch.sh --yes` for a box of any size with 40 GB GPUs (a g6e.12xlarge
   is four of them), then `live/bootstrap.sh ACTOR_ID --actor --yes`. Every GPU serves one vLLM instance (ports 8000
   upward), the rt engine image is built, and the power-off is armed; there is no trainer image. A different subnet is
   fine; the box needs S3 and SSM, nothing else. A different region is not supported by these scripts: each env file
   names one region, bucket and SSM prefix, so an actor in another region needs its own env file, bucket and SSM key,
   and a transport prefix that box's role can reach.
3. IAM, for the learner and every actor in one command (the role reads and writes only the `run/` prefix):
   `live/train-iam.sh create LEARNER_ID ACTOR_ID... --yes`. Run again later for another actor, it reuses the role and
   profile and only attaches the new box.
4. Configs. Add the `[transport]` block from `live/train-run.example.toml` to the learner's config, with `uri` under
   `s3://BUCKET/zeroad-train/run/RUN_ID/` and `role = "learner"`. Copy `live/actors/actor.example.toml` once per actor:
   a unique `actor_id` and `root`, `box_usd_per_hour` that box's own rate, the same `uri`, and the learner's `run_id`,
   `seed`, `[games]`, `[mix]` and `[held_out]` copied exactly (an actor with different ones refuses to start), and
   `[servers] urls` the box's own instances (`actor.sh start` compares them with `/etc/zeroad-live/vllm.urls`). An actor
   config with panels or hard states is refused.

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
  `games_per_h`, `unshipped`, `box_hours`, `box_usd`, `frontier_usd`, `instance_id` and `last_error`.
- `actors.ingested` and `actors.dropped` (by reason: `stale`, `config`, `rejected`, `unfetchable`; see above). Many stale
  drops mean the actors are slow to take new versions or `max_version_lag` is small for the learner's update rate; any
  `config` drop means an actor plays positions the learner holds out.
- `counts`: `published`, `remote_credited`, `remote_failed`, `remote_conflict`, `transport_error_*`.
- `spend` includes every actor; `alarms` has `actor:ID` while one is stale, `duplicate:ID` for two boxes with one id and
  `conflict:...` for a game shipped twice with different contents.

On an actor (`actor.sh ID status`): `transport.ready`, `transport.unshipped` (finished games waiting to upload; it
should hover near zero), `last_error`, and `counts` (`pulled`, `shipped`, `ship_failed`). The transport can also be read
from anywhere with S3 access: `python -m agentenv_zeroad.train.transport status --uri URI`.

## Stopping

`stop.json` is written only at a final stop of the learner. It is not written for a crash, SIGTERM or `train.sh stop`'s
systemd stop, so actors survive a learner restart; if the learner is really gone, its pulse goes
`learner_stale_minutes` old and they stop on that.

- **The run ends by `duration` or `complete`.** `stop.json` has `drain = true`: each actor finishes its games in flight
  (up to `drain_timeout_s`), ships them, and sends its final heartbeat. The learner keeps ingesting until every actor has
  sent it (or the same timeout), so those games are trained on.
- **The run ends by `budget`, or a tripwire pause and stop.** `drain = false`: actors cancel their games at once, ship
  what is finished and end, since nothing would ingest what they played out.
- **By hand, from anywhere:** `python -m agentenv_zeroad.train.transport stop --uri URI --reason "..." [--drain]` writes
  a stop for the epoch now running.
- **One actor, now:** `live/actor.sh ACTOR_ID stop --yes` cancels its games in flight and ships the finished ones.
- An actor also stops on its own `budget_usd` or `duration_hours`, on a games config that differs from the learner's, and
  on SIGTERM. After an actor's orchestrator ends, `actor-run.sh` runs `transport flush`: games that could not be
  uploaded are retried with backoff for up to 30 minutes, and only then is the final heartbeat sent, with `unshipped` and
  `last_error`. A box with `shutdown = true` powers off only after a clean flush; one that still holds games stays up and
  says so (`actor.sh ID status`, `actor.sh ID fetch --yes` for its files). Run `transport flush --config actor.toml` by
  hand once the network is back.
- `live/actor.sh ID fetch --yes` copies an actor's orchestrator folder and log to `live/fetched/actor-<time>/`. The games
  themselves are in the transport prefix.

## What fails and what happens

| Failure | Result |
|---|---|
| An actor dies mid-game | The game is lost; the learner never sees a manifest. After `actor_stale_minutes` the learner shows it stale. A replacement with the same `actor_id` carries on past its ids. |
| An upload dies halfway | No marker, so it is ignored; the actor retries the whole game with backoff (5 s, doubling to 5 min). |
| A corrupt or truncated bundle or adapter | Refused by its sha256 or size. A game is retried three times, then rejected and counted; a game the store cannot give is dropped after five tries. An actor retries an adapter at the next poll. |
| S3 or the network is down for a box | Reads and uploads fail and are retried every poll. The learner keeps training on its own games. An actor that can reach nothing sees the learner's pulse go stale and stops. |
| The learner restarts | It continues from the published version and its `ingested.json`, resolves any crash intent, and clears the old `stop.json`. It is a new epoch; actors keep running through it if it is back within `learner_stale_minutes`. |
| An actor restarts | It keeps its queue of unshipped games (`ROOT/orchestrator/ship.json`) and its game stream. |

## Cost

A g6e.12xlarge actor is $10.49/h on demand in us-west-2, four vLLM servers' worth of games. The learner's
`LIVE_BUDGET_USD` check at `train.sh start` counts only the learner's box; the run's `budget_usd` is what stops a
multi-box run, and it counts all boxes. Set each actor's own `budget_usd` and `duration_hours` too, and keep the
`LIVE_MAX_HOURS` power-off on every box.
