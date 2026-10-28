# Livestream runbook: a 0 A.D. arena match on X

This is how to stream one rendered 0 A.D. match from one AWS GPU box to X, or to any RTMP target. The scripts are in
`live/`, and the stack follows `agentenv-airops-plugin/live`: one g6.4xlarge (NVIDIA L4), Xorg on the GPU, Chrome
for the overlay, ffmpeg with NVENC, a local relay that records everything, and a push to RTMP. Secrets live only in
SSM Parameter Store.

**Steps marked GO-AHEAD cost money or are public.** Each one needs the user's explicit approval at the time, with the
spend so far (spec §7 item 5). The scripts enforce this: any script that launches, changes or deletes something only
prints its plan unless it is given `--yes`.

## Costs

| What | Cost | Source |
|---|---|---|
| The box, g6.4xlarge (1 L4, 16 vCPU, 64 GB), on demand | about **$1.32/h** while it runs, stopped or not yet terminated | spec §8; `LIVE_BOX_USD_PER_HOUR` |
| Setup: launch, bootstrap with the images built on the box | about 45 min of box time, about $1 | |
| `$0` rehearsal: scripted bot vs Petra, 5 game-min (`livestream-petra`) | $0 of model spend, about 15 min of box time ($0.33) | |
| One model match: `livestream`, Setup A, 40 game-min | **up to $9** of model spend (strategists $4 + $4, Luna reflexes $0.50 + $0.50), plus about 1 h of box time ($1.32) | the task's caps |
| Egress while pushing at 6 Mbit/s | about 2.7 GB/h, about $0.25/h | |
| Root volume of a stopped box, 200 GB gp3 | about $16 per month | |

So a session with setup, a rehearsal and one model match comes to about **$9 + $3.50 ≈ $12.50** at most. The box
powers itself off, and so terminates, `LIVE_MAX_HOURS` (default 12) after launch, which caps the box at about $16.

**The budget guard.** Before every match, `box/budget.sh` adds up the model spend in the box ledger, the box hours
over every boot of the box × $1.32 (the `zeroad-boxtime` timer records each boot's running time, so a stop-box and
start keeps the earlier hours), this match's caps, and its expected box hours. It refuses to start the match if that total would
pass `LIVE_BUDGET_USD`. A model match is also refused if its task's caps add up to more than `LIVE_MATCH_CAP_USD`
($9). When a match ends, its estimated spend from `results.json` goes into the ledger; with no results, the full
caps go in. Afterwards, add the match to the loop ledger: `python -m agentenv_zeroad.spend add --ledger
<loop>/spend.jsonl --what "livestream <name>" --usd <usd>`.

## What runs where

```
laptop: live/*.sh  --SSM-->  box (g6.4xlarge, DLAMI Ubuntu 22.04, NVIDIA driver and container toolkit)
  zeroad-x       Xorg :0 on the L4, a virtual 1920x1080 screen (box/display.sh); local clients allowed (xhost)
  zeroad-relay   mediamtx on :1935; records path "zeroad" to /var/lib/zeroad-live/rec (one-hour fMP4 segments)
  zeroad-match   agent-env run <bundle> --task livestream[-petra], started by box/match.sh (transient unit)
                   env container mcp-server-zeroad-live, with the GPU, the X socket and the relay added by the
                   docker shim: the -rl-realtime engine renders the observer view on :0 with the director mod,
                   Chrome shows the overlay (/live?stream) beside it, ffmpeg (NVENC) records an MP4 and sends the
                   same stream to rtmp://host.docker.internal:1935/zeroad
                   agents zeroad-commander x2 (model match only), with the LiteLLM key from SSM
  zeroad-push    only when asked: relay -> RTMP server/key from SSM, reconnecting; SIGINT on stop
  zeroad-health  timer, every 30 min: box/repair.sh --auto (repairs what it can, appends to health.jsonl)
```

The env container gets the GPU and the display through `box/docker-shim.sh`. agent-env's server provider can't add
devices or mounts, so the shim sits first on the match process's PATH and adds them to the live env image's
`docker run` only. The stream key never enters the env container: the env publishes to the box's relay, and
`zeroad-push` sends the relay on.

## 0. Prerequisites (once)

1. **AWS account:**
   - SSM-managed instances, and G-instance quota of at least 16 vCPU;
   - subnets with internet access, one per zone in `LIVE_SUBNETS`;
   - a security group with **no inbound rule**;
   - an instance profile with `AmazonSSMManagedInstanceCore`, plus `ssm:GetParameter` on `/zeroad-live/*`;
   - an S3 bucket in the same region.
2. **On the laptop:**
   - bash, the AWS CLI v2, python3 (with boto3 or uv), git, tar and zstd;
   - this checkout, with the code to stream **committed**. Only HEAD ships.
3. **Settings:** `cp live/live.env.example live/live.env`, then fill in every value. The file is git-ignored and
   holds no secrets. Set `LIVE_BUDGET_USD` to the total the user approved for model and box together.
4. **Credentials:** `aws sso login` (or your profile's login), then check them with
   `aws sts get-caller-identity`.
5. **The model key**, typed by the user, never pasted into a chat:
   ```bash
   live/put-secret.sh litellm-key --yes    # reads without echo; stored as /zeroad-live/litellm-key (SecureString)
   ```
   The RTMP server and key can wait until step 5.

Every script prints its plan without `--yes`. Run a script without `--yes` first, read the plan, then run it again
with `--yes`.

## 1. Launch the box — GO-AHEAD (about $1.32/h from now on)

```bash
live/launch.sh                              # the plan
ID=$(live/launch.sh --yes)                  # tries each type and subnet; prints the instance id once SSM sees it
```

`--max-hours H` sets the self-power-off (default `LIVE_MAX_HOURS`, 12). Power-off **terminates** the box. To cancel
it, run `sudo shutdown -c` on the box.

## 2. Bootstrap (about 30-45 min of box time; nothing plays or streams)

```bash
live/bootstrap.sh $ID                       # the plan
live/bootstrap.sh $ID --yes                 # or --images pull with LIVE_IMAGE_REGISTRY set
live/status.sh $ID
```

Bootstrap does the following:
- checks the L4 and its driver;
- adds Docker address pools and keeps the NVIDIA runtime;
- installs Xorg, ffmpeg, mediamtx v1.21.2 and uv;
- unpacks the code at `/opt/zeroad-live/repo`;
- runs `agent-env zeroad setup --live` (the `-rl-realtime` engine build is most of the time) and
  `setup --agents-only`;
- installs the docker shim and the units, and starts the display, the relay and the health timer.

`status.sh` should show `zeroad-x=active zeroad-relay=active`.

## 3. Rehearsal at $0 (bot vs Petra, about 6 min; recorded, not pushed)

The model spend is $0, but box time is still running. Ask before this step if the box is idling for it.

```bash
live/match.sh $ID rehearsal                 # the plan: task livestream-petra, caps $0.00, the budget line
live/match.sh $ID rehearsal --yes
live/health.sh $ID                          # repeat while it plays
live/status.sh $ID
```

**Healthy** looks like this:
- `status` is `ok`;
- `units.match` is `active`;
- `relay.ready` is true, and `relay.bytes_in` grows;
- `recording.age_s` is under 120;
- `gpu.nvenc_pct` is above 0;
- `env_container` is `Up`.

Then fetch the recording and watch it, checking for:
- the observer view with no fog;
- director cuts held at least 8 s;
- the overlay beside the game, with the clock, the scoreboard and per-empire panels;
- smooth motion at 30 fps.

```bash
live/fetch.sh $ID <match name> --yes        # to live/fetched/ (match folder + its relay segments)
```

## 4. Model match — GO-AHEAD (up to $9 model spend plus about $1.32 box)

```bash
live/match.sh $ID model                     # the plan: caps $9.00 of the $9 match cap, the budget line
live/match.sh $ID model --yes               # recorded only; add --push to go public at once (step 5)
```

This plays the `livestream` task:
- Opus 5.5 against GPT-6 Astra, each with a Luna reflex;
- 40 game-minutes in real time;
- each brain stops at its own cap (`ZEROAD_MAX_COST_USD`, `reflex_max_usd`).

The LiteLLM key is read from SSM into the match process's environment only. A second match is refused while one is
running.

## 5. Go live on X — GO-AHEAD (public)

1. In X Media Studio (studio.x.com), go to **Producer**, then **Sources**, then **Create source**. Pick RTMP and the
   nearest region. The user stores the **RTMP URL** and the **stream key** in SSM:
   ```bash
   live/put-secret.sh rtmp-server --yes     # e.g. rtmps://<region>.pscp.tv:443/x
   live/put-secret.sh rtmp-key --yes        # the stream key ("-" if the URL already holds it)
   ```
2. Start the push. This restarts nothing else; the capture and the recording go on:
   ```bash
   live/push.sh $ID start                   # the plan
   live/push.sh $ID start --yes             # or: live/match.sh $ID model --push --yes
   ```
3. In Producer, go to **Broadcasts**, then **Create broadcast**. Pick the source, set the title, wait for the
   preview, then press **Go Live**.

The key is read from SSM each time the push starts, and is never printed or logged. Logs show `<rtmp url>`. If X's
RTMP connection drops, only the push restarts (backing off from 1 s, doubling to 30 s). A new key in SSM takes effect
at the next `push.sh start`. The push ends with its match: when the match ends or is stopped, the push stops too, and a
later match started without `--push` stops any push still on before it starts, so it is recorded only.

## 6. Monitor every 30 minutes, and repair

```bash
live/health.sh $ID                          # one JSON line; status ok or warn with problems
live/status.sh $ID 30                       # units, the match's and the push's last 30 log lines, spend, GPU, disk
live/repair.sh $ID --yes                    # restart what is down (never a running match), free disk
```

The box also runs `box/repair.sh --auto` every 30 minutes and appends to `/var/lib/zeroad-live/health.jsonl`.

| Problem | Meaning | Fix |
|---|---|---|
| `display :0 down` | Xorg died; a running match lost its picture | `repair.sh --yes`, then `stop.sh --yes` and start the match again |
| `relay down` | mediamtx died | `repair.sh --yes`; the env's RTMP leg doesn't reconnect, so the recording resumes with the next match (the env's own MP4 is unaffected) |
| `push wanted but not running` | the push unit stopped | `repair.sh --yes` (or `push.sh start --yes`); check the SSM parameters |
| `match running but nothing publishes to the relay` | the env's ffmpeg is not streaming | check that `rtmp: true` is in the match bundle, the shim log (`/var/log/zeroad-live/docker-shim.log`) and `status.sh` |
| `recording is not growing` | as above, or the disk is full | `repair.sh --yes` |
| `disk N GB free` | under `LIVE_MIN_FREE_GB` | `repair.sh` deletes the oldest segments (never the newest) |

## 7. Stop, save, tear down

```bash
live/stop.sh $ID --push-only --yes          # end the push cleanly (ffmpeg SIGINT); the match plays on, recorded
live/stop.sh $ID --yes                      # end the push, then stop the match (agent-env tears it down)
live/fetch.sh $ID --yes                     # all match folders, ledger, health log and recordings to live/fetched/
```

A match ends by itself at the 40-minute cap or by conquest. End the broadcast in Producer before stopping the push.

**Tear down — GO-AHEAD (deletes the box and everything on it):**

```bash
live/teardown.sh $ID                        # checks Name and purpose tags; the plan
live/teardown.sh $ID --yes
```

Or pause instead: `live/stop-box.sh $ID --yes` stops the GPU charge but keeps the volume (about $16/month).

Then clean up by hand:
- reset the stream key in Media Studio;
- delete the SSM parameters (`teardown.sh` prints the command);
- empty `s3://$LIVE_BUCKET/$LIVE_S3_PREFIX/`;
- add the match spend to the loop ledger.

## Script reference

| Script | Kind | What |
|---|---|---|
| `live/launch.sh` | `--yes` | launch the box (types × subnets), power-off after H hours |
| `live/bootstrap.sh` | `--yes` | ship HEAD and run `box/bootstrap.sh` |
| `live/match.sh` | `--yes` | `rehearsal` or `model`, `--push`, `--name` |
| `live/push.sh` | `--yes` | `start`, `stop` |
| `live/stop.sh` | `--yes` | stop the push, then the match (`--push-only`) |
| `live/repair.sh` | `--yes` | the repairs now |
| `live/fetch.sh` | `--yes` | match folders and recordings to `live/fetched/` |
| `live/stop-box.sh` | `--yes` | stop (not terminate) the box |
| `live/teardown.sh` | `--yes` | terminate the box (tag-checked) |
| `live/status.sh` | read | EC2 state, units, logs, spend, GPU, disk, power-off |
| `live/health.sh` | read | one JSON line |
| `live/put-secret.sh` | `--yes` (interactive) | store `litellm-key`, `rtmp-server` or `rtmp-key` in SSM |

The box side is in `live/box/`:
- `bootstrap.sh`, `display.sh` and `display-ready.sh` set up the box and the display;
- `docker-shim.sh` gives the env container the GPU, the display and the relay;
- `match.sh`, `run-match.sh`, `with-model.sh` and `budget.sh` run a match and guard its spend;
- `push.sh`, `stop.sh`, `health.sh`, `repair.sh`, `status.sh` and `fetch.sh` manage the stream;
- the units are in `units/`, and the relay's config is `mediamtx.yml`.

Every box script prints its plan with `DRY_RUN=1`.

## Known limits (not yet proven on a real box)

- The shim's flags are proven against a fake docker only:
  - `--gpus all` with `NVIDIA_DRIVER_CAPABILITIES=all` (GLX and NVENC in the container);
  - the X socket mount, with `xhost +local:`;
  - `host.docker.internal` reaching the relay.

  The first rehearsal checks them all: check `gpu.nvenc_pct` and a fetched recording.
- If the env container can't use the host's Xorg, there are two fallbacks:
  - set no `ZEROAD_DISPLAY`: the env uses its own Xvfb, still encoding with NVENC, with software GL. This is low fps
    (U6 measured ~11 fps at 2/3 scale);
  - render with VirtualGL. Neither is scripted here.
- The push URL, stream key included, is on ffmpeg's command line on the box. Bootstrap remounts `/proc` with
  `hidepid=2` (and keeps it in `/etc/fstab`), so only root can see other users' command lines with `ps`; the push
  runs as root. The URL is never logged or written to a file.
