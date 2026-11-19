#!/usr/bin/env bash
# showcase-run.sh --budget USD --stamp YYYYMMDD-HHMMSS [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay]
#                 [--no-shutdown]: the zeroad-showcase unit's process (box/showcase.sh starts it).
# Runs the showcase session (python -m agentenv_zeroad.showcase run) in the zeroad-showcase container, on display :0 with
# the GPU, the X socket and the box's relay, as docker-shim.sh gives them to the env container (host networking, so the
# relay and vLLM are on the loopback). The LiteLLM key comes from SSM through with-model.sh and vLLM's from its root-only
# file; both reach the container's environment, never a command line. The session's folder is
# $STATE/showcase/showcase-STAMP (STAMP is the session's start time in UTC, also written to its `stamp` file).
# When the run ends, however it ends, the folder and the relay's segments written at or after STAMP are copied to
# s3://BUCKET/PREFIX/run/showcase-STAMP/ (segments under video/) with the box's instance profile. Then, unless
# --no-shutdown, a stop by hand (SIGTERM: the container is stopped, which records the game in play as no result and
# saves session.json) or a failed upload, it runs `shutdown -h now`, which stops the box (never terminates it).
# The bucket's name is never printed.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
BUDGET= STAMP= SHUTDOWN=1 PASS=()
while [ $# -gt 0 ]; do
  case $1 in
    --budget) BUDGET=${2:?--budget needs USD}; [[ $BUDGET =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget needs a number"; shift 2 ;;
    --stamp) STAMP=${2:?--stamp needs a time}; [[ $STAMP =~ ^[0-9]{8}-[0-9]{6}$ ]] || die "--stamp is YYYYMMDD-HHMMSS (UTC)"; shift 2 ;;
    --hours) [[ ${2:-} =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--hours needs a number"; PASS+=("$1" "$2"); shift 2 ;;
    --max-games) [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--max-games needs a number"; PASS+=("$1" "$2"); shift 2 ;;
    --rotation) [[ ${2:-} =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || die "--rotation needs a,b,c"; PASS+=("$1" "$2"); shift 2 ;;
    --mode) [[ ${2:-} =~ ^(live|replay)$ ]] || die "--mode is live or replay"; PASS+=("$1" "$2"); shift 2 ;;
    --no-shutdown) SHUTDOWN=0; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$BUDGET" ] && [ -n "$STAMP" ] || die "usage: showcase-run.sh --budget USD --stamp YYYYMMDD-HHMMSS [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay] [--no-shutdown]"
: "${ZEROAD_BOX_USD_PER_HOUR:?the box has no rate: ZEROAD_BOX_USD_PER_HOUR is not set in $CONF}"
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
DEADLINE_FILE=${ZEROAD_DEADLINE_FILE:-$ETC/deadline}
BUCKET=${ZEROAD_S3_BUCKET:-}
PREFIX=${ZEROAD_S3_PREFIX:-zeroad-train}
SESSION=$STATE/showcase/showcase-$STAMP
DEST=s3://${BUCKET:-<bucket>}/$PREFIX/run/showcase-$STAMP/
NAME=zeroad-showcase
export PATH="$PATH:/usr/local/bin"

# hide: whatever aws prints goes through this, so the bucket's name never reaches the output.
hide() { if [ -n "$BUCKET" ]; then sed "s#$BUCKET#<bucket>#g"; else cat; fi; }
s3cp() {
  if [ "$DRY_RUN" = 1 ]; then { printf '+ aws s3 cp'; printf ' %q' "$@"; printf '\n'; } | hide; return 0; fi
  aws s3 cp --only-show-errors "$@" 2>&1 | hide >&2
}

# upload: the session's folder, then each relay segment whose filename time (box time, UTC) is not before STAMP. A
# one-hour segment that began before the session is left out. Tries everything; fails if any copy failed.
upload() {
  local failed=0 seg name key
  s3cp --recursive "$SESSION" "$DEST" || failed=1
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ aws s3 cp each relay segment of $STATE/rec/zeroad written at or after $STAMP to ${DEST}video/" | hide
    return 0
  fi
  for seg in "$STATE"/rec/zeroad/*.mp4; do
    [ -e "$seg" ] || continue
    name=${seg##*/}
    key=${name:0:19}; key=${key//-/}; key=${key/_/-}  # 2026-11-19_12-00-00 -> 20261119-120000
    if [[ $key < $STAMP ]]; then continue; fi
    s3cp "$seg" "${DEST}video/$name" || failed=1
  done
  return "$failed"
}

say "showcase session $STAMP: ${PASS[*]-} (budget \$$BUDGET), folder $SESSION"
run mkdir -p "$SESSION"
[ "$DRY_RUN" = 1 ] || printf '%s\n' "$STAMP" > "$SESSION/stamp"
run docker rm -f "$NAME" >/dev/null 2>&1 || true
# The keys: LITELLM_API_KEY is in the environment from with-model.sh, VLLM_API_KEY comes from the root-only env file.
container=(docker run --rm --init --name "$NAME" --network host --gpus all
  -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size 2g
  -e DISPLAY=:0 -e ZEROAD_DISPLAY=:0 -e ZEROAD_ENCODER=nvenc -e ZEROAD_GRAPHICS=high -e "ZEROAD_RTMP_URL=$RELAY"
  --env-file "$ETC/vllm.env" -e "ZEROAD_VLLM_URL=http://127.0.0.1:8000" -e LITELLM_BASE_URL -e LITELLM_API_KEY
  -e "LIVE_BUDGET_USD=$BUDGET" -e "ZEROAD_BOX_USD_PER_HOUR=$ZEROAD_BOX_USD_PER_HOUR"
  -e "ZEROAD_LIVE_HOME=$BOX_HOME" -e "ZEROAD_LIVE_STATE=$STATE" -e "ZEROAD_DEADLINE_FILE=$DEADLINE_FILE"
  -v "$BOX_HOME:$BOX_HOME:ro" -v "$STATE:$STATE" -v "$DEADLINE_FILE:$DEADLINE_FILE:ro"
  -v "$REPO/live:/work/live:ro" -v "$SESSION:/out" -w /work
  --entrypoint python zeroad-showcase
  -m agentenv_zeroad.showcase run --out /out --config live/showcase/showcase.toml "${PASS[@]+"${PASS[@]}"}")
cmd=("$BOX_HOME/box/with-model.sh" "${container[@]}")

rc=0 STOPPED=0
if [ "$DRY_RUN" = 1 ]; then
  "${cmd[@]}"  # with-model.sh prints its own plan
else
  # In the background so that a SIGTERM to this script (systemctl stop) is handled here: the container is stopped, which
  # makes the session record the game in play as no result and save session.json, and the upload below still runs.
  ( "${cmd[@]}" 2>&1 | redact | tee -a "$SESSION/run.log" ) &
  pid=$!
  trap 'STOPPED=1; say "showcase: stop requested, stopping the container"; docker stop -t 600 "$NAME" >/dev/null 2>&1 || true' TERM INT
  while :; do
    if wait "$pid"; then rc=0; else rc=$?; fi  # a trapped signal interrupts wait while the run still goes on
    kill -0 "$pid" 2>/dev/null || break
  done
  trap - TERM INT
fi
say "showcase session $STAMP ended (exit $rc); uploading to $DEST" | hide
up=0
upload || up=1
if [ "$up" = 1 ]; then
  echo "ERROR: the upload failed: not shutting down, the box stays up with the session in $SESSION" >&2
  exit 1
fi
if [ "$STOPPED" = 1 ]; then say "stopped by hand: the box stays up"; exit 0; fi
if [ "$SHUTDOWN" = 0 ]; then say "--no-shutdown: the box stays up"; exit "$rc"; fi
run shutdown -h now
exit "$rc"
