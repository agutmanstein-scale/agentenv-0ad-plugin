# shellcheck shell=bash
# Sourced by the box scripts (installed in /opt/zeroad-live/box): the paths, the box config and `run`, which prints a
# command instead of running it when DRY_RUN=1. The laptop scripts' dry runs run the box scripts that way, with the
# paths pointed at a temporary folder, so a plan shows what the box would do without touching anything.
# shellcheck disable=SC2034
set -euo pipefail

BOX_HOME=${ZEROAD_LIVE_HOME:-/opt/zeroad-live}
CONF=${ZEROAD_LIVE_CONF:-/etc/zeroad-live.conf}
STATE=${ZEROAD_LIVE_STATE:-/var/lib/zeroad-live}
LOGDIR=${ZEROAD_LIVE_LOG:-/var/log/zeroad-live}
REPO=${ZEROAD_LIVE_REPO:-$BOX_HOME/repo}
DRY_RUN=${DRY_RUN:-0}
LEDGER=$STATE/spend.jsonl
RELAY=rtmp://127.0.0.1:1935/zeroad
UNITS=(zeroad-x zeroad-relay zeroad-push zeroad-match)
if [ -f "$CONF" ]; then
  # shellcheck source=/dev/null
  source "$CONF"
fi
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2} AWS_PAGER=""

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "$*"; }
run() {
  if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
# The RTMP URL carries the stream key: every log line that could hold it goes through redact.
redact() { sed -u -E 's#rtmps?://[^[:space:]"'"'"']+#<rtmp url>#g'; }
active() { [ "$DRY_RUN" = 1 ] && { echo inactive; return; }; systemctl is-active "$1" 2>/dev/null || true; }
# require_engine_image IMAGE [PATCH...]: die unless docker has IMAGE and its zeroad.patches label names every PATCH. The
# rt engine is built by scripts/build-engine.sh rt (pilot-bootstrap.sh --train runs it); an older rt image built without
# a patch keeps its tag but not the label, and a run that expects the patch must not start on it.
require_engine_image() {
  local image=$1 have patch
  shift
  if [ "$DRY_RUN" = 1 ]; then say "+ docker image inspect $image (its zeroad.patches label must name: ${*:-nothing})"; return 0; fi
  have=$(docker image inspect -f '{{index .Config.Labels "zeroad.patches"}}' "$image" 2>/dev/null) \
    || die "engine image $image is not built on this box: $REPO/scripts/build-engine.sh rt (pilot-bootstrap.sh --train does it)"
  [ "$have" = "<no value>" ] && have=
  for patch in "$@"; do
    case " $have " in
      *" $patch "*) ;;
      *) die "engine image $image lacks the patch $patch (its zeroad.patches label is \"$have\"): REBUILD=1 $REPO/scripts/build-engine.sh rt" ;;
    esac
  done
}

# The vLLM and box-time steps pilot-bootstrap.sh and showcase-bootstrap.sh both do. A bootstrap uses them after it has
# unpacked the payload (this file comes with it); a dry run prints the same plan lines either way.

# make_vllm_key ETC: the vLLM API key, made once in ETC/vllm.env (0600, root only) and never printed.
make_vllm_key() {
  local etc=$1
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ make $etc/vllm.env (0600): VLLM_API_KEY=<random>, unless it exists"
  elif [ ! -s "$etc/vllm.env" ]; then
    (umask 077; printf 'VLLM_API_KEY=%s\n' "$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$etc/vllm.env")
  fi
}

# wait_vllm_up BRIDGE PORT...: waits (30 min at most) until each port's /health answers; dumps the container's log if not.
wait_vllm_up() {
  local bridge=$1 port i
  shift
  for port in "$@"; do
    for i in $(seq 1 180); do
      curl -fsS "http://$bridge:$port/health" >/dev/null 2>&1 && break
      [ "$i" = 180 ] && { docker logs --tail 40 "$(docker ps -a --filter "publish=$port" --format '{{.Names}}' | head -1)" >&2; echo "vLLM on $port never came up" >&2; exit 1; }
      sleep 10
    done
  done
}

# start_vllm NAME DEVICES PORT TP MOUNT MEM MAX_LEN [EXTRA_FLAG...]: one vLLM container, as Run 1 serves it (the extra
# flags are the mode's dtype, LoRA and generation-config flags). Needs BRIDGE, ETC, STATE, VLLM_IMAGE and MODEL from the
# caller. Recreated when it is not running or runs with another --max-model-len or memory share; a dry run always
# prints the commands.
start_vllm() {
  local name=$1 devices=$2 port=$3 tp=$4 mount=$5 mem=$6 max_len=$7
  shift 7
  if [ "$DRY_RUN" = 1 ] || [ -z "$(docker ps -q --filter name="^$name\$")" ] \
      || ! docker inspect --format '{{join .Args " "}}' "$name" 2>/dev/null \
        | grep -q -- "--max-model-len $max_len --gpu-memory-utilization $mem "; then
    run docker rm -f "$name" >/dev/null 2>&1 || true
    run docker run -d --name "$name" --restart unless-stopped --gpus "\"device=$devices\"" --ipc=host \
      --env-file "$ETC/vllm.env" -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True -p "$BRIDGE:$port:8000" -p "127.0.0.1:$port:8000" \
      -v "$STATE/hf:/root/.cache/huggingface" -v "$mount" \
      "$VLLM_IMAGE" --model "$MODEL" --served-model-name local/qwen-base --tensor-parallel-size "$tp" \
      --max-model-len "$max_len" --gpu-memory-utilization "$mem" --enable-lora --max-lora-rank 32 "$@" \
      --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching
  fi
}

# run_engine_test REPO PYTEST_ARG...: runs the engine tests (`pytest -m engine`) from REPO and dies unless at least one
# passed and none were skipped: pytest exits 0 when every test skips (no engine image, no Docker), which proves nothing.
run_engine_test() {
  local repo=$1 out
  shift
  if [ "$DRY_RUN" = 1 ]; then
    run uv run --directory "$repo" --frozen --group dev pytest -m engine "$@" -q -rs
    return 0
  fi
  out=$(mktemp)
  uv run --directory "$repo" --frozen --group dev pytest -m engine "$@" -q -rs 2>&1 | tee "$out" \
    || { rm -f "$out"; die "the engine test failed: $*"; }
  if ! grep -Eq '[0-9]+ passed' "$out" || grep -Eq '[0-9]+ skipped' "$out"; then
    rm -f "$out"
    die "the engine test did not pass in full (nothing passed or some tests skipped): $*"
  fi
  rm -f "$out"
}

# enable_boxtime_timer: the timer the budget guard reads box hours from.
enable_boxtime_timer() {
  run sh -c 'cp "$1"/box/units/zeroad-boxtime.service "$1"/box/units/zeroad-boxtime.timer /etc/systemd/system/' _ "$BOX_HOME"
  run systemctl daemon-reload
  run systemctl enable --now zeroad-boxtime.timer
}

# load_adapter URL NAME PATH ETC: loads the LoRA adapter at PATH (inside the vLLM container) as NAME unless /v1/models
# lists it, then checks it is listed. The key is read from ETC/vllm.env into a 0600 header file, never printed. vLLM
# forgets runtime-loaded adapters when it restarts, so a box stopped and started needs this again.
load_adapter() {
  local url=$1 name=$2 path=$3 etc=$4 headers key
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ POST $url/v1/load_lora_adapter {\"lora_name\": \"$name\", \"lora_path\": \"$path\"} (key from $etc/vllm.env, never printed), unless /v1/models lists it"
    return 0
  fi
  key=$(sed -n 's/^VLLM_API_KEY=//p' "$etc/vllm.env")
  [ -n "$key" ] || die "no VLLM_API_KEY in $etc/vllm.env"
  headers=$(umask 077; mktemp)
  printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$key" > "$headers"
  if ! curl -fsS -H "@$headers" "$url/v1/models" | grep -q "\"$name\""; then
    curl -fsS -H "@$headers" -d "{\"lora_name\": \"$name\", \"lora_path\": \"$path\"}" "$url/v1/load_lora_adapter" >/dev/null \
      || { rm -f "$headers"; die "vLLM would not load the adapter as $name"; }
  fi
  curl -fsS -H "@$headers" "$url/v1/models" | grep -q "\"$name\"" || { rm -f "$headers"; die "vLLM does not list $name after loading it"; }
  rm -f "$headers"
}

# prepare_ledger: the showcase container runs as uid 1000 and appends to the spend ledger as one bind-mounted file, so
# the file (made if missing) gets group 1000 and mode 664. Re-applied at every start: `budget.sh reset` recreates it
# root-owned.
prepare_ledger() {
  run mkdir -p "$STATE"
  run touch "$LEDGER"
  run chgrp 1000 "$LEDGER"
  run chmod 664 "$LEDGER"
}

# showcase_mounts SESSION: sets MOUNTS, the showcase container's mounts: the state folder read-only (sibling sessions,
# the ledger and boxtime.json are read from it), SESSION and the ledger file read-write, nothing else writable. The
# deadline file is a --mount so that docker fails, rather than creating a directory, when it is missing.
showcase_mounts() {
  MOUNTS=(-v "$BOX_HOME:$BOX_HOME:ro" -v "$STATE:$STATE:ro" -v "$1:$1" -v "$LEDGER:$LEDGER"
    --mount "type=bind,source=$DEADLINE_FILE,target=$DEADLINE_FILE,readonly"
    -v "$REPO/live:/work/live:ro" -v /tmp/.X11-unix:/tmp/.X11-unix)
}
