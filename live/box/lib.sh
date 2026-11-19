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
redact() { sed -E 's#rtmps?://[^[:space:]"'"'"']+#<rtmp url>#g'; }
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

# enable_boxtime_timer: the timer the budget guard reads box hours from.
enable_boxtime_timer() {
  run sh -c 'cp "$1"/box/units/zeroad-boxtime.service "$1"/box/units/zeroad-boxtime.timer /etc/systemd/system/' _ "$BOX_HOME"
  run systemctl daemon-reload
  run systemctl enable --now zeroad-boxtime.timer
}
