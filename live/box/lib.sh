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
