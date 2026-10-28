# shellcheck shell=bash
# Sourced by every laptop script in live/: the settings (live/live.env, or the file LIVE_ENV names), the dry-run
# switch, and the aws and SSM helpers.
#
# Why a dry run by default: launching a GPU box, a model match and a push to X each cost money or are public, and the
# user approves each one (spec §7 item 5). A script that changes anything prints its plan and stops unless it is given
# --yes; a read-only one runs unless it is given --dry-run. A dry run makes no aws call at all, not even
# `sts get-caller-identity`: every aws command goes through aws_run, which only prints it then.
# shellcheck disable=SC2034
set -euo pipefail

LIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$LIVE_DIR/.." && pwd)"
SCRIPT="$(basename "$0")"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0" >&2; exit 2; }
plan() { echo "+ $*" >&2; }

# The settings. A dry run without live/live.env reads live.env.example, so the plan can be read before anything is set.
LIVE_ENV="${LIVE_ENV:-$LIVE_DIR/live.env}"
load_settings() {
  local file=$LIVE_ENV
  if [ ! -f "$file" ]; then
    [ "${DRY_RUN:-1}" = 1 ] || die "no $file: copy live/live.env.example there and fill it in"
    file=$LIVE_DIR/live.env.example
    log "no $LIVE_ENV; planning with $file"
  fi
  # shellcheck source=/dev/null
  source "$file"
  if [ -n "${LIVE_AWS_PROFILE:-}" ]; then  # forced: a shell's own (e.g. production) profile must never be used here
    export AWS_PROFILE="$LIVE_AWS_PROFILE"
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  fi
  export AWS_REGION="${LIVE_AWS_REGION:-us-west-2}" AWS_DEFAULT_REGION="${LIVE_AWS_REGION:-us-west-2}" AWS_PAGER=""
  PURPOSE="${LIVE_TAG_PURPOSE:-zeroad-live}"
  BOX_NAME="${LIVE_BOX_NAME:-zeroad-live-onair}"
  RATE="${LIVE_BOX_USD_PER_HOUR:-1.32}"
  SSM_PREFIX="${LIVE_SSM_PREFIX:-/zeroad-live}"
  PREFIX="${LIVE_S3_PREFIX:-zeroad-live}"
}

need() { local v; for v in "$@"; do [ -n "${!v:-}" ] || die "set $v in $LIVE_ENV"; done; }

# parse_mode change|read ARGS...: takes --yes and --dry-run out of ARGS (the rest stays in the array ARGS) and sets
# DRY_RUN. A `change` script is a dry run unless --yes; a `read` script runs unless --dry-run.
parse_mode() {
  local kind=$1; shift
  ARGS=()
  DRY_RUN=$([ "$kind" = change ] && echo 1 || echo 0)
  local a
  for a in "$@"; do
    case $a in
      --yes) DRY_RUN=0 ;;
      --dry-run) DRY_RUN=1 ;;
      -h|--help) usage ;;
      *) ARGS+=("$a") ;;
    esac
  done
  export DRY_RUN
  load_settings
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN ($SCRIPT): nothing is launched, changed or read in AWS.$([ "$kind" = change ] && echo ' Add --yes to do it.')" >&2
  fi
}

# aws_run PLACEHOLDER ARGS...: runs `aws ARGS`; in a dry run prints it and writes PLACEHOLDER as its output instead.
AWS_CHECKED=0
aws_run() {
  local placeholder=$1; shift
  if [ "$DRY_RUN" = 1 ]; then
    plan "aws $*"
    [ -z "$placeholder" ] || printf '%s\n' "$placeholder"
    return 0
  fi
  if [ "$AWS_CHECKED" = 0 ]; then
    AWS_CHECKED=1
    if [ -n "${LIVE_AWS_ACCOUNT:-}" ]; then
      [ "$(aws sts get-caller-identity --query Account --output text)" = "$LIVE_AWS_ACCOUNT" ] \
        || die "these credentials are not account $LIVE_AWS_ACCOUNT"
    fi
  fi
  aws "$@"
}

wait_ssm() {
  local id=$1 i
  if [ "$DRY_RUN" = 1 ]; then plan "wait until $id is Online in SSM (aws ssm describe-instance-information, 15 min max)"; return; fi
  for i in $(seq 1 90); do
    [ "$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$id" \
         --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" = Online ] && return 0
    [ "$i" = 90 ] || sleep 10
  done
  die "$id never came online in SSM"
}

# box_conf: /etc/zeroad-live.conf for the box, from live.env. No secret is in it: only the SSM parameter names.
box_conf() {
  cat <<EOF
# /etc/zeroad-live.conf: written by live/bootstrap.sh from live/live.env. No secrets: keys stay in SSM.
AWS_DEFAULT_REGION=$AWS_REGION
ZEROAD_SSM_LITELLM_KEY=$SSM_PREFIX/litellm-key
ZEROAD_SSM_RTMP_SERVER=$SSM_PREFIX/rtmp-server
ZEROAD_SSM_RTMP_KEY=$SSM_PREFIX/rtmp-key
LITELLM_BASE_URL=${LIVE_LITELLM_BASE_URL:-}
ZEROAD_BOX_USD_PER_HOUR=$RATE
ZEROAD_MIN_FREE_GB=${LIVE_MIN_FREE_GB:-8}
EOF
}

# ssm_box ID TIMEOUT_S BOX_SCRIPT [ARGS...]: runs live/box/BOX_SCRIPT as root on the box through SSM and prints its
# output (SSM keeps ~24 KB of stdout). The script and its arguments travel base64-encoded in the command, so nothing
# secret may be an argument. In a dry run it prints the send, then the box script's own plan (DRY_RUN=1, run here).
ssm_box() {
  local id=$1 timeout=$2 name=$3; shift 3
  local file="$LIVE_DIR/box/$name"
  [ -f "$file" ] || die "no box script $file"
  if [ "$DRY_RUN" = 1 ]; then
    plan "aws ssm send-command --instance-ids $id --document-name AWS-RunShellScript (box/$name $*, timeout ${timeout}s)"
    local tmp rc=0
    tmp=$(mktemp -d)
    box_conf > "$tmp/zeroad-live.conf"
    echo "  on the box, box/$name would:" >&2
    DRY_RUN=1 ZEROAD_LIVE_CONF="$tmp/zeroad-live.conf" ZEROAD_LIVE_STATE="$tmp/state" ZEROAD_LIVE_LOG="$tmp/log" \
      ZEROAD_LIVE_HOME="$LIVE_DIR" ZEROAD_LIVE_REPO="$REPO_DIR" bash "$file" "$@" 2>&1 | sed 's/^/  box| /' >&2 || rc=$?
    rm -rf "$tmp"
    return "$rc"
  fi
  local wrapped b64 params cid status=Pending out
  wrapped="$(printf 'set --'; [ $# -eq 0 ] || printf ' %q' "$@"; printf '\n'; printf 'export ZEROAD_LIVE_HOME=/opt/zeroad-live\n'; cat "$file")"
  b64=$(printf '%s\n' "$wrapped" | base64 | tr -d '\n')
  params=$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": [sys.argv[2]]}))' \
    "f=\$(mktemp); echo $b64 | base64 -d > \$f; bash \$f" "$timeout")
  cid=$(aws_run "" ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript --timeout-seconds 600 \
        --comment "zeroad-live $name" --parameters "$params" --query Command.CommandId --output text)
  while :; do
    sleep "${LIVE_SSM_POLL_S:-5}"
    status=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query Status --output text \
             2>/dev/null || echo Pending)
    case $status in Pending|InProgress|Delayed|Cancelling) ;; *) break ;; esac
  done
  out=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" --output json)
  python3 -c '
import json, sys
r = json.loads(sys.stdin.read())
sys.stdout.write(r.get("StandardOutputContent", ""))
err = r.get("StandardErrorContent", "")
if err.strip():
    sys.stderr.write("--- stderr ---\n" + err)
' <<<"$out"
  [ "$status" = Success ] || { log "SSM command $cid ended $status"; return 1; }
}

# tags_of ID: "STATE NAME PURPOSE" of an instance (in a dry run, the expected ones).
tags_of() {
  # shellcheck disable=SC2016  # the backticks are JMESPath literals
  aws_run "running ${2:-$BOX_NAME} $PURPOSE" ec2 describe-instances --instance-ids "$1" --query \
    'Reservations[0].Instances[0].[State.Name, Tags[?Key==`Name`].Value|[0], Tags[?Key==`purpose`].Value|[0]]' \
    --output text
}

check_instance_id() { [[ ${1:-} =~ ^i-[0-9a-f]{8,17}$ ]] || die "not an instance id: '${1:-}'"; }
