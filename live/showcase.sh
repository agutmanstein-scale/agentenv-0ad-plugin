#!/usr/bin/env bash
# showcase.sh ID start [--hours H] [--max-games N] [--rotation LIST] [--mode live|replay] | status [LINES] | stop
#        | fetch [STAMP] [--with-video] | watch  [--yes|--dry-run]
# The showcase session (the RL-trained Qwen against Opus, Astra and Petra-3, rendered) on box ID, bootstrapped with
# bootstrap.sh --showcase; run every command with LIVE_ENV=live/showcase.env. `start` runs the session as the unit
# zeroad-showcase, through box/showcase.sh: H hours (6 unless given), refused when LIVE_BUDGET_USD would be passed by the
# box hours so far plus H plus one frontier game's cap, or when the box's power-off leaves less than H + 1 hours
# (extend-poweroff.sh moves it). LIST is a comma-separated rotation of opponents (live/showcase/showcase.toml's if not
# given); --mode replay plays the recorded games again. `status` shows the unit, the game in play, the session table,
# the spend, the power-off and the GPU memory. `stop` ends the session early: the game in play is recorded as no result,
# the session uploads, and the box stays up. `fetch` copies the newest session (or STAMP's), s3://BUCKET/LIVE_S3_PREFIX/run/showcase-
# STAMP/ (STAMP is its start time in UTC), to live/fetched/showcase-STAMP/, without the relay's video unless
# --with-video. `watch` forwards the relay's RTMP port 1935 through SSM to this laptop's loopback and prints the URL to
# open in VLC or ffplay; nothing leaves the box otherwise. start, stop and fetch are dry runs unless --yes; status and
# watch run unless --dry-run. The bucket's name is never printed.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status|watch) KIND=read ;; start|stop|fetch) KIND=change ;; *) echo "showcase.sh ID start|status|stop|fetch|watch: not '${2:-}'" >&2; usage ;; esac
export LIVE_REDACT_BUCKET=1 BOX_CONF_S3=1  # before load_settings: the plan and aws's errors show <bucket>
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
WATCH_URL=rtmp://127.0.0.1:1935/zeroad
case $SUB in
  start)
    HOURS=6; out=()
    while [ $# -gt 0 ]; do
      case $1 in
        --hours) [[ ${2:-} =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--hours needs a number"; HOURS=$2; shift 2 ;;
        --max-games) [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--max-games needs a number"; out+=("$1" "$2"); shift 2 ;;
        --rotation) [[ ${2:-} =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || die "--rotation needs a,b,c"; out+=("$1" "$2"); shift 2 ;;
        --mode) [[ ${2:-} =~ ^(live|replay)$ ]] || die "--mode is live or replay"; out+=("$1" "$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
      esac
    done
    [ "$DRY_RUN" = 1 ] && LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-60}
    need LIVE_BUDGET_USD
    log "showcase on $ID for $HOURS h against \$$LIVE_BUDGET_USD (box hours at \$$RATE/h + one frontier game's cap), box power-off in at most ${LIVE_MAX_HOURS:-?} h; the box powers off when the session ends on its own (a stop by hand keeps it up)"
    ssm_box "$ID" 300 showcase.sh start --budget "$LIVE_BUDGET_USD" --hours "$HOURS" "${out[@]+"${out[@]}"}"
    log "started; follow it with live/showcase.sh $ID status; live/showcase.sh $ID watch to see it; live/showcase.sh $ID fetch --yes afterwards" ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 showcase.sh status "$@" ;;
  stop)
    [ $# -eq 0 ] || die "stop takes no argument"
    ssm_box "$ID" 2400 showcase.sh stop ;;
  fetch)
    WITH_VIDEO=0 stamp=
    for a in "$@"; do
      case $a in
        --with-video) WITH_VIDEO=1 ;;
        [0-9]*-[0-9]*) [[ $a =~ ^[0-9]{8}-[0-9]{6}$ ]] || die "STAMP is YYYYMMDD-HHMMSS (UTC)"; stamp=$a ;;
        *) echo "unknown argument: $a" >&2; usage ;;
      esac
    done
    [ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER}
    need LIVE_BUCKET
    if [ -z "$stamp" ]; then
      listing=$(aws_run "PRE showcase-<stamp>/" s3 ls "s3://$LIVE_BUCKET/$PREFIX/run/")
      if [ "$DRY_RUN" = 1 ]; then
        stamp='<stamp>'
      else
        stamp=$(printf '%s\n' "$listing" | sed -n 's#.*PRE showcase-\([0-9]\{8\}-[0-9]\{6\}\)/.*#\1#p' | sort | tail -1)
        [ -n "$stamp" ] || die "there is no showcase session on S3 yet (nothing under s3://<bucket>/$PREFIX/run/showcase-*): the box uploads one when a session ends or is stopped"
      fi
    fi
    FETCHED=${LIVE_FETCHED_DIR:-$LIVE_DIR/fetched}
    dest="$FETCHED/showcase-$stamp"
    [ "$DRY_RUN" = 1 ] || mkdir -p "$dest"
    skip=(--exclude "video/*"); [ "$WITH_VIDEO" = 0 ] || skip=()
    aws_run "" s3 sync "s3://$LIVE_BUCKET/$PREFIX/run/showcase-$stamp/" "$dest" --only-show-errors "${skip[@]+"${skip[@]}"}"
    log "session $stamp in live/fetched/showcase-$stamp$([ "$WITH_VIDEO" = 1 ] || echo ' (no video: --with-video fetches the relay segments too)')" ;;
  watch)
    [ $# -eq 0 ] || die "watch takes no argument"
    log "forwarding the relay's RTMP port (box :1935) to this laptop's loopback; stop with Ctrl-C. Open this in VLC or ffplay:"
    echo "$WATCH_URL"
    aws_run "" ssm start-session --target "$ID" --document-name AWS-StartPortForwardingSession \
      --parameters '{"portNumber":["1935"],"localPortNumber":["1935"]}' ;;
esac
