#!/usr/bin/env bash
# actor.sh ID start CONFIG | status [LINES] | stop | fetch  [--yes|--dry-run]
# A rollout actor of a multi-box run on box ID (bootstrapped with bootstrap.sh --actor, on the instance role train-iam.sh
# made), through box/actor.sh (docs/s4-actors.md). `start` sends the actor config file CONFIG (TOML, [transport] role =
# "actor", live/actors/actor.example.toml) to the box, checks it there, and starts the actor as the unit zeroad-actor; it
# is refused when LIVE_BUDGET_USD would be passed or a vLLM instance is down. The actor plays games on every GPU, pulls the
# learner's adapters and ships each finished game through the transport prefix; the learner box has to be running first
# (train.sh with a [transport] role = "learner" config), or the actor gives up after learner_stale_minutes. `stop` ends
# this actor now (games in flight are cancelled, finished ones shipped, with up to 30 minutes of retries for any that
# could not go; the learner's stop.json at the end of a run is the gentler stop for every actor). `status` is read-only. `fetch` copies the actor's orchestrator files and log to
# live/fetched/actor-<time>/ (the games are in the transport prefix on S3). start, stop and fetch are dry runs unless
# --yes; status runs unless --dry-run.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status) KIND=read ;; start|stop|fetch) KIND=change ;; *) echo "actor.sh ID start|status|stop|fetch: not '${2:-}'" >&2; usage ;; esac
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
case $SUB in
  start)
    FILE=${1:-}; shift || true
    [ -f "$FILE" ] || die "start needs the actor config file: actor.sh ID start CONFIG"
    [ $# -eq 0 ] || die "start takes only CONFIG"
    [ "$DRY_RUN" = 1 ] && LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-150}
    need LIVE_BUDGET_USD
    log "actor on $ID from $FILE: budget \$$LIVE_BUDGET_USD (box hours at \$$RATE/h + frontier spend)"
    ssm_box "$ID" 120 actor.sh put-config "$(base64 < "$FILE" | tr -d '\n')"
    ssm_box "$ID" 300 actor.sh start --budget "$LIVE_BUDGET_USD" ;;
  stop)
    [ $# -eq 0 ] || die "stop takes no argument"
    ssm_box "$ID" 4200 actor.sh stop ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 actor.sh status "$@" ;;
  fetch)
    [ $# -eq 0 ] || die "fetch takes no argument"
    "$LIVE_DIR/fetch.sh" "$ID" --only train/actor $([ "$DRY_RUN" = 1 ] || echo --yes)
    if [ "$DRY_RUN" = 1 ]; then
      plan "unpack the newest live/fetched/$ID-*.tar.zst into live/fetched/actor-<time>/ (zstd -dc | tar -x)"
      exit 0
    fi
    command -v zstd >/dev/null || die "zstd is not installed here (brew install zstd): unpack the archive in live/fetched/ by hand"
    FETCHED=${LIVE_FETCHED_DIR:-$LIVE_DIR/fetched}
    archive=$(ls -1 "$FETCHED/$ID"-*.tar.zst | sort | tail -1)
    stamp=${archive##*/"$ID"-}; stamp=${stamp%.tar.zst}
    dir="$FETCHED/actor-$stamp"
    mkdir -p "$dir"
    zstd -dc "$archive" | tar -x -C "$dir"
    log "results in live/fetched/actor-$stamp/ (the run's orchestrator/ folder: status.json, final.json, ship.json; train/actor-logs/)" ;;
esac
