#!/usr/bin/env bash
# s1.sh ID start [--n N] [--replays K] [--wall-hours H] [--engines E] | status [LINES] | fetch | stop  [--yes|--dry-run]
# The S1 tool check (docs/post-training-v2.md, docs/s1-tool-check.md) on CPU box ID (bootstrapped with bootstrap.sh
# --cpu), through box/s1.sh. No game has a model seat; the batch is the scripted bot against Petra 1, 2 and 3 on
# interface v1 and v2 (N games per interface and level, LIVE_S1_PER_CELL, default 100), K pilot games replayed under v1
# and v2 (LIVE_S1_REPLAYS, default 40, picked by scripts/s1_specs.py from LIVE_S1_PILOT_RESULTS), and the E1 CPU
# calibration, all through the lockstep runner on a pool of engines (LIVE_S1_ENGINES, default the box's vCPUs minus
# 4), then scripts/s1_summary.py's s1-summary.txt and s1-summary.json with the gates as PASS or FAIL.
# `start` picks and packs the replay specs here (with only the pilot results.json files they name), sends them to the
# box through S3 (presigned), and starts the batch as the unit zeroad-s1, which survives an SSM disconnect and ends
# itself H hours (LIVE_S1_WALL_HOURS, default 2.5) after it started; it is refused when LIVE_BUDGET_USD would be passed
# or when H leaves LIVE_MAX_HOURS less than an hour for the bootstrap and the fetch. `status` shows the phase, the games
# done, the load, the memory and the spend. `fetch` copies the results (s1/out, the ledger) back to
# live/fetched/s1-<time>/ with fetch.sh's mechanism and prints the summary. `stop` ends the batch early (the games
# done stay in s1/out; no summary is written). start, fetch and stop are dry runs unless --yes; status runs unless
# --dry-run.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status) KIND=read ;; start|fetch|stop) KIND=change ;; *) echo "s1.sh ID start|status|fetch|stop: not '${2:-}'" >&2; usage ;; esac
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
num() { [[ ${2:-} =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "$1 needs a number"; }
case $SUB in
  start)
    N=${LIVE_S1_PER_CELL:-100} K=${LIVE_S1_REPLAYS:-40} WALL=${LIVE_S1_WALL_HOURS:-2.5} ENGINES=${LIVE_S1_ENGINES:-}
    while [ $# -gt 0 ]; do
      case $1 in
        --n) N=${2:-}; shift 2 ;;
        --replays) K=${2:-}; shift 2 ;;
        --wall-hours) WALL=${2:-}; shift 2 ;;
        --engines) ENGINES=${2:-}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
      esac
    done
    num --n "$N"; num --replays "$K"; num --wall-hours "$WALL"
    [[ $N =~ ^[1-9][0-9]*$ ]] || die "--n must be a whole number of at least 1"
    [[ $K =~ ^[1-9][0-9]*$ ]] || die "--replays must be a whole number of at least 1"
    [ -z "$ENGINES" ] || [[ $ENGINES =~ ^[1-9][0-9]*$ ]] || die "--engines must be a whole number of at least 1"
    awk -v w="$WALL" -v m="${LIVE_MAX_HOURS:-12}" 'BEGIN { exit !(w > 0 && w + 1 <= m) }' \
      || die "--wall-hours $WALL leaves LIVE_MAX_HOURS ${LIVE_MAX_HOURS:-12} less than an hour for the bootstrap and the fetch"
    [ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER} LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-15}
    need LIVE_BUCKET LIVE_BUDGET_USD
    PILOT=${LIVE_S1_PILOT_RESULTS:-live/fetched/pilot-final}
    [[ $PILOT = /* ]] || PILOT=$REPO_DIR/$PILOT
    log "S1 on $ID: $((N * 6)) bot games, $((K * 2)) replay games from $K pilot games, then the CPU calibration; at most ${WALL} h of box time at \$$RATE/h against \$$LIVE_BUDGET_USD"
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    key="$PREFIX/s1/inputs-$(date -u +%Y%m%d-%H%M%S).tgz"
    if [ "$DRY_RUN" = 1 ]; then
      plan "uv run python scripts/s1_specs.py $PILOT --out <tmp>/specs --n $K --petra-only --delay-ms 5000   (K pilot games against Petra, policy v0)"
      plan "uv run python -m agentenv_zeroad.s1_batch package $PILOT <tmp>/specs --out <tmp>   (the specs and only the results.json files they name)"
    else
      [ -d "$PILOT" ] || die "no pilot results at $PILOT: set LIVE_S1_PILOT_RESULTS in $LIVE_ENV"
      (cd "$REPO_DIR" && uv run python scripts/s1_specs.py "$PILOT" --out "$tmp/specs" --n "$K" --petra-only --delay-ms 5000)
      (cd "$REPO_DIR" && uv run python -m agentenv_zeroad.s1_batch package "$PILOT" "$tmp/specs" --out "$tmp")
    fi
    aws_run "" s3 cp "$tmp/s1-inputs.tgz" "s3://$LIVE_BUCKET/$key" --only-show-errors
    url=$(aws_run https://PRESIGNED-inputs-url s3 presign "s3://$LIVE_BUCKET/$key" --expires-in 3600)
    ssm_box "$ID" 600 s1.sh put-inputs "$url"
    ssm_box "$ID" 300 s1.sh start --n "$N" --wall-hours "$WALL" ${ENGINES:+--engines "$ENGINES"} --budget "$LIVE_BUDGET_USD"
    log "started; follow it with live/s1.sh $ID status, then live/s1.sh $ID fetch --yes when it is done" ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 s1.sh status "$@" ;;
  stop)
    [ $# -eq 0 ] || die "stop takes no argument"
    ssm_box "$ID" 600 s1.sh stop ;;
  fetch)
    [ $# -eq 0 ] || die "fetch takes no argument"
    "$LIVE_DIR/fetch.sh" "$ID" --only s1/out $([ "$DRY_RUN" = 1 ] || echo --yes)
    if [ "$DRY_RUN" = 1 ]; then
      plan "unpack the newest live/fetched/$ID-*.tar.zst into live/fetched/s1-<time>/ (zstd -dc | tar -x), then print s1/out/s1-summary.txt"
      exit 0
    fi
    command -v zstd >/dev/null || die "zstd is not installed here (brew install zstd): unpack the archive in live/fetched/ by hand"
    FETCHED=${LIVE_FETCHED_DIR:-$LIVE_DIR/fetched}
    archive=$(ls -1 "$FETCHED/$ID"-*.tar.zst | sort | tail -1)
    stamp=${archive##*/"$ID"-}; stamp=${stamp%.tar.zst}
    dir="$FETCHED/s1-$stamp"
    mkdir -p "$dir"
    zstd -dc "$archive" | tar -x -C "$dir"
    log "results in live/fetched/s1-$stamp/s1/out (games per phase, calibration.json, s1-summary.txt and .json)"
    if [ -f "$dir/s1/out/s1-summary.txt" ]; then cat "$dir/s1/out/s1-summary.txt"; else log "no s1-summary.txt: the batch did not reach its summary (see s1/out/progress.log)"; fi
    log "Next: live/teardown.sh $ID --yes (the box only stops at its power-off, and its disk is billed until then)" ;;
esac
