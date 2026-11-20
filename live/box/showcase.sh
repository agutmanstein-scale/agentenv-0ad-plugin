#!/usr/bin/env bash
# showcase.sh start --budget USD [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay]
#             [--continue STAMP] [--no-shutdown] | status [LINES] | stop
# The showcase session on this box (bootstrapped with bootstrap.sh --showcase) in the transient unit zeroad-showcase
# (showcase-run.sh), which survives an SSM disconnect. `start` refuses while a session runs, while vLLM or the showcase
# image is missing, when the power-off deadline leaves less than H + 1 hours (live/extend-poweroff.sh moves it), and when
# budget.sh says --budget would be passed by the box hours so far plus H at the box rate plus one frontier game's cap
# (ZEROAD_MATCH_CAP_USD; the runner checks every later game itself). H is 6 unless given. --continue STAMP starts the
# table from the rows of the earlier session $STATE/showcase/STAMP; --no-shutdown keeps the box up at the end. `status` prints the unit, the game in play, the
# session table and its last game (from session.json), the spend, the power-off and the GPU memory. `stop` sends SIGTERM: the game in play is recorded as no
# result, the session uploads, and the box stays up (no power-off after a stop by hand).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
# session_summary FILE: how the session ended and one line per opponent, from the rows the runner saves.
session_summary() {
  jq -c '{ended, ended_why, games: (.rows | length), table: (.rows | group_by(.opponent) | map({opponent: .[0].opponent,
    w: (map(select(.result == "win")) | length), l: (map(select(.result == "loss")) | length),
    d: (map(select(.result == "draw")) | length), nr: (map(select(.result == "no_result")) | length)}))}' "$1" 2>/dev/null \
    | cut -c1-3000 || echo "(none yet)"
}
DEADLINE_FILE=${ZEROAD_DEADLINE_FILE:-$ETC/deadline}
VLLM=$(cat "$ETC/vllm.url" 2>/dev/null || true)  # showcase-bootstrap.sh writes it
SHOW=$STATE/showcase
USAGE="usage: showcase.sh start --budget USD [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay] [--continue STAMP] [--no-shutdown] | status [LINES] | stop"
case $CMD in
  start)
    BUDGET= HOURS=6 MAX_GAMES= ROTATION= MODE= CONTINUE= NO_SHUTDOWN=0
    while [ $# -gt 0 ]; do
      case $1 in
        --budget) BUDGET=${2:?--budget needs USD}; shift 2 ;;
        --hours) HOURS=${2:-}; shift 2 ;;
        --max-games) MAX_GAMES=${2:-}; shift 2 ;;
        --rotation) ROTATION=${2:-}; shift 2 ;;
        --mode) MODE=${2:-}; shift 2 ;;
        --continue) CONTINUE=${2:-}; shift 2 ;;
        --no-shutdown) NO_SHUTDOWN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the showcase live env"
    [[ $BUDGET =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget needs a number"
    [[ $HOURS =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v h="$HOURS" 'BEGIN { exit !(h > 0) }' || die "--hours needs a positive number"
    [ -z "$MAX_GAMES" ] || [[ $MAX_GAMES =~ ^[1-9][0-9]*$ ]] || die "--max-games needs a number"
    [ -z "$ROTATION" ] || [[ $ROTATION =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || die "--rotation needs a,b,c"
    [ -z "$MODE" ] || [[ $MODE =~ ^(live|replay)$ ]] || die "--mode is live or replay"
    [ -z "$CONTINUE" ] || [[ $CONTINUE =~ ^[0-9]{8}-[0-9]{6}$ ]] || die "--continue needs a STAMP (YYYYMMDD-HHMMSS, UTC)"
    [ -z "$CONTINUE" ] || [ "$DRY_RUN" = 1 ] || [ -f "$SHOW/$CONTINUE/session.json" ] \
      || die "no earlier session $CONTINUE with a session.json in $SHOW"
    [ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set in $CONF: the box has no rate"
    [ "$(active zeroad-showcase)" != active ] || die "a session is running: showcase.sh status, or stop it first"
    if [ "$DRY_RUN" = 0 ]; then
      [ -n "$VLLM" ] || die "no $ETC/vllm.url: the showcase bootstrap did not finish (live/bootstrap.sh ID --showcase)"
      docker image inspect zeroad-showcase >/dev/null 2>&1 || die "no zeroad-showcase image: live/bootstrap.sh ID --showcase first"
      curl -fsS "$VLLM/health" >/dev/null || die "vLLM is not up at $VLLM (docker logs zeroad-vllm)"
    else
      VLLM=${VLLM:-http://172.17.0.1:8000}
    fi
    # vLLM forgets runtime-loaded adapters on a restart (a box stopped and started), so the policy is loaded again here.
    load_adapter "$VLLM" local/policy-v1 /adapters/v1 "$ETC"
    for unit in zeroad-x zeroad-relay; do
      [ "$(active $unit)" = active ] || run systemctl start "$unit"
    done
    # One frontier game's cap is planned with the box hours; the runner's own budget.sh check before every game refuses
    # each later game that would pass the budget.
    cap=${ZEROAD_MATCH_CAP_USD:-9}
    if [ ! -f "$DEADLINE_FILE" ] && [ "$DRY_RUN" = 0 ]; then
      die "no power-off deadline file $DEADLINE_FILE: the box was not launched with its power-off armed"
    fi
    if [ -f "$DEADLINE_FILE" ]; then
      left=$(awk -v d="$(cat "$DEADLINE_FILE")" -v n="${ZEROAD_NOW_EPOCH:-$(date +%s)}" 'BEGIN { printf "%.2f", (d - n) / 3600 }')
      awk -v l="$left" -v h="$HOURS" 'BEGIN { exit !(l >= h + 1) }' \
        || die "the box powers off in $left h and the session needs $HOURS h + 1: live/extend-poweroff.sh ID HOURS moves the power-off"
    fi
    "$BOX_HOME/box/budget.sh" check --planned-usd "$cap" --planned-hours "$HOURS" --budget "$BUDGET"
    hours_usd=$(awk -v h="$HOURS" -v r="$ZEROAD_BOX_USD_PER_HOUR" 'BEGIN { printf "%.2f", h * r }')
    say "showcase: $HOURS h of box at \$$ZEROAD_BOX_USD_PER_HOUR/h = \$$hours_usd, plus one frontier game's cap of \$$cap, against budget \$$BUDGET (each later game is checked by the runner)"
    stamp=$(date -u +%Y%m%d-%H%M%S)
    # The container runs as uid 1000: a throwaway folder of the session's shape and the ledger must be writable to it.
    prepare_ledger
    probe=$SHOW/.preflight-$stamp
    run mkdir -p "$SHOW"
    run install -d -o 1000 -g 1000 -m 755 "$probe"
    showcase_mounts "$probe"
    run docker run --rm --entrypoint sh "${MOUNTS[@]}" zeroad-showcase -c \
      'touch "$1/.w" && rm "$1/.w" && : >> "$2"' _ "$probe" "$LEDGER" \
      || { rm -rf "$probe"; die "the showcase container (uid 1000) cannot write its session folder or open the spend ledger for append"; }
    run rm -rf "$probe"
    args=(--budget "$BUDGET" --stamp "$stamp" --hours "$HOURS")
    [ -z "$MAX_GAMES" ] || args+=(--max-games "$MAX_GAMES")
    [ -z "$ROTATION" ] || args+=(--rotation "$ROTATION")
    [ -z "$MODE" ] || args+=(--mode "$MODE")
    [ -z "$CONTINUE" ] || args+=(--continue "$SHOW/$CONTINUE")
    [ "$NO_SHUTDOWN" = 0 ] || args+=(--no-shutdown)
    # TimeoutStopSec: a stop makes the container finish its game record, then the session uploads (video included).
    # KillMode=mixed: the stop's SIGTERM reaches the script alone, which stops the container itself.
    run systemd-run --unit zeroad-showcase --collect -p KillMode=mixed -p TimeoutStopSec=1800 \
      -p "After=network-online.target docker.service" -p "Wants=network-online.target docker.service" \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/showcase-run.sh" "${args[@]}"
    say "the box powers off when a session that ran ends on its own; a stop by hand, a failed upload, a refusal before the first game or a crash keeps it up"
    say "started session $stamp; follow it with live/showcase.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: showcase.sh status [LINES]" >&2; exit 2; }
    say "zeroad-showcase: $(active zeroad-showcase)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ the newest session's session.json (the table and the last game); journalctl -u zeroad-showcase -n $N; budget.sh show; poweroff.sh status; nvidia-smi memory; vLLM health"
      exit 0
    fi
    session=$(ls -d "$SHOW"/[0-9]*-[0-9]* 2>/dev/null | sort | tail -1 || true)
    if [ -n "$session" ]; then
      echo "--- session ${session##*/}"
      echo "--- session table"; session_summary "$session/session.json"
      echo "--- last game"; jq -c '.rows | last // "(none yet)"' "$session/session.json" 2>/dev/null | cut -c1-1500 || echo "(none yet)"
      [ ! -f "$session/box.txt" ] || { echo "--- box"; cat "$session/box.txt"; }
    else
      echo "(no session yet)"
    fi
    echo "--- log"; journalctl -u zeroad-showcase -n "$N" --no-pager 2>/dev/null | redact || true
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- power-off"; "$BOX_HOME/box/poweroff.sh" status
    echo "--- gpu memory"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
    echo "--- vllm $VLLM $(curl -s -o /dev/null -w '%{http_code}' "$VLLM/health" || echo down)"
    ;;
  stop)
    if [ "$DRY_RUN" = 1 ] || [ "$(active zeroad-showcase)" = active ]; then
      say "stopping zeroad-showcase: the game in play is recorded as no result, the session uploads, the box stays up"
      run systemctl stop zeroad-showcase
    else
      say "no session running"
    fi
    ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
