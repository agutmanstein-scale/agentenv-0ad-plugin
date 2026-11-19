#!/usr/bin/env bash
# showcase.sh start --budget USD [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay] | status [LINES] | stop
# The showcase session on this box (bootstrapped with bootstrap.sh --showcase) in the transient unit zeroad-showcase
# (showcase-run.sh), which survives an SSM disconnect. `start` refuses while a session runs, while vLLM or the showcase
# image is missing, when the power-off deadline leaves less than H + 1 hours (live/extend-poweroff.sh moves it), and when
# budget.sh says --budget would be passed by the box hours so far plus H plus the caps of the frontier games the
# session could play (at most --max-games, else two an hour; the rotation's ids that are not petra bots, as a share of
# the rotation, at ZEROAD_MATCH_CAP_USD each). H is 4 unless given. `status` prints the unit, the game in play, the
# session table, the spend, the power-off and the GPU memory. `stop` sends SIGTERM: the game in play is recorded as no
# result, the session uploads, and the box stays up (no power-off after a stop by hand).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
DEADLINE_FILE=${ZEROAD_DEADLINE_FILE:-$ETC/deadline}
VLLM=$(cat "$ETC/vllm.url" 2>/dev/null || echo http://172.17.0.1:8000)
SHOW=$STATE/showcase
USAGE="usage: showcase.sh start --budget USD [--hours H] [--max-games N] [--rotation a,b,c] [--mode live|replay] | status [LINES] | stop"
case $CMD in
  start)
    BUDGET= HOURS=4 MAX_GAMES= ROTATION= MODE=
    while [ $# -gt 0 ]; do
      case $1 in
        --budget) BUDGET=${2:?--budget needs USD}; shift 2 ;;
        --hours) HOURS=${2:-}; shift 2 ;;
        --max-games) MAX_GAMES=${2:-}; shift 2 ;;
        --rotation) ROTATION=${2:-}; shift 2 ;;
        --mode) MODE=${2:-}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the showcase live env"
    [[ $BUDGET =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget needs a number"
    [[ $HOURS =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v h="$HOURS" 'BEGIN { exit !(h > 0) }' || die "--hours needs a positive number"
    [ -z "$MAX_GAMES" ] || [[ $MAX_GAMES =~ ^[1-9][0-9]*$ ]] || die "--max-games needs a number"
    [ -z "$ROTATION" ] || [[ $ROTATION =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || die "--rotation needs a,b,c"
    [ -z "$MODE" ] || [[ $MODE =~ ^(live|replay)$ ]] || die "--mode is live or replay"
    [ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set in $CONF: the box has no rate"
    [ "$(active zeroad-showcase)" != active ] || die "a session is running: showcase.sh status, or stop it first"
    if [ "$DRY_RUN" = 0 ]; then
      docker image inspect zeroad-showcase >/dev/null 2>&1 || die "no zeroad-showcase image: live/bootstrap.sh ID --showcase first"
      curl -fsS "$VLLM/health" >/dev/null || die "vLLM is not up at $VLLM (docker logs zeroad-vllm)"
    fi
    # The frontier games' caps: the rotation's ids other than the petra bots, as a share of the games it could play.
    rotation=$ROTATION
    [ -n "$rotation" ] || rotation=$(sed -n 's/^rotation *= *\[\(.*\)\].*/\1/p' "$REPO/live/showcase/showcase.toml" 2>/dev/null | tr -d '" ')
    [ -n "$rotation" ] || die "no rotation: give --rotation or set it in live/showcase/showcase.toml"
    frontier=$(tr ',' '\n' <<<"$rotation" | grep -vc '^petra' || true)
    total=$(tr ',' '\n' <<<"$rotation" | wc -l | tr -d ' ')
    cap=${ZEROAD_MATCH_CAP_USD:-9}
    games=$(awk -v h="$HOURS" -v m="${MAX_GAMES:-0}" 'BEGIN { n = int(h * 2); if (n < h * 2) n++; if (m > 0 && m < n) n = m; print n }')
    usd=$(awk -v g="$games" -v f="$frontier" -v t="$total" -v c="$cap" 'BEGIN { k = g * f / t; n = int(k); if (n < k) n++; printf "%.2f", n * c }')
    if [ -f "$DEADLINE_FILE" ]; then
      left=$(awk -v d="$(cat "$DEADLINE_FILE")" -v n="${ZEROAD_NOW_EPOCH:-$(date +%s)}" 'BEGIN { printf "%.2f", (d - n) / 3600 }')
      awk -v l="$left" -v h="$HOURS" 'BEGIN { exit !(l >= h + 1) }' \
        || die "the box powers off in $left h and the session needs $HOURS h + 1: live/extend-poweroff.sh ID HOURS moves the power-off"
    fi
    "$BOX_HOME/box/budget.sh" check --planned-usd "$usd" --planned-hours "$HOURS" --budget "$BUDGET"
    say "showcase: $HOURS h, up to $games games, $frontier of $total rotation ids are frontier: \$$usd of caps at \$$cap, budget \$$BUDGET"
    stamp=$(date -u +%Y%m%d-%H%M%S)
    args=(--budget "$BUDGET" --stamp "$stamp" --hours "$HOURS")
    [ -z "$MAX_GAMES" ] || args+=(--max-games "$MAX_GAMES")
    [ -z "$ROTATION" ] || args+=(--rotation "$ROTATION")
    [ -z "$MODE" ] || args+=(--mode "$MODE")
    # TimeoutStopSec: a stop makes the container finish its game record, then the session uploads (video included).
    # KillMode=mixed: the stop's SIGTERM reaches the script alone, which stops the container itself.
    run systemd-run --unit zeroad-showcase --collect -p KillMode=mixed -p TimeoutStopSec=1800 \
      -p "After=network-online.target docker.service" -p "Wants=network-online.target docker.service" \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/showcase-run.sh" "${args[@]}"
    say "started session $stamp; follow it with live/showcase.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: showcase.sh status [LINES]" >&2; exit 2; }
    say "zeroad-showcase: $(active zeroad-showcase)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ the newest session's state.json (the game in play) and session.json (the table); journalctl -u zeroad-showcase -n $N; budget.sh show; poweroff.sh status; nvidia-smi memory; vLLM health"
      exit 0
    fi
    session=$(ls -d "$SHOW"/showcase-* 2>/dev/null | sort | tail -1 || true)
    if [ -n "$session" ]; then
      echo "--- session ${session##*/}"
      echo "--- game in play"; jq -c . "$session/state.json" 2>/dev/null | cut -c1-1500 || echo "(none yet)"
      echo "--- session table"; jq -c . "$session/session.json" 2>/dev/null | cut -c1-3000 || echo "(none yet)"
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
