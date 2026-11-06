#!/usr/bin/env bash
# s1.sh put-inputs URL | start [--n N] [--wall-hours H] [--engines E] --budget USD | status [LINES] | stop
# The S1 tool check on this CPU box (bootstrapped with bootstrap.sh --cpu), in the transient unit zeroad-s1 (s1-run.sh:
# the lockstep runner on an engine pool for the bot games, the replays and the calibration, then the summary).
# `put-inputs` downloads and unpacks the replay specs and the pilot results they name (a tarball from live/s1.sh, at a
# presigned URL) into $STATE/s1/inputs. `start` refuses while a batch is going, without inputs, or when budget.sh says
# --budget would be passed by the box hours so far plus H hours (the batch's wall-clock limit, default 2.5) at the
# box's rate; it moves an earlier s1/out aside, starts the unit with a hard stop at H hours plus 10 minutes, and
# the batch itself ends at H hours. --n is the bot games per interface and Petra level (default 100), --engines the
# pool's size (default the vCPUs minus 4, at least 4). `status` prints the unit, the batch's log, the games done per
# phase, the calibration levels done, the engines, the load, the memory and the spend. `stop` ends the unit and removes
# its engines (the games done are kept in s1/out).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
S1=$STATE/s1
INPUTS=$S1/inputs
OUT=$S1/out
USAGE="usage: s1.sh put-inputs URL | start [--n N] [--wall-hours H] [--engines E] --budget USD | status [LINES] | stop"
case $CMD in
  put-inputs)
    URL=${1:?$USAGE}
    [[ $URL =~ ^https://[^[:space:]\'\"]+$ ]] || die "put-inputs needs an https URL"
    [ "$(active zeroad-s1)" != active ] || die "a batch is running: s1.sh status, or stop it first"
    run mkdir -p "$S1"
    run sh -c 'rm -rf "$1/inputs.new" && mkdir -p "$1/inputs.new" && curl -fsSL "$2" | tar xz -C "$1/inputs.new" && rm -rf "$1/inputs" && mv "$1/inputs.new" "$1/inputs"' _ "$S1" "$URL"
    if [ "$DRY_RUN" = 0 ]; then
      [ -d "$INPUTS/specs" ] && [ -d "$INPUTS/pilot" ] || die "the inputs have no specs/ and pilot/ folders"
      say "inputs in $INPUTS: $(find "$INPUTS/specs" -name '*.json' ! -name index.json | wc -l) replay specs, $(find "$INPUTS/pilot" -name '*results.json' | wc -l) pilot results"
    fi
    ;;
  start)
    N=100 WALL=2.5 ENGINES= BUDGET=
    while [ $# -gt 0 ]; do
      case $1 in
        --n) N=${2:-}; [[ $N =~ ^[1-9][0-9]*$ ]] || die "--n needs a whole number"; shift 2 ;;
        --wall-hours) WALL=${2:-}; [[ $WALL =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--wall-hours needs a number"; shift 2 ;;
        --engines) ENGINES=${2:-}; [[ $ENGINES =~ ^[1-9][0-9]*$ ]] || die "--engines needs a whole number"; shift 2 ;;
        --budget) BUDGET=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the S1 live env"
    awk -v w="$WALL" 'BEGIN { exit !(w > 0) }' || die "--wall-hours must be positive"
    [ "$(active zeroad-s1)" != active ] || die "a batch is running: s1.sh status, or stop it first"
    [ "$DRY_RUN" = 1 ] || [ -d "$INPUTS/specs" ] || die "no inputs in $INPUTS: live/s1.sh start sends them (put-inputs)"
    [ -n "$ENGINES" ] || ENGINES=$(awk -v n="$(nproc 2>/dev/null || echo 48)" 'BEGIN { n -= 4; print (n < 4 ? 4 : n) }')
    WALL_S=$(awk -v w="$WALL" 'BEGIN { printf "%d", w * 3600 }')
    "$BOX_HOME/box/budget.sh" check --planned-usd 0 --planned-hours "$WALL" --budget "$BUDGET"
    say "S1: $((N * 6)) bot games, the replays, the CPU calibration, on $ENGINES engines, at most $WALL h"
    [ ! -e "$OUT" ] || run mv "$OUT" "$S1/out-$(date -u +%Y%m%d-%H%M%S)"
    # The unit's own limit is the batch's plus 10 minutes: the batch ends itself at the wall-clock limit and writes its summary.
    run systemd-run --unit zeroad-s1 --collect -p KillMode=control-group -p TimeoutStopSec=300 \
      -p "RuntimeMaxSec=$((WALL_S + 600))" -p "After=network-online.target docker.service" \
      -p "Wants=network-online.target docker.service" -p "WorkingDirectory=$REPO" \
      "$BOX_HOME/box/s1-run.sh" --n "$N" --wall-s "$WALL_S" --engines "$ENGINES"
    say "started; follow it with live/s1.sh ID status"
    ;;
  status)
    N=${1:-15}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: s1.sh status [LINES]" >&2; exit 2; }
    say "zeroad-s1: $(active zeroad-s1)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ tail s1/out/progress.log; games.jsonl per phase; calibration.json levels; engines; load; free; budget.sh show; df; power-off"
      exit 0
    fi
    echo "--- batch log"; tail -n "$N" "$OUT/progress.log" 2>/dev/null || echo "(no batch yet)"
    echo "--- games done (finished/all lines of games.jsonl)"
    for phase in replay bot; do
      f=$OUT/$phase/games.jsonl
      if [ -f "$f" ]; then echo "$phase: $(grep -c '"status": "finished"' "$f" || true) finished of $(wc -l < "$f")"; else echo "$phase: none yet"; fi
    done
    echo "--- calibration levels done"; jq -r '.levels | keys | join(" ")' "$OUT/calibration/calibration.json" 2>/dev/null || echo "none yet"
    echo "--- engines"; echo "$(docker ps -q --filter name=zeroad-engine- | wc -l) containers"
    echo "--- cpu and memory"; echo "$(nproc) vCPU, load $(cut -d' ' -f1-3 /proc/loadavg)"; free -m | awk '/^Mem:/ {print $3 " of " $2 " MB used"}'
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- disk"; df -h "$STATE" | tail -1
    echo "--- power-off"; busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null || echo none
    ;;
  stop)
    say "stopping zeroad-s1: the games in flight are lost, the games done stay in $OUT"
    run systemctl stop zeroad-s1
    ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
