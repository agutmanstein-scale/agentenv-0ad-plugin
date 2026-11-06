#!/usr/bin/env bash
# s1-run.sh --n N --wall-s S --engines E [--calibration-only] [--cal-args "ARGS"]: the zeroad-s1 unit's process (s1.sh starts it).
# The S1 batch, in order, each phase under its own time limit and all of them inside S seconds from the start:
#   1. build the game specs (agentenv_zeroad.s1_batch build): bot.jsonl, replay.jsonl;
#   2. the replays: 2 games per pilot game, v1 and v2 (limit 25 min);
#   3. the E1 CPU calibration: 1, 20, 40 and 60 engines in flight (agentenv_zeroad.s1_calibrate; limit 50 min);
#   4. the bot games: v1 and v2 against Petra 1, 2 and 3 (the rest of the time but the summary's 3 minutes). The specs
#      are ordered game by game, so a phase cut at its limit leaves the cells equal;
#   5. scripts/s1_summary.py: s1-summary.txt and s1-summary.json (the gates, PASS or FAIL, and the calibration).
# The games of phases 2 and 4 are played by `python -m agentenv_zeroad.train.runner` on a pool of E engines in docker
# (host network, one container each), in worker processes of 4 engines each (S1_ENGINES_PER_WORKER) on the idle-wait
# engine image zeroad-engine:0.28-rt (S1_ENGINE_IMAGE; the PPA's zeroad-engine:0.28 spins a core per engine). A phase
# that reaches its limit is cut (its engines are removed); one that fails is logged and the batch goes on. The log is
# s1/out/progress.log.
# The calibration (phase 3) runs `s1_calibrate --levels 12,24,44 --per-worker 4,24 --idle-engines 1,20` (levels of 12, 24
# and 44 engines, each in workers of 4 and in one loop of up to 24; idle engines on the PPA image and the patched one; a
# same-seed parity game on both) unless --cal-args gives its flags (e.g. "--levels 20,40 --per-worker 2,4,11").
# The engine image must carry the zeroad.patches label (rl-realtime rl-idle-wait) or the batch refuses to start; S1_ALLOW_UNPATCHED=1
# lets a comparison run use another image. With --calibration-only the batch is only that phase
# and s1/out/calibration.txt, the report of s1/out/calibration/calibration.json (no specs, replays, bot games or gates).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
N= WALL_S= ENGINES= CAL_ONLY=0 CAL_ARGS="--levels 12,24,44 --per-worker 4,24 --idle-engines 1,20"
while [ $# -gt 0 ]; do
  case $1 in
    --n) N=${2:?}; shift 2 ;;
    --wall-s) WALL_S=${2:?}; shift 2 ;;
    --engines) ENGINES=${2:?}; shift 2 ;;
    --calibration-only) CAL_ONLY=1; shift ;;
    --cal-args) CAL_ARGS=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
for v in N WALL_S ENGINES; do [[ ${!v} =~ ^[1-9][0-9]*$ ]] || die "usage: s1-run.sh --n N --wall-s S --engines E"; done
S1=$STATE/s1
INPUTS=$S1/inputs
OUT=$S1/out
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; a dry run elsewhere may have it elsewhere
export PATH="/usr/local/bin:/usr/bin:/bin${UV_BIN:+:${UV_BIN%/*}}"
PY=(uv run --project "$REPO" --frozen --no-dev python)
[[ $CAL_ARGS =~ ^[A-Za-z0-9._,=:/\ -]*$ ]] || die "--cal-args has characters outside A-Za-z0-9._,=:/ -"
ENGINES_PER_WORKER=${S1_ENGINES_PER_WORKER:-4}
ENGINE_IMAGE=${S1_ENGINE_IMAGE:-zeroad-engine:0.28-rt}
UNPATCHED=()
if [ "${S1_ALLOW_UNPATCHED:-0}" = 1 ]; then UNPATCHED=(--allow-unpatched); else require_engine_image "$ENGINE_IMAGE" rl-realtime rl-idle-wait; fi
require_engine_image zeroad-engine:0.28  # the calibration compares its idle engines and a parity game to the PPA image
WORKERS=$(( (ENGINES + ENGINES_PER_WORKER - 1) / ENGINES_PER_WORKER ))
SUMMARY_RESERVE_S=180
REPLAY_LIMIT_S=1500
CALIBRATION_LIMIT_S=3000
LEVEL_TIMEOUT_S=800
START=$(date +%s)
DEADLINE=$((START + WALL_S))
run mkdir -p "$OUT"
note() { echo "$(date -u +%H:%M:%S) $*" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a "$OUT/progress.log"; fi; }; }
engines_gone() { [ "$DRY_RUN" = 1 ] || { ids=$(docker ps -aq --filter name=zeroad-engine- 2>/dev/null || true); [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true; }; }
trap engines_gone EXIT

# phase NAME MAX_SECONDS COMMAND...: runs COMMAND under `timeout` for at most MAX_SECONDS, or what is left before the
# deadline (less the summary's reserve) if that is less; skipped when under a minute is left.
phase() {
  local name=$1 max=$2; shift 2
  local left=$(( DEADLINE - $(date +%s) - SUMMARY_RESERVE_S )) limit began rc=0
  limit=$(( max < left ? max : left ))
  if [ "$limit" -lt 60 ]; then note "phase $name skipped: ${left}s left before the summary"; return 0; fi
  began=$(date +%s)
  note "phase $name starts (limit ${limit}s)"
  run timeout -k 30 "$limit" "$@" || rc=$?
  engines_gone
  case $rc in
    0) note "phase $name done in $(( $(date +%s) - began ))s" ;;
    124) note "phase $name cut at its ${limit}s limit: the games done are kept" ;;
    *) note "phase $name FAILED (exit $rc) after $(( $(date +%s) - began ))s; the batch goes on" ;;
  esac
}
# runner_cmd NAME: RUNNER, the command that plays $OUT/specs/NAME.jsonl into $OUT/NAME on the pool.
runner_cmd() {
  RUNNER=("${PY[@]}" -m agentenv_zeroad.train.runner --specs "$OUT/specs/$1.jsonl" --out "$OUT/$1"
    --engines "$ENGINES" --engines-per-worker "$ENGINES_PER_WORKER" --image "$ENGINE_IMAGE" "${UNPATCHED[@]+"${UNPATCHED[@]}"}")
}

if [ "$CAL_ONLY" = 1 ]; then
  note "S1 calibration only: s1_calibrate $CAL_ARGS on $ENGINE_IMAGE; ${WALL_S}s in all"
  # shellcheck disable=SC2086 # CAL_ARGS is a checked list of flags
  phase calibration "$WALL_S" "${PY[@]}" -m agentenv_zeroad.s1_calibrate --out "$OUT/calibration" \
    --image "$ENGINE_IMAGE" "${UNPATCHED[@]+"${UNPATCHED[@]}"}" --level-timeout-s "$LEVEL_TIMEOUT_S" $CAL_ARGS
  run "${PY[@]}" -c 'import json, sys; from agentenv_zeroad.s1_gates import format_calibration; print(format_calibration(json.load(open(sys.argv[1]))))' \
    "$OUT/calibration/calibration.json" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee "$OUT/calibration.txt" | tee -a "$OUT/progress.log"; fi; }
  [ "$DRY_RUN" = 1 ] || date -u +%FT%TZ > "$OUT/DONE"
  note "S1 calibration done"
  exit 0
fi
note "S1 batch: $((N * 6)) bot games, the replays, the calibration; $ENGINES engines in $WORKERS workers of $ENGINES_PER_WORKER; ${WALL_S}s in all"
run "${PY[@]}" -m agentenv_zeroad.s1_batch build --inputs "$INPUTS" --out "$OUT/specs" --n "$N" | { if [ "$DRY_RUN" = 1 ]; then cat; else tee -a "$OUT/progress.log"; fi; }
runner_cmd replay
phase replay "$REPLAY_LIMIT_S" "${RUNNER[@]}"
# shellcheck disable=SC2086 # CAL_ARGS is a checked list of flags
phase calibration "$CALIBRATION_LIMIT_S" "${PY[@]}" -m agentenv_zeroad.s1_calibrate --out "$OUT/calibration" \
  --image "$ENGINE_IMAGE" "${UNPATCHED[@]+"${UNPATCHED[@]}"}" --level-timeout-s "$LEVEL_TIMEOUT_S" $CAL_ARGS
runner_cmd bot
phase bot "$WALL_S" "${RUNNER[@]}"
note "summary"
run "${PY[@]}" scripts/s1_summary.py "$OUT/bot" "$OUT/replay" "$INPUTS/pilot" --calibration "$OUT/calibration/calibration.json" --out "$OUT" \
  || note "the summary found no finished games (see s1/out/bot and s1/out/replay)"
[ "$DRY_RUN" = 1 ] || date -u +%FT%TZ > "$OUT/DONE"
note "S1 batch done"
