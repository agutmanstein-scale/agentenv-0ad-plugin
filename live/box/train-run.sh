#!/usr/bin/env bash
# train-run.sh CONFIG [--init DIR [--start-version N]] [--anchor DIR|base] [--baseline FILE] [--drift trip|alarm] [--grad-norm trip|alarm] [--anchor-beta0 X] [--warmup-updates N] [--decisions-per-update N] [--full-prox] [--kl-check-fraction F] [--probe-rows N] [--whole-groups] [--play-only] | [--sft EPOCHS [--lr LR]] | [--probe ROWS] | [--stats-only ROWS] | [--bench DIR [--bench-updates U]]: the zeroad-train unit's process (train.sh starts it).
# Starts the learner in the trainer image on the training GPUs (torchrun, one process per GPU, its --root the run's
# folder, which the config names and which must be under $STATE/train: that folder is also what the vLLM containers
# mount at /train to hot-load adapters from), then runs the orchestrator with vLLM's key from its root-only file, in
# the foreground, so stopping the unit (SIGTERM) lets the orchestrator stop the learner and upload. When the
# orchestrator ends (or this script is stopped), the learner container is stopped. With --play-only there is no
# learner and nothing is credited. Two learner-only modes (the S2 sequence's, box/s2-run.sh): --sft EPOCHS starts the learner
# in SFT mode on the run's root (its inbox holds the distillation labels) and waits for sft-done.json, no orchestrator;
# --probe ROWS starts it with the box-day gates (`learner.py --probe`) and waits for probe.json; --stats-only ROWS starts it
# on the base model to measure v0's health records on those rows without training (`learner.py --stats-only`, the tripwire
# baseline) and waits for baseline.json; --bench DIR starts it on the decision rows in DIR to time U updates (default 5) with the
# learner flags given and waits for bench.json (`learner.py --bench`, no publish, no checkpoint). The learner flags are passed through to
# learner.py (see its docstring): --anchor-beta0, --warmup-updates, --decisions-per-update, --kl-check-fraction, --probe-rows, --full-prox,
# --whole-groups, --start-version (the --init adapter's version, already published in the root for the games to play). The container is stopped when any of the four is done or this script is. When a panel has a frontier seat the orchestrator runs under with-model.sh, which
# reads the LiteLLM key from SSM into its environment only (never printed, written or put on a command line).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CONFIG=${1:?usage: train-run.sh CONFIG [--init DIR] [--anchor DIR] [--baseline FILE] [--drift trip|alarm] [--grad-norm trip|alarm] [--play-only]}; shift
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; elsewhere (a dry run, the tests' stubs) it goes first
export PATH="${UV_BIN:+${UV_BIN%/*}:}/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
PLAY_ONLY=0 LEARNER=() SFT_EPOCHS= SFT_LR= PROBE= STATS= BENCH= BENCH_UPDATES=
while [ $# -gt 0 ]; do
  case $1 in
    --play-only) PLAY_ONLY=1; shift ;;
    --sft) SFT_EPOCHS=${2:?--sft needs the number of epochs}; [[ $SFT_EPOCHS =~ ^[1-9][0-9]*$ ]] || die "--sft needs a whole number of epochs"; shift 2 ;;
    --lr) SFT_LR=${2:?--lr needs a number}; [[ $SFT_LR =~ ^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$ ]] || die "--lr needs a number"; shift 2 ;;
    --probe) PROBE=${2:?--probe needs the sampled rows file}; [[ $PROBE =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad --probe path"; shift 2 ;;
    --stats-only) STATS=${2:?--stats-only needs the v0 rows file}; [[ $STATS =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad --stats-only path"; shift 2 ;;
    --bench) BENCH=${2:?--bench needs the directory of decision rows}; [[ $BENCH =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad --bench path"; shift 2 ;;
    --bench-updates) BENCH_UPDATES=${2:-}; [[ $BENCH_UPDATES =~ ^[1-9][0-9]*$ ]] || die "--bench-updates needs a whole number"; shift 2 ;;
    --anchor-beta0) [[ ${2:-} =~ ^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$ ]] || die "--anchor-beta0 needs a number"; LEARNER+=("$1" "$2"); shift 2 ;;
    --kl-check-fraction) [[ ${2:-} =~ ^(1(\.0+)?|0?\.[0-9]*[1-9][0-9]*)$ ]] || die "--kl-check-fraction needs a number in (0, 1]"; LEARNER+=("$1" "$2"); shift 2 ;;
    --start-version) [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "--start-version needs a whole number of at least 1"; LEARNER+=("$1" "$2"); shift 2 ;;
    --warmup-updates|--decisions-per-update) [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number"; LEARNER+=("$1" "$2"); shift 2 ;;
    --probe-rows) [[ ${2:-} =~ ^[0-9]+$ ]] || die "--probe-rows needs a whole number"; LEARNER+=("$1" "$2"); shift 2 ;;
    --full-prox|--whole-groups) LEARNER+=("$1"); shift ;;
    --drift) [[ ${2:-} =~ ^(trip|alarm)$ ]] || die "--drift needs trip or alarm"; LEARNER+=("$1" "$2"); shift 2 ;;
    --grad-norm) [[ ${2:-} =~ ^(trip|alarm)$ ]] || die "--grad-norm needs trip or alarm"; LEARNER+=("$1" "$2"); shift 2 ;;
    --init|--anchor|--baseline) LEARNER+=("$1" "$2"); shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -z "$SFT_EPOCHS$PROBE$STATS$BENCH" ] || [ "$PLAY_ONLY" = 0 ] || die "--sft, --probe, --stats-only and --bench run the learner: not with --play-only"
MODES=0; for m in "$SFT_EPOCHS" "$PROBE" "$STATS" "$BENCH"; do [ -z "$m" ] || MODES=$((MODES + 1)); done
[ "$MODES" -le 1 ] || die "--sft, --probe, --stats-only and --bench are separate runs"
[ -z "$BENCH_UPDATES" ] || [ -n "$BENCH" ] || die "--bench-updates goes with --bench"
[ -z "$STATS" ] || [ ${#LEARNER[@]} -eq 0 ] || die "--stats-only measures the base model: no --init, --anchor or --baseline"
PY=(uv run --project "$REPO" --frozen --no-dev --extra train python)
if [ "$DRY_RUN" = 1 ]; then
  say "+ source $ETC/vllm.env (VLLM_API_KEY, never printed)"
  eval "$("${PY[@]}" -m agentenv_zeroad.train.config "$CONFIG")"
  TRAIN_ROOT=$TRAIN/RUN_ID TOOLS=hold,set_economy
else
  set -a; source "$ETC/vllm.env"; set +a
  eval "$("${PY[@]}" -m agentenv_zeroad.train.config "$CONFIG")"
  TOOLS=$("${PY[@]}" -m agentenv_zeroad.train.orchestrate tools)
fi
export ZEROAD_MIN_FREE_GB=${ZEROAD_MIN_FREE_GB:-8} ZEROAD_MATCH_CAP_USD=${ZEROAD_MATCH_CAP_USD:-9}
case $TRAIN_ROOT in "$TRAIN"/*) ;; *) die "the run's root $TRAIN_ROOT must be under $TRAIN (vLLM mounts that for the adapters)" ;; esac
PILOT_TRAIN_GPUS=2,3
# shellcheck source=/dev/null
[ ! -f "$ETC/pilot.conf" ] || source "$ETC/pilot.conf"
TRAIN_GPUS=$PILOT_TRAIN_GPUS
run mkdir -p "$TRAIN_ROOT"
DONE=  # a learner-only run's result file, which a stale one must not stand in for
[ -z "$PROBE" ] || DONE=$TRAIN_ROOT/probe.json
[ -z "$SFT_EPOCHS" ] || DONE=$TRAIN_ROOT/sft-done.json
[ -z "$STATS" ] || DONE=$TRAIN_ROOT/baseline.json
[ -z "$BENCH" ] || DONE=$TRAIN_ROOT/bench.json
[ -z "$DONE" ] || run rm -f "$DONE"
# A stop (SIGTERM reaches this script as well as the orchestrator) must still stop the learner container: the trap
# keeps bash alive until the orchestrator, which stops the learner and uploads, has ended, then runs on exit.
stop_learner() { docker stop -t 60 zeroad-learner >/dev/null 2>&1 || true; }
if [ "$DRY_RUN" = 0 ] && [ "$PLAY_ONLY" = 0 ]; then trap ':' TERM; trap stop_learner EXIT; fi
if [ "$PLAY_ONLY" = 0 ]; then
  # The learner is the repo's trainer/learner.py baked into the image: rebuild (a cached no-op when trainer/ is unchanged)
  # so a box bootstrapped before an update does not train with the old one.
  run docker build -q -t zeroad-trainer "$REPO/trainer"
  run docker rm -f zeroad-learner >/dev/null 2>&1 || true
  run docker run -d --name zeroad-learner --gpus "\"device=$TRAIN_GPUS\"" --ipc=host -v "$TRAIN:$TRAIN" \
    -v "$STATE/hf:/root/.cache/huggingface" zeroad-trainer torchrun --nproc_per_node "$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")" \
    /app/learner.py --base "${ZEROAD_PILOT_MODEL:-Qwen/Qwen3-8B}" --root "$TRAIN_ROOT" --known-tools "$TOOLS" \
    ${SFT_EPOCHS:+--sft --sft-epochs "$SFT_EPOCHS"} ${SFT_LR:+--lr "$SFT_LR"} ${PROBE:+--probe "$PROBE"} ${STATS:+--stats-only "$STATS"} ${BENCH:+--bench "$BENCH" --bench-updates "${BENCH_UPDATES:-5}"} "${LEARNER[@]+"${LEARNER[@]}"}"
fi
if [ -n "$DONE" ]; then
  # Learner-only: no orchestrator. The learner exits itself when done (SFT: after its last epoch; probe: after its
  # numbers; stats-only: after its records; bench: after its updates); wait for its result file, and fail if the container ends without one. A stop ends the wait.
  [ "$DRY_RUN" = 1 ] && { say "+ wait for $DONE (the learner container ending without it is a failure)"; exit 0; }
  trap 'exit 143' TERM
  while [ ! -f "$DONE" ]; do
    [ -n "$(docker ps -q --filter name=zeroad-learner)" ] || { sleep 5; [ -f "$DONE" ] || die "the learner ended without writing $DONE (docker logs zeroad-learner)"; }
    sleep 10
  done
  exit 0
fi
orchestrate=("${PY[@]}" -m agentenv_zeroad.train.orchestrate run --config "$CONFIG")
[ "$PLAY_ONLY" = 0 ] || orchestrate+=(--play-only)
[ "${TRAIN_FRONTIER:-0}" != 1 ] || orchestrate=("$BOX_HOME/box/with-model.sh" "${orchestrate[@]}")
if [ "$DRY_RUN" = 1 ]; then
  if [ "${TRAIN_FRONTIER:-0}" = 1 ]; then "${orchestrate[@]}"; else run "${orchestrate[@]}"; fi
  exit 0
fi
rc=0
"${orchestrate[@]}" || rc=$?
exit "$rc"
