#!/usr/bin/env bash
# train-run.sh CONFIG [--init DIR] [--anchor DIR|base] [--baseline FILE] [--play-only] | [--sft EPOCHS [--lr LR]] | [--probe ROWS]: the zeroad-train unit's process (train.sh starts it).
# Starts the learner in the trainer image on the training GPUs (torchrun, one process per GPU, its --root the run's
# folder, which the config names and which must be under $STATE/train: that folder is also what the vLLM containers
# mount at /train to hot-load adapters from), then runs the orchestrator with vLLM's key from its root-only file, in
# the foreground, so stopping the unit (SIGTERM) lets the orchestrator stop the learner and upload. When the
# orchestrator ends (or this script is stopped), the learner container is stopped. With --play-only there is no
# learner and nothing is credited. Two learner-only modes (the S2 sequence's, box/s2-run.sh): --sft EPOCHS starts the learner
# in SFT mode on the run's root (its inbox holds the distillation labels) and waits for sft-done.json, no orchestrator;
# --probe ROWS starts it with the box-day gates (`learner.py --probe`) and waits for probe.json. The container is stopped
# when either is done or this script is. When a panel has a frontier seat the orchestrator runs under with-model.sh, which
# reads the LiteLLM key from SSM into its environment only (never printed, written or put on a command line).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CONFIG=${1:?usage: train-run.sh CONFIG [--init DIR] [--anchor DIR] [--baseline FILE] [--play-only]}; shift
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; elsewhere (a dry run, the tests' stubs) it goes first
export PATH="${UV_BIN:+${UV_BIN%/*}:}/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
PLAY_ONLY=0 LEARNER=() SFT_EPOCHS= SFT_LR= PROBE=
while [ $# -gt 0 ]; do
  case $1 in
    --play-only) PLAY_ONLY=1; shift ;;
    --sft) SFT_EPOCHS=${2:?--sft needs the number of epochs}; [[ $SFT_EPOCHS =~ ^[1-9][0-9]*$ ]] || die "--sft needs a whole number of epochs"; shift 2 ;;
    --lr) SFT_LR=${2:?--lr needs a number}; [[ $SFT_LR =~ ^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$ ]] || die "--lr needs a number"; shift 2 ;;
    --probe) PROBE=${2:?--probe needs the sampled rows file}; [[ $PROBE =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad --probe path"; shift 2 ;;
    --init|--anchor|--baseline) LEARNER+=("$1" "$2"); shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -z "$SFT_EPOCHS$PROBE" ] || [ "$PLAY_ONLY" = 0 ] || die "--sft and --probe run the learner: not with --play-only"
[ -z "$SFT_EPOCHS" ] || [ -z "$PROBE" ] || die "--sft and --probe are separate runs"
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
    ${SFT_EPOCHS:+--sft --sft-epochs "$SFT_EPOCHS"} ${SFT_LR:+--lr "$SFT_LR"} ${PROBE:+--probe "$PROBE"} "${LEARNER[@]+"${LEARNER[@]}"}"
fi
if [ -n "$DONE" ]; then
  # Learner-only: no orchestrator. The learner exits itself when done (SFT: after its last epoch; probe: after its
  # numbers); wait for its result file, and fail if the container ends without one. A stop ends the wait.
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
