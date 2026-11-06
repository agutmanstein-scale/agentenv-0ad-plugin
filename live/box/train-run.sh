#!/usr/bin/env bash
# train-run.sh CONFIG [--init DIR] [--baseline FILE] [--play-only]: the zeroad-train unit's process (train.sh starts it).
# Starts the learner in the trainer image on the training GPUs (torchrun, one process per GPU, its --root the run's
# folder, which the config names and which must be under $STATE/train: that folder is also what the vLLM containers
# mount at /train to hot-load adapters from), then runs the orchestrator with vLLM's key from its root-only file, in
# the foreground, so stopping the unit (SIGTERM) lets the orchestrator stop the learner and upload. When the
# orchestrator ends (or this script is stopped), the learner container is stopped. With --play-only there is no
# learner and nothing is credited. When a panel has a frontier seat the orchestrator runs under with-model.sh, which
# reads the LiteLLM key from SSM into its environment only (never printed, written or put on a command line).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CONFIG=${1:?usage: train-run.sh CONFIG [--init DIR] [--baseline FILE] [--play-only]}; shift
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; elsewhere (a dry run, the tests' stubs) it goes first
export PATH="${UV_BIN:+${UV_BIN%/*}:}/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
PLAY_ONLY=0 LEARNER=()
while [ $# -gt 0 ]; do
  case $1 in
    --play-only) PLAY_ONLY=1; shift ;;
    --init|--baseline) LEARNER+=("$1" "$2"); shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
PY=(uv run --project "$REPO" --frozen --no-dev python)
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
# A stop (SIGTERM reaches this script as well as the orchestrator) must still stop the learner container: the trap
# keeps bash alive until the orchestrator, which stops the learner and uploads, has ended, then runs on exit.
stop_learner() { docker stop -t 60 zeroad-learner >/dev/null 2>&1 || true; }
if [ "$DRY_RUN" = 0 ] && [ "$PLAY_ONLY" = 0 ]; then trap ':' TERM; trap stop_learner EXIT; fi
if [ "$PLAY_ONLY" = 0 ]; then
  run docker rm -f zeroad-learner >/dev/null 2>&1 || true
  run docker run -d --name zeroad-learner --gpus "\"device=$TRAIN_GPUS\"" --ipc=host -v "$TRAIN:$TRAIN" \
    -v "$STATE/hf:/root/.cache/huggingface" zeroad-trainer torchrun --nproc_per_node "$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")" \
    /app/learner.py --base "${ZEROAD_PILOT_MODEL:-Qwen/Qwen3-8B}" --root "$TRAIN_ROOT" --known-tools "$TOOLS" "${LEARNER[@]+"${LEARNER[@]}"}"
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
