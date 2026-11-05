#!/usr/bin/env bash
# pilot-run.sh FRONTIER [RUN ARGS...]: the zeroad-pilot unit's process (pilot.sh starts it). Runs the pilot's round
# loop (python -m agentenv_zeroad.pilot.run) with agent-env's state and config, vLLM's key from its root-only file, and
# the budget guard and ledger. With FRONTIER 1 it runs under with-model.sh, so agent-env's own endpoint is LiteLLM
# with the key from SSM (the Sonnet probe); otherwise agent-env's endpoint is the box's vLLM. The policy and every
# snapshot always play on vLLM: their tasks name it.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
FRONTIER=${1:?usage: pilot-run.sh FRONTIER [RUN ARGS...]}; shift
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
export PATH="/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
if [ "$DRY_RUN" = 1 ]; then
  say "+ source $ETC/vllm.env (VLLM_API_KEY, never printed)"
else
  set -a; source "$ETC/vllm.env"; set +a
fi
VLLM=$(cat "$ETC/vllm.url" 2>/dev/null || echo http://172.17.0.1:8000)
PILOT_TRAIN_GPUS=2,3 PILOT_SHARE_GPU=0
# shellcheck source=/dev/null
[ ! -f "$ETC/pilot.conf" ] || source "$ETC/pilot.conf"
gpu=(--train-gpus "$PILOT_TRAIN_GPUS")
[ "$PILOT_SHARE_GPU" != 1 ] || gpu+=(--share-gpu)
cmd=(uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.pilot.run --home "$STATE/pilot"
     --vllm-url "$VLLM" --vllm-host "$VLLM" --budget-script "$BOX_HOME/box/budget.sh" --ledger "$LEDGER"
     --hf-cache "$STATE/hf" "${gpu[@]}" "$@")
if [ "$FRONTIER" = 1 ]; then
  cmd=("$BOX_HOME/box/with-model.sh" "${cmd[@]}")
else
  export LITELLM_BASE_URL=$VLLM LITELLM_API_KEY=${VLLM_API_KEY:-}
fi
if [ "$DRY_RUN" = 1 ]; then
  if [ "$FRONTIER" = 1 ]; then "${cmd[@]}"; else run "${cmd[@]}"; fi
  exit 0
fi
exec "${cmd[@]}"
