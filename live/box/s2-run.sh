#!/usr/bin/env bash
# s2-run.sh --budget USD [--no-shutdown]: the zeroad-s2 unit's process (box/s2.sh starts it).
# Runs the whole box-day sequence in one Python process (agentenv_zeroad.train.s2 run, docs/s2-box-day.md): E1 GPU gates,
# the E2 positive control, the baselines, the distillation probe, then Run 1, each step through train-run.sh (the learner
# container and the orchestrator, or one of them) under its own wall-clock cap, with the gate verdicts in
# $STATE/train/s2/gates.json. It reads vLLM's key from its root-only file into its environment (never printed) and gives
# its children the box config's LiteLLM URL; the LiteLLM key itself is read from SSM by with-model.sh, only for the
# children that need it (frontier panels, the labels). At its end, win or lose, it writes the report, uploads the
# sequence's files, and, unless --no-shutdown or the upload failed, runs `shutdown -h now`, which stops the box (it is
# never terminated from here).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
BUDGET= EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --budget) BUDGET=${2:?--budget needs USD}; [[ $BUDGET =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget needs a number"; shift 2 ;;
    --no-shutdown) EXTRA+=(--no-shutdown); shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$BUDGET" ] || die "usage: s2-run.sh --budget USD [--no-shutdown]"
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; a dry run elsewhere may have it elsewhere
export PATH="${UV_BIN:+${UV_BIN%/*}:}/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
export ZEROAD_MIN_FREE_GB=${ZEROAD_MIN_FREE_GB:-8} ZEROAD_MATCH_CAP_USD=${ZEROAD_MATCH_CAP_USD:-9}
export LITELLM_BASE_URL=${LITELLM_BASE_URL:-}
sequence=(uv run --project "$REPO" --frozen --no-dev --extra train python -m agentenv_zeroad.train.s2 run
  --config-dir "$TRAIN/s2-config" --budget "$BUDGET" --box-home "$BOX_HOME" "${EXTRA[@]+"${EXTRA[@]}"}")
if [ "$DRY_RUN" = 1 ]; then
  say "+ source $ETC/vllm.env (VLLM_API_KEY, never printed)"
  run "${sequence[@]}"
  exit 0
fi
set -a; source "$ETC/vllm.env"; set +a
exec "${sequence[@]}"
