#!/usr/bin/env bash
# actor-run.sh CONFIG: the zeroad-actor unit's process (actor.sh starts it).
# Runs the orchestrator as a rollout actor (its config has [transport] role = "actor") in the foreground, with vLLM's key
# from its root-only file, so stopping the unit (SIGTERM) lets it ship what is finished, send its last heartbeat and
# end. There is no learner container: the actor's GPUs all serve, and the learner is another box.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CONFIG=${1:?usage: actor-run.sh CONFIG}
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
UV_BIN=$(command -v uv || true)  # on the box it is /usr/local/bin; elsewhere (a dry run, the tests' stubs) it goes first
export PATH="${UV_BIN:+${UV_BIN%/*}:}/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=$ETC/agent-env.toml
PY=(uv run --project "$REPO" --frozen --no-dev --extra train python)
if [ "$DRY_RUN" = 1 ]; then
  say "+ source $ETC/vllm.env (VLLM_API_KEY, never printed)"
else
  set -a; source "$ETC/vllm.env"; set +a
fi
eval "$("${PY[@]}" -m agentenv_zeroad.train.config "$CONFIG")"
[ "$TRAIN_TRANSPORT_ROLE" = actor ] || die "$CONFIG is not an actor's config ([transport] enabled, role = \"actor\")"
export ZEROAD_MIN_FREE_GB=${ZEROAD_MIN_FREE_GB:-8} ZEROAD_MATCH_CAP_USD=${ZEROAD_MATCH_CAP_USD:-9}
case $TRAIN_ROOT in "$TRAIN"/*) ;; *) [ "$DRY_RUN" = 1 ] || die "the actor's root $TRAIN_ROOT must be under $TRAIN (vLLM mounts that for the adapters)" ;; esac
run mkdir -p "$TRAIN_ROOT"
run "${PY[@]}" -m agentenv_zeroad.train.orchestrate run --config "$CONFIG"
