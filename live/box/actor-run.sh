#!/usr/bin/env bash
# actor-run.sh CONFIG: the zeroad-actor unit's process (actor.sh starts it).
# Runs the orchestrator as a rollout actor (its config has [transport] role = "actor") in the foreground, with vLLM's key
# from its root-only file, so stopping the unit (SIGTERM) lets it ship what is finished, send its last heartbeat and
# end. There is no learner container: the actor's GPUs all serve, and the learner is another box. When the orchestrator
# has ended it runs `transport flush`: games the orchestrator could not upload (S3 or the network was down at the end)
# are retried with backoff for up to 30 minutes, and the actor's final heartbeat says what, if anything, is left. Only
# a clean flush lets a config with `shutdown = true` power the box off; an actor that still holds games stays up.
# ZEROAD_INSTANCE_ID (from IMDSv2) goes in its heartbeats, so the learner can tell two boxes using one actor_id.
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
if [ "$DRY_RUN" = 1 ]; then
  say "+ ZEROAD_INSTANCE_ID from IMDSv2 (curl http://169.254.169.254/latest/meta-data/instance-id)"
else
  token=$(curl -fsS -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)
  ZEROAD_INSTANCE_ID=$(curl -fsS -m 3 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
  export ZEROAD_INSTANCE_ID
  trap ':' TERM  # a stop reaches this script too; the orchestrator and the flush after it must still run to their ends
fi
rc=0
run "${PY[@]}" -m agentenv_zeroad.train.orchestrate run --config "$CONFIG" || rc=$?
flush=0
run "${PY[@]}" -m agentenv_zeroad.train.transport flush --config "$CONFIG" --timeout-minutes 30 || flush=$?
if [ "$flush" != 0 ]; then
  echo "NOT powering off: games are still unshipped on this box (see the flush output above; live/actor.sh ID status)" >&2
elif [ "$TRAIN_SHUTDOWN" = 1 ]; then
  run shutdown -h now
fi
exit "$rc"
