#!/usr/bin/env bash
# with-model.sh CMD [ARGS...]: runs CMD with the model endpoint in its environment: LITELLM_BASE_URL from the box
# config and LITELLM_API_KEY read from SSM ($ZEROAD_SSM_LITELLM_KEY, a SecureString) by the instance profile. The key
# goes only into CMD's environment (agent-env passes it to the agents, the strategist hands it to its reflex); it is
# never printed, written to a file or put on a command line.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
[ $# -gt 0 ] || { echo "usage: with-model.sh CMD [ARGS...]" >&2; exit 2; }
PARAM=${ZEROAD_SSM_LITELLM_KEY:-/zeroad-live/litellm-key}
[ -n "${LITELLM_BASE_URL:-}" ] || die "no LITELLM_BASE_URL in $CONF (LIVE_LITELLM_BASE_URL in live.env)"
if [ "$DRY_RUN" = 1 ]; then
  say "+ aws ssm get-parameter --name $PARAM --with-decryption (into LITELLM_API_KEY, never printed)"
  say "+ LITELLM_BASE_URL=$LITELLM_BASE_URL LITELLM_API_KEY=<from SSM> $*"
  exit 0
fi
key=$(aws ssm get-parameter --name "$PARAM" --with-decryption --query Parameter.Value --output text) \
  || die "cannot read $PARAM from SSM: store it with live/put-secret.sh litellm-key"
if [ -z "$key" ] || [ "$key" = None ]; then die "$PARAM is empty"; fi
export LITELLM_BASE_URL LITELLM_API_KEY=$key
unset key
exec "$@"
