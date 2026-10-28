#!/usr/bin/env bash
# run-match.sh NAME KIND TASK CAP: the zeroad-match unit's process (match.sh starts it). Runs `agent-env run` on the
# match's bundle with the docker shim first on PATH (the env container gets the GPU, the display and the relay), and
# for a model match with the LiteLLM key from SSM (with-model.sh). Afterwards it copies the match's results.json
# next to the bundle and adds the match's spend to the box ledger: the results' estimate, or the full caps when
# there are no results (the guard counts what could have been spent). A stop (systemctl stop zeroad-match sends
# SIGTERM to this script alone, KillMode=mixed) is forwarded to agent-env, which tears its containers down; this script
# waits for it and still books the spend. The match's push, if any, ends with the match.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
NAME=${1:?usage: run-match.sh NAME KIND TASK CAP}; KIND=${2:?}; TASK=${3:?}; CAP=${4:?}
DIR=$STATE/matches/$NAME
export PATH="$BOX_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_STATE_HOME=$STATE/agentenv AGENT_ENV_CONFIG=/etc/zeroad-live/agent-env.toml
AE=(uv run --project "$REPO" --frozen --no-dev)
run mkdir -p "$DIR"
run touch "$DIR/.started"
cmd=("${AE[@]}" agent-env run "$DIR/bundle" --task "$TASK")
[ "$KIND" != model ] || cmd=("$BOX_HOME/box/with-model.sh" "${cmd[@]}")
rc=0
if [ "$DRY_RUN" = 1 ]; then
  if [ "$KIND" = model ]; then "${cmd[@]}"; else run "${cmd[@]}"; fi  # with-model.sh prints its own plan
else
  # agent-env runs in the background so that a SIGTERM to this script reaches it (uv run forwards it) instead of
  # killing this script mid-pipeline, before the ledger line.
  "${cmd[@]}" > >(redact | tee "$DIR/run.log") 2>&1 &
  pid=$!
  trap 'say "match $NAME: stop requested, forwarding SIGTERM to agent-env"; kill -TERM "$pid" 2>/dev/null || true' TERM INT
  while :; do
    if wait "$pid"; then rc=0; else rc=$?; fi  # a trapped signal interrupts wait while agent-env still runs
    kill -0 "$pid" 2>/dev/null || break
  done
  trap - TERM INT
fi
results=
if [ "$DRY_RUN" = 0 ]; then
  results=$(find "$XDG_STATE_HOME" -newer "$DIR/.started" -name '*results*.json' -size +0 -print -quit 2>/dev/null || true)
fi
if [ -n "$results" ]; then
  cp "$results" "$DIR/results.json"
  usd=$("${AE[@]}" python -m agentenv_zeroad.spend results "$DIR/results.json")
else
  usd=$CAP
  say "no results.json found: counting the full caps (\$$CAP)"
fi
[ "$KIND" = rehearsal ] && [ -z "$results" ] && usd=0
run "${AE[@]}" python -m agentenv_zeroad.spend add --ledger "$LEDGER" --what "$TASK $NAME" --usd "$usd"
say "match $NAME ended (exit $rc), spend \$$usd"
if [ -f "$STATE/push.wanted" ] || [ "$(active zeroad-push)" = active ]; then "$BOX_HOME/box/push.sh" stop; fi
exit "$rc"
