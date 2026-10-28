#!/usr/bin/env bash
# match.sh start rehearsal|model [--task TASK] [--name NAME] --budget USD [--match-cap USD] [--push] | status
# Starts one rendered match on this box, in the transient unit zeroad-match (run-match.sh), and returns:
#   rehearsal  livestream-petra: the scripted bot against Petra, 5 game-minutes, no model ($0)
#   model      livestream: Setup A (Opus 5.5 vs GPT-6 Astra, Luna reflexes), 40 game-minutes, capped at $9; --task
#              plays another rendered model task of the bundle instead (e.g. livestream-4), its box time from its cap
# The match renders on the L4's Xorg (display :0) through the docker shim, with the director mod and the overlay,
# and its stream goes to the relay, which records it (always). It is pushed to the RTMP target only with --push.
# Without --push, a push still on from before is stopped first. Refused when a match (or its env container) is
# already running, when a model match's caps (the task's ZEROAD_MAX_COST_USD and
# reflex_max_usd) pass --match-cap, or when budget.sh says the budget would be passed.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
KIND=; NAME=; BUDGET=; MATCH_CAP=9; PUSH=0; MODEL_TASK=livestream
if [ "$CMD" = start ]; then KIND=${1:-}; shift || true; fi
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=${2:?}; shift 2 ;;
    --task) MODEL_TASK=${2:?}; shift 2 ;;
    --budget) BUDGET=${2:?}; shift 2 ;;
    --match-cap) MATCH_CAP=${2:?}; shift 2 ;;
    --push) PUSH=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$CMD" = status ]; then
  say "zeroad-match: $(active zeroad-match)"
  run journalctl -u zeroad-match -n 15 --no-pager -o cat
  exit 0
fi
[ "$CMD" = start ] || { echo "usage: match.sh start rehearsal|model [--task TASK] [--name NAME] --budget USD [--match-cap USD] [--push] | status" >&2; exit 2; }
case $KIND in
  rehearsal) TASK=livestream-petra HOURS=0.25 ;;
  model) TASK=$MODEL_TASK HOURS= ;;
  *) echo "match.sh start: rehearsal or model, not '$KIND'" >&2; exit 2 ;;
esac
NAME=${NAME:-$TASK-$(date -u +%Y%m%d-%H%M%S)}
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "--name: letters, digits, . _ - only"
[ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in live.env"
SRC=$REPO/src/agentenv_zeroad/bundles/zeroad
[[ $TASK =~ ^[A-Za-z0-9._-]+$ ]] || die "--task: letters, digits, . _ - only"
[ -f "$SRC/tasks/$TASK.json" ] || die "no task $TASK in $SRC (bootstrap first)"
if [ -z "$HOURS" ]; then  # the game-minutes cap plus 20 minutes of setup and teardown
  HOURS=$(python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); l = next(x for x in s if x["type"] == "open_lobby"); print(round((l["game_settings"].get("time_cap_minutes", 40) + 20) / 60, 2))' "$SRC/tasks/$TASK.json")
fi

CAP=$(python3 - "$SRC/tasks/$TASK.json" <<'PY'
import json, sys
steps = json.load(open(sys.argv[1]))
cap = sum(float(s.get("env_vars", {}).get("ZEROAD_MAX_COST_USD", 0)) for s in steps if s["type"] == "deploy_agent")
cap += sum(float(s.get("game_settings", {}).get("reflex_max_usd", 0)) for s in steps if s["type"] == "add_player_slot")
print(f"{cap:.2f}")
PY
)
say "match $NAME: task $TASK, model caps \$$CAP (match cap \$$MATCH_CAP), about $HOURS h of box time"
if [ "$KIND" = model ] && ! awk -v c="$CAP" -v m="$MATCH_CAP" 'BEGIN { exit !(c + 0 <= m + 0) }'; then
  die "the task's caps \$$CAP pass the match cap \$$MATCH_CAP (LIVE_MATCH_CAP_USD)"
fi
[ "$(active zeroad-match)" != active ] || die "a match is running: live/stop.sh first, or wait for it"
if [ "$DRY_RUN" = 0 ] && [[ $(docker ps --format '{{.Image}}' 2>/dev/null || true) == *mcp-server-zeroad-live* ]]; then
  die "a live env container is still up (a match that did not tear down): docker ps, then docker rm -f it"
fi
"$BOX_HOME/box/budget.sh" check --planned-usd "$CAP" --planned-hours "$HOURS" --budget "$BUDGET"

# The match's own bundle: the installed one with the lobby's rtmp on, so the env's stream goes to the relay.
if [ "$DRY_RUN" = 1 ]; then DIR=$(mktemp -d); else DIR=$STATE/matches/$NAME; fi
[ ! -e "$DIR/bundle" ] || die "$DIR exists: pick another --name"
mkdir -p "$DIR" && cp -R "$SRC" "$DIR/bundle"
python3 - "$DIR/bundle/tasks/$TASK.json" <<'PY'
import json, sys
path = sys.argv[1]
steps = json.load(open(path))
lobby = next(s for s in steps if s["type"] == "open_lobby")
lobby["game_settings"]["rtmp"] = True
json.dump(steps, open(path, "w"), indent=1)
print(f"bundle task {path.split('/')[-1]}: lobby {json.dumps(lobby['game_settings'])}")
PY
[ "$DRY_RUN" = 0 ] || rm -rf "$DIR"

for unit in zeroad-x zeroad-relay; do
  [ "$(active $unit)" = active ] || run systemctl start "$unit"
done
if [ "$PUSH" = 1 ]; then
  "$BOX_HOME/box/push.sh" start
else
  # A push left on (by an earlier match or by hand) would put this match on air: without --push it is ended first.
  if [ -f "$STATE/push.wanted" ] || [ "$(active zeroad-push)" = active ]; then
    say "push: a push was still on; this match is not to be pushed, so it is stopped first"
    "$BOX_HOME/box/push.sh" stop
  fi
  say "push: off (recorded only; live/push.sh ID start --yes pushes)"
fi
run systemd-run --unit zeroad-match --collect -p KillMode=mixed -p TimeoutStopSec=300 \
  -p "WorkingDirectory=$STATE/matches/$NAME" \
  "$BOX_HOME/box/run-match.sh" "$NAME" "$KIND" "$TASK" "$CAP"
if [ "$DRY_RUN" = 1 ]; then
  say "in zeroad-match, run-match.sh would:"
  "$BOX_HOME/box/run-match.sh" "$NAME" "$KIND" "$TASK" "$CAP" | sed 's/^/  /'
fi
say "started $NAME; follow it with live/status.sh, live/health.sh"
