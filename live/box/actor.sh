#!/usr/bin/env bash
# actor.sh put-config B64 | start [--config PATH] --budget USD | status [LINES] | stop
# A rollout actor of a multi-box run on this box (bootstrapped with bootstrap.sh --actor), in the transient unit
# zeroad-actor (actor-run.sh: the orchestrator, src/agentenv_zeroad/train/orchestrate.py, with [transport] role = "actor":
# it plays games on every GPU's vLLM, pulls the learner's adapters and ships each finished game through S3).
# `put-config` writes the actor config (base64, as the laptop script sends it) to $STATE/train/actor.toml and checks it,
# including that it is an actor's. `start` refuses while an actor is going, while a vLLM instance is down, when the
# config's box_usd_per_hour is not this box's rate (ZEROAD_BOX_USD_PER_HOUR), or when budget.sh says --budget would be
# passed by the box hours and ledger so far plus the config's whole duration_hours at the rate. `status` prints the unit,
# the orchestrator's status, the containers, the GPUs, vLLM and the spend. `stop` sends SIGTERM: the orchestrator
# cancels the games in flight, ships what is finished (then `transport flush` retries for up to 30 minutes what could
# not go), sends its last heartbeat and ends (a learner's stop.json at the end of a run, or `transport stop --drain`, lets
# the actor play its games out first).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
CONFIG=$TRAIN/actor.toml
PYRUN=(uv run --project "$REPO" --frozen --no-dev --extra train python)
case $CMD in
  put-config)
    B64=${1:?usage: actor.sh put-config B64}
    [[ $B64 =~ ^[A-Za-z0-9+/=]+$ ]] || die "put-config needs base64"
    run mkdir -p "$TRAIN"
    if [ "$DRY_RUN" = 1 ]; then say "+ write the decoded config to $CONFIG and check it (train.config; it must be [transport] role = \"actor\")"; exit 0; fi
    printf '%s' "$B64" | base64 -d > "$CONFIG.new"
    vars=$("${PYRUN[@]}" -m agentenv_zeroad.train.config "$CONFIG.new") || { rm -f "$CONFIG.new"; die "the config is not valid"; }
    eval "$vars"
    [ "$TRAIN_TRANSPORT_ROLE" = actor ] || { rm -f "$CONFIG.new"; die "the config is not an actor's ([transport] enabled, role = \"actor\")"; }
    mv "$CONFIG.new" "$CONFIG"
    say "config in place: $CONFIG"
    ;;
  start)
    BUDGET=
    while [ $# -gt 0 ]; do
      case $1 in
        --config) CONFIG=${2:?--config needs a path}; shift 2 ;;
        --budget) BUDGET=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the training live env"
    [ "$DRY_RUN" = 1 ] || [ -f "$CONFIG" ] || die "no actor config at $CONFIG (actor.sh put-config)"
    [ "$(active zeroad-actor)" != active ] || die "an actor is going: actor.sh status, or stop it first"
    [ "$(active zeroad-train)" != active ] || die "this box is running a learner (zeroad-train): an actor box does not train"
    if [ "$DRY_RUN" = 0 ]; then
      for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do
        curl -fsS "$url/health" >/dev/null || die "vLLM is not up at $url (docker logs zeroad-vllm-N)"
      done
    fi
    if [ -f "$CONFIG" ]; then
      cfg_vars=$("${PYRUN[@]}" -m agentenv_zeroad.train.config "$CONFIG") || die "the actor config $CONFIG is not a runnable run"
      eval "$cfg_vars"
      [ "$TRAIN_TRANSPORT_ROLE" = actor ] || die "$CONFIG is not an actor's config ([transport] enabled, role = \"actor\")"
      if [ "$DRY_RUN" = 0 ]; then
        have=$(sort "$ETC/vllm.urls" 2>/dev/null | tr '\n' ' ') want=$(tr ' ' '\n' <<<"$TRAIN_SERVER_URLS" | sort | tr '\n' ' ')
        [ "$have" = "$want" ] || die "the config's [servers] urls ($want) are not this box's vLLM instances ($have in $ETC/vllm.urls): one url per GPU, as bootstrap.sh --actor made them"
      fi
    else
      TRAIN_DURATION_HOURS=0.5 TRAIN_BOX_USD_PER_HOUR=${ZEROAD_BOX_USD_PER_HOUR:-0}
      say "(dry run: no config at $CONFIG yet; planning half an hour)"
    fi
    # shellcheck disable=SC2086 # TRAIN_ENGINE_PATCHES is a space-separated list of patch names
    [ -z "${TRAIN_ENGINE_IMAGE:-}" ] || require_engine_image "$TRAIN_ENGINE_IMAGE" ${TRAIN_ENGINE_PATCHES:-}
    [ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set in $CONF: the box has no rate"
    awk -v a="$TRAIN_BOX_USD_PER_HOUR" -v b="$ZEROAD_BOX_USD_PER_HOUR" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.005) }' \
      || die "the actor config's box_usd_per_hour ($TRAIN_BOX_USD_PER_HOUR) is not this box's rate ($ZEROAD_BOX_USD_PER_HOUR in $CONF): price it at its own"
    "$BOX_HOME/box/budget.sh" check --planned-usd 0 --planned-hours "$TRAIN_DURATION_HOURS" --budget "$BUDGET"
    say "actor ${TRAIN_ACTOR_ID:-?} from $CONFIG: ${TRAIN_ENGINES:-?} engines (${TRAIN_CONCURRENCY:-?} games at once) on ${TRAIN_ENGINE_IMAGE:-?}"
    # TimeoutStopSec: the orchestrator's final upload and then `transport flush`'s 30 minutes of retries (actor-run.sh).
    run systemd-run --unit zeroad-actor --collect -p KillMode=control-group -p TimeoutStopSec=4200 \
      -p "After=network-online.target docker.service" -p "Wants=network-online.target docker.service" \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/actor-run.sh" "$CONFIG"
    say "started; follow it with live/actor.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: actor.sh status [LINES]" >&2; exit 2; }
    say "zeroad-actor: $(active zeroad-actor)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ orchestrator/status.json; journalctl -u zeroad-actor -n $N; docker ps; nvidia-smi; vLLM health; budget.sh show; df"
      exit 0
    fi
    eval "$("${PYRUN[@]}" -m agentenv_zeroad.train.config "$CONFIG" 2>/dev/null)" || TRAIN_ROOT=
    echo "--- orchestrator"; cat "$TRAIN_ROOT/orchestrator/status.json" 2>/dev/null || echo "(no status yet)"
    echo "--- log"; journalctl -u zeroad-actor -n "$N" --no-pager -o cat 2>/dev/null || true
    echo "--- containers"; docker ps --format '{{.Names}} {{.Status}}' | grep -E 'zeroad-vllm' || true
    echo "--- gpus"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true
    echo "--- vllm"
    for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do echo "$url $(curl -s -o /dev/null -w '%{http_code}' "$url/health" || echo down)"; done
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- disk"; df -h "$STATE" | tail -1
    echo "--- power-off"; "$BOX_HOME/box/poweroff.sh" status
    ;;
  stop)
    say "stopping zeroad-actor: games in flight are cancelled, finished ones are shipped, then it ends"
    run systemctl stop zeroad-actor
    ;;
  *) echo "usage: actor.sh put-config B64 | start --budget USD ... | status [LINES] | stop" >&2; exit 2 ;;
esac
