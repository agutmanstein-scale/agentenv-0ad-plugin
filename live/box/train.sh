#!/usr/bin/env bash
# train.sh put-config B64 | start [--config PATH] [--init DIR] [--baseline FILE] [--play-only] --budget USD | status [LINES] | stop
# A post-training run on this box (bootstrapped with bootstrap.sh --train), in the transient unit zeroad-train
# (train-run.sh: the learner container on the training GPUs, then the orchestrator, src/agentenv_zeroad/train/orchestrate.py).
# `put-config` writes the run config (base64, as the laptop script sends it) to $STATE/train/config.toml and checks it.
# `start` refuses while a run is going, while a vLLM instance is down, when the run config's box_usd_per_hour is not this
# box's rate (ZEROAD_BOX_USD_PER_HOUR), or when budget.sh says --budget would be passed by the box hours and ledger so
# far plus the config's whole duration_hours at the rate; --init is a PEFT adapter to start the learner from, --baseline the tripwire baseline (JSON from baseline_from), both
# under $STATE/train; --play-only starts no learner and credits nothing (a baseline or panel run). `status` prints the
# unit, the orchestrator's status, the learner's latest health records, the containers, the GPUs, vLLM and the spend.
# `stop` sends SIGTERM: the orchestrator stops the learner, uploads the run (each of its 3 tries is cut off after 10 min;
# the unit's TimeoutStopSec is 40 min) and ends.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
CONFIG=$TRAIN/config.toml
case $CMD in
  put-config)
    B64=${1:?usage: train.sh put-config B64}
    [[ $B64 =~ ^[A-Za-z0-9+/=]+$ ]] || die "put-config needs base64"
    run mkdir -p "$TRAIN"
    if [ "$DRY_RUN" = 1 ]; then say "+ write the decoded config to $CONFIG and check it (train.config)"; exit 0; fi
    printf '%s' "$B64" | base64 -d > "$CONFIG.new"
    uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.train.config "$CONFIG.new" || { rm -f "$CONFIG.new"; die "the config is not valid"; }
    mv "$CONFIG.new" "$CONFIG"
    say "config in place: $CONFIG"
    ;;
  start)
    ARGS=(); BUDGET=
    while [ $# -gt 0 ]; do
      case $1 in
        --config) CONFIG=${2:?--config needs a path}; shift 2 ;;
        --init|--baseline)
          [[ ${2:-} == "$TRAIN"/* && ${2:-} =~ ^[A-Za-z0-9._/-]+$ ]] || die "$1 needs a path under $TRAIN"
          ARGS+=("$1" "$2"); shift 2 ;;
        --play-only) ARGS+=(--play-only); shift ;;
        --budget) BUDGET=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the training live env"
    [ "$DRY_RUN" = 1 ] || [ -f "$CONFIG" ] || die "no run config at $CONFIG (train.sh put-config)"
    [ "$(active zeroad-train)" != active ] || die "a run is going: train.sh status, or stop it first"
    if [ "$DRY_RUN" = 0 ]; then
      for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do
        curl -fsS "$url/health" >/dev/null || die "vLLM is not up at $url (docker logs zeroad-vllm-0 / zeroad-vllm-1)"
      done
    fi
    # The run's own numbers: its duration and its rate (the config is checked here too, panel files and a sample game).
    if [ -f "$CONFIG" ]; then
      cfg_vars=$(uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.train.config "$CONFIG") || die "the run config $CONFIG is not a runnable run"
      eval "$cfg_vars"
    else
      TRAIN_DURATION_HOURS=0.5 TRAIN_BOX_USD_PER_HOUR=${ZEROAD_BOX_USD_PER_HOUR:-0}
      say "(dry run: no config at $CONFIG yet; planning half an hour)"
    fi
    [ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set in $CONF: the box has no rate"
    awk -v a="$TRAIN_BOX_USD_PER_HOUR" -v b="$ZEROAD_BOX_USD_PER_HOUR" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.005) }' \
      || die "the run config's box_usd_per_hour ($TRAIN_BOX_USD_PER_HOUR) is not this box's rate ($ZEROAD_BOX_USD_PER_HOUR in $CONF): a run on another instance type must be priced at its own"
    "$BOX_HOME/box/budget.sh" check --planned-usd 0 --planned-hours "$TRAIN_DURATION_HOURS" --budget "$BUDGET"
    say "training run from $CONFIG ${ARGS[*]-}"
    # TimeoutStopSec covers the final upload's 3 tries of 10 minutes (sync.py) plus the learner's grace.
    run systemd-run --unit zeroad-train --collect -p KillMode=control-group -p TimeoutStopSec=2400 \
      -p "After=network-online.target docker.service" -p "Wants=network-online.target docker.service" \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/train-run.sh" "$CONFIG" "${ARGS[@]+"${ARGS[@]}"}"
    say "started; follow it with live/train.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: train.sh status [LINES]" >&2; exit 2; }
    say "zeroad-train: $(active zeroad-train)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ orchestrator/status.json; health.jsonl; journalctl -u zeroad-train -n $N; docker ps; nvidia-smi; vLLM health; budget.sh show; df"
      exit 0
    fi
    eval "$(uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.train.config "$CONFIG" 2>/dev/null)" || TRAIN_ROOT=
    echo "--- orchestrator"; cat "$TRAIN_ROOT/orchestrator/status.json" 2>/dev/null || echo "(no status yet)"
    echo "--- learner (latest health records)"; tail -n 3 "$TRAIN_ROOT/health.jsonl" 2>/dev/null | cut -c1-400 || true
    echo "--- log"; journalctl -u zeroad-train -n "$N" --no-pager -o cat 2>/dev/null || true
    echo "--- containers"; docker ps --format '{{.Names}} {{.Status}}' | grep -E 'zeroad-(vllm|learner)' || true
    echo "--- gpus"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true
    echo "--- vllm"
    for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do echo "$url $(curl -s -o /dev/null -w '%{http_code}' "$url/health" || echo down)"; done
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- disk"; df -h "$STATE" | tail -1
    echo "--- power-off"; busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null || echo none
    ;;
  stop)
    say "stopping zeroad-train: the learner is told to stop, the run is uploaded, then it ends"
    run systemctl stop zeroad-train
    ;;
  *) echo "usage: train.sh put-config B64 | start --budget USD ... | status [LINES] | stop" >&2; exit 2 ;;
esac
