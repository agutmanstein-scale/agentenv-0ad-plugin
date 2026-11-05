#!/usr/bin/env bash
# pilot.sh start [--baseline] [--rounds N] [--train-games N] [--train-samples N] [--speed S] [--cap-minutes M]
#        [--seed N] [--frontier id=model,...] [--frontier-per-round N] [--frontier-offset N] [--no-frontier]
#        [--objective pg|best] [--train-lr X] [--best-threshold X] [--fresh]
#        [--pool s3://... --pool-self NAME --pool-members a,b,c] [--pool-wait-s S] --budget USD
#        | status [LINES] | stop [--now] | recollect ROUND | cleanup
# The post-training pilot on this box (src/agentenv_zeroad/pilot/run.py) in the transient unit zeroad-pilot
# (pilot-run.sh). `start` refuses while it runs or vLLM is down; --baseline plays one round and trains nothing;
# without --no-frontier the frontier probes (Sonnet, Opus, Astra, Sol, or --frontier's) play on the LiteLLM key from
# SSM (with-model.sh). The driver asks budget.sh
# before every round whether --budget still holds. `status` prints the unit, the ratings and the latest round,
# the log, the containers, the GPUs, vLLM's models and the spend. --fresh first moves the pilot's folder aside
# (pilot-solo-<time>) and restarts vLLM on the empty one, so the run starts again from the base model; with --pool
# the round loop pools every member's update each round (run.py). `stop` lets the round in play finish; --now ends
# it at once (every process of the unit gets SIGTERM, so agent-env tears its games down). `recollect` reads a played
# round's games again and replaces its records, decisions, progress page and report (the loop's state stays).
# `cleanup` (only while the pilot is stopped) removes the game and agent containers and networks a stopped round
# left (a stop --now ends agent-env before it tears its games down); vLLM and the image registry stay.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
HOME_DIR=$STATE/pilot
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
VLLM=$(cat "$ETC/vllm.url" 2>/dev/null || echo http://172.17.0.1:8000)
case $CMD in
  start)
    ARGS=(); BUDGET=; FRONTIER=1; FRESH=0
    while [ $# -gt 0 ]; do
      case $1 in
        --baseline) ARGS+=(--baseline); shift ;;
        --rounds|--train-games|--train-samples|--train-batch|--speed|--cap-minutes|--seed|--frontier-per-round|--frontier-offset|--best-threshold|--pool-wait-s)
          [[ ${2:-} =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "$1 needs a number"
          ARGS+=("$1" "$2"); shift 2 ;;
        --no-frontier) FRONTIER=0; ARGS+=(--frontier ""); shift ;;
        --objective) [[ ${2:-} =~ ^(pg|best)$ ]] || die "--objective is pg or best"; ARGS+=("$1" "$2"); shift 2 ;;
        --fresh) FRESH=1; shift ;;
        --pool) [[ ${2:-} =~ ^s3://[a-z0-9.-]+/[A-Za-z0-9._/-]+$ ]] || die "--pool needs s3://bucket/prefix"; ARGS+=("$1" "$2"); shift 2 ;;
        --pool-self) [[ ${2:-} =~ ^[a-z0-9]+$ ]] || die "--pool-self needs a name"; ARGS+=("$1" "$2"); shift 2 ;;
        --pool-members) [[ ${2:-} =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || die "--pool-members needs a,b,..."; ARGS+=("$1" "$2"); shift 2 ;;
        --train-lr) [[ ${2:-} =~ ^[0-9.]+(e-?[0-9]+)?$ ]] || die "--train-lr needs a number"; ARGS+=("$1" "$2"); shift 2 ;;
        --frontier)
          [[ ${2:-} =~ ^[a-z0-9_-]+=[A-Za-z0-9._/-]+(,[a-z0-9_-]+=[A-Za-z0-9._/-]+)*$ ]] || die "--frontier needs id=model,..."
          ARGS+=(--frontier "$2"); shift 2 ;;
        --budget) BUDGET=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the pilot's live env"
    [ "$(active zeroad-pilot)" != active ] || die "the pilot is running: pilot.sh status, or stop it first"
    if [ "$FRESH" = 1 ]; then
      run mv "$HOME_DIR" "$STATE/pilot-solo-$(date -u +%Y%m%d-%H%M%S)"
      run mkdir -p "$HOME_DIR/adapters"
      run docker restart zeroad-vllm  # its adapters mount is $HOME_DIR/adapters: the new, empty folder
      if [ "$DRY_RUN" = 0 ]; then
        for _ in $(seq 1 60); do curl -fsS "$VLLM/health" >/dev/null 2>&1 && break; sleep 10; done
      fi
    fi
    if [ "$DRY_RUN" = 0 ]; then
      curl -fsS "$VLLM/health" >/dev/null || die "vLLM is not up at $VLLM (docker logs zeroad-vllm)"
      rm -f "$HOME_DIR/STOP"
    fi
    "$BOX_HOME/box/budget.sh" check --planned-usd 0 --planned-hours 0.5 --budget "$BUDGET"
    say "pilot: $([ "$FRONTIER" = 1 ] && echo 'with frontier probes (LiteLLM key from SSM)' || echo 'no frontier probes') ${ARGS[*]-}"
    run systemd-run --unit zeroad-pilot --collect -p KillMode=control-group -p TimeoutStopSec=600 \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/pilot-run.sh" "$FRONTIER" --budget "$BUDGET" "${ARGS[@]+"${ARGS[@]}"}"
    say "started; follow it with live/pilot.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: pilot.sh status [LINES]" >&2; exit 2; }
    say "zeroad-pilot: $(active zeroad-pilot)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ state.json; progress.md (ratings, latest round); journalctl -u zeroad-pilot -n $N; docker ps; nvidia-smi; vLLM models; budget.sh show; df"
      exit 0
    fi
    echo "--- state"; cat "$HOME_DIR/state.json" 2>/dev/null || echo "(no round finished yet)"
    if [ -f "$HOME_DIR/progress.md" ]; then
      echo "--- ratings and latest round"
      awk '/^## Ratings/{r=1} /^## Round/{r=0; last=""} r{print} /^## Round/{cap=1} cap{last=last $0 "\n"} END{printf "%s", last}' "$HOME_DIR/progress.md"
    fi
    echo "--- log"; journalctl -u zeroad-pilot -n "$N" --no-pager -o cat 2>/dev/null || true
    echo "--- containers"; docker ps --format '{{.Image}}' | sed 's/:.*//' | sort | uniq -c || true
    echo "--- cpu"; echo "$(nproc) vCPU, load $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "--- gpus"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true
    echo "--- vllm"
    if [ -s "$ETC/vllm.env" ]; then
      set -a; source "$ETC/vllm.env"; set +a
      curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM/v1/models" 2>/dev/null | jq -r '[.data[].id] | join(" ")' || echo "vLLM not answering"
    fi
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- pilot folders"; du -sh "$STATE"/pilot "$STATE"/pilot-solo-* 2>/dev/null | sed "s#$STATE/##" || true
    echo "--- disk"; df -h "$STATE" | tail -1
    echo "--- power-off"; busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null || echo none
    ;;
  stop)
    if [ "${1:-}" = --now ]; then
      say "stopping zeroad-pilot now: its games are torn down"
      run systemctl stop zeroad-pilot
    else
      run touch "$HOME_DIR/STOP"
      say "the pilot stops after the round in play (pilot.sh stop --now ends it at once)"
    fi
    ;;
  recollect)
    [[ ${1:-} =~ ^[0-9]+$ ]] || die "recollect needs a round number"
    [ "$(active zeroad-pilot)" != active ] || die "the pilot is running: stop it first"
    if [ "$DRY_RUN" = 1 ]; then say "+ source $ETC/vllm.env"; else set -a; source "$ETC/vllm.env"; set +a; fi
    run env XDG_STATE_HOME="$STATE/agentenv" uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.pilot.run \
      --home "$HOME_DIR" --recollect "$1"
    ;;
  cleanup)
    [ "$(active zeroad-pilot)" != active ] || die "the pilot is running: stop it first"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ docker rm -f <every container whose image is in the box's registry (localhost:5000/...)>; docker network prune -f"
      exit 0
    fi
    ids=$(docker ps -a --format '{{.ID}} {{.Image}}' | awk '$2 ~ /^localhost:5000\// {print $1}')
    say "removing $(echo "$ids" | grep -c . || true) game and agent containers"
    [ -z "$ids" ] || echo "$ids" | xargs docker rm -f >/dev/null
    docker network prune -f >/dev/null
    say "left: $(docker ps --format '{{.Image}}' | sed 's/:.*//' | sort | uniq -c | tr -s ' ' | paste -sd, -)"
    ;;
  *) echo "usage: pilot.sh start ... --budget USD | status [LINES] | stop [--now] | recollect ROUND | cleanup" >&2; exit 2 ;;
esac
