#!/usr/bin/env bash
# pilot.sh ID start [--baseline] [--rounds N] [--train-games N] [--train-samples N] [--speed S] [--cap-minutes M] [--no-frontier]
#        | status [LINES] | stop [--now] | recollect ROUND | cleanup  [--yes|--dry-run]
# The post-training pilot on box ID (bootstrapped with bootstrap.sh --pilot), through box/pilot.sh: `start` starts the
# round loop (with --baseline one round of the base model and no training), refused when LIVE_BUDGET_USD would be
# passed; `stop` lets the round in play finish (--now ends it at once); `status` is read-only. Without
# --no-frontier the frontier probes (Sonnet, Opus, Astra and Sol; --frontier-per-round N plays N of them a round, in
# turn) spend real model money on the LiteLLM key in SSM (put-secret.sh litellm-key). Other start options
# (--seed, --train-games, --train-samples, --frontier, --frontier-offset) pass through to box/pilot.sh. --pool ID
# pools this box (PILOT_BOX) with --pool-members a,b,... through s3://LIVE_BUCKET/LIVE_S3_PREFIX/pool/ID (the boxes
# need pilot-iam.sh's role for it).
# start and stop are dry runs unless --yes; status runs unless --dry-run.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status) KIND=read ;; start|stop|recollect|cleanup) KIND=change ;; *) echo "pilot.sh ID start|status|stop|recollect|cleanup: not '${2:-}'" >&2; usage ;; esac
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
case $SUB in
  start)
    [ "$DRY_RUN" = 1 ] && LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-150}
    need LIVE_BUDGET_USD
    for a in "$@"; do [[ $a =~ ^[A-Za-z0-9._,=/-]+$ ]] || die "bad argument: '$a'"; done
    out=()
    while [ $# -gt 0 ]; do
      if [ "$1" = --pool ]; then
        [[ ${2:-} =~ ^[a-z0-9-]+$ ]] || die "--pool needs an id (letters, digits, -)"
        [ -n "${PILOT_BOX:-}" ] || die "--pool needs PILOT_BOX (this box's name in the pool)"
        out+=(--pool "s3://$LIVE_BUCKET/$PREFIX/pool/$2" --pool-self "$PILOT_BOX"); shift 2
      else
        out+=("$1"); shift
      fi
    done
    set -- "${out[@]}"
    log "pilot on $ID: budget \$$LIVE_BUDGET_USD (box hours at \$$RATE/h + the frontier probes), $*"
    ssm_box "$ID" 300 pilot.sh start "$@" --budget "$LIVE_BUDGET_USD" ;;
  stop)
    [ $# -eq 0 ] || [ "$1" = --now ] || die "stop takes only --now"
    ssm_box "$ID" 900 pilot.sh stop "$@" ;;
  recollect)
    [[ ${1:-} =~ ^[0-9]+$ ]] && [ $# -eq 1 ] || die "recollect takes a round number"
    ssm_box "$ID" 600 pilot.sh recollect "$1" ;;
  cleanup)
    [ $# -eq 0 ] || die "cleanup takes no argument"
    ssm_box "$ID" 300 pilot.sh cleanup ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 pilot.sh status "$@" ;;
esac
