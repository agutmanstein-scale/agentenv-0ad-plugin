#!/usr/bin/env bash
# train.sh ID start CONFIG [--init PATH] [--baseline PATH] [--play-only] | status [LINES] | stop  [--yes|--dry-run]
# A post-training run on box ID (bootstrapped with bootstrap.sh --train), through box/train.sh. `start` sends the run
# config file CONFIG (TOML, train/config.py) to the box, checks it there, and starts the learner and the orchestrator,
# refused when LIVE_BUDGET_USD would be passed or a vLLM instance is down; --init and --baseline name files under
# /var/lib/zeroad-live/train on the box, --play-only starts no learner and credits nothing (a baseline or a panel run).
# `stop` SIGTERMs the orchestrator, which stops the learner, uploads the run to its s3 prefix and ends (the box stays up:
# `aws ec2 stop-instances`, live/stop-box.sh, when the upload is checked). `status` is read-only. start and stop are dry
# runs unless --yes; status runs unless --dry-run. The run copies itself to S3 with the box's role (train-iam.sh).
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status) KIND=read ;; start|stop) KIND=change ;; *) echo "train.sh ID start|status|stop: not '${2:-}'" >&2; usage ;; esac
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
case $SUB in
  start)
    FILE=${1:-}; shift || true
    [ -f "$FILE" ] || die "start needs the run config file: train.sh ID start CONFIG"
    [ "$DRY_RUN" = 1 ] && LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-150}
    need LIVE_BUDGET_USD
    for a in "$@"; do [[ $a =~ ^[A-Za-z0-9._,=/-]+$ ]] || die "bad argument: '$a'"; done
    log "training run on $ID from $FILE: budget \$$LIVE_BUDGET_USD (box hours at \$$RATE/h + frontier spend), $*"
    ssm_box "$ID" 120 train.sh put-config "$(base64 < "$FILE" | tr -d '\n')"
    ssm_box "$ID" 300 train.sh start "$@" --budget "$LIVE_BUDGET_USD" ;;
  stop)
    [ $# -eq 0 ] || die "stop takes no argument"
    ssm_box "$ID" 1800 train.sh stop ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 train.sh status "$@" ;;
esac
