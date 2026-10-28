#!/usr/bin/env bash
# status.sh ID [LINES] [--dry-run]
# The box's EC2 state, type, launch time and tags, then box/status.sh: the units, the match's and the push's last
# LINES log lines (redacted), spend and budget, the newest recordings, GPU, disk and the power-off time. Read-only.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode read "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; N=${2:-15}
check_instance_id "$ID"
[[ $N =~ ^[0-9]+$ ]] || { echo "LINES must be a number" >&2; usage; }
# shellcheck disable=SC2016  # the backticks are JMESPath literals
aws_run "running g6.4xlarge 2026-01-01T00:00:00Z $BOX_NAME $PURPOSE" ec2 describe-instances --instance-ids "$ID" --query \
  'Reservations[0].Instances[0].[State.Name, InstanceType, LaunchTime, Tags[?Key==`Name`].Value|[0], Tags[?Key==`purpose`].Value|[0]]' \
  --output text
ssm_box "$ID" 60 status.sh "$N"
