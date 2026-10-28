#!/usr/bin/env bash
# launch.sh [--name NAME] [--max-hours H] [--root-gb GB] [--yes|--dry-run]
# Launches the stream box: a g6.4xlarge (NVIDIA L4) on the AWS Deep Learning Base OSS NVIDIA driver AMI (Ubuntu
# 22.04), reached only through SSM. GPU capacity moves between zones, so it tries each type of LIVE_GPU_TYPES in each
# subnet of LIVE_SUBNETS until one launches, waits until SSM sees it and prints its instance id. The box powers itself
# off, and so terminates, H hours (LIVE_MAX_HOURS, default 12) after launch. If a box named NAME is already up, that
# one's id is printed. Costs about $1.32/h: a dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
NAME=$BOX_NAME
MAX_HOURS=${LIVE_MAX_HOURS:-12}
ROOT_GB=200
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --max-hours) MAX_HOURS=${2:?--max-hours needs a value}; shift 2 ;;
    --root-gb) ROOT_GB=${2:?--root-gb needs a value}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
if ! [[ $MAX_HOURS =~ ^[0-9]+$ ]] || [ "$MAX_HOURS" -lt 1 ] || [ "$MAX_HOURS" -gt 72 ]; then die "--max-hours must be 1-72"; fi
if ! [[ $ROOT_GB =~ ^[0-9]+$ ]] || [ "$ROOT_GB" -lt 100 ]; then die "--root-gb must be at least 100"; fi
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "--name: letters, digits, . _ - only"
TYPES=${LIVE_GPU_TYPES:-g6.4xlarge}
SUBNETS=${LIVE_SUBNETS:-${LIVE_SUBNET:-}}
if [ "$DRY_RUN" = 1 ]; then
  SUBNETS=${SUBNETS:-subnet-PLACEHOLDER} LIVE_SECURITY_GROUP=${LIVE_SECURITY_GROUP:-sg-PLACEHOLDER}
  LIVE_INSTANCE_PROFILE=${LIVE_INSTANCE_PROFILE:-PROFILE-PLACEHOLDER}
fi
[ -n "$SUBNETS" ] || die "set LIVE_SUBNETS in $LIVE_ENV"
need LIVE_SECURITY_GROUP LIVE_INSTANCE_PROFILE
log "box $NAME: one of [$TYPES], about \$$RATE/h, powers off after $MAX_HOURS h (at most about \$$(awk -v r="$RATE" -v h="$MAX_HOURS" 'BEGIN { printf "%.2f", r * h }'))"

id=$(aws_run None ec2 describe-instances \
  --filters "Name=tag:Name,Values=$NAME" "Name=tag:purpose,Values=$PURPOSE" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[0].InstanceId' --output text)
if [ -n "$id" ] && [ "$id" != None ]; then
  log "$NAME is already up as $id"
else
  id=
  ami=$(aws_run ami-DLAMI-PLACEHOLDER ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --query Parameter.Value --output text)
  userdata=$(printf '#!/bin/bash\nshutdown -h +%d "zeroad-live: MAX_HOURS reached"\n' $((MAX_HOURS * 60)) | base64 | tr -d '\n')
  tags="{Key=Name,Value=$NAME},{Key=owner,Value=${LIVE_TAG_OWNER:-${USER:-zeroad}}},{Key=purpose,Value=$PURPOSE}"
  for t in $TYPES; do
    for s in $SUBNETS; do
      if id=$(aws_run i-0123456789abcdef0 ec2 run-instances --image-id "$ami" --instance-type "$t" --count 1 \
          --subnet-id "$s" --security-group-ids "$LIVE_SECURITY_GROUP" \
          --iam-instance-profile "Name=$LIVE_INSTANCE_PROFILE" \
          --instance-initiated-shutdown-behavior terminate \
          --metadata-options HttpTokens=required,HttpEndpoint=enabled \
          --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
          --user-data "$userdata" \
          --tag-specifications "ResourceType=instance,Tags=[$tags]" "ResourceType=volume,Tags=[$tags]" \
          --query 'Instances[0].InstanceId' --output text); then
        log "launched $id ($t in $s, $ROOT_GB GB root, powers off after $MAX_HOURS h)"
        break 2
      fi
      log "$t in $s: no capacity or refused; trying the next"
    done
  done
  [ -n "$id" ] || die "no box came up: no capacity for $TYPES in $SUBNETS"
fi
aws_run "" ec2 wait instance-running --instance-ids "$id"
wait_ssm "$id"
log "$id online in SSM. Next: live/bootstrap.sh $id"
echo "$id"
