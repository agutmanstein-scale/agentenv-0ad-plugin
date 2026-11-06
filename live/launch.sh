#!/usr/bin/env bash
# launch.sh [--name NAME] [--max-hours H] [--root-gb GB] [--shutdown-behavior terminate|stop] [--yes|--dry-run]
# Launches the stream box: a g6.4xlarge (NVIDIA L4) on the AWS Deep Learning Base OSS NVIDIA driver AMI (Ubuntu
# 22.04), reached only through SSM. GPU capacity moves between zones, so it tries each type of LIVE_GPU_TYPES in each
# subnet of LIVE_SUBNETS until one launches, waits until SSM sees it and prints its instance id. The box powers itself
# off, and so terminates, H hours (LIVE_MAX_HOURS, default 12) after launch: the power-off is armed on every boot (a
# cloud-init boothook with the absolute deadline, at least 30 min out), so a box that was stopped and started again has
# it too. If a box named NAME is already up, that one's id is printed. Costs about $1.32/h: a dry run unless --yes.
# Every type after the first is a fallback with a price of its own: LIVE_GPU_RATES ("TYPE=USD_PER_HOUR ...") names it, a
# fallback with no rate is refused, and the launched type's rate is tagged on the box (bootstrap.sh writes it to the
# box's config), so a run on the larger type is not priced at the smaller one's rate.
# The power-off terminates the box (and deletes its volume) unless --shutdown-behavior stop (or LIVE_SHUTDOWN_BEHAVIOR):
# then it only stops, the disk stays (billed as storage) and `aws ec2 start-instances` brings it back, which a training
# run wants: its power-off at MAX_HOURS must not destroy results that were not yet copied out.
# The types are LIVE_GPU_TYPES whatever they are: a CPU box (the S1 tool check, live/s1.env.example) lists CPU types
# there, and sets LIVE_AMI_SSM_PARAM to the public SSM parameter of a plain Ubuntu 24.04 AMI instead of the GPU one.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
NAME=$BOX_NAME
MAX_HOURS=${LIVE_MAX_HOURS:-12}
ROOT_GB=200
SHUTDOWN=${LIVE_SHUTDOWN_BEHAVIOR:-terminate}
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --max-hours) MAX_HOURS=${2:?--max-hours needs a value}; shift 2 ;;
    --root-gb) ROOT_GB=${2:?--root-gb needs a value}; shift 2 ;;
    --shutdown-behavior) SHUTDOWN=${2:?--shutdown-behavior needs a value}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
if ! [[ $MAX_HOURS =~ ^[0-9]+$ ]] || [ "$MAX_HOURS" -lt 1 ] || [ "$MAX_HOURS" -gt 72 ]; then die "--max-hours must be 1-72"; fi
case $SHUTDOWN in terminate|stop) ;; *) die "--shutdown-behavior must be terminate or stop" ;; esac
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
# rate_for TYPE: its price in LIVE_GPU_RATES, or LIVE_BOX_USD_PER_HOUR for the first type. A fallback with no price is
# refused (nothing printed) when LIVE_REQUIRE_RATE=1, a training box, whose types cost several times each other; the
# stream box keeps LIVE_BOX_USD_PER_HOUR for it and the launch warns.
rate_for() {
  local pair
  for pair in ${LIVE_GPU_RATES:-}; do
    if [ "${pair%%=*}" = "$1" ]; then echo "${pair#*=}"; return 0; fi
  done
  if [ "$1" = "${TYPES%% *}" ] || [ "${LIVE_REQUIRE_RATE:-0}" != 1 ]; then echo "$RATE"; fi
  return 0
}
priced_at_default() { # TYPE: a fallback type with no entry in LIVE_GPU_RATES
  local pair
  [ "$1" != "${TYPES%% *}" ] || return 1
  for pair in ${LIVE_GPU_RATES:-}; do [ "${pair%%=*}" != "$1" ] || return 1; done
}
for t in $TYPES; do
  [ -n "$(rate_for "$t")" ] || die "no rate for the fallback type $t: add $t=USD_PER_HOUR to LIVE_GPU_RATES in $LIVE_ENV (a box is never priced at another type's rate)"
done
log "box $NAME: one of [$TYPES], about \$$RATE/h, powers off ($SHUTDOWN) after $MAX_HOURS h (at most about \$$(awk -v r="$RATE" -v h="$MAX_HOURS" 'BEGIN { printf "%.2f", r * h }'))"

id=$(aws_run None ec2 describe-instances \
  --filters "Name=tag:Name,Values=$NAME" "Name=tag:purpose,Values=$PURPOSE" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[0].InstanceId' --output text)
if [ -n "$id" ] && [ "$id" != None ]; then
  log "$NAME is already up as $id"
else
  id=
  ami_param=${LIVE_AMI_SSM_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id}
  ami=$(aws_run "$([ -z "${LIVE_AMI_SSM_PARAM:-}" ] && echo ami-DLAMI-PLACEHOLDER || echo ami-PLACEHOLDER)" ssm get-parameter \
    --name "$ami_param" --query Parameter.Value --output text)
  # The boothook only records the absolute deadline and installs zeroad-poweroff.service, which arms the power-off after
  # systemd-logind on every boot: a `shutdown -h` from the boothook itself runs before logind can take it and is lost.
  userdata=$(sed "s/@DEADLINE@/$(( $(date +%s) + MAX_HOURS * 3600 ))/" "$LIVE_DIR/lib/poweroff-boothook.sh" | base64 | tr -d '\n')
  for t in $TYPES; do
    trate=$(rate_for "$t")
    tags="{Key=Name,Value=$NAME},{Key=owner,Value=${LIVE_TAG_OWNER:-${USER:-zeroad}}},{Key=purpose,Value=$PURPOSE},{Key=zeroad-box-usd-per-hour,Value=$trate},{Key=zeroad-instance-type,Value=$t}"
    for s in $SUBNETS; do
      if id=$(aws_run i-0123456789abcdef0 ec2 run-instances --image-id "$ami" --instance-type "$t" --count 1 \
          --subnet-id "$s" --security-group-ids "$LIVE_SECURITY_GROUP" \
          --iam-instance-profile "Name=$LIVE_INSTANCE_PROFILE" \
          --instance-initiated-shutdown-behavior "$SHUTDOWN" \
          --metadata-options HttpTokens=required,HttpEndpoint=enabled \
          --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
          --user-data "$userdata" \
          --tag-specifications "ResourceType=instance,Tags=[$tags]" "ResourceType=volume,Tags=[$tags]" \
          --query 'Instances[0].InstanceId' --output text); then
        log "launched $id ($t in $s at \$$trate/h, $ROOT_GB GB root, powers off ($SHUTDOWN) after $MAX_HOURS h)"
        if priced_at_default "$t"; then
          log "WARNING: no rate for $t in LIVE_GPU_RATES: it is priced at the \$$RATE/h of LIVE_BOX_USD_PER_HOUR, which may not be its cost"
        elif [ "$trate" != "$RATE" ]; then
          log "WARNING: $t costs \$$trate/h, not the \$$RATE/h in $LIVE_ENV: the run config's box_usd_per_hour must be $trate (bootstrap.sh writes it to the box)"
        fi
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
