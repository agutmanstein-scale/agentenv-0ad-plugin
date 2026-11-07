#!/usr/bin/env bash
# start-box.sh ID [--yes|--dry-run]
# Starts box ID again after a stop (its own power-off, a Spot reclaim or stop-box.sh), if its Name is LIVE_BOX_NAME and
# its purpose tag LIVE_TAG_PURPOSE, waits until SSM sees it, and checks that the MAX_HOURS power-off is armed again (the
# zeroad-poweroff unit re-arms it from the launch deadline on every boot). vLLM's containers restart with Docker. A
# sequence that was running does not: live/s2.sh ID start CONFIGDIR --resume. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
read -r state name purpose < <(tags_of "$ID")
[ "$name" = "$BOX_NAME" ] || die "$ID is named '$name', not '$BOX_NAME'; not starting"
[ "$purpose" = "$PURPOSE" ] || die "$ID has purpose '$purpose', not '$PURPOSE'; not starting"
case $state in
  running) log "$ID is already running" ;;
  stopped)
    aws_run pending ec2 start-instances --instance-ids "$ID" --query 'StartingInstances[0].CurrentState.Name' --output text
    aws_run "" ec2 wait instance-running --instance-ids "$ID"
    ;;
  *) die "$ID is $state: wait until it is stopped, then start it" ;;
esac
wait_ssm "$ID"
ssm_box "$ID" 120 poweroff.sh check
log "$ID running and in SSM. Next, for a stopped sequence: live/s2.sh $ID start CONFIGDIR --resume --yes"
