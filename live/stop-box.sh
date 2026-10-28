#!/usr/bin/env bash
# stop-box.sh ID [--yes|--dry-run]
# Stops (does not terminate) box ID, if its Name is LIVE_BOX_NAME and its purpose tag LIVE_TAG_PURPOSE: the GPU
# charge stops, the 200 GB root volume (images, recordings) stays and is billed as storage (about $16 a month).
# `aws ec2 start-instances` brings it back; the scheduled power-off is gone after a stop. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
read -r state name purpose < <(tags_of "$ID")
[ "$name" = "$BOX_NAME" ] || die "$ID is named '$name', not '$BOX_NAME'; not stopping"
[ "$purpose" = "$PURPOSE" ] || die "$ID has purpose '$purpose', not '$PURPOSE'; not stopping"
if [ "$state" = stopped ]; then log "$ID is already stopped"; exit 0; fi
aws_run stopping ec2 stop-instances --instance-ids "$ID" --query 'StoppingInstances[0].CurrentState.Name' --output text
aws_run "" ec2 wait instance-stopped --instance-ids "$ID"
log "$ID stopped"
