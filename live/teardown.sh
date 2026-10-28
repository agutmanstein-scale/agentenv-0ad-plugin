#!/usr/bin/env bash
# teardown.sh ID [--yes|--dry-run]
# Terminates box ID, only if its Name is LIVE_BOX_NAME and its purpose tag LIVE_TAG_PURPOSE, and waits until it is
# gone. Everything on it is lost, recordings included: live/fetch.sh first. An instance already terminated is
# reported, not an error. Then it lists what to clean up by hand (SSM parameters, S3 objects, keys to rotate).
# A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
read -r state name purpose < <(tags_of "$ID")
[ "$name" = "$BOX_NAME" ] || die "$ID is named '$name', not '$BOX_NAME'; not terminating"
[ "$purpose" = "$PURPOSE" ] || die "$ID has purpose '$purpose', not '$PURPOSE'; not terminating"
if [ "$state" != terminated ]; then
  log "terminating $ID ($name, $state)"
  aws_run shutting-down ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
  aws_run "" ec2 wait instance-terminated --instance-ids "$ID"
fi
log "$ID terminated. By hand, when the stream is over:"
log "  reset the stream key in X Media Studio, then: aws ssm delete-parameters --names $SSM_PREFIX/rtmp-server $SSM_PREFIX/rtmp-key $SSM_PREFIX/litellm-key"
log "  aws s3 rm --recursive s3://${LIVE_BUCKET:-BUCKET}/$PREFIX/   (the shipped code and fetched archives)"
