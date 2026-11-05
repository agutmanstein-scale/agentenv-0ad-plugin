#!/usr/bin/env bash
# pool-check.sh ID POOL_ID [--dry-run]
# From box ID (box/pool-check.sh): whether its own role can put, list, get and delete under
# s3://LIVE_BUCKET/LIVE_S3_PREFIX/pool/POOL_ID/check/ (pilot-iam.sh create gives it that role). Writes and deletes one
# small object only.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode read "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; POOL_ID=${2:-}
check_instance_id "$ID"
[[ $POOL_ID =~ ^[a-z0-9-]+$ ]] || die "POOL_ID: letters, digits, -"
[ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER}
need LIVE_BUCKET
ssm_box "$ID" 120 pool-check.sh "s3://$LIVE_BUCKET/$PREFIX/pool/$POOL_ID"
