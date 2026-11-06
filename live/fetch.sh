#!/usr/bin/env bash
# fetch.sh ID [NAME] [--only PATH] [--yes|--dry-run]
# Copies the box's match folders (results.json, run.log), its ledger and health log, and the relay's recordings
# (NAME's only, when given) to live/fetched/ID-<time>.tar.zst through S3 (a presigned PUT, so the box needs no S3
# rights). One PUT takes at most 5 GB and a one-hour segment is about 2.7 GB, so for a match past about 90 minutes
# fetch its parts one at a time: --only PATH sends just that path under the box's state folder (matches/NAME, or
# one segment such as rec/zeroad/<segment>.mp4; status.sh lists them) with the ledger and health log. A dry run
# unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ONLY=; POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --only) ONLY=${2:?--only needs a path}; shift 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
ID=${POS[0]:-}; NAME=${POS[1]:-}
check_instance_id "$ID"
[ -z "$NAME" ] || [[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "NAME: letters, digits, . _ - only"
[ -z "$ONLY" ] || { [[ $ONLY =~ ^[A-Za-z0-9._/-]+$ ]] && [[ $ONLY != *..* ]] && [[ $ONLY != /* ]]; } ||
  die "--only: a relative path under the box's state folder (letters, digits, . _ - /, no ..)"
[ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER}
need LIVE_BUCKET
key="$PREFIX/fetch/$ID-$(date -u +%Y%m%d-%H%M%S).tar.zst"
if [ "$DRY_RUN" = 1 ]; then
  plan "presign a PUT for s3://$LIVE_BUCKET/$key (boto3, 2 h)"
  url=https://PRESIGNED-PUT-url
else
  py=(python3); python3 -c 'import boto3' 2>/dev/null || py=(uv run --quiet --no-project --with boto3 python)
  url=$("${py[@]}" -c '
import sys, boto3
from botocore.config import Config
bucket, key, region = sys.argv[1:4]
s3 = boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))
print(s3.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7200))
' "$LIVE_BUCKET" "$key" "$AWS_REGION")
fi
ssm_box "$ID" 3600 fetch.sh "$url" "$NAME" ${ONLY:+"$ONLY"}
FETCHED=${LIVE_FETCHED_DIR:-$LIVE_DIR/fetched}  # where the archive lands (tests point it elsewhere)
[ "$DRY_RUN" = 1 ] || mkdir -p "$FETCHED"
aws_run "" s3 cp "s3://$LIVE_BUCKET/$key" "$FETCHED/" --only-show-errors
log "fetched to live/fetched/$(basename "$key") (zstd -d, then tar x)"
