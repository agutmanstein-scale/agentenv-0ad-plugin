#!/usr/bin/env bash
# pool-check.sh POOL_URI: whether this box's own role can use the pooled pilot's S3 prefix: put, list, get and delete
# one small object under POOL_URI/check/. Prints what each step did.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
POOL=${1:?usage: pool-check.sh s3://bucket/prefix/pool/ID}
[[ $POOL =~ ^s3://[a-z0-9.-]+/[A-Za-z0-9._/-]+$ ]] || die "not an s3:// pool: $POOL"
KEY="$POOL/check/$(hostname -s).txt"
if [ "$DRY_RUN" = 1 ]; then say "+ put, list, get and delete $KEY with the box's own role"; exit 0; fi
say "role: $(aws sts get-caller-identity --query Arn --output text | sed 's#.*:assumed-role/##; s#/.*##')"
echo "pool check $(date -u +%FT%TZ)" | aws s3 cp - "$KEY" --only-show-errors && say "put: ok"
aws s3 ls "$POOL/check/" | grep -q "$(hostname -s).txt" && say "list: ok"
[ -n "$(aws s3 cp "$KEY" - 2>/dev/null)" ] && say "get: ok"
aws s3 rm "$KEY" --only-show-errors && say "delete: ok"
