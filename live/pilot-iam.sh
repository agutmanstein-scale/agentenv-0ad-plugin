#!/usr/bin/env bash
# pilot-iam.sh create ID... | delete ID... [--yes|--dry-run]
# The pooled pilot's own IAM role, so its boxes can swap adapters through S3 with no laptop in the loop.
#   create: makes the role and instance profile zeroad-pilot-box (AmazonSSMManagedInstanceCore, as the boxes have now,
#           plus read/write on s3://LIVE_BUCKET/LIVE_S3_PREFIX/pool/* and read on the LIVE_SSM_PREFIX parameters only)
#           and puts it on each box ID in place of its current profile. Nothing else in the account changes.
#   delete: puts each box ID that is still running back on LIVE_INSTANCE_PROFILE, then deletes the role and profile.
# Run with LIVE_ENV=live/pilot.env. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
CMD=${1:-}; shift || true
case $CMD in create|delete) ;; *) echo "pilot-iam.sh create|delete ID..." >&2; usage ;; esac
[ $# -gt 0 ] || { echo "pilot-iam.sh $CMD ID...: name the pilot boxes" >&2; usage; }
for id in "$@"; do check_instance_id "$id"; done
[ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER} LIVE_AWS_ACCOUNT=${LIVE_AWS_ACCOUNT:-123456789012}
need LIVE_BUCKET LIVE_AWS_ACCOUNT LIVE_INSTANCE_PROFILE
NAME=zeroad-pilot-box
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

swap_profile() {  # swap_profile ID PROFILE: the box's instance profile becomes PROFILE
  local id=$1 profile=$2 assoc
  assoc=$(aws_run iip-assoc-PLACEHOLDER ec2 describe-iam-instance-profile-associations \
    --filters "Name=instance-id,Values=$id" "Name=state,Values=associated" --query 'IamInstanceProfileAssociations[0].AssociationId' --output text)
  if [ -z "$assoc" ] || [ "$assoc" = None ]; then
    aws_run "" ec2 associate-iam-instance-profile --instance-id "$id" --iam-instance-profile "Name=$profile" --query 'IamInstanceProfileAssociation.State' --output text
  else
    aws_run "" ec2 replace-iam-instance-profile-association --association-id "$assoc" --iam-instance-profile "Name=$profile" --query 'IamInstanceProfileAssociation.State' --output text
  fi
  log "$id now on instance profile $profile"
}

if [ "$CMD" = create ]; then
  cat > "$tmp/trust.json" <<'EOF'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
EOF
  cat > "$tmp/pool.json" <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Sid": "PoolObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
   "Resource": "arn:aws:s3:::$LIVE_BUCKET/$PREFIX/pool/*"},
  {"Sid": "PoolList", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$LIVE_BUCKET",
   "Condition": {"StringLike": {"s3:prefix": ["$PREFIX/pool/*"]}}},
  {"Sid": "PilotSecrets", "Effect": "Allow", "Action": "ssm:GetParameter",
   "Resource": "arn:aws:ssm:$AWS_REGION:$LIVE_AWS_ACCOUNT:parameter$SSM_PREFIX/*"}
]}
EOF
  log "role $NAME: AmazonSSMManagedInstanceCore + s3://$LIVE_BUCKET/$PREFIX/pool/* + $SSM_PREFIX/* parameters; then on $*"
  aws_run "" iam create-role --role-name "$NAME" --assume-role-policy-document "file://$tmp/trust.json" \
    --description "zeroad pilot boxes: SSM, the $SSM_PREFIX parameters, the pool S3 prefix" \
    --tags "Key=purpose,Value=$PURPOSE" "Key=owner,Value=${LIVE_TAG_OWNER:-${USER:-zeroad}}" --query Role.Arn --output text
  aws_run "" iam attach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws_run "" iam put-role-policy --role-name "$NAME" --policy-name zeroad-pilot-pool --policy-document "file://$tmp/pool.json"
  aws_run "" iam create-instance-profile --instance-profile-name "$NAME" --tags "Key=purpose,Value=$PURPOSE" \
    --query InstanceProfile.Arn --output text
  aws_run "" iam add-role-to-instance-profile --instance-profile-name "$NAME" --role-name "$NAME"
  [ "$DRY_RUN" = 1 ] || sleep 15  # a new instance profile takes a few seconds before EC2 accepts it
  for id in "$@"; do swap_profile "$id" "$NAME"; done
  log "done. Undo with: LIVE_ENV=$LIVE_ENV live/pilot-iam.sh delete $* --yes"
else
  for id in "$@"; do
    state=$(aws_run running ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
    if [ "$state" = running ] || [ "$state" = stopped ]; then swap_profile "$id" "$LIVE_INSTANCE_PROFILE"; else log "$id is $state: nothing to swap back"; fi
  done
  aws_run "" iam remove-role-from-instance-profile --instance-profile-name "$NAME" --role-name "$NAME"
  aws_run "" iam delete-instance-profile --instance-profile-name "$NAME"
  aws_run "" iam delete-role-policy --role-name "$NAME" --policy-name zeroad-pilot-pool
  aws_run "" iam detach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws_run "" iam delete-role --role-name "$NAME"
  log "role and instance profile $NAME deleted"
fi
