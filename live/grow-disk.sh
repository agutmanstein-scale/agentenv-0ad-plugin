#!/usr/bin/env bash
# grow-disk.sh ID SIZE_GB [--yes|--dry-run]
# Grows box ID's root EBS volume to SIZE_GB (never shrinks it), waits until AWS has applied the new size, then grows
# the root partition and filesystem on the box (box/grow-disk.sh) with no reboot, while a run goes on. Only a box whose
# Name is LIVE_BOX_NAME and purpose tag LIVE_TAG_PURPOSE. gp3 costs about $0.08 per GB-month.
# A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-} SIZE=${2:-}
check_instance_id "$ID"
[[ $SIZE =~ ^[1-9][0-9]*$ ]] || usage
[ $# -le 2 ] || { echo "unknown argument: $3" >&2; usage; }
read -r state name purpose < <(tags_of "$ID")
[ "$name" = "$BOX_NAME" ] || die "$ID is named '$name', not '$BOX_NAME'; not growing its disk"
[ "$purpose" = "$PURPOSE" ] || die "$ID has purpose '$purpose', not '$PURPOSE'; not growing its disk"
[ "$state" = running ] || die "$ID is $state: the box must be running to grow its filesystem"
# shellcheck disable=SC2016  # the backticks are JMESPath literals
vol=$(aws_run vol-0123456789abcdef0 ec2 describe-instances --instance-ids "$ID" --output text \
  --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/sda1`].Ebs.VolumeId|[0]')
[[ $vol =~ ^vol- ]] || die "no root volume (/dev/sda1) on $ID"
cur=$(aws_run 200 ec2 describe-volumes --volume-ids "$vol" --query 'Volumes[0].Size' --output text)
[ "$SIZE" -gt "$cur" ] || die "$vol is already $cur GB; a volume only grows"
log "growing $vol from $cur to $SIZE GB (about \$$(awk -v g=$((SIZE - cur)) 'BEGIN { printf "%.2f", g * 0.08 / 30 }') a day more)"
aws_run modifying ec2 modify-volume --volume-id "$vol" --size "$SIZE" --query VolumeModification.ModificationState --output text
if [ "$DRY_RUN" = 1 ]; then
  plan "wait until aws ec2 describe-volumes-modifications says optimizing or completed (10 min max)"
else
  for _ in $(seq 1 120); do
    mstate=$(aws ec2 describe-volumes-modifications --volume-ids "$vol" \
      --query 'VolumesModifications[0].ModificationState' --output text 2>/dev/null || echo modifying)
    case $mstate in optimizing|completed) break ;; failed) die "AWS failed to resize $vol" ;; esac
    sleep 5
  done
  case $mstate in optimizing|completed) ;; *) die "$vol still $mstate after 10 min: rerun once it is optimizing" ;; esac
fi
ssm_box "$ID" 600 grow-disk.sh
log "$ID's root filesystem fills $SIZE GB"
