#!/usr/bin/env bash
# box/grow-disk.sh: grows the root partition and filesystem into the root volume's size, after live/grow-disk.sh has
# resized the volume, with no reboot; then reports what fills the disk. Safe to repeat: growpart says NOCHANGE when the
# partition already fills the volume.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
if [ "$DRY_RUN" = 1 ]; then
  say "+ growpart <the root partition's disk> <its number>; resize2fs (ext4) or xfs_growfs (xfs) on /"
else
  part=$(lsblk -rno NAME,MOUNTPOINT | awk '$2 == "/" { print $1; exit }')
  [ -n "$part" ] || die "no block device is mounted on /"
  disk=$(lsblk -no PKNAME "/dev/$part" | head -1)
  num=$(cat "/sys/class/block/$part/partition")
  growpart "/dev/$disk" "$num" || [ $? -eq 1 ]  # 1: NOCHANGE, the partition already fills the disk
  case $(findmnt -no FSTYPE /) in
    ext4) resize2fs "/dev/$part" ;;
    xfs) xfs_growfs / ;;
    *) die "unknown root filesystem: $(findmnt -no FSTYPE /)" ;;
  esac
fi
run df -h /
run docker system df
run sh -c 'du -sh "$1"/* 2>/dev/null | sort -h | tail -8' _ "$STATE"
