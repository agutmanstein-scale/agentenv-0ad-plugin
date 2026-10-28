#!/usr/bin/env bash
# health.sh ID [--dry-run]
# One JSON line from box/health.sh: status ok|warn and the problems, the units, the relay, the recording's age, the
# push, the GPU and NVENC, free disk, spend and the power-off time. Run it every 30 minutes while a match is on; the
# box's own timer also checks and repairs every 30 minutes (box/repair.sh --auto). Read-only.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode read "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
ssm_box "$ID" 120 health.sh
