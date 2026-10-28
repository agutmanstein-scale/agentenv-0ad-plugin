#!/usr/bin/env bash
# repair.sh ID [--yes|--dry-run]
# box/repair.sh now: restarts the display, the relay or a wanted push when down, re-allows local X clients and frees
# disk by deleting the oldest recording segments. It never restarts a running match (it says when one needs it).
# A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
ssm_box "$ID" 300 repair.sh
