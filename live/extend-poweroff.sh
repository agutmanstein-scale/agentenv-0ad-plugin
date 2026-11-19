#!/usr/bin/env bash
# extend-poweroff.sh ID HOURS [--yes|--dry-run]
# Moves box ID's MAX_HOURS power-off HOURS later (box/poweroff.sh extend), for a run that needs longer than the launch
# allowed. Only a box whose Name is LIVE_BOX_NAME and purpose tag LIVE_TAG_PURPOSE. The box costs its rate for the
# extra hours; the run's budget still stops it at budget_usd.
# A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-} HOURS=${2:-}
check_instance_id "$ID"
[[ $HOURS =~ ^[1-9][0-9]?$ ]] || usage
[ $# -le 2 ] || { echo "unknown argument: $3" >&2; usage; }
read -r state name purpose < <(tags_of "$ID")
[ "$name" = "$BOX_NAME" ] || die "$ID is named '$name', not '$BOX_NAME'; not moving its power-off"
[ "$purpose" = "$PURPOSE" ] || die "$ID has purpose '$purpose', not '$PURPOSE'; not moving its power-off"
[ "$state" = running ] || die "$ID is $state: the box must be running"
log "moving $ID's power-off $HOURS h later (at most about \$$(awk -v r="$RATE" -v h="$HOURS" 'BEGIN { printf "%.2f", r * h }') more)"
ssm_box "$ID" 120 poweroff.sh extend "$HOURS"
