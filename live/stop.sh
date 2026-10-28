#!/usr/bin/env bash
# stop.sh ID [--push-only] [--yes|--dry-run]
# Ends the push cleanly, then stops the running match (agent-env tears its containers down); --push-only leaves the
# match playing and recording. The box, the display and the relay stay up. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; shift || true
check_instance_id "$ID"
case ${1:-} in --push-only|"") ;; *) echo "unknown argument: $1" >&2; usage ;; esac
ssm_box "$ID" 400 stop.sh "$@"
