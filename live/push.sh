#!/usr/bin/env bash
# push.sh ID start|stop [--yes|--dry-run]
# start: pushes the box's relay (the match on air, or nothing between matches) to the RTMP target whose server and
# key are in SSM (put-secret.sh rtmp-server / rtmp-key). Public: X shows it once a broadcast uses the source.
# stop: ends the push cleanly; the match and its recording go on. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; CMD=${2:-}
check_instance_id "$ID"
case $CMD in start|stop) ;; *) echo "push.sh ID start|stop: not '$CMD'" >&2; usage ;; esac
[ $# -le 2 ] || { echo "unknown argument: $3" >&2; usage; }
[ "$CMD" = stop ] || log "push to the RTMP target in $SSM_PREFIX/rtmp-server + $SSM_PREFIX/rtmp-key: PUBLIC"
ssm_box "$ID" 120 push.sh "$CMD"
