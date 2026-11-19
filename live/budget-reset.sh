#!/usr/bin/env bash
# budget-reset.sh ID [--yes|--dry-run]
# box/budget.sh reset: the box's spend counter starts again from $0 (model spend and box hours). History is kept: the
# old ledger and boxtime.json are moved to spend-before-reset-STAMP.jsonl and boxtime-before-reset-STAMP.json in the
# box's state folder, and budget-resets.jsonl there gets one line with the counts before the reset. Safe while a
# training sequence is running. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
ssm_box "$ID" 120 budget.sh reset
