#!/usr/bin/env bash
# match.sh ID rehearsal|model [--task TASK] [--name NAME] [--push] [--yes|--dry-run]
# Starts one rendered match on box ID (box/match.sh): `rehearsal` is the scripted bot against Petra (livestream-petra,
# $0, about 6 min); `model` is the 40-minute livestream match, Setup A, capped at LIVE_MATCH_CAP_USD ($9), or with
# --task another rendered model task of the bundle (e.g. livestream-4, four models for 10 minutes). It is
# recorded on the box either way and pushed to the RTMP target only with --push. Refused when LIVE_BUDGET_USD would
# be passed (model spend so far + this match's caps + box hours). Costs money and, with --push, is public: a dry run
# unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; KIND=${2:-}; shift 2 || true
check_instance_id "$ID"
case $KIND in rehearsal|model) ;; *) echo "match.sh ID rehearsal|model: not '$KIND'" >&2; usage ;; esac
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --name)
      [[ ${2:-} =~ ^[A-Za-z0-9._-]+$ ]] || die "--name: letters, digits, . _ - only"
      EXTRA+=(--name "$2"); shift 2 ;;
    --task)
      [ "$KIND" = model ] || die "--task: model matches only"
      [[ ${2:-} =~ ^[A-Za-z0-9._-]+$ ]] || die "--task: letters, digits, . _ - only"
      EXTRA+=(--task "$2"); shift 2 ;;
    --push) EXTRA+=(--push); shift ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ "$DRY_RUN" = 1 ] && LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-20}
need LIVE_BUDGET_USD
log "$KIND match on $ID: budget \$$LIVE_BUDGET_USD, match cap \$${LIVE_MATCH_CAP_USD:-9}$([[ " ${EXTRA[*]-} " == *" --push "* ]] && echo ', PUSHED to the RTMP target (public)')"
ssm_box "$ID" 300 match.sh start "$KIND" --budget "$LIVE_BUDGET_USD" --match-cap "${LIVE_MATCH_CAP_USD:-9}" "${EXTRA[@]+"${EXTRA[@]}"}"
