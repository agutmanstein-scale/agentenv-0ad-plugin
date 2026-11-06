#!/usr/bin/env bash
# s2.sh ID start CONFIGDIR [--resume|--fresh] | status [LINES] | fetch | stop  [--yes|--dry-run]
# Box day 1 (S2) and Run 1 (S3) end to end, unattended, on one training box ID (bootstrapped with bootstrap.sh --train,
# on the instance role train-iam.sh made), through box/s2.sh (docs/s2-box-day.md). `start` checks CONFIGDIR (the step
# configs and sequence.toml, live/s2/ is the example) here, replaces the s3://BUCKET/ placeholder of every s3.uri with
# LIVE_BUCKET, sends the directory to the box and starts the sequence as the unit zeroad-s2, which survives an SSM
# disconnect: E1 GPU gates, the E2 positive control, the v0 baselines and teacher calibration, the distillation probe,
# then Run 1, each step under its own wall-clock cap and starting only if the gates of the one before passed. It is
# refused when LIVE_BUDGET_USD would be passed or the box's power-off leaves less than the sequence's expected hours.
# --resume continues an earlier sequence (the steps that passed are skipped), --fresh moves it aside. `status` shows the
# gate verdicts, the step under way, its orchestrator and the learner's health, the spend and the power-off. `fetch`
# copies the sequence's own files (gates.json, logs, s2-report.json, the labels) to live/fetched/s2-<time>/; the step
# roots (games, panels, adapters' manifests) are on S3 already. `stop` ends the sequence early: the step stops and
# uploads, and the box stays up. start, fetch and stop are dry runs unless --yes; status runs unless --dry-run.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
case ${2:-} in status) KIND=read ;; start|fetch|stop) KIND=change ;; *) echo "s2.sh ID start|status|fetch|stop: not '${2:-}'" >&2; usage ;; esac
parse_mode "$KIND" "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; SUB=$2; shift 2
check_instance_id "$ID"
case $SUB in
  start)
    DIR=${1:-}; shift || true
    MODE=
    for a in "$@"; do
      case $a in
        --resume|--fresh) [ -z "$MODE" ] || die "--resume and --fresh are alternatives"; MODE=$a ;;
        *) echo "unknown argument: $a" >&2; usage ;;
      esac
    done
    [ -f "$DIR/sequence.toml" ] || die "start needs the config directory (with sequence.toml): s2.sh ID start CONFIGDIR"
    [ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER} LIVE_BUDGET_USD=${LIVE_BUDGET_USD:-350}
    need LIVE_BUCKET LIVE_BUDGET_USD
    log "S2 + Run 1 on $ID from $DIR against \$$LIVE_BUDGET_USD (box hours at \$$RATE/h + frontier spend), box power-off in at most ${LIVE_MAX_HOURS:-?} h"
    if [ "$DRY_RUN" = 1 ]; then
      plan "uv run --project $REPO_DIR --extra train python -m agentenv_zeroad.train.s2 check $DIR --budget $LIVE_BUDGET_USD   (every step's config, panels and sample game)"
    else
      (cd "$REPO_DIR" && uv run --extra train python -m agentenv_zeroad.train.s2 check "$DIR" --budget "$LIVE_BUDGET_USD") >&2 \
        || die "the config directory is not valid"
    fi
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    mkdir "$tmp/cfg"
    for f in "$DIR"/*.toml; do sed "s#s3://BUCKET/#s3://$LIVE_BUCKET/#g" "$f" > "$tmp/cfg/$(basename "$f")"; done
    tar -C "$tmp/cfg" -czf "$tmp/config.tgz" .
    plan "send $(ls "$tmp/cfg" | wc -l | tr -d ' ') config files ($(wc -c < "$tmp/config.tgz" | tr -d ' ') bytes gzipped), s3://BUCKET/ -> s3://$LIVE_BUCKET/"
    ssm_box "$ID" 300 s2.sh put-config "$(base64 < "$tmp/config.tgz" | tr -d '\n')"
    ssm_box "$ID" 300 s2.sh start --budget "$LIVE_BUDGET_USD" ${MODE:+"$MODE"}
    log "started; follow it with live/s2.sh $ID status; live/s2.sh $ID fetch --yes when gates.json says complete or stopped" ;;
  status)
    [ $# -le 1 ] || die "status takes at most LINES"
    ssm_box "$ID" 120 s2.sh status "$@" ;;
  stop)
    [ $# -eq 0 ] || die "stop takes no argument"
    ssm_box "$ID" 4200 s2.sh stop ;;
  fetch)
    [ $# -eq 0 ] || die "fetch takes no argument"
    "$LIVE_DIR/fetch.sh" "$ID" --only train/s2 $([ "$DRY_RUN" = 1 ] || echo --yes)
    if [ "$DRY_RUN" = 1 ]; then
      plan "unpack the newest live/fetched/$ID-*.tar.zst into live/fetched/s2-<time>/ (zstd -dc | tar -x), then print gates.json's verdicts"
      exit 0
    fi
    command -v zstd >/dev/null || die "zstd is not installed here (brew install zstd): unpack the archive in live/fetched/ by hand"
    FETCHED=${LIVE_FETCHED_DIR:-$LIVE_DIR/fetched}
    archive=$(ls -1 "$FETCHED/$ID"-*.tar.zst | sort | tail -1)
    stamp=${archive##*/"$ID"-}; stamp=${stamp%.tar.zst}
    dir="$FETCHED/s2-$stamp"
    mkdir -p "$dir"
    zstd -dc "$archive" | tar -x -C "$dir"
    log "results in live/fetched/s2-$stamp/train/s2 (gates.json, progress.log, logs/, s2-report.json, distill/)"
    if [ -f "$dir/train/s2/gates.json" ]; then
      python3 -c 'import json,sys
d=json.load(open(sys.argv[1])); print("verdict:", d["verdict"], d.get("reason") or "")
for k,v in d["steps"].items(): print(k + ": " + v["status"], v.get("reason") or "")' "$dir/train/s2/gates.json"
    else log "no gates.json: the sequence never started a step"; fi
    log "The step roots (games, panels, adapter manifests) are under the s3.uri prefixes of the configs. Next: live/teardown.sh $ID --yes once they are checked (the box only stops, and its disk is billed until then)" ;;
esac
