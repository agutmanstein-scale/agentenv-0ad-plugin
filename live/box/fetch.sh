#!/usr/bin/env bash
# fetch.sh PUT_URL [NAME] [ONLY]: packs the match folders (results.json, run.log, the bundle) or just NAME's, the box
# ledger and health log, and the relay's recordings (all, or those written since NAME started), and uploads the
# archive to PUT_URL (presigned by live/fetch.sh; the box needs no S3 rights). With ONLY, a path under the state
# folder, it packs just that path and the ledger and health log; train/s2 is the S2 sequence's verdicts, logs, report and
# labels only (live/s2.sh fetch); train/actor is a rollout actor's orchestrator folder and log (live/actor.sh fetch).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
URL=${1:?usage: fetch.sh PUT_URL [NAME] [ONLY]}; NAME=${2:-}; ONLY=${3:-}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
list=$tmp/list
if [ "$ONLY" = train/s2 ]; then
  # The S2 sequence's own files only (its step roots hold hundreds of adapters and every game's token ids; they sync to
  # S3 as they run): the verdicts, the logs, the report and the distillation labels.
  [ "$DRY_RUN" = 1 ] || [ -f "$STATE/train/s2/gates.json" ] || die "no S2 sequence in $STATE/train/s2"
  : > "$list"
  for f in gates.json status.json progress.log s2-report.json distill logs generated; do
    [ "$DRY_RUN" = 1 ] || [ ! -e "$STATE/train/s2/$f" ] || echo "train/s2/$f" >> "$list"
  done
elif [ "$ONLY" = train/actor ]; then
  # A rollout actor's own files (live/actor.sh fetch): its orchestrator folder (status, final, ship state, config) and
  # the unit's log. Its games were shipped through the transport prefix and its adapters are the learner's.
  [ "$DRY_RUN" = 1 ] || [ -f "$STATE/train/actor.toml" ] || die "no actor config in $STATE/train/actor.toml"
  : > "$list"
  if [ "$DRY_RUN" = 0 ]; then
    eval "$(uv run --project "$REPO" --frozen --no-dev --extra train python -m agentenv_zeroad.train.config "$STATE/train/actor.toml")"
    mkdir -p "$STATE/train/actor-logs"
    journalctl -u zeroad-actor --no-pager -o short-iso > "$STATE/train/actor-logs/zeroad-actor.log" 2>/dev/null || true
    echo "train/actor-logs" >> "$list"
    [ ! -d "$TRAIN_ROOT/orchestrator" ] || echo "${TRAIN_ROOT#"$STATE"/}/orchestrator" >> "$list"
  fi
elif [ -n "$ONLY" ]; then
  [ "$DRY_RUN" = 1 ] || [ -e "$STATE/$ONLY" ] || die "no $ONLY in $STATE"
  echo "$ONLY" > "$list"
elif [ -n "$NAME" ]; then
  [ "$DRY_RUN" = 1 ] || [ -d "$STATE/matches/$NAME" ] || die "no match $NAME"
  echo "matches/$NAME" > "$list"
  [ "$DRY_RUN" = 1 ] || find "$STATE/rec" -type f -newer "$STATE/matches/$NAME/.started" -printf 'rec/%P\n' >> "$list"
else
  printf 'matches\nrec\n' > "$list"
fi
for f in spend.jsonl health.jsonl; do [ "$DRY_RUN" = 1 ] || [ ! -f "$STATE/$f" ] || echo "$f" >> "$list"; done
# S3 takes no chunked upload on a presigned PUT, so the archive is written first and sent with its length.
run sh -c 'tar -C "$1" -cf - -T "$2" | zstd -q -T0 > "$3" && curl -fsS -X PUT -T "$3" "$4"' _ "$STATE" "$list" "$tmp/fetch.tar.zst" "$URL"
say "uploaded $(wc -l < "$list") paths"
