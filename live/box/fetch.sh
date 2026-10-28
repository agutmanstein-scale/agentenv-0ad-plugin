#!/usr/bin/env bash
# fetch.sh PUT_URL [NAME] [ONLY]: packs the match folders (results.json, run.log, the bundle) or just NAME's, the box
# ledger and health log, and the relay's recordings (all, or those written since NAME started), and uploads the
# archive to PUT_URL (presigned by live/fetch.sh; the box needs no S3 rights). With ONLY, a path under the state
# folder, it packs just that path and the ledger and health log.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
URL=${1:?usage: fetch.sh PUT_URL [NAME] [ONLY]}; NAME=${2:-}; ONLY=${3:-}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
list=$tmp/list
if [ -n "$ONLY" ]; then
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
