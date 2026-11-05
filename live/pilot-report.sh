#!/usr/bin/env bash
# pilot-report.sh BOX=ID [BOX=ID ...] [--pooled] [--yes|--dry-run]
# Fetches each pilot box's report.json (live/fetch.sh ID --only pilot/report.json, with PILOT_BOX=BOX) into
# live/fetched/pilot-BOX/report.json, then rates every run on one scale (python -m agentenv_zeroad.pilot.combine)
# into live/fetched/pilot-dashboard.json, the file the progress dashboard (docs/pilot-dashboard.html) reads as
# report.json; --pooled when the boxes pooled one policy (pilot.sh --pool), so a version is one player in every run.
# Run with LIVE_ENV=live/pilot.env. A dry run unless --yes (each fetch is one presigned S3 upload).
set -euo pipefail
LIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$LIVE_DIR/.." && pwd)"
YES=(); BOXES=(); SHARED=()
for a in "$@"; do
  case $a in
    --yes) YES=(--yes) ;;
    --dry-run) YES=() ;;
    --pooled) SHARED=(--shared-versions) ;;
    *=i-*) BOXES+=("$a") ;;
    *) echo "pilot-report.sh BOX=ID ...: not '$a'" >&2; exit 2 ;;
  esac
done
[ ${#BOXES[@]} -gt 0 ] || { echo "usage: pilot-report.sh BOX=ID [BOX=ID ...] [--yes]" >&2; exit 2; }
runs=()
for pair in "${BOXES[@]}"; do
  box=${pair%%=*} id=${pair#*=}
  [[ $box =~ ^[a-z0-9]+$ ]] || { echo "BOX must be letters or digits: '$box'" >&2; exit 2; }
  out=$(PILOT_BOX=$box bash "$LIVE_DIR/fetch.sh" "$id" --only pilot/report.json ${YES[@]+"${YES[@]}"} 2>&1) || { echo "$out" >&2; exit 1; }
  [ ${#YES[@]} -gt 0 ] || { echo "$out" | sed "s/^/[$box] /" >&2; continue; }
  archive=$(echo "$out" | sed -n 's/.*fetched to live\/fetched\/\([^ ]*\.tar\.zst\).*/\1/p' | tail -1)
  mkdir -p "$LIVE_DIR/fetched/pilot-$box"
  zstd -q -dc "$LIVE_DIR/fetched/$archive" | tar -x -C "$LIVE_DIR/fetched/pilot-$box" --strip-components 1 pilot/report.json
  rm -f "$LIVE_DIR/fetched/$archive"
  runs+=(--run "$box=$LIVE_DIR/fetched/pilot-$box/report.json")
done
[ ${#YES[@]} -gt 0 ] || { echo "+ combine the fetched reports into live/fetched/pilot-dashboard.json" >&2; exit 0; }
(cd "$REPO_DIR" && uv run --frozen python -m agentenv_zeroad.pilot.combine "${runs[@]}" ${SHARED[@]+"${SHARED[@]}"} --out "$LIVE_DIR/fetched/pilot-dashboard.json")
echo "live/fetched/pilot-dashboard.json: $((${#runs[@]} / 2)) run(s)"
