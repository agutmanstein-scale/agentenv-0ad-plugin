#!/usr/bin/env bash
# repair.sh [--auto]: fixes what can be fixed without touching a running match, and says what it did:
#   the display down -> restart zeroad-x and re-allow local clients (a running match's engine loses its window: the
#   match must be stopped and started again); the relay down -> restart it (the env's RTMP leg does not reconnect,
#   so the recording of a running match resumes only with the next match); the push wanted but down -> start it;
#   free disk under ZEROAD_MIN_FREE_GB -> delete the oldest recording segments, never the newest.
# --auto (the half-hourly zeroad-health timer) also appends health.sh's line to $STATE/health.jsonl.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
AUTO=0
case ${1:-} in --auto) AUTO=1 ;; "") ;; *) echo "usage: repair.sh [--auto]" >&2; exit 2 ;; esac
did=0
if [ "$DRY_RUN" = 1 ] || [ "$(active zeroad-x)" != active ]; then
  say "repair: display :0 down -> restart zeroad-x"; run systemctl restart zeroad-x; did=1
  [ "$DRY_RUN" = 1 ] || [ "$(active zeroad-match)" != active ] || say "repair: WARNING a match is running on the old display: stop it and start it again"
fi
if [ "$DRY_RUN" = 1 ]; then say "+ display-ready.sh (xhost +local: on :0)"; else "$BOX_HOME/box/display-ready.sh" >/dev/null; fi
if [ "$DRY_RUN" = 1 ] || [ "$(active zeroad-relay)" != active ]; then
  say "repair: relay down -> restart zeroad-relay"; run systemctl restart zeroad-relay; did=1
fi
if [ "$DRY_RUN" = 1 ] || { [ -f "$STATE/push.wanted" ] && [ "$(active zeroad-push)" != active ]; }; then
  say "repair: push wanted but down -> start zeroad-push"; run systemctl start zeroad-push; did=1
fi
min=${ZEROAD_MIN_FREE_GB:-8}
if [ "$DRY_RUN" = 1 ]; then
  say "repair: if under $min GB free -> delete the oldest segments in $STATE/rec (never the newest)"
else
  while :; do
    free=$(df -BG --output=avail "$STATE" | tail -1 | tr -dc 0-9)
    [ "$free" -lt "$min" ] || break
    mapfile -t segs < <(find "$STATE/rec" -type f -name '*.mp4' -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)
    [ "${#segs[@]}" -gt 1 ] || { say "repair: $free GB free and no old segment left to delete"; break; }
    say "repair: $free GB free -> delete ${segs[0]}"; rm -f "${segs[0]}"; did=1
  done
fi
[ "$did" = 1 ] || say "repair: nothing to do"
if [ "$AUTO" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then say "+ health.sh >> $STATE/health.jsonl"; else mkdir -p "$STATE"; "$BOX_HOME/box/health.sh" >> "$STATE/health.jsonl"; fi
fi
