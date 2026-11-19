#!/usr/bin/env bash
# poweroff.sh check|status|extend HOURS: whether the MAX_HOURS power-off is scheduled with systemd-logind (the
# zeroad-poweroff unit arms it after logind on every boot; see live/lib/poweroff-boothook.sh).
#   check         re-arms from /etc/zeroad-live/deadline if nothing is scheduled, then exits 1 loudly if it still is not
#   status        prints one line, "power-off armed: ..." or "power-off NOT ARMED: ...", and always exits 0
#   extend HOURS  moves the deadline HOURS later and re-arms from it (a new shutdown -h +N replaces the scheduled one),
#                 for a run that needs longer than the launch allowed; later boots keep the new deadline
set -uo pipefail
ARM=${ZEROAD_POWEROFF_ARM:-/usr/local/sbin/zeroad-poweroff-arm}
DEADLINE_FILE=${ZEROAD_DEADLINE_FILE:-/etc/zeroad-live/deadline}
MAX=18446744073709551615

# Prints the scheduled epoch seconds, or nothing when nothing is scheduled (type "" or USEC_INFINITY).
scheduled_epoch() {
  local out type usec
  out=$(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null) || return 0
  set -- $out
  type=${2:-}; type=${type//\"/}; usec=${!#:-}
  [[ $usec =~ ^[0-9]+$ ]] || return 0
  [ -n "$type" ] && [ "$usec" != "$MAX" ] && [ "$usec" -gt 0 ] && echo $(( usec / 1000000 ))
  return 0
}

report() {
  local at when
  at=$(scheduled_epoch)
  if [ -n "$at" ]; then
    when=$(date -u -d "@$at" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "epoch $at")
    echo "power-off armed: $when (in $(( (at - $(date +%s)) / 60 )) min)"
  else
    echo "power-off NOT ARMED: no shutdown is scheduled, so this box will not stop at MAX_HOURS (re-arm: sudo systemctl start zeroad-poweroff)"
    return 1
  fi
}

case ${1:-} in
  status) report || true ;;
  check)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ poweroff.sh check: re-arm with $ARM if no shutdown is scheduled, fail if still none"; exit 0; fi
    [ -z "$(scheduled_epoch)" ] && { echo "power-off not scheduled: arming it from the deadline file" >&2; "$ARM" >&2 || true; }
    if report; then exit 0; fi
    echo "ERROR: the MAX_HOURS power-off is NOT ARMED and re-arming failed; stop or terminate this box yourself" >&2
    exit 1
    ;;
  extend)
    hours=${2:-}
    [[ $hours =~ ^[1-9][0-9]?$ ]] || { echo "usage: poweroff.sh extend HOURS (1-99)" >&2; exit 2; }
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ move the deadline in $DEADLINE_FILE $hours h later, then $ARM"; exit 0; fi
    old=$(cat "$DEADLINE_FILE" 2>/dev/null) || { echo "ERROR: no deadline file $DEADLINE_FILE" >&2; exit 1; }
    [[ $old =~ ^[0-9]+$ ]] || { echo "ERROR: $DEADLINE_FILE does not hold epoch seconds" >&2; exit 1; }
    new=$(( old + hours * 3600 ))
    printf '%s\n' "$new" > "$DEADLINE_FILE.tmp" && mv "$DEADLINE_FILE.tmp" "$DEADLINE_FILE"
    echo "deadline $(date -u -d "@$old" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "$old") -> $(date -u -d "@$new" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "$new")"
    "$ARM" || { echo "ERROR: re-arming failed" >&2; exit 1; }
    report || exit 1
    ;;
  *) echo "usage: poweroff.sh check|status|extend HOURS" >&2; exit 2 ;;
esac
