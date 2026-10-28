#!/usr/bin/env bash
# stop.sh [--push-only]: ends the push cleanly first (ffmpeg gets SIGINT and closes the FLV stream), then, unless
# --push-only, stops the match: run-match.sh forwards the SIGTERM to agent-env, which tears its containers down, and
# then books the match's spend (the env's own MP4 is lost when the match is cut short; the relay's recording up to
# that moment is kept). The display and the relay stay up.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
case ${1:-} in
  --push-only|"") ;;
  *) echo "usage: stop.sh [--push-only]" >&2; exit 2 ;;
esac
"$BOX_HOME/box/push.sh" stop
if [ "${1:-}" != --push-only ]; then
  if [ "$DRY_RUN" = 1 ] || [ "$(active zeroad-match)" = active ]; then
    run systemctl stop zeroad-match
  else
    say "no match running"
  fi
fi
say "units: $(for u in "${UNITS[@]}"; do printf '%s=%s ' "$u" "$(active "$u")"; done)"
