#!/usr/bin/env bash
# display-ready.sh: zeroad-x's ExecStartPost (and repair.sh): waits up to 30 s for display :0, then lets local
# clients (the env container through the mounted socket) draw on it. X listens on no TCP port (-nolisten tcp), and
# the box has no inbound rule, so "local" means processes and containers on this box only.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
if [ "$DRY_RUN" = 0 ]; then
  for _ in $(seq 1 "${ZEROAD_DISPLAY_WAIT_S:-30}"); do [ -S /tmp/.X11-unix/X0 ] && DISPLAY=:0 xdpyinfo >/dev/null 2>&1 && break; sleep 1; done
fi
run env DISPLAY=:0 xhost +local:
