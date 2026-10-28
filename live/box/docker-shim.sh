#!/usr/bin/env bash
# docker-shim.sh, installed as /opt/zeroad-live/bin/docker and first on PATH for the match only (run-match.sh).
# agent-env's server provider creates the env container with a fixed set of arguments (its name, port, --add-host
# and three variables), with no way to add a device or a mount. The stream env needs the L4 and the box's Xorg, so
# for a `docker run` or `docker create` of the live env image (mcp-server-zeroad-live) this adds: the GPU with every
# driver capability (GLX for the engine, NVENC for ffmpeg), the X socket, a 2 GB /dev/shm for Chrome, and the
# variables the image reads (src/agentenv_zeroad/live.py): ZEROAD_DISPLAY, ZEROAD_ENCODER=nvenc, full graphics and
# ZEROAD_RTMP_URL = this box's relay (the stream key never enters the container; push.sh sends the relay on).
# Every other docker command passes through unchanged.
set -euo pipefail
REAL=${ZEROAD_REAL_DOCKER:-/usr/bin/docker}
LOG=${ZEROAD_SHIM_LOG:-/var/log/zeroad-live/docker-shim.log}
sub=${1:-}
if [ "$sub" = run ] || [ "$sub" = create ]; then
  live=0
  for a in "${@:2}"; do
    if [[ $a =~ ^([^[:space:]=]+/)?mcp-server-zeroad-live(:[A-Za-z0-9._-]+)?$ ]]; then live=1; fi
  done
  if [ "$live" = 1 ]; then
    inject=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size 2g
            -e DISPLAY=:0 -e ZEROAD_DISPLAY=:0 -e ZEROAD_ENCODER=nvenc -e ZEROAD_GRAPHICS=high
            -e ZEROAD_RTMP_URL=rtmp://host.docker.internal:1935/zeroad)
    { echo "$(date -u +%FT%TZ) $sub of the live env: added ${inject[*]}" >> "$LOG"; } 2>/dev/null || true
    exec "$REAL" "$sub" "${inject[@]}" "${@:2}"
  fi
fi
exec "$REAL" "$@"
