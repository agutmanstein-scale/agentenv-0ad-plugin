#!/usr/bin/env bash
# Throwaway spike: start headless 0 A.D. 0.28 with the RL interface in Docker and wait for the port.
# 0.28 quits nonvisual mode unless an -autostart map is given; /reset replaces that map.
set -euo pipefail
docker rm -f zad-spike >/dev/null 2>&1 || true
docker run -d -t --init --name zad-spike -u 1000:1000 -e HOME=/tmp -e XDG_RUNTIME_DIR=/tmp --network host \
  -v "$(dirname "$(readlink -f "$0")")":/spike:ro zeroad-spike:0.28 \
  /usr/games/pyrogenesis --rl-interface=127.0.0.1:6000 -autostart-nonvisual -autostart=random/mainland \
  -autostart-size=128 -mod=public "$@" >/dev/null
for _ in $(seq 1 60); do docker exec zad-spike grep -qi ":1770" /proc/net/tcp && exit 0; sleep 2; done
echo "engine did not open port 6000" >&2; exit 1
