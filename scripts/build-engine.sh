#!/usr/bin/env bash
# Build the 0 A.D. 0.28 engine images.
#   scripts/build-engine.sh [docker build args]      zeroad-engine:0.28 (ppa:wfg/0ad on ubuntu:24.04), the env's engine
#   scripts/build-engine.sh rt [docker build args]   also zeroad-engine:0.28-rt (patches/rl-realtime.patch, the stream's,
#                                                    and patches/rl-idle-wait.patch, which stops idle engines spinning
#                                                    and adds /step?state=0: training's engine; run its games on this
#                                                    image, not the PPA's). The image is labelled zeroad.patches; an
#                                                    rt image that already has that label is kept (REBUILD=1 forces).
# engine/Dockerfile matches the spike image so the docker layer cache is reused; the rt package build takes ~15 min the
# first time and is cached after that.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
RT_IMAGE=zeroad-engine:0.28-rt
RT_PATCHES="rl-realtime rl-idle-wait"
target=base
if [[ "${1:-}" == "rt" ]]; then
  target=rt
  shift
fi
docker build -t zeroad-engine:0.28 engine/ "$@"
if [[ "$target" == "rt" ]]; then
  have=$(docker image inspect -f '{{index .Config.Labels "zeroad.patches"}}' "$RT_IMAGE" 2>/dev/null || true)
  if [[ "$have" == "$RT_PATCHES" && "${REBUILD:-}" != 1 ]]; then
    echo "$RT_IMAGE already carries zeroad.patches=\"$have\"; not rebuilding (REBUILD=1 forces)"
  else
    docker build -t "$RT_IMAGE" -f engine/Dockerfile.rt patches/ "$@"
  fi
fi
