#!/usr/bin/env bash
# Build the 0 A.D. 0.28 engine images.
#   scripts/build-engine.sh [docker build args]      zeroad-engine:0.28 (ppa:wfg/0ad on ubuntu:24.04), the env's engine
#   scripts/build-engine.sh rt [docker build args]   also zeroad-engine:0.28-rt (patches/rl-realtime.patch, the stream's,
#                                                    and patches/rl-idle-wait.patch, which stops idle engines spinning
#                                                    and adds /step?state=0: training's engine; run its games on this
#                                                    image, not the PPA's). The image is labelled zeroad.patches; an
#                                                    rt image that already has that label is kept (REBUILD=1 forces).
#   scripts/build-engine.sh show [docker build args] also zeroad-engine:0.28-show (the three patches, the last being
#                                                    patches/rl-paced-render.patch: the showcase's paced render
#                                                    engine). It never replaces the rt image. Same label check.
# engine/Dockerfile matches the spike image so the docker layer cache is reused; the rt package build takes ~15 min the
# first time and is cached after that.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
RT_IMAGE=zeroad-engine:0.28-rt
RT_PATCHES="rl-realtime rl-idle-wait"
SHOW_IMAGE=zeroad-engine:0.28-show
SHOW_PATCHES="rl-realtime rl-idle-wait rl-paced-render"
target=base
if [[ "${1:-}" == "rt" || "${1:-}" == "show" ]]; then
  target=$1
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
if [[ "$target" == "show" ]]; then
  have=$(docker image inspect -f '{{index .Config.Labels "zeroad.patches"}}' "$SHOW_IMAGE" 2>/dev/null || true)
  if [[ "$have" == "$SHOW_PATCHES" && "${REBUILD:-}" != 1 ]]; then
    echo "$SHOW_IMAGE already carries zeroad.patches=\"$have\"; not rebuilding (REBUILD=1 forces)"
  else
    docker build -t "$SHOW_IMAGE" -f engine/Dockerfile.show patches/ "$@"
  fi
fi
