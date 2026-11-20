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
# A label of patch names cannot tell that a patch's content changed, so the image also carries a hash of the patch files
# in the order the Dockerfile applies them.
patch_sha() {
  local name
  for name in "$@"; do cat "patches/$name.patch"; done | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-16
}
# build_patched IMAGE DOCKERFILE NAMES [docker build args]: skip only when names and hash both match.
build_patched() {
  local image=$1 dockerfile=$2 names=$3 have have_sha sha
  shift 3
  # shellcheck disable=SC2086  # NAMES is a space-separated list
  sha=$(patch_sha $names)
  have=$(docker image inspect -f '{{index .Config.Labels "zeroad.patches"}}' "$image" 2>/dev/null || true)
  have_sha=$(docker image inspect -f '{{index .Config.Labels "zeroad.patches.sha256"}}' "$image" 2>/dev/null || true)
  if [[ "$have" == "$names" && "$have_sha" == "$sha" && "${REBUILD:-}" != 1 ]]; then
    echo "$image already carries zeroad.patches=\"$have\" and zeroad.patches.sha256=$have_sha; not rebuilding (REBUILD=1 forces)"
  else
    docker build -t "$image" -f "$dockerfile" --label zeroad.patches.sha256="$sha" patches/ "$@"
  fi
}
if [[ "$target" == "rt" ]]; then
  build_patched "$RT_IMAGE" engine/Dockerfile.rt "$RT_PATCHES" "$@"
fi
if [[ "$target" == "show" ]]; then
  build_patched "$SHOW_IMAGE" engine/Dockerfile.show "$SHOW_PATCHES" "$@"
fi
