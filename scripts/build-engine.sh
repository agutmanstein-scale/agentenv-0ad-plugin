#!/usr/bin/env bash
# Build the 0 A.D. 0.28 engine images.
#   scripts/build-engine.sh [docker build args]      zeroad-engine:0.28 (ppa:wfg/0ad on ubuntu:24.04), the env's engine
#   scripts/build-engine.sh rt [docker build args]   also zeroad-engine:0.28-rt (patches/rl-realtime.patch, the stream's,
#                                                    and patches/rl-idle-wait.patch, which stops idle engines spinning:
#                                                    training's engine; run its games on this image, not the PPA's)
# engine/Dockerfile matches the spike image so the docker layer cache is reused; the rt package build takes ~15 min the
# first time and is cached after that.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
target=base
if [[ "${1:-}" == "rt" ]]; then
  target=rt
  shift
fi
docker build -t zeroad-engine:0.28 engine/ "$@"
if [[ "$target" == "rt" ]]; then
  docker build -t zeroad-engine:0.28-rt -f engine/Dockerfile.rt patches/ "$@"
fi
