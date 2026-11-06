#!/usr/bin/env bash
# cpu-bootstrap.sh PAYLOAD_URL REPO_URL [--images build | --images pull --registry REG --tag TAG]
# Run as root on a fresh CPU box (a c7i.12xlarge or like, plain Ubuntu 24.04 AMI) by live/bootstrap.sh --cpu. Makes it
# the S1 tool-check box: no GPU or driver check, no Xorg, ffmpeg, mediamtx or vLLM, no display or stream. It installs
# Docker (the engines run in it, on the host network), jq, zstd and uv, unpacks the box scripts and the repo, builds the
# engine images (zeroad-engine:0.28 and the idle-wait zeroad-engine:0.28-rt the batch runs on, scripts/build-engine.sh
# rt, about 10 min plus the engine's own build) or pulls REG/zeroad-engine:TAG and REG/zeroad-engine:TAG-rt and tags them so,
# and starts the box-time timer the budget guard reads. Nothing is played. Idempotent.
set -euo pipefail
USAGE='usage: cpu-bootstrap.sh PAYLOAD_URL REPO_URL [--images build | --images pull --registry REG --tag TAG]'
PAYLOAD_URL=${1:?$USAGE}
REPO_URL=${2:?$USAGE}
shift 2
IMAGES=build REGISTRY='' TAG=latest
while [ $# -gt 0 ]; do
  case $1 in
    --images) IMAGES=${2:?$USAGE}; shift 2 ;;
    --registry) REGISTRY=${2:?$USAGE}; shift 2 ;;
    --tag) TAG=${2:?$USAGE}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case $IMAGES in build) ;; pull) [ -n "$REGISTRY" ] || { echo "--images pull needs --registry" >&2; exit 2; } ;;
  *) echo "--images must be build or pull" >&2; exit 2 ;; esac

BOX_HOME=${ZEROAD_LIVE_HOME:-/opt/zeroad-live}
STATE=${ZEROAD_LIVE_STATE:-/var/lib/zeroad-live}
LOGDIR=${ZEROAD_LIVE_LOG:-/var/log/zeroad-live}
DRY_RUN=${DRY_RUN:-0}
UV=0.12.13
ENGINE_IMAGE=zeroad-engine:0.28
RT_IMAGE=zeroad-engine:0.28-rt  # patches/rl-idle-wait.patch: an engine waiting for /step sleeps (the PPA's spins a core)
run() { if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
step() { echo "== $*"; }
export DEBIAN_FRONTEND=noninteractive

step "the box: CPUs, memory, and the cgroup version the calibration reads (no GPU here)"
if [ "$DRY_RUN" = 1 ]; then
  echo "+ nproc; free -m; stat -fc %T /sys/fs/cgroup"
else
  echo "cpus: $(nproc), memory: $(free -m | awk '/^Mem:/ {print $2}') MB, cgroup: $(stat -fc %T /sys/fs/cgroup)"
  for _ in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break; sleep 5; done
fi
run systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer

step "packages: Docker, jq, curl, zstd, uv"
run apt-get -o Acquire::Retries=5 update -qq
run apt-get -o Acquire::Retries=5 install -y -qq --no-install-recommends docker.io jq curl zstd python3 ca-certificates
run systemctl enable --now docker
if [ "$DRY_RUN" = 1 ] || ! command -v uv >/dev/null || ! uv --version | grep -q "$UV"; then
  run sh -c "curl -LsSf https://astral.sh/uv/$UV/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
fi

step "the box scripts and the repo"
run mkdir -p "$BOX_HOME/bin" "$STATE/s1" "$LOGDIR" /etc/zeroad-live
run sh -c 'rm -rf "$1/box.new" && mkdir -p "$1/box.new" && curl -fsSL "$2" | tar xz -C "$1/box.new"' _ "$BOX_HOME" "$PAYLOAD_URL"
run sh -c 'rm -rf "$1/box" && mv "$1/box.new/box" "$1/box" && install -m 644 "$1/box.new/zeroad-live.conf" /etc/zeroad-live.conf && rm -rf "$1/box.new"' _ "$BOX_HOME"
run sh -c 'rm -rf "$1/repo.new" && mkdir -p "$1/repo.new" && curl -fsSL "$2" | tar xz -C "$1/repo.new" && rm -rf "$1/repo" && mv "$1/repo.new" "$1/repo"' _ "$BOX_HOME" "$REPO_URL"
run uv sync --project "$BOX_HOME/repo" --frozen --no-dev

step "the engine image ($IMAGES): $ENGINE_IMAGE, 0 A.D. 0.28 on ubuntu:24.04, what the lockstep runner starts per engine"
if [ "$IMAGES" = pull ]; then
  run docker pull "$REGISTRY/zeroad-engine:$TAG"
  run docker tag "$REGISTRY/zeroad-engine:$TAG" "$ENGINE_IMAGE"
  run docker pull "$REGISTRY/zeroad-engine:$TAG-rt"
  run docker tag "$REGISTRY/zeroad-engine:$TAG-rt" "$RT_IMAGE"
else
  run sh -c 'cd "$1/repo" && scripts/build-engine.sh rt' _ "$BOX_HOME"
fi
run docker run --rm "$ENGINE_IMAGE" test -x /usr/games/pyrogenesis
run docker run --rm "$RT_IMAGE" test -x /usr/games/pyrogenesis

step "the box-time timer (the budget guard's box hours)"
run sh -c 'cp "$1"/box/units/zeroad-boxtime.service "$1"/box/units/zeroad-boxtime.timer /etc/systemd/system/' _ "$BOX_HOME"
run systemctl daemon-reload
run systemctl enable --now zeroad-boxtime.timer
run "$BOX_HOME/box/poweroff.sh" check
echo "cpu bootstrap done; nothing is played (live/s1.sh ID start)"
