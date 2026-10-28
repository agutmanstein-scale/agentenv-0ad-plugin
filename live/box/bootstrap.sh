#!/usr/bin/env bash
# bootstrap.sh PAYLOAD_URL REPO_URL [--images build | --images pull --registry REG --tag TAG]
# Run as root on a fresh g6.4xlarge (Deep Learning Base OSS NVIDIA driver AMI, Ubuntu 22.04) by live/bootstrap.sh.
# Makes it a stream box: checks the L4 and its driver (the AMI's; it installs none), keeps the AMI's Docker and NVIDIA
# runtime and adds address pools, installs Xorg, ffmpeg, mediamtx and uv, unpacks the box scripts and the repo, builds
# (or pulls) the images and registers them with agent-env, installs the docker shim and the units, and starts the
# display, the relay and the half-hourly health timer. Nothing is played or pushed. Idempotent.
set -euo pipefail
PAYLOAD_URL=${1:?usage: bootstrap.sh PAYLOAD_URL REPO_URL [--images build|pull ...]}
REPO_URL=${2:?usage: bootstrap.sh PAYLOAD_URL REPO_URL [--images build|pull ...]}
shift 2
IMAGES=build REGISTRY='' TAG=latest
while [ $# -gt 0 ]; do
  case $1 in
    --images) IMAGES=$2; shift 2 ;;
    --registry) REGISTRY=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case $IMAGES in build) ;; pull) [ -n "$REGISTRY" ] || { echo "--images pull needs --registry" >&2; exit 2; } ;;
  *) echo "--images must be build or pull" >&2; exit 2 ;; esac

BOX_HOME=${ZEROAD_LIVE_HOME:-/opt/zeroad-live}
STATE=${ZEROAD_LIVE_STATE:-/var/lib/zeroad-live}
LOGDIR=${ZEROAD_LIVE_LOG:-/var/log/zeroad-live}
DRY_RUN=${DRY_RUN:-0}
MEDIAMTX=v1.21.2
UV=0.12.13
run() { if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
step() { echo "== $*"; }
export DEBIAN_FRONTEND=noninteractive

step "GPU and driver (the AMI's)"
run nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
if [ "$DRY_RUN" = 0 ]; then
  nvidia-smi -L | grep -q L4 || echo "WARNING: not an NVIDIA L4; the settings are tuned for the L4" >&2
  command -v nvidia-ctk >/dev/null || { echo "no NVIDIA container toolkit: use the Deep Learning Base OSS NVIDIA driver AMI" >&2; exit 1; }
  for _ in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break; sleep 5; done
fi
run systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer

step "procfs: hidepid=2, so the push's RTMP URL on ffmpeg's command line is visible to root only"
if [ "$DRY_RUN" = 1 ] || ! grep -q '^proc /proc proc .*hidepid=2' /etc/fstab; then
  run sh -c "sed -i '/^proc \/proc /d' /etc/fstab && echo 'proc /proc proc defaults,hidepid=2 0 0' >> /etc/fstab"
fi
run mount -o remount,hidepid=2 /proc

step "packages: Xorg (with the AMI's NVIDIA X driver), ffmpeg, tools"
run apt-get -o Acquire::Retries=5 update -qq
run apt-get -o Acquire::Retries=5 install -y -qq --no-install-recommends xserver-xorg-core x11-xserver-utils \
  x11-utils ffmpeg jq curl zstd python3
if [ "$DRY_RUN" = 1 ] || ! command -v uv >/dev/null || ! uv --version | grep -q "$UV"; then
  run sh -c "curl -LsSf https://astral.sh/uv/$UV/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
fi

step "Docker: the AMI's, with address pools for the env's networks (the NVIDIA runtime kept)"
run python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
d = json.loads(p.read_text()) if p.exists() else {}
if "default-address-pools" not in d:
    d["default-address-pools"] = [{"base": "172.20.0.0/14", "size": 24}]
    p.write_text(json.dumps(d, indent=2))
    print("added address pools")
PY
if [ "$DRY_RUN" = 1 ] || [ -z "$(docker ps -q)" ]; then run systemctl restart docker; fi

step "the box scripts and the repo"
run mkdir -p "$BOX_HOME/bin" "$STATE/rec" "$STATE/matches" "$STATE/agentenv" "$LOGDIR" /etc/zeroad-live
run sh -c 'rm -rf "$1/box.new" && mkdir -p "$1/box.new" && curl -fsSL "$2" | tar xz -C "$1/box.new"' _ "$BOX_HOME" "$PAYLOAD_URL"
run sh -c 'rm -rf "$1/box" && mv "$1/box.new/box" "$1/box" && install -m 644 "$1/box.new/zeroad-live.conf" /etc/zeroad-live.conf && rm -rf "$1/box.new"' _ "$BOX_HOME"
run sh -c 'rm -rf "$1/repo.new" && mkdir -p "$1/repo.new" && curl -fsSL "$2" | tar xz -C "$1/repo.new" && rm -rf "$1/repo" && mv "$1/repo.new" "$1/repo"' _ "$BOX_HOME" "$REPO_URL"
run uv sync --project "$BOX_HOME/repo" --frozen --no-dev
# agent-env passes the model endpoint to each agent from these variables, which with-model.sh sets per match.
run sh -c 'printf "[model]\nbase_url = \"env:LITELLM_BASE_URL\"\napi_key = \"env:LITELLM_API_KEY\"\n" > /etc/zeroad-live/agent-env.toml'
if [ ! -x "$BOX_HOME/bin/mediamtx" ] || [ "$DRY_RUN" = 1 ]; then
  run sh -c "curl -fsSL https://github.com/bluenviron/mediamtx/releases/download/$MEDIAMTX/mediamtx_${MEDIAMTX}_linux_amd64.tar.gz | tar xz -C $BOX_HOME/bin mediamtx"
fi
run install -m 755 "$BOX_HOME/box/docker-shim.sh" "$BOX_HOME/bin/docker"

step "images ($IMAGES) and agent-env registration (env zeroad-live, agents zeroad-commander and zeroad-luna)"
AE=(env XDG_STATE_HOME="$STATE/agentenv" AGENT_ENV_CONFIG=/etc/zeroad-live/agent-env.toml
    uv run --project "$BOX_HOME/repo" --frozen --no-dev agent-env)
if [ "$IMAGES" = pull ]; then
  run docker pull "$REGISTRY/mcp-server-zeroad-live:$TAG"
  run docker tag "$REGISTRY/mcp-server-zeroad-live:$TAG" mcp-server-zeroad-live
  run "${AE[@]}" zeroad setup --live --image mcp-server-zeroad-live
else
  run "${AE[@]}" zeroad setup --live --source "$BOX_HOME/repo"
fi
run "${AE[@]}" zeroad setup --agents-only --source "$BOX_HOME/repo"

step "units: the display on the L4, the relay (records every match), the push (not started), the health timer"
run sh -c 'cp "$1"/box/units/*.service "$1"/box/units/*.timer /etc/systemd/system/' _ "$BOX_HOME"
run systemctl daemon-reload
run systemctl enable --now zeroad-x zeroad-relay zeroad-health.timer zeroad-boxtime.timer
run systemctl disable zeroad-push
if [ "$DRY_RUN" = 0 ]; then
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "ffmpeg: $(ffmpeg -version | head -1)"
  echo "mediamtx: $("$BOX_HOME/bin/mediamtx" --version 2>&1 | head -1)"
  echo "units: $(systemctl is-active zeroad-x zeroad-relay zeroad-health.timer | paste -sd' ') (x relay health)"
fi
echo "bootstrap done; no match is running and nothing is pushed"
