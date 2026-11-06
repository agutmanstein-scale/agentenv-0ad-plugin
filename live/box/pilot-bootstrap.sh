#!/usr/bin/env bash
# pilot-bootstrap.sh PAYLOAD_URL REPO_URL [--train]
# Run as root on a fresh GPU box (g6e.12xlarge or g6e.8xlarge, Deep Learning Base OSS NVIDIA driver AMI, Ubuntu 22.04)
# by live/bootstrap.sh --pilot. Makes it a post-training pilot box: no display, relay or stream; instead the headless
# env image (`agent-env zeroad setup`, the engine from the PPA) and the agents, vLLM serving Qwen with runtime LoRA
# loading (bound to the Docker bridge and loopback only, behind a key made here and kept in a root-only file), the
# trainer image (trainer/, on vLLM's image) and the box-time timer the budget guard reads. With 4 GPUs vLLM serves on
# 0 and 1 and the trainer takes 2 and 3; with fewer they share GPU 0, the driver stopping vLLM while it trains
# ($ETC/pilot.conf). It ends with one strategist call through vLLM that must come back with a tool call and its token
# ids. Nothing is played or trained. Idempotent.
# With --train (a 4-GPU 48 GB box, g6e.12xlarge) it makes a post-training run box instead: two vLLM instances, TP=1,
# bf16, on GPU 0 and GPU 1 (ports 8000 and 8001, both containers' --restart unless-stopped), sampling from vLLM's own
# generation config, 8 LoRA slots on the GPU and 32 in CPU memory, and runtime LoRA updating on; GPU 2 and 3 stay free
# for the learner. Both mount $STATE/train read-only at /train, where each run keeps its adapters. It ends with
# train.smoke against both: every reply must carry one log-prob per generated token.
set -euo pipefail
PAYLOAD_URL=${1:?usage: pilot-bootstrap.sh PAYLOAD_URL REPO_URL}
REPO_URL=${2:?usage: pilot-bootstrap.sh PAYLOAD_URL REPO_URL [--train]}
TRAIN=0
case ${3:-} in --train) TRAIN=1 ;; "") ;; *) echo "unknown argument: $3" >&2; exit 2 ;; esac
# A training box also installs the `train` extra (numpy, scikit-learn): the orchestrator's critic and board statistics
# run in the venv; a pilot or play-only box keeps the base install.
EXTRA=(); [ "$TRAIN" = 0 ] || EXTRA=(--extra train)
BOX_HOME=${ZEROAD_LIVE_HOME:-/opt/zeroad-live}
STATE=${ZEROAD_LIVE_STATE:-/var/lib/zeroad-live}
LOGDIR=${ZEROAD_LIVE_LOG:-/var/log/zeroad-live}
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
DRY_RUN=${DRY_RUN:-0}
UV=0.12.13
VLLM_IMAGE=vllm/vllm-openai:v0.11.0
MODEL=${ZEROAD_PILOT_MODEL:-Qwen/Qwen3-8B}
run() { if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
step() { echo "== $*"; }
export DEBIAN_FRONTEND=noninteractive

step "GPUs and driver (the AMI's)"
run nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPUS=4
if [ "$DRY_RUN" = 0 ]; then
  GPUS=$(nvidia-smi -L | wc -l)
  [ "$GPUS" -ge 1 ] || { echo "no GPU" >&2; exit 1; }
  command -v nvidia-ctk >/dev/null || { echo "no NVIDIA container toolkit: use the Deep Learning Base OSS NVIDIA driver AMI" >&2; exit 1; }
  for _ in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock >/dev/null 2>&1 || break; sleep 5; done
  if [ "$TRAIN" = 1 ] && { [ "$GPUS" -lt 4 ] || [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 40000 ]; }; then
    echo "--train needs 4 GPUs of 40 GB or more (a g6e.12xlarge)" >&2; exit 1
  fi
fi
run systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer

step "packages and uv"
run apt-get -o Acquire::Retries=5 update -qq
run apt-get -o Acquire::Retries=5 install -y -qq --no-install-recommends jq curl zstd python3
if [ "$DRY_RUN" = 1 ] || ! command -v uv >/dev/null || ! uv --version | grep -q "$UV"; then
  run sh -c "curl -LsSf https://astral.sh/uv/$UV/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
fi

step "Docker: the AMI's, with address pools for every game's env network"
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
run mkdir -p "$BOX_HOME/bin" "$STATE/pilot/adapters" "$STATE/train" "$STATE/agentenv" "$STATE/hf" "$LOGDIR" "$ETC"
run sh -c 'rm -rf "$1/box.new" && mkdir -p "$1/box.new" && curl -fsSL "$2" | tar xz -C "$1/box.new"' _ "$BOX_HOME" "$PAYLOAD_URL"
run sh -c 'rm -rf "$1/box" && mv "$1/box.new/box" "$1/box" && install -m 644 "$1/box.new/zeroad-live.conf" /etc/zeroad-live.conf && rm -rf "$1/box.new"' _ "$BOX_HOME"
run sh -c 'rm -rf "$1/repo.new" && mkdir -p "$1/repo.new" && curl -fsSL "$2" | tar xz -C "$1/repo.new" && rm -rf "$1/repo" && mv "$1/repo.new" "$1/repo"' _ "$BOX_HOME" "$REPO_URL"
run uv sync --project "$BOX_HOME/repo" --frozen --no-dev ${EXTRA[@]+"${EXTRA[@]}"}
# agent-env passes the model endpoint to each agent from these variables; pilot-run.sh sets them.
run sh -c 'printf "[model]\nbase_url = \"env:LITELLM_BASE_URL\"\napi_key = \"env:LITELLM_API_KEY\"\n" > "$1/agent-env.toml"' _ "$ETC"

step "images: the headless env (engine from the PPA) and the agents"
AE=(env XDG_STATE_HOME="$STATE/agentenv" AGENT_ENV_CONFIG="$ETC/agent-env.toml"
    uv run --project "$BOX_HOME/repo" --frozen --no-dev ${EXTRA[@]+"${EXTRA[@]}"} agent-env)
run "${AE[@]}" zeroad setup --source "$BOX_HOME/repo"
run "${AE[@]}" zeroad setup --agents-only --source "$BOX_HOME/repo"

# The agents' containers reach the host at the default bridge's gateway (172.17.0.1 unless Docker's address pools
# moved it); vLLM listens there and on loopback only, and its URL goes in $ETC/vllm.url for the pilot's scripts.
if [ "$DRY_RUN" = 1 ]; then BRIDGE=172.17.0.1; else
  BRIDGE=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
fi
if [ "$GPUS" -ge 4 ]; then SERVE=0,1 TP=2 TRAIN_GPUS=2,3 SHARE=0; else SERVE=0 TP=1 TRAIN_GPUS=0 SHARE=1; fi
# On a 24 GB GPU (A10G, L4) an 8B model in bf16 leaves about 3 GB for the request cache, too little for a round's
# games calling at once, and an L4's memory is slow for it (about 19 s a reply). FP8 weights halve both.
QUANT=()
if [ "$DRY_RUN" = 0 ] && [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 40000 ]; then
  QUANT=(--quantization fp8)
fi
# One vLLM container per entry: NAME|GPUS|HOST PORT|TENSOR PARALLEL. Its adapters mount and engine flags follow the mode.
INSTANCES=("zeroad-vllm|$SERVE|8000|$TP")
MOUNT="$STATE/pilot/adapters:/adapters:ro"
MAX_LEN=12288
VLLM_FLAGS=(--max-loras 4 --max-cpu-loras 16 ${QUANT[@]+"${QUANT[@]}"})
if [ "$TRAIN" = 1 ]; then
  INSTANCES=("zeroad-vllm-0|0|8000|1" "zeroad-vllm-1|1|8001|1")
  SERVE=0,1 TRAIN_GPUS=2,3 SHARE=0
  MOUNT="$STATE/train:/train:ro"
  VLLM_FLAGS=(--dtype bfloat16 --generation-config vllm --max-loras 8 --max-cpu-loras 32)
  # The learner (trainer/learner.py max_len) rejects a decision longer than 8192 tokens: nothing longer is played.
  MAX_LEN=8192
  # A pilot box's single vLLM holds GPUs 0 and 1 and port 8000: it goes before the two TP=1 instances come up.
  run docker rm -f zeroad-vllm 2>/dev/null || true
fi
step "vLLM ($VLLM_IMAGE): $MODEL on GPUs $SERVE (training on $TRAIN_GPUS, shared: $SHARE), at $BRIDGE:8000 and loopback only, behind a key made here"
run sh -c 'echo "http://$1:8000" > "$2/vllm.url"' _ "$BRIDGE" "$ETC"
PORTS=()
for instance in "${INSTANCES[@]}"; do IFS='|' read -r _ _ port _ <<<"$instance"; PORTS+=("$port"); done
run sh -c 'out=$1; shift; for p in "$@"; do echo "http://$0:$p"; done > "$out"' "$BRIDGE" "$ETC/vllm.urls" "${PORTS[@]}"
run sh -c 'printf "PILOT_TRAIN_GPUS=%s\nPILOT_SHARE_GPU=%s\n" "$1" "$2" > "$3/pilot.conf"' _ "$TRAIN_GPUS" "$SHARE" "$ETC"
if [ "$DRY_RUN" = 1 ]; then
  echo "+ make $ETC/vllm.env (0600): VLLM_API_KEY=<random>, unless it exists"
elif [ ! -s "$ETC/vllm.env" ]; then
  (umask 077; printf 'VLLM_API_KEY=%s\n' "$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$ETC/vllm.env")
fi
run docker pull -q "$VLLM_IMAGE"
for instance in "${INSTANCES[@]}"; do
  IFS='|' read -r NAME DEVICES PORT TPN <<<"$instance"
  # Recreated when it is not running, or runs with another --max-model-len than this mode's.
  if [ "$DRY_RUN" = 1 ] || [ -z "$(docker ps -q --filter name="^$NAME\$")" ] \
      || ! docker inspect --format '{{join .Args " "}}' "$NAME" 2>/dev/null | grep -q -- "--max-model-len $MAX_LEN "; then
    run docker rm -f "$NAME" >/dev/null 2>&1 || true
    run docker run -d --name "$NAME" --restart unless-stopped --gpus "\"device=$DEVICES\"" --ipc=host \
      --env-file "$ETC/vllm.env" -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True -p "$BRIDGE:$PORT:8000" -p "127.0.0.1:$PORT:8000" \
      -v "$STATE/hf:/root/.cache/huggingface" -v "$MOUNT" \
      "$VLLM_IMAGE" --model "$MODEL" --served-model-name local/qwen-base --tensor-parallel-size "$TPN" \
      --max-model-len $MAX_LEN --gpu-memory-utilization 0.9 --enable-lora --max-lora-rank 32 "${VLLM_FLAGS[@]}" \
      --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching
  fi
done

step "the trainer image (trainer/ on $VLLM_IMAGE, with PEFT; train-run.sh rebuilds it at every start)"
run docker build -q -t zeroad-trainer --build-arg "VLLM_IMAGE=$VLLM_IMAGE" "$BOX_HOME/repo/trainer"

if [ "$TRAIN" = 1 ]; then
  step "the training engine (zeroad-engine:0.28-rt, labelled zeroad.patches; the PPA source build takes ~15 min, none when already built)"
  run "$BOX_HOME/repo/scripts/build-engine.sh" rt
  if [ "$DRY_RUN" = 0 ]; then
    [ "$(docker image inspect -f '{{index .Config.Labels "zeroad.patches"}}' zeroad-engine:0.28-rt)" = "rl-realtime rl-idle-wait" ] \
      || { echo "zeroad-engine:0.28-rt is not labelled zeroad.patches=\"rl-realtime rl-idle-wait\"" >&2; exit 1; }
  fi
fi

step "the box-time timer (the budget guard's box hours)"
run sh -c 'cp "$1"/box/units/zeroad-boxtime.service "$1"/box/units/zeroad-boxtime.timer /etc/systemd/system/' _ "$BOX_HOME"
run systemctl daemon-reload
run systemctl enable --now zeroad-boxtime.timer

step "vLLM up, then one strategist call through it (a tool call and its token ids$([ "$TRAIN" = 1 ] && echo ', and one log-prob per token') must come back)"
if [ "$DRY_RUN" = 1 ]; then
  echo "+ wait for http://$BRIDGE:<each vLLM port>/health (the model downloads once, about 16 GB), then $([ "$TRAIN" = 1 ] && echo train.smoke || echo pilot.smoke)"
else
  for port in "${PORTS[@]}"; do
    for i in $(seq 1 180); do
      curl -fsS "http://$BRIDGE:$port/health" >/dev/null 2>&1 && break
      [ "$i" = 180 ] && { docker logs --tail 40 "$(docker ps -a --filter "publish=$port" --format '{{.Names}}' | head -1)" >&2; echo "vLLM on $port never came up" >&2; exit 1; }
      sleep 10
    done
  done
  set -a; source "$ETC/vllm.env"; set +a
  for port in "${PORTS[@]}"; do
    smoke=agentenv_zeroad.pilot.smoke; [ "$TRAIN" = 1 ] && smoke=agentenv_zeroad.train.smoke
    uv run --project "$BOX_HOME/repo" --frozen --no-dev ${EXTRA[@]+"${EXTRA[@]}"} python -m "$smoke" --url "http://$BRIDGE:$port"
  done
  docker run --rm curlimages/curl:8.10.1 -fsS -o /dev/null -w 'a container reaches vLLM: %{http_code}\n' "http://$BRIDGE:${PORTS[0]}/health"
  echo "gpus: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' ')"
fi
step "power-off armed (zeroad-poweroff, from the launch deadline)"
run "$BOX_HOME/box/poweroff.sh" check
if [ "$TRAIN" = 1 ]; then
  echo "training box done; nothing is played or trained (live/train.sh ID start CONFIG --yes)"
else
  echo "pilot bootstrap done; nothing is played or trained (live/pilot.sh ID start --baseline)"
fi
