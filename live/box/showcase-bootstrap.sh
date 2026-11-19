#!/usr/bin/env bash
# showcase-bootstrap.sh PAYLOAD_URL REPO_URL
# Run as root on a fresh g6e.2xlarge (one 48 GB L40S, Deep Learning Base OSS NVIDIA driver AMI, Ubuntu 22.04) by
# live/bootstrap.sh --showcase. Makes it the showcase box:
#   1. the GPU check (one GPU of 40 GB or more; nothing is changed on a smaller box);
#   2. the stream setup of box/bootstrap.sh: Xorg, ffmpeg, mediamtx, uv, the display, the relay and the units (it runs
#      that script with --images none: the images below are built on the show engine instead);
#   3. the show engine (scripts/build-engine.sh show: zeroad-engine:0.28-show, with the paced-render patch), the env image
#      on it and the live image on that, tagged zeroad-showcase (the manual builds of streamer/Dockerfile);
#   4. one vLLM, as Run 1 served it (box/pilot-bootstrap.sh --train's flags) but for a smaller memory share and one LoRA
#      slot, behind a key made in a root-only file;
#   5. Run 1's adapters/v50 copied from S3 to $STATE/showcase/adapters/v1 and checked against the sizes and checksums
#      S3 holds for its files;
#   6. the adapter loaded as local/policy-v1 and one strategist call through it, which must come back with a tool call
#      and token ids;
#   7. the paced engine's test (tests/engine/test_paced_engine.py);
#   8. the GPU memory in use.
# Nothing is played or streamed. The bucket's name (ZEROAD_S3_BUCKET in the box config) is never printed. Idempotent.
set -euo pipefail
PAYLOAD_URL=${1:?usage: showcase-bootstrap.sh PAYLOAD_URL REPO_URL}
REPO_URL=${2:?usage: showcase-bootstrap.sh PAYLOAD_URL REPO_URL}
[ $# -eq 2 ] || { echo "usage: showcase-bootstrap.sh PAYLOAD_URL REPO_URL" >&2; exit 2; }
BOX_HOME=${ZEROAD_LIVE_HOME:-/opt/zeroad-live}
STATE=${ZEROAD_LIVE_STATE:-/var/lib/zeroad-live}
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
DRY_RUN=${DRY_RUN:-0}
VLLM_IMAGE=vllm/vllm-openai:v0.11.0
MODEL=${ZEROAD_PILOT_MODEL:-Qwen/Qwen3-8B}
SHOW_ENGINE=zeroad-engine:0.28-show
IMAGE=zeroad-showcase
POLICY=local/policy-v1
RUN1_ADAPTER=run/s2-run1/adapters/v50
ADAPTER_DIR=$STATE/showcase/adapters/v1
run() { if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }
step() { echo "== $*"; }

step "GPU (one of 40 GB or more)"
GPUS=1 MEM=46068
run nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
if [ "$DRY_RUN" = 0 ]; then
  GPUS=$(nvidia-smi -L | wc -l)
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "$GPUS" -eq 1 ] || { echo "the showcase box has one GPU (a g6e.2xlarge); this one has $GPUS" >&2; exit 1; }
  [ "$MEM" -ge 40000 ] || { echo "the showcase needs a GPU of 40 GB or more (an L40S); this one has $MEM MiB" >&2; exit 1; }
fi

step "the box scripts (the stream setup below unpacks them again, and the repo)"
run mkdir -p "$BOX_HOME/bin" "$ETC"
run sh -c 'rm -rf "$1/box.new" && mkdir -p "$1/box.new" && curl -fsSL "$2" | tar xz -C "$1/box.new"' _ "$BOX_HOME" "$PAYLOAD_URL"
run sh -c 'rm -rf "$1/box" && mv "$1/box.new/box" "$1/box" && install -m 644 "$1/box.new/zeroad-live.conf" /etc/zeroad-live.conf && rm -rf "$1/box.new"' _ "$BOX_HOME"
# shellcheck source=lib.sh
source "$BOX_HOME/box/lib.sh"
BUCKET=${ZEROAD_S3_BUCKET:-}
# hide: whatever aws or docker prints goes through this, so the bucket's name never reaches the output.
hide() { if [ -n "$BUCKET" ]; then sed "s#$BUCKET#<bucket>#g"; else cat; fi; }

step "the stream setup (box/bootstrap.sh --images none): Xorg on the GPU, ffmpeg, mediamtx, uv, the display, the relay"
# Not through run: in a dry run it prints its own plan, which belongs in this one.
bash "$BOX_HOME/box/bootstrap.sh" "$PAYLOAD_URL" "$REPO_URL" --images none

step "the images: the show engine ($SHOW_ENGINE, the paced-render patch), the env on it, the live image as $IMAGE"
GAME_ENV=$(sed -n 's/^ *"agentenv-game-env @ \([^"]*\)".*/\1/p' "$REPO/pyproject.toml" | head -1)
[ -n "$GAME_ENV" ] || die "no agentenv-game-env archive in $REPO/pyproject.toml"
run "$REPO/scripts/build-engine.sh" show
require_engine_image "$SHOW_ENGINE" rl-paced-render
run docker build -q --build-arg "GAME_ENV=$GAME_ENV" --build-arg "ENGINE_IMAGE=$SHOW_ENGINE" -t "$IMAGE-env" "$REPO"
run docker build -q -f "$REPO/streamer/Dockerfile" --build-arg "ENV_IMAGE=$IMAGE-env" -t "$IMAGE" "$REPO"

if [ "$DRY_RUN" = 1 ]; then BRIDGE=172.17.0.1; else
  BRIDGE=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
fi
# Run 1's engine flags (pilot-bootstrap.sh --train): only the memory share and the LoRA slot count differ, because the
# renderer shares the GPU and one adapter is served. Neither changes what the model sees.
MEM_SHARE=0.55 MAX_LEN=8192
VLLM_FLAGS=(--dtype bfloat16 --generation-config vllm --max-loras 1 --max-cpu-loras 32)
step "vLLM ($VLLM_IMAGE): $MODEL on the GPU at $BRIDGE:8000 and loopback only, behind a key made here, $MEM_SHARE of its memory"
run sh -c 'echo "http://$1:8000" > "$2/vllm.url"' _ "$BRIDGE" "$ETC"
run mkdir -p "$STATE/showcase/adapters" "$STATE/hf"
make_vllm_key "$ETC"
run docker pull -q "$VLLM_IMAGE"
start_vllm zeroad-vllm 0 8000 1 "$STATE/showcase/adapters:/adapters:ro" "$MEM_SHARE" "$MAX_LEN" "${VLLM_FLAGS[@]}"

# verify_adapter: every object under Run 1's v50 is in ADAPTER_DIR with the size S3 holds and a matching checksum. S3's
# ETag is the MD5 of an object uploaded in one part, and for a multipart one (the CLI's parts of 8 MiB and up) the MD5 of
# its parts' MD5s and the part count: both are recomputed here. Exits 0 when all match, 1 with the first difference.
verify_adapter() {
  local listing rc=0
  listing=$(mktemp)
  if ! aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$PREFIX/$RUN1_ADAPTER/" --output json > "$listing" 2> "$listing.err"; then
    hide < "$listing.err" >&2
    rm -f "$listing" "$listing.err"
    return 1
  fi
  python3 - "$listing" "$ADAPTER_DIR" "$PREFIX/$RUN1_ADAPTER/" <<'PY' || rc=$?
import hashlib
import json
import sys
from pathlib import Path

listing, root, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
files = [o for o in json.loads(listing.read_text()).get("Contents") or [] if not o["Key"].endswith("/")]
if not files:
    sys.exit("no adapter files in S3")
MIB = 1 << 20


def etags_of(path, parts):
    """The ETags S3 would give `path` uploaded in one part (parts == 0) or in `parts` parts of each size the CLI uses."""
    if parts == 0:
        digest = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(MIB), b""):
                digest.update(block)
        return {digest.hexdigest()}
    found = set()
    for part_mib in (8, 16, 32, 64, 128, 256):
        if -(-path.stat().st_size // (part_mib * MIB)) != parts:
            continue
        sums = []
        with path.open("rb") as f:
            for block in iter(lambda: f.read(part_mib * MIB), b""):
                sums.append(hashlib.md5(block).digest())
        found.add(f"{hashlib.md5(b''.join(sums)).hexdigest()}-{parts}")
    return found


total = 0
for o in files:
    name = o["Key"][len(prefix):]
    path = root / name
    if not path.is_file():
        sys.exit(f"missing: {name}")
    if path.stat().st_size != o["Size"]:
        sys.exit(f"{name}: {path.stat().st_size} bytes here, {o['Size']} in S3")
    etag = o["ETag"].strip('"')
    if etag not in etags_of(path, int(etag.split("-")[1]) if "-" in etag else 0):
        sys.exit(f"{name}: checksum differs from S3 (ETag {etag})")
    total += o["Size"]
names = {o["Key"][len(prefix):] for o in files}
for needed in ("adapter_config.json", "adapter_model.safetensors"):
    if needed not in names:
        sys.exit(f"{needed} is not in the adapter")
print(f"adapter checked: {len(files)} files, {total} bytes, sizes and checksums match S3")
PY
  rm -f "$listing" "$listing.err"
  return "$rc"
}

step "Run 1's adapter (adapters/v50) from S3 to $ADAPTER_DIR, checked against the sizes and checksums in S3"
PREFIX=${ZEROAD_S3_PREFIX:-zeroad-train}
if [ "$DRY_RUN" = 1 ]; then
  echo "+ aws s3 cp --recursive s3://<bucket>/$PREFIX/$RUN1_ADAPTER/ $ADAPTER_DIR (unless the copy there already checks out)"
  echo "+ aws s3api list-objects-v2 --prefix $PREFIX/$RUN1_ADAPTER/, then compare each file's size and checksum (ETag) with the copy"
else
  [ -n "$BUCKET" ] || die "no ZEROAD_S3_BUCKET in $CONF: bootstrap.sh --showcase writes it from LIVE_BUCKET"
  if ! verify_adapter >/dev/null 2>&1; then
    mkdir -p "$ADAPTER_DIR"
    if ! aws s3 cp --recursive --only-show-errors "s3://$BUCKET/$PREFIX/$RUN1_ADAPTER/" "$ADAPTER_DIR" 2>&1 | hide; then
      die "the copy of the adapter from S3 failed"
    fi
  fi
  verify_adapter || die "the adapter on this box does not match S3: $ADAPTER_DIR"
fi

step "vLLM up, the adapter loaded as $POLICY, then one strategist call through it (a tool call and token ids must come back)"
if [ "$DRY_RUN" = 1 ]; then
  echo "+ wait for http://$BRIDGE:8000/health (the model downloads once, about 16 GB)"
  echo "+ POST http://$BRIDGE:8000/v1/load_lora_adapter {\"lora_name\": \"$POLICY\", \"lora_path\": \"/adapters/v1\"} (key from $ETC/vllm.env, never printed), unless /v1/models lists it"
  echo "+ uv run python -m agentenv_zeroad.pilot.smoke --url http://$BRIDGE:8000 --model $POLICY"
else
  wait_vllm_up "$BRIDGE" 8000
  set -a; source "$ETC/vllm.env"; set +a
  headers=$(umask 077; mktemp)
  trap 'rm -f "$headers"' EXIT
  printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$VLLM_API_KEY" > "$headers"
  if ! curl -fsS -H "@$headers" "http://$BRIDGE:8000/v1/models" | grep -q "\"$POLICY\""; then
    curl -fsS -H "@$headers" -d "{\"lora_name\": \"$POLICY\", \"lora_path\": \"/adapters/v1\"}" \
      "http://$BRIDGE:8000/v1/load_lora_adapter" >/dev/null || die "vLLM would not load the adapter as $POLICY"
  fi
  uv run --project "$REPO" --frozen --no-dev python -m agentenv_zeroad.pilot.smoke \
    --url "http://$BRIDGE:8000" --model "$POLICY"
fi

step "the paced engine's test, on $SHOW_ENGINE with the repo's code (pytest -m engine tests/engine/test_paced_engine.py)"
# The test launches engine containers through Docker, so it runs here with the repo's dev environment, as the engine
# tests do; the engine container it starts is the image built above.
run_engine_test "$REPO" tests/engine/test_paced_engine.py

step "box time and the power-off"
enable_boxtime_timer
run "$BOX_HOME/box/poweroff.sh" check

if [ "$DRY_RUN" = 1 ]; then
  echo "+ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
else
  echo "gpu memory used: $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | paste -sd' ')"
fi
echo "showcase bootstrap done; nothing is played or streamed"
