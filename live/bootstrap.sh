#!/usr/bin/env bash
# bootstrap.sh ID [--images build|pull] [--pilot|--train|--cpu] [--yes|--dry-run]
# Makes box ID a stream box: ships this checkout's HEAD (`git archive`; uncommitted changes don't go) and live/box
# with the box config from live.env through S3 (presigned, so the box needs no S3 rights), then runs box/bootstrap.sh
# through SSM: the GPU and driver check, Docker's address pools, Xorg on the L4, ffmpeg, mediamtx, uv, the images
# (`agent-env zeroad setup --live` and `--agents-only`, or pulled from LIVE_IMAGE_REGISTRY), the docker shim and the
# units. Nothing is streamed or played. About 30 min when the images are built. With --pilot it makes a post-training
# pilot box instead (box/pilot-bootstrap.sh: the headless env, vLLM serving Qwen, the trainer image; no display or
# stream; about 25 min). With --train it makes a post-training run box (g6e.12xlarge: two vLLM instances on GPU 0 and 1,
# the learner's GPUs 2 and 3 left free, a log-prob smoke check on both servers; live/train.sh runs the run). With --cpu
# it makes a CPU box on a plain Ubuntu 24.04 AMI (the S1 tool check, live/s1.sh; box/cpu-bootstrap.sh): no GPU or
# driver check, no Xorg, ffmpeg, mediamtx or vLLM; Docker, uv, the zeroad-engine image (--images build, about 10 min,
# or pull zeroad-engine:TAG from LIVE_IMAGE_REGISTRY), the repo and the box-time timer. A dry run unless --yes.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}; shift || true
check_instance_id "$ID"
IMAGES=${LIVE_IMAGES:-build}
PILOT=0 TRAIN=0 CPU=0
while [ $# -gt 0 ]; do
  case $1 in
    --images) IMAGES=${2:?--images needs build or pull}; shift 2 ;;
    --pilot) PILOT=1; shift ;;
    --train) PILOT=1 TRAIN=1; shift ;;
    --cpu) CPU=1; shift ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ "$CPU" = 0 ] || [ "$PILOT" = 0 ] || die "--cpu is not combined with --pilot or --train"
case $IMAGES in
  build) IMAGE_ARGS=(--images build) ;;
  pull)
    [ -n "${LIVE_IMAGE_REGISTRY:-}" ] || die "--images pull needs LIVE_IMAGE_REGISTRY in $LIVE_ENV"
    IMAGE_ARGS=(--images pull --registry "$LIVE_IMAGE_REGISTRY" --tag "${LIVE_IMAGE_TAG:-latest}") ;;
  *) die "--images must be build or pull" ;;
esac
[ "$DRY_RUN" = 1 ] && LIVE_BUCKET=${LIVE_BUCKET:-BUCKET-PLACEHOLDER}
need LIVE_BUCKET

# A box launched on a fallback type costs that type's rate, which launch.sh tagged on it; live.env's is the first type's.
if [ -n "${LIVE_GPU_RATES:-}" ]; then
  # shellcheck disable=SC2016  # the backticks are JMESPath literals
  tagged=$(aws_run "" ec2 describe-instances --instance-ids "$ID" --output text \
    --query 'Reservations[0].Instances[0].Tags[?Key==`zeroad-box-usd-per-hour`].Value|[0]')
  if [ -n "$tagged" ] && [ "$tagged" != None ]; then
    [ "$tagged" = "$RATE" ] || log "WARNING: $ID costs \$$tagged/h (its type's rate), not live.env's \$$RATE/h: the box is configured with $tagged, and the run config's box_usd_per_hour must be $tagged"
    RATE=$tagged
  fi
fi

[ -z "$(git -C "$REPO_DIR" status --porcelain)" ] || log "WARNING: uncommitted changes in $REPO_DIR are not shipped (HEAD is)"
rev=$(git -C "$REPO_DIR" rev-parse --short HEAD)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/payload/box"
cp -R "$LIVE_DIR/box/." "$tmp/payload/box/"
box_conf > "$tmp/payload/zeroad-live.conf"
tar -czf "$tmp/payload.tgz" -C "$tmp/payload" .
git -C "$REPO_DIR" archive --format=tar.gz -o "$tmp/repo.tgz" HEAD
log "shipping $rev ($(du -h "$tmp/repo.tgz" | cut -f1) code, $(du -h "$tmp/payload.tgz" | cut -f1) box payload)"

aws_run "" s3 cp "$tmp/payload.tgz" "s3://$LIVE_BUCKET/$PREFIX/box/payload.tgz" --only-show-errors
aws_run "" s3 cp "$tmp/repo.tgz" "s3://$LIVE_BUCKET/$PREFIX/box/repo-$rev.tgz" --only-show-errors
payload_url=$(aws_run https://PRESIGNED-payload-url s3 presign "s3://$LIVE_BUCKET/$PREFIX/box/payload.tgz" --expires-in 3600)
repo_url=$(aws_run https://PRESIGNED-repo-url s3 presign "s3://$LIVE_BUCKET/$PREFIX/box/repo-$rev.tgz" --expires-in 3600)
wait_ssm "$ID"
if [ "$CPU" = 1 ]; then
  ssm_box "$ID" 5400 cpu-bootstrap.sh "$payload_url" "$repo_url" "${IMAGE_ARGS[@]}"
  log "bootstrapped CPU box $ID at $rev. Next: live/s1.sh $ID start --yes"
  exit 0
fi
if [ "$PILOT" = 1 ]; then
  if [ "$TRAIN" = 1 ]; then
    ssm_box "$ID" 5400 pilot-bootstrap.sh "$payload_url" "$repo_url" --train
    log "bootstrapped training box $ID at $rev. Next: live/train-iam.sh create $ID --yes, then live/train.sh $ID start CONFIG --yes"
  else
    ssm_box "$ID" 5400 pilot-bootstrap.sh "$payload_url" "$repo_url"
    log "bootstrapped pilot box $ID at $rev. Next: live/pilot.sh $ID start --baseline --yes"
  fi
  exit 0
fi
ssm_box "$ID" 5400 bootstrap.sh "$payload_url" "$repo_url" "${IMAGE_ARGS[@]}"
log "bootstrapped $ID at $rev. Next: live/match.sh $ID rehearsal --yes (\$0)"
