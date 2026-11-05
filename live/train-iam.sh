#!/usr/bin/env bash
# train-iam.sh create ID... | delete ID... [--yes|--dry-run]
# The post-training run's IAM role, so its box can copy a run to S3 with no laptop in the loop (the orchestrator's
# `aws s3 sync` every 15 minutes uses the box's instance role). create makes the role and instance profile
# zeroad-train-box: AmazonSSMManagedInstanceCore, read/write on s3://LIVE_BUCKET/LIVE_S3_PREFIX/run/* and read on the
# LIVE_SSM_PREFIX parameters only, and puts it on each box ID in place of its current profile. delete puts each running
# box back on LIVE_INSTANCE_PROFILE and removes the role. This is pilot-iam.sh with --train. Run with
# LIVE_ENV=live/train.env. A dry run unless --yes.
exec bash "$(dirname "$0")/pilot-iam.sh" "$@" --train
