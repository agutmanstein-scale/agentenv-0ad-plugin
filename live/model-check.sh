#!/usr/bin/env bash
# model-check.sh ID [--dry-run]
# From box ID (box/model-check.sh): whether the LiteLLM proxies in live.env (LIVE_LITELLM_BASE_URL and the optional
# LIVE_LITELLM_ALT_BASE_URL; bootstrap.sh puts them in the box config) resolve, answer, and accept
# the key in SSM (put-secret.sh litellm-key) and the second one beside it (put-secret.sh litellm-key-alt). One GET
# /v1/models per proxy and key; no model call. Prints status codes only, never a key. Read-only.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode read "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ID=${1:-}
check_instance_id "$ID"
ssm_box "$ID" 120 model-check.sh
