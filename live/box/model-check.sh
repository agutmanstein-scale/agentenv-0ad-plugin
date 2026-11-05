#!/usr/bin/env bash
# model-check.sh: from this box, whether each LiteLLM proxy in the box config (LITELLM_BASE_URL, and
# LITELLM_ALT_BASE_URL when set) resolves, answers, and accepts each key in SSM ($ZEROAD_SSM_LITELLM_KEY and its -alt beside it): one GET /v1/models
# per proxy and key, no model call, nothing spent. A key goes to curl on stdin (-H @-), never on a command line, and
# only status codes are printed. Read-only.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
KEY_PARAM=${ZEROAD_SSM_LITELLM_KEY:-/zeroad-live/litellm-key}
PROXIES=()
for url in "${LITELLM_BASE_URL:-}" "${LITELLM_ALT_BASE_URL:-}"; do [ -z "$url" ] || PROXIES+=("${url%/}"); done
[ ${#PROXIES[@]} -gt 0 ] || die "no LITELLM_BASE_URL in $CONF (LIVE_LITELLM_BASE_URL in live.env)"
if [ "$DRY_RUN" = 1 ]; then
  for url in "${PROXIES[@]}"; do say "+ resolve ${url#https://}; GET $url/v1/models without a key, then with $KEY_PARAM and $KEY_PARAM-alt (status codes only)"; done
  exit 0
fi
keys=()
for name in "$KEY_PARAM" "$KEY_PARAM-alt"; do
  v=$(aws ssm get-parameter --name "$name" --with-decryption --query Parameter.Value --output text || true)
  if [ -n "$v" ] && [ "$v" != None ]; then keys+=("$name"); eval "key_$((${#keys[@]} - 1))=\$v"; else say "$name: not in SSM"; fi
  unset v
done
for url in "${PROXIES[@]}"; do
  host=${url#https://}; host=${host%%/*}
  say "$host: resolves to $(getent ahosts "$host" | awk 'NR==1 {print $1}' || true)"
  say "  no key: $(curl -sS -o /dev/null -w '%{http_code} in %{time_total}s' -m 15 "$url/v1/models" 2>&1 || true)"
  for i in "${!keys[@]}"; do
    var=key_$i
    code=$(printf 'Authorization: Bearer %s\n' "${!var}" | curl -sS -o /dev/null -w '%{http_code}' -m 15 -H @- "$url/v1/models" 2>&1 || true)
    say "  ${keys[$i]##*/}: $code"
  done
done
for i in "${!keys[@]}"; do unset "key_$i"; done
