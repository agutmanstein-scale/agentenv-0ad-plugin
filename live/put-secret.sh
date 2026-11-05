#!/usr/bin/env bash
# put-secret.sh litellm-key|litellm-key-alt|rtmp-server|rtmp-key [--yes|--dry-run]
# Reads one secret without echoing it and stores it as a SecureString at $LIVE_SSM_PREFIX/<name>. The value travels
# to aws in a JSON body, never on a command line, and never reaches the terminal, the shell history or a log. The body
# is a 0600 file in a fresh 0700 temporary folder, removed on exit: the macOS aws CLI reads no JSON from a pipe
# (file:///dev/stdin arrives empty). rtmp-server is the ingest URL (X Media Studio: Producer, Sources, RTMP: rtmps://<region>.pscp.tv:443/x);
# rtmp-key is the stream key ("-" when the server URL already holds it). litellm-key-alt is a second LiteLLM key that
# only model-check.sh reads, to compare two keys from a box.
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
parse_mode change "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
case ${1:-} in litellm-key|litellm-key-alt|rtmp-server|rtmp-key) ;; *) echo "put-secret.sh: litellm-key, litellm-key-alt, rtmp-server or rtmp-key" >&2; usage ;; esac
[ $# -le 1 ] || { echo "unknown argument: $2" >&2; usage; }
NAME=$SSM_PREFIX/$1
if [ "$DRY_RUN" = 1 ]; then
  plan "read the value for $NAME without echo"
  plan "aws ssm put-parameter --cli-input-json file://<private temp file>   ({Name: $NAME, Type: SecureString, Overwrite: true, Value: <read>})"
  exit 0
fi
read -rsp "value for $NAME: " v; echo >&2
[ -n "$v" ] || die "empty value; nothing stored"
[ "$1" != rtmp-server ] || [[ $v =~ ^rtmps?:// ]] || die "rtmp-server must be an rtmp:// or rtmps:// URL; nothing stored"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(umask 077; printf '%s' "$v" | python3 -c '
import json, sys
print(json.dumps({"Name": sys.argv[1], "Type": "SecureString", "Overwrite": True, "Value": sys.stdin.read()}))
' "$NAME" > "$tmp/body.json")
unset v
aws_run "" ssm put-parameter --cli-input-json "file://$tmp/body.json" --query Version --output text >/dev/null
log "stored $NAME"
