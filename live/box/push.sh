#!/usr/bin/env bash
# push.sh start|stop|run: the push of the relay (rtmp://127.0.0.1:1935/zeroad, the match as the env encodes it) to
# the RTMP target, X or any rtmp(s):// server.
#   start  checks that both SSM parameters exist (without reading their values), marks the push wanted and starts
#          zeroad-push.service. This makes the stream public: live/push.sh and live/match.sh --push ask for --yes.
#   stop   unmarks it and stops the unit; the unit's stop reaches ffmpeg as SIGINT, which ends the FLV stream
#          cleanly. The match, the relay and its recording go on.
#   run    the unit's loop: reads the server and the key from SSM at every (re)start, so a new key takes effect at the
#          next start; copies the relay to SERVER/KEY with no re-encode; reconnects 1 s, doubling to 30 s. The URL
#          is never logged (ffmpeg's messages go through redact) and never written to a file.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}
SERVER_PARAM=${ZEROAD_SSM_RTMP_SERVER:-/zeroad-live/rtmp-server}
KEY_PARAM=${ZEROAD_SSM_RTMP_KEY:-/zeroad-live/rtmp-key}
PUSH_LOG=$LOGDIR/push.log
WANTED=$STATE/push.wanted

param_exists() {
  if [ "$DRY_RUN" = 1 ]; then say "+ aws ssm get-parameter --name $1 (that it exists; the value is not read)"; return; fi
  aws ssm get-parameter --name "$1" --query Parameter.Name --output text >/dev/null \
    || die "SSM parameter $1 is missing or unreadable: store it with live/put-secret.sh"
}

# target_url: SERVER/KEY from SSM, on stdout only (captured by the caller, never echoed). A key of "-" means the
# server URL already holds it.
target_url() {
  local server key
  server=$(aws ssm get-parameter --name "$SERVER_PARAM" --with-decryption --query Parameter.Value --output text) || return 1
  key=$(aws ssm get-parameter --name "$KEY_PARAM" --with-decryption --query Parameter.Value --output text) || return 1
  [[ $server =~ ^rtmps?:// ]] || { echo "the RTMP server in $SERVER_PARAM is not an rtmp:// or rtmps:// URL" >&2; return 1; }
  if [ "$key" = - ]; then printf '%s' "$server"; else printf '%s/%s' "${server%/}" "$key"; fi
}

push_loop() {
  local backoff=1 url pid='' started stopping=0
  mkdir -p "$LOGDIR"
  trap 'stopping=1; [ -z "$pid" ] || kill -INT "$pid" 2>/dev/null || true' INT TERM
  while [ "$stopping" = 0 ]; do
    if ! url=$(target_url); then
      echo "$(date -u +%FT%TZ) cannot read the target from SSM ($SERVER_PARAM, $KEY_PARAM); retrying in 30 s" >> "$PUSH_LOG"
      [ -z "${ZEROAD_PUSH_ONCE:-}" ] || return 1
      sleep 30 & wait $! || true
      continue
    fi
    echo "$(date -u +%FT%TZ) push: relay -> <rtmp url> ($SERVER_PARAM + $KEY_PARAM)" >> "$PUSH_LOG"
    started=$(date +%s)
    ffmpeg -hide_banner -nostdin -loglevel warning -rw_timeout 15000000 -i "$RELAY" -c copy \
      -f flv -flvflags no_duration_filesize "$url" 2> >(redact >> "$PUSH_LOG") &
    pid=$!  # a background job here starts with SIGINT ignored; ffmpeg installs its own handler, so the INT lands
    unset url
    wait "$pid" || true
    [ "$stopping" = 0 ] || { wait "$pid" 2>/dev/null || true; break; }
    pid=
    echo "$(date -u +%FT%TZ) push: ffmpeg exited after $(($(date +%s) - started)) s; reconnecting in $backoff s" >> "$PUSH_LOG"
    [ -z "${ZEROAD_PUSH_ONCE:-}" ] || return 0
    [ $(($(date +%s) - started)) -lt 60 ] || backoff=1
    sleep "$backoff" & wait $! || true
    backoff=$((backoff * 2 > 30 ? 30 : backoff * 2))
  done
  echo "$(date -u +%FT%TZ) push: stopped" >> "$PUSH_LOG"
}

case $CMD in
  start)
    param_exists "$SERVER_PARAM"
    param_exists "$KEY_PARAM"
    run touch "$WANTED"
    run systemctl start zeroad-push
    say "push: started (relay -> the target in $SERVER_PARAM + $KEY_PARAM)"
    ;;
  stop)
    run rm -f "$WANTED"
    run systemctl stop zeroad-push
    say "push: stopped; the match and its recording go on"
    ;;
  run)
    if [ "$DRY_RUN" = 1 ]; then
      say "+ aws ssm get-parameter --name $SERVER_PARAM --with-decryption (into a variable, never printed)"
      say "+ aws ssm get-parameter --name $KEY_PARAM --with-decryption (into a variable, never printed)"
      say "+ ffmpeg -i $RELAY -c copy -f flv <rtmp url>   (messages redacted into $PUSH_LOG; reconnect 1-30 s)"
      exit 0
    fi
    push_loop
    ;;
  *) echo "usage: push.sh start|stop|run" >&2; exit 2 ;;
esac
