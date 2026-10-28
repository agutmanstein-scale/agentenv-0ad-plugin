#!/usr/bin/env bash
# health.sh: one JSON line about the stream: the units, the display, the env container, the relay (publishing and
# its bytes), the newest recording segment and its age, the push (wanted, running, recent errors), the GPU and NVENC,
# free disk, spend and budget, and when the box powers off. `status` is "ok", or "warn" with `problems`. Read-only.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
if [ "$DRY_RUN" = 1 ]; then
  say "+ systemctl is-active ${UNITS[*]}; xdpyinfo -display :0"
  say "+ docker ps (the mcp-server-zeroad-live container); curl 127.0.0.1:9997/v3/paths/get/zeroad (the relay)"
  say "+ the newest segment in $STATE/rec/zeroad; tail $LOGDIR/push.log; nvidia-smi; df $STATE; $LEDGER"
  say "+ print one JSON line: status ok|warn, problems, units, relay, recording, push, gpu, disk, spend, poweroff"
  exit 0
fi
export H_X H_RELAY H_PUSH H_MATCH H_WANTED H_DISPLAY H_ENV H_RELAY_JSON H_REC H_REC_AGE H_REC_MB H_PUSH_ERR H_GPU H_DISK H_SPEND H_POWEROFF H_UPTIME
H_X=$(active zeroad-x); H_RELAY=$(active zeroad-relay); H_PUSH=$(active zeroad-push); H_MATCH=$(active zeroad-match)
H_WANTED=$([ -f "$STATE/push.wanted" ] && echo 1 || echo 0)
H_DISPLAY=$(DISPLAY=:0 timeout 5 xdpyinfo >/dev/null 2>&1 && echo 1 || echo 0)
H_ENV=$(docker ps --format '{{.Image}} {{.Status}}' 2>/dev/null | grep 'mcp-server-zeroad-live' | head -1 || true)
H_RELAY_JSON=$(curl -fsS -m 5 http://127.0.0.1:9997/v3/paths/get/zeroad 2>/dev/null || echo '{}')
H_REC=$(find "$STATE/rec" -type f -name '*.mp4' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -1 || true)
H_REC_AGE=$([ -n "$H_REC" ] && echo $(( $(date +%s) - ${H_REC%%.*} )) || echo)
H_REC_MB=$([ -n "$H_REC" ] && awk '{printf "%d", $2/1048576}' <<<"$H_REC" || echo)
H_PUSH_ERR=$(tail -n 200 "$LOGDIR/push.log" 2>/dev/null | grep -ci 'error\|reconnecting' || true)
H_GPU=$(nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
H_DISK=$(df -BG --output=avail "$STATE" 2>/dev/null | tail -1 | tr -dc 0-9)
H_SPEND=$("$BOX_HOME/box/budget.sh" show 2>/dev/null | tail -1 || true)
H_POWEROFF=$(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null || true)
H_UPTIME=$(awk '{printf "%.1f", $1/3600}' /proc/uptime)
python3 - "${ZEROAD_MIN_FREE_GB:-8}" <<'PY'
import datetime, json, os, sys

e = os.environ.get
min_free = float(sys.argv[1])
relay = json.loads(e("H_RELAY_JSON") or "{}")
gpu = [x.strip() for x in (e("H_GPU") or "").split(",")]
units = {"x": e("H_X"), "relay": e("H_RELAY"), "push": e("H_PUSH"), "match": e("H_MATCH")}
rec_age = int(e("H_REC_AGE")) if e("H_REC_AGE") else None
disk = int(e("H_DISK")) if e("H_DISK") else None
problems = []
if units["x"] != "active" or e("H_DISPLAY") != "1":
    problems.append("display :0 down")
if units["relay"] != "active":
    problems.append("relay down")
if e("H_WANTED") == "1" and units["push"] != "active":
    problems.append("push wanted but not running")
if units["match"] == "active":
    if not relay.get("ready"):
        problems.append("match running but nothing publishes to the relay")
    if rec_age is None or rec_age > 120:
        problems.append("match running but the recording is not growing")
if disk is not None and disk < min_free:
    problems.append(f"disk {disk} GB free")
poweroff = None
parts = (e("H_POWEROFF") or "").split()
if len(parts) >= 3 and parts[2].isdigit() and int(parts[2]):
    poweroff = datetime.datetime.fromtimestamp(int(parts[2]) / 1e6, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print(json.dumps({
    "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "warn" if problems else "ok", "problems": problems, "units": units,
    "push_wanted": e("H_WANTED") == "1", "env_container": e("H_ENV") or None,
    "relay": {"ready": bool(relay.get("ready")), "bytes_in": relay.get("bytesReceived")},
    "recording": {"age_s": rec_age, "mb": int(e("H_REC_MB")) if e("H_REC_MB") else None},
    "push_errors_recent": int(e("H_PUSH_ERR") or 0),
    "gpu": dict(zip(["gpu_pct", "nvenc_pct", "mem_mb", "temp_c"], gpu)) if len(gpu) == 4 else None,
    "disk_free_gb": disk, "spend": e("H_SPEND") or None, "uptime_h": float(e("H_UPTIME") or 0), "poweroff_utc": poweroff,
}))
PY
