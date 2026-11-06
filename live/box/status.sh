#!/usr/bin/env bash
# status.sh [LINES]: the units, the match's last log lines (redacted), the push log, the spend and budget, the
# GPU, the disk and when the box powers off. Read-only.
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
N=${1:-15}
[[ $N =~ ^[0-9]+$ ]] || { echo "usage: status.sh [LINES]" >&2; exit 2; }
say "units: $(for u in "${UNITS[@]}"; do printf '%s=%s ' "$u" "$(active "$u")"; done)"
if [ "$DRY_RUN" = 1 ]; then
  say "+ journalctl -u zeroad-match -n $N | redact; tail -n $N $LOGDIR/push.log; budget.sh show; nvidia-smi; df -h; shutdown schedule"
  exit 0
fi
echo "--- match"; journalctl -u zeroad-match -n "$N" --no-pager -o cat 2>/dev/null | redact || true
echo "--- push"; tail -n "$N" "$LOGDIR/push.log" 2>/dev/null | redact || true
echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
echo "--- recordings"; find "$STATE/rec" -type f -name '*.mp4' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -5 || true
echo "--- gpu"; nvidia-smi --query-gpu=name,utilization.gpu,utilization.encoder,memory.used,temperature.gpu --format=csv,noheader || true
echo "--- disk"; df -h "$STATE" | tail -1
echo "--- power-off"; "$BOX_HOME/box/poweroff.sh" status
