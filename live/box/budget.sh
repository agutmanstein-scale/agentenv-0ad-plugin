#!/usr/bin/env bash
# budget.sh check --planned-usd USD --planned-hours H --budget USD | show [--budget USD] | tick
# The stream's budget guard. Model spend is the box ledger's (spend.jsonl, agentenv_zeroad.spend's format: one
# {"what", "usd_est"} per match); box spend is the box's running hours at ZEROAD_BOX_USD_PER_HOUR, over every boot,
# not just this one: `tick` (the zeroad-boxtime timer, every minute) records this boot's start and its latest minute
# in boxtime.json, so a stop-box and start (or a reboot) keeps the earlier hours. `check` exits 1 (and says why) when
# spent + the planned match's caps + box hours so far and the match's hours would pass the budget.
# ZEROAD_NOW_EPOCH and ZEROAD_BOOT_EPOCH override the clock (tests).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
PLANNED_USD=0 PLANNED_HOURS=0 BUDGET=
while [ $# -gt 0 ]; do
  case $1 in
    --planned-usd) PLANNED_USD=${2:?}; shift 2 ;;
    --planned-hours) PLANNED_HOURS=${2:?}; shift 2 ;;
    --budget) BUDGET=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case $CMD in check|show|tick) ;; *) echo "usage: budget.sh check|show [--planned-usd USD --planned-hours H] --budget USD | tick" >&2; exit 2 ;; esac
[ "$CMD" != check ] || [ -n "$BUDGET" ] || die "budget.sh check needs --budget (LIVE_BUDGET_USD)"
[ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set: the box's rate comes from its /etc/zeroad-live.conf (LIVE_BOX_USD_PER_HOUR); there is no default"
NOW=${ZEROAD_NOW_EPOCH:-$(date +%s)}
if [ -n "${ZEROAD_BOOT_EPOCH:-}" ]; then BOOT=$ZEROAD_BOOT_EPOCH
elif [ "$DRY_RUN" = 1 ]; then BOOT=$NOW; echo "(dry run: as on a box booted now)"
else BOOT=$(awk '/^btime/ {print $2}' /proc/stat); fi
if [ "$CMD" = tick ] && [ "$DRY_RUN" = 1 ]; then say "+ record boot $BOOT up to $NOW in $STATE/boxtime.json"; exit 0; fi
[ "$DRY_RUN" = 1 ] || mkdir -p "$STATE"
python3 - "$CMD" "$LEDGER" "$PLANNED_USD" "$PLANNED_HOURS" "${BUDGET:-}" "$ZEROAD_BOX_USD_PER_HOUR" "$BOOT" "$NOW" \
  "$STATE/boxtime.json" "$DRY_RUN" <<'PY'
import json, math, os, pathlib, sys

cmd, ledger, planned_usd, planned_h, budget, rate, boot, now, boxtime, dry_run = sys.argv[1:]
try:
    planned_usd, planned_h, rate = float(planned_usd), float(planned_h), float(rate)
    budget = float(budget) if budget else None
    boot, now = int(boot), int(now)
except ValueError as e:
    sys.exit(f"ERROR: not a number: {e}")
amounts = [planned_usd, planned_h, rate] + ([budget] if budget is not None else [])
if not all(math.isfinite(x) for x in amounts):
    sys.exit("ERROR: amounts must be finite numbers")
if min(planned_usd, planned_h, rate) < 0 or (budget is not None and budget <= 0):
    sys.exit("ERROR: amounts must be positive")

# boxtime.json: {"<boot epoch>": [boot epoch, last seen epoch]}, one entry per boot of this box.
bt_path = pathlib.Path(boxtime)
boots = json.loads(bt_path.read_text()) if bt_path.exists() else {}
boots[str(boot)] = [boot, max(now, boots.get(str(boot), [boot, boot])[1])]
if dry_run != "1":
    tmp = bt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(boots))
    os.replace(tmp, bt_path)
if cmd == "tick":
    sys.exit(0)

path = pathlib.Path(ledger)
lines = path.read_text().splitlines() if path.exists() else []
model = sum(float(json.loads(line).get("usd_est", 0)) for line in lines if line.strip())
earlier_h = sum(max(0, last - start) for key, (start, last) in boots.items() if key != str(boot)) / 3600
box_h = earlier_h + max(0.0, (now - boot) / 3600)
box = box_h * rate
ahead = planned_h * rate
total = model + box + planned_usd + ahead
boots_note = f", {len(boots)} boots" if len(boots) > 1 else ""
line = (f"model ${model:.2f} ({len([x for x in lines if x.strip()])} matches) + box ${box:.2f} ({box_h:.1f} h at"
        f" ${rate:g}/h{boots_note}) + this match ${planned_usd:.2f} caps + ${ahead:.2f} box ({planned_h:g} h)"
        f" = ${total:.2f}")
if budget is not None:
    line += f" of ${budget:g}"
print(line)
if cmd == "check" and total > budget:
    print(f"REFUSED: ${total:.2f} would pass the ${budget:g} budget (LIVE_BUDGET_USD)")
    sys.exit(1)
PY
