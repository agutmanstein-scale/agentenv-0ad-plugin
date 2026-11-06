#cloud-boothook
#!/bin/bash
# User-data for the stream, pilot and training boxes (launch.sh substitutes the deadline, epoch seconds, on the line that writes it). cloud-init runs a
# boothook very early on every boot, before systemd-logind can take a scheduled shutdown, so scheduling the shutdown here
# is silently dropped. The boothook only records the deadline and installs a unit that arms the power-off after
# logind on every boot: first boot and after each stop/start, at the same absolute deadline (at least 30 minutes
# after the box is up, so its results can still be fetched).
ROOT=${ZEROAD_ROOT:-}
mkdir -p "$ROOT/etc/zeroad-live" "$ROOT/usr/local/sbin" "$ROOT/etc/systemd/system"
[ -s "$ROOT/etc/zeroad-live/deadline" ] || echo @DEADLINE@ > "$ROOT/etc/zeroad-live/deadline"
cat > "$ROOT/usr/local/sbin/zeroad-poweroff-arm" <<'ARM'
#!/bin/sh
set -eu
deadline=$(cat "${ZEROAD_DEADLINE_FILE:-/etc/zeroad-live/deadline}")
mins=$(( (deadline - $(date +%s)) / 60 ))
[ "$mins" -ge 30 ] || mins=30
shutdown -h +"$mins" "zeroad-live: MAX_HOURS reached"
ARM
chmod 755 "$ROOT/usr/local/sbin/zeroad-poweroff-arm"
cat > "$ROOT/etc/systemd/system/zeroad-poweroff.service" <<'UNIT'
[Unit]
Description=Arm the zeroad-live MAX_HOURS power-off
Wants=systemd-logind.service
After=systemd-logind.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/zeroad-poweroff-arm
TimeoutStartSec=60

[Install]
WantedBy=multi-user.target
UNIT
# enable only, never start: starting a unit from a boothook would wait on boot; multi-user.target starts it after logind
[ -n "$ROOT" ] || systemctl enable zeroad-poweroff.service || ln -sf /etc/systemd/system/zeroad-poweroff.service /etc/systemd/system/multi-user.target.wants/zeroad-poweroff.service
exit 0
