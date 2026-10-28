#!/usr/bin/env bash
# display.sh: Xorg on the NVIDIA L4 (zeroad-x.service), display :0, a virtual 1920x1080 screen with no monitor, no
# screen saver and no DPMS. The stream's engine renders on it with the GPU (GLX on the NVIDIA driver), Chrome draws
# the overlay beside it, and the env's ffmpeg grabs it. The env runs in a container: the docker shim mounts
# /tmp/.X11-unix into it and display-ready.sh lets local clients in. The output stays at the driver's default mode
# (airops' trap: a 1920x1080 output mode syncs every client to 60 Hz and costs frames).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
SIZE=${ZEROAD_SIZE:-1920x1080}
if [ "$DRY_RUN" = 1 ]; then
  BUS=PCI:0:30:0
else
  BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 \
    | awk -F'[:.]' '{printf "PCI:%d:%d:%d", strtonum("0x"$2), strtonum("0x"$3), strtonum("0x"$4)}')
fi
CONFIG=$(cat <<XC
Section "ServerFlags"
  Option "AutoAddDevices" "false"
  Option "AutoAddGPU" "false"
  Option "BlankTime" "0"
  Option "StandbyTime" "0"
  Option "SuspendTime" "0"
  Option "OffTime" "0"
EndSection
Section "Device"
  Identifier "nvidia"
  Driver "nvidia"
  BusID "$BUS"
  Option "AllowEmptyInitialConfiguration" "true"
  Option "HardDPMS" "false"
EndSection
Section "Screen"
  Identifier "screen0"
  Device "nvidia"
  DefaultDepth 24
  Option "AllowEmptyInitialConfiguration" "true"
  SubSection "Display"
    Depth 24
    Virtual ${SIZE%x*} ${SIZE#*x}
  EndSubSection
EndSection
XC
)
if [ "$DRY_RUN" = 1 ]; then
  echo "+ write /etc/X11/xorg-zeroad.conf (BusID $BUS, virtual $SIZE)"
else
  printf '%s\n' "$CONFIG" > /etc/X11/xorg-zeroad.conf
fi
run exec Xorg :0 -config /etc/X11/xorg-zeroad.conf -noreset -nolisten tcp -novtswitch -sharevts -s 0 -dpms vt7
