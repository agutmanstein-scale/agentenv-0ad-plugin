#!/usr/bin/env bash
# s2.sh put-config B64 | start --budget USD [--resume|--fresh] | status [LINES] | stop
# Box day 1 (S2) and Run 1 (S3) as one unattended sequence on this training box (bootstrapped with bootstrap.sh --train),
# in the transient unit zeroad-s2 (s2-run.sh), which survives an SSM disconnect.
# `put-config` takes the config directory as base64 of a gzipped tar (live/s2.sh sends it), refuses a path that leaves the
# folder, unpacks it to $STATE/train/s2-config and checks it there (agentenv_zeroad.train.s2 check: every step's config,
# its panels and one sample game, the guards the sequence relies on). `start` refuses while a sequence is going, while
# a vLLM instance is down, when the configs' box rate is not this box's, when a sequence already left gates.json
# (--resume continues it, skipping the steps that passed; --fresh moves it aside), when the power-off deadline leaves less
# than the sequence's expected hours (on --resume, those of the steps it still runs), and when budget.sh says --budget
# would be passed by the box hours so far plus those. `status` prints the unit, the gate verdicts, the step's orchestrator status and the learner's latest health.
# `stop` sends SIGTERM: the running step stops and uploads, the report and the sequence's files are uploaded, and the
# box stays up (no power-off after a stop by hand).
# shellcheck source=lib.sh
source "${ZEROAD_LIVE_HOME:-/opt/zeroad-live}/box/lib.sh"
CMD=${1:-}; shift || true
ETC=${ZEROAD_PILOT_ETC:-/etc/zeroad-live}
TRAIN=$STATE/train
S2=$TRAIN/s2
CONFIGDIR=$TRAIN/s2-config
DEADLINE_FILE=${ZEROAD_DEADLINE_FILE:-$ETC/deadline}
PY=(uv run --project "$REPO" --frozen --no-dev --extra train python -m agentenv_zeroad.train.s2)
USAGE="usage: s2.sh put-config B64 | start --budget USD [--resume|--fresh] | status [LINES] | stop"
case $CMD in
  put-config)
    B64=${1:?$USAGE}
    [[ $B64 =~ ^[A-Za-z0-9+/=]+$ ]] || die "put-config needs base64"
    if [ "$DRY_RUN" = 1 ]; then say "+ unpack the config directory to $CONFIGDIR and check it (train.s2 check)"; exit 0; fi
    [ "$(active zeroad-s2)" != active ] || die "a sequence is running: s2.sh status, or stop it first"
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    printf '%s' "$B64" | base64 -d > "$tmp/config.tgz"
    if tar tzf "$tmp/config.tgz" | grep -Eq '(^/|\.\.)'; then die "the config archive names a path outside its folder"; fi
    mkdir -p "$tmp/new" "$TRAIN"
    tar xzf "$tmp/config.tgz" -C "$tmp/new"
    # The configs name vLLM at its URLs on this box (bootstrap's Docker address pools move the bridge off 172.17.0.1).
    [ -s "$ETC/vllm.urls" ] || die "no $ETC/vllm.urls: live/bootstrap.sh ID --train first"
    python3 - "$tmp/new" $(cat "$ETC/vllm.urls") <<'PY' || die "could not point the configs at this box's vLLM"
import pathlib, re, sys
root, urls = pathlib.Path(sys.argv[1]), sys.argv[2:]
line = "urls = [" + ", ".join(f'"{u}"' for u in urls) + "]"
for path in sorted(root.glob("*.toml")):
    text = path.read_text()
    new, n = re.subn(r"(?m)^urls = \[[^\]]*\]", line, text)
    if n:
        path.write_text(new)
        print(f"{path.name}: [servers] urls = {', '.join(urls)}")
PY
    "${PY[@]}" check "$tmp/new" || die "the config directory is not valid"
    rm -rf "$CONFIGDIR.old"; [ ! -d "$CONFIGDIR" ] || mv "$CONFIGDIR" "$CONFIGDIR.old"
    mv "$tmp/new" "$CONFIGDIR"
    say "config in place: $CONFIGDIR"
    ;;
  start)
    BUDGET= MODE=
    while [ $# -gt 0 ]; do
      case $1 in
        --budget) BUDGET=${2:?--budget needs USD}; shift 2 ;;
        --resume|--fresh) [ -z "$MODE" ] || die "--resume and --fresh are alternatives"; MODE=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$BUDGET" ] || die "no budget: set LIVE_BUDGET_USD in the S2 live env"
    [[ $BUDGET =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget needs a number"
    [ "$(active zeroad-s2)" != active ] || die "a sequence is going: s2.sh status, or stop it first"
    [ "$(active zeroad-train)" != active ] || die "a training run is going: train.sh status, or stop it first"
    if [ "$DRY_RUN" = 0 ]; then
      [ -d "$CONFIGDIR" ] || die "no config directory at $CONFIGDIR (s2.sh put-config)"
      for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do
        curl -fsS "$url/health" >/dev/null || die "vLLM is not up at $url (docker logs zeroad-vllm-0 / zeroad-vllm-1)"
      done
    fi
    if [ -d "$CONFIGDIR" ]; then
      resume=(); [ "$MODE" != --resume ] || resume=(--resume-from "$S2")  # a resume is held to the steps it still runs
      numbers=$("${PY[@]}" check "$CONFIGDIR" --budget "$BUDGET" ${resume[@]+"${resume[@]}"}) || die "the config directory $CONFIGDIR is not valid for a budget of \$$BUDGET"
      eval "$numbers"
    else
      S2_EXPECTED_HOURS=0.5 S2_BOX_USD_PER_HOUR=${ZEROAD_BOX_USD_PER_HOUR:-0}
      say "(dry run: no config directory at $CONFIGDIR yet; planning half an hour)"
    fi
    [ -n "${ZEROAD_BOX_USD_PER_HOUR:-}" ] || die "ZEROAD_BOX_USD_PER_HOUR is not set in $CONF: the box has no rate"
    awk -v a="$S2_BOX_USD_PER_HOUR" -v b="$ZEROAD_BOX_USD_PER_HOUR" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.005) }' \
      || die "the configs' box_usd_per_hour ($S2_BOX_USD_PER_HOUR) is not this box's rate ($ZEROAD_BOX_USD_PER_HOUR in $CONF)"
    if [ -f "$DEADLINE_FILE" ]; then
      left=$(awk -v d="$(cat "$DEADLINE_FILE")" -v n="${ZEROAD_NOW_EPOCH:-$(date +%s)}" 'BEGIN { printf "%.2f", (d - n) / 3600 }')
      awk -v l="$left" -v e="$S2_EXPECTED_HOURS" 'BEGIN { exit !(l >= e + 0.5) }' \
        || die "the box powers off in $left h and the sequence is expected to take $S2_EXPECTED_HOURS h (+ half an hour): relaunch with a larger LIVE_MAX_HOURS"
    fi
    if [ -f "$S2/gates.json" ]; then
      case $MODE in
        --resume) say "resuming the sequence in $S2: steps that passed are skipped" ;;
        --fresh) run mv "$S2" "$S2-old-$(date -u +%Y%m%d-%H%M%S)" ;;
        *) die "$S2/gates.json is from an earlier sequence: --resume continues it, --fresh moves it aside" ;;
      esac
    fi
    "$BOX_HOME/box/budget.sh" check --planned-usd 0 --planned-hours "$S2_EXPECTED_HOURS" --budget "$BUDGET"
    say "S2 + Run 1 from $CONFIGDIR: about $S2_EXPECTED_HOURS h, budget \$$BUDGET"
    # TimeoutStopSec: a stop makes the running step upload (3 tries of 10 minutes) and then the sequence uploads its own.
    run systemd-run --unit zeroad-s2 --collect -p KillMode=control-group -p TimeoutStopSec=4200 \
      -p "After=network-online.target docker.service" -p "Wants=network-online.target docker.service" \
      -p "WorkingDirectory=$REPO" "$BOX_HOME/box/s2-run.sh" --budget "$BUDGET"
    say "started; follow it with live/s2.sh ID status"
    ;;
  status)
    N=${1:-20}
    [[ $N =~ ^[0-9]+$ ]] || { echo "usage: s2.sh status [LINES]" >&2; exit 2; }
    say "zeroad-s2: $(active zeroad-s2)"
    if [ "$DRY_RUN" = 1 ]; then
      say "+ s2 status (gates.json); status.json; progress.log; the step's orchestrator/status.json and health.jsonl; docker ps; nvidia-smi; vLLM health; budget.sh show; df; power-off"
      exit 0
    fi
    echo "--- gates"; "${PY[@]}" status "$S2" 2>/dev/null || echo "(no gates yet)"
    echo "--- now"; cat "$S2/status.json" 2>/dev/null || echo "(no status yet)"
    echo "--- progress"; tail -n "$N" "$S2/progress.log" 2>/dev/null || true
    step=$(jq -r '.step // empty' "$S2/status.json" 2>/dev/null || true)
    case $step in
      e1) dir=e1 ;; e2) dir=e2 ;; baselines|calibration|tripwire|labels) dir=baselines ;; sft|sft_panels|init) dir=sft ;; run1) dir=run1 ;; *) dir= ;;
    esac
    # E2 runs its v0 baseline games (e2-baseline) and then its training (e2b; e2 is the first attempt): the newest one.
    if [ "$step" = e2 ]; then
      newest=$(ls -t "$S2"/e2*/orchestrator/status.json 2>/dev/null | head -n 1 || true)
      [ -z "$newest" ] || { newest=${newest%/orchestrator/status.json}; dir=${newest##*/}; }
    fi
    if [ -n "$dir" ]; then
      echo "--- $dir orchestrator"; jq -c '{run_hours, spend, current, counts, panels, briefs, alarms, stop_reason}' "$S2/$dir/orchestrator/status.json" 2>/dev/null || echo "(none)"
      echo "--- $dir learner (latest health records)"; tail -n 3 "$S2/$dir/health.jsonl" 2>/dev/null | cut -c1-400 || true
    fi
    echo "--- containers"; docker ps --format '{{.Names}} {{.Status}}' | grep -E 'zeroad-(vllm|learner|engine)' | head -8 || true
    echo "--- gpus"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader || true
    echo "--- vllm"
    for url in $(cat "$ETC/vllm.urls" 2>/dev/null); do echo "$url $(curl -s -o /dev/null -w '%{http_code}' "$url/health" || echo down)"; done
    echo "--- spend"; "$BOX_HOME/box/budget.sh" show || true
    echo "--- disk"; df -h "$STATE" | tail -1
    echo "--- power-off"; "$BOX_HOME/box/poweroff.sh" status
    ;;
  stop)
    say "stopping zeroad-s2: the running step stops and uploads, then the sequence's files are uploaded; the box stays up"
    run systemctl stop zeroad-s2
    ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
