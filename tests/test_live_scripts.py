"""The GPU-box stream scripts in live/ (U7), run for real but against stubs: no AWS call, box or RTMP connection.

Every script that changes anything is a dry run unless given --yes, and a dry run must make no aws call at all, so
those tests put an `aws` on PATH that fails the test if it is ever run. The tests that exercise the real paths (the
secret handling, the push, the teardown's tag check, SSM) use a recording stub that answers by rules instead. The box
scripts run here with their paths pointed at a temporary folder, as the laptop scripts' dry runs do.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "live"
BOX = LIVE / "box"
ID = "i-0123456789abcdef0"
SECRET = "sk-test-SECRET-value"

STUB_AWS = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    argv = sys.argv[1:]
    inputs = [a[len("file://"):] for a in argv if a.startswith("file://")]
    stdin = open(inputs[0]).read() if inputs else None
    if os.environ.get("STUB_AWS_FORBID"):
        sys.stderr.write("STUB AWS CALLED: " + " ".join(argv) + "\\n")
        sys.exit(99)
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({{"argv": argv, "stdin": stdin}}) + "\\n")
    joined = " ".join(argv)
    for needle, out, rc in json.loads(os.environ.get("STUB_AWS_RULES", "[]")):
        if all(part in joined for part in needle.split("&&")):
            sys.stdout.write(out)
            sys.exit(rc)
    """
)


def write_exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def stubs(tmp_path):
    """A bin folder first on PATH with the aws stub, and the env every script runs with."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_exe(bin_dir / "aws", STUB_AWS.format(python=sys.executable))
    live_env = tmp_path / "live.env"
    live_env.write_text(
        "LIVE_AWS_REGION=us-west-2\nLIVE_SUBNETS='subnet-a subnet-b'\nLIVE_SECURITY_GROUP=sg-1\n"
        "LIVE_INSTANCE_PROFILE=zeroad-live-box\nLIVE_BUCKET=bucket-1\nLIVE_BUDGET_USD=20\nLIVE_MATCH_CAP_USD=9\n"
        "LIVE_LITELLM_BASE_URL=https://litellm.example\n"
    )
    log = tmp_path / "aws.jsonl"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "LIVE_ENV": str(live_env),
        "STUB_LOG": str(log),
        "LIVE_SSM_POLL_S": "0",
    }
    return {"bin": bin_dir, "env": env, "log": log, "tmp": tmp_path, "live_env": live_env}


def run(stubs, script: str, *args: str, extra: dict | None = None, stdin: str | None = None, forbid_aws=False):
    env = {**stubs["env"], **(extra or {})}
    if forbid_aws:
        env["STUB_AWS_FORBID"] = "1"
    return subprocess.run(
        ["bash", str(LIVE / script), *args],
        env=env,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def box_env(stubs, **extra) -> dict:
    state = stubs["tmp"] / "state"
    conf = stubs["tmp"] / "zeroad-live.conf"
    conf.write_text(
        "AWS_DEFAULT_REGION=us-west-2\nZEROAD_SSM_LITELLM_KEY=/zeroad-live/litellm-key\n"
        "ZEROAD_SSM_RTMP_SERVER=/zeroad-live/rtmp-server\nZEROAD_SSM_RTMP_KEY=/zeroad-live/rtmp-key\n"
        "LITELLM_BASE_URL=https://litellm.example\nZEROAD_BOX_USD_PER_HOUR=1.32\n"
    )
    return {
        **stubs["env"],
        "ZEROAD_LIVE_HOME": str(LIVE),
        "ZEROAD_LIVE_CONF": str(conf),
        "ZEROAD_LIVE_STATE": str(state),
        "ZEROAD_LIVE_LOG": str(stubs["tmp"] / "log"),
        "ZEROAD_LIVE_REPO": str(ROOT),
        **extra,
    }


def run_box(stubs, script: str, *args: str, **extra):
    return subprocess.run(
        ["bash", str(BOX / script), *args],
        env=box_env(stubs, **extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def aws_calls(stubs) -> list[dict]:
    log = stubs["log"]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


CHANGE_SCRIPTS = [
    ("launch.sh",),
    ("launch.sh", "--max-hours", "3", "--name", "zeroad-live-test"),
    ("bootstrap.sh", ID),
    ("bootstrap.sh", ID, "--images", "build"),
    ("match.sh", ID, "rehearsal"),
    ("match.sh", ID, "model"),
    ("match.sh", ID, "model", "--push", "--name", "first-live"),
    ("push.sh", ID, "start"),
    ("push.sh", ID, "stop"),
    ("stop.sh", ID),
    ("stop.sh", ID, "--push-only"),
    ("repair.sh", ID),
    ("stop-box.sh", ID),
    ("teardown.sh", ID),
    ("fetch.sh", ID),
    ("fetch.sh", ID, "livestream-1"),
    ("fetch.sh", ID, "--only", "rec/zeroad/2026-10-29_01-00-00-000000.mp4"),
    ("put-secret.sh", "litellm-key"),
    ("bootstrap.sh", ID, "--pilot"),
    ("pilot.sh", ID, "start"),
    ("pilot.sh", ID, "start", "--baseline", "--no-frontier", "--speed", "4"),
    ("pilot.sh", ID, "stop"),
    ("pilot.sh", ID, "stop", "--now"),
    ("launch.sh", "--shutdown-behavior", "stop"),
    ("bootstrap.sh", ID, "--train"),
    ("train-iam.sh", "create", ID),
    ("train-iam.sh", "delete", ID),
    ("train.sh", ID, "start", str(LIVE / "train-run.example.toml")),
    ("train.sh", ID, "start", str(LIVE / "train-run.example.toml"), "--play-only"),
    ("train.sh", ID, "stop"),
]


@pytest.mark.parametrize("cmd", CHANGE_SCRIPTS, ids=" ".join)
def test_every_script_that_changes_anything_is_a_dry_run_without_yes_and_calls_no_aws(stubs, cmd):
    out = run(stubs, *cmd, forbid_aws=True)
    text = out.stdout + out.stderr
    assert out.returncode == 0, text
    assert "DRY RUN" in text and "Add --yes" in text
    assert "STUB AWS CALLED" not in text and not stubs["log"].exists()
    assert not (LIVE / "fetched").exists()


@pytest.mark.parametrize(
    "cmd",
    [
        ("status.sh", ID, "--dry-run"),
        ("health.sh", ID, "--dry-run"),
        ("pilot.sh", ID, "status", "--dry-run"),
        ("train.sh", ID, "status", "--dry-run"),
    ],
)
def test_the_read_only_scripts_plan_with_dry_run_and_call_no_aws(stubs, cmd):
    out = run(stubs, *cmd, forbid_aws=True)
    assert out.returncode == 0, out.stderr
    assert "DRY RUN" in out.stderr and "STUB AWS CALLED" not in out.stderr
    assert not stubs["log"].exists()


def test_a_dry_run_plans_from_the_example_settings_when_live_env_is_missing(stubs):
    out = run(stubs, "launch.sh", extra={"LIVE_ENV": str(stubs["tmp"] / "missing.env")}, forbid_aws=True)
    assert out.returncode == 0, out.stderr
    assert "planning with" in out.stderr and "live.env.example" in out.stderr
    out = run(stubs, "launch.sh", "--yes", extra={"LIVE_ENV": str(stubs["tmp"] / "missing.env")}, forbid_aws=True)
    assert out.returncode != 0 and "copy live/live.env.example" in out.stderr


def test_the_launch_plan_is_one_l4_box_on_the_nvidia_dlami_that_powers_itself_off(stubs):
    out = run(stubs, "launch.sh", "--max-hours", "3", forbid_aws=True)
    plan = out.stderr
    assert "--instance-type g6.4xlarge" in plan and "--subnet-id subnet-a" in plan
    assert "base-oss-nvidia-driver-gpu-ubuntu-22.04" in plan
    assert "--instance-initiated-shutdown-behavior terminate" in plan and "HttpTokens=required" in plan
    assert "Key=purpose,Value=zeroad-live" in plan and "Key=Name,Value=zeroad-live-onair" in plan
    user_data = re.search(r"--user-data (\S+)", plan).group(1)
    script = base64.b64decode(user_data).decode()
    assert script.startswith("#cloud-boothook\n") and 'shutdown -h +"$mins" "zeroad-live: MAX_HOURS reached"' in script
    assert "at most about $3.96" in plan
    assert out.stdout.strip() == ID  # the id is the only stdout, for ID=$(live/launch.sh --yes)


def test_launch_tries_the_next_subnet_when_one_has_no_capacity(stubs):
    rules = [
        ["describe-instances", "None\n", 0],
        ["ssm&&get-parameter", "ami-1\n", 0],
        ["run-instances&&subnet-a", "", 254],
        ["run-instances&&subnet-b", "i-00000000000000001\n", 0],
        ["describe-instance-information", "Online\n", 0],
    ]
    out = run(stubs, "launch.sh", "--yes", extra={"STUB_AWS_RULES": json.dumps(rules)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "i-00000000000000001"
    runs = [c["argv"] for c in aws_calls(stubs) if "run-instances" in c["argv"]]
    assert [a[a.index("--subnet-id") + 1] for a in runs] == ["subnet-a", "subnet-b"]


@pytest.mark.parametrize(
    "cmd",
    [
        ("launch.sh", "--max-hours", "0"),
        ("launch.sh", "--max-hours", "abc"),
        ("launch.sh", "--bogus"),
        ("bootstrap.sh", "not-an-id"),
        ("bootstrap.sh", ID, "--images", "rsync"),
        ("match.sh", ID, "final"),
        ("match.sh", "i-nope", "model"),
        ("match.sh", ID, "model", "--name", "bad name"),
        ("push.sh", ID, "restart"),
        ("stop.sh", ID, "--everything"),
        ("teardown.sh", ID, "extra"),
        ("put-secret.sh", "x-key"),
        ("status.sh", ID, "many"),
        ("launch.sh", "--shutdown-behavior", "reboot"),
        ("train-iam.sh", "create"),
        ("train.sh", ID, "start", "/no/such/config.toml"),
        ("train.sh", ID, "start", str(LIVE / "train-run.example.toml"), "bad arg"),
        ("train.sh", ID, "stop", "--now"),
    ],
    ids=" ".join,
)
def test_bad_arguments_are_refused_before_anything_is_called(stubs, cmd):
    out = run(stubs, *cmd, "--yes", forbid_aws=True)
    assert out.returncode != 0
    assert "STUB AWS CALLED" not in out.stderr and not stubs["log"].exists()


def test_the_model_match_reads_the_key_from_ssm_is_capped_at_9_and_pushes_only_with_push(stubs):
    out = run(stubs, "match.sh", ID, "model", forbid_aws=True)
    plan = out.stderr
    assert "task livestream, model caps $9.00 (match cap $9)" in plan
    assert '"render": true, "rtmp": true' in plan and '"time_cap_minutes": 40' in plan
    assert "aws ssm get-parameter --name /zeroad-live/litellm-key --with-decryption" in plan
    assert "LITELLM_API_KEY=<from SSM>" in plan and "agent-env run" in plan
    assert "zeroad-push" not in plan and "push: off" in plan
    pushed = run(stubs, "match.sh", ID, "model", "--push", forbid_aws=True).stderr
    assert "systemctl start zeroad-push" in pushed and "PUSHED to the RTMP target (public)" in pushed
    assert "get-parameter --name /zeroad-live/rtmp-key (that it exists; the value is not read)" in pushed


def test_the_rehearsal_is_free_and_uses_no_model(stubs):
    plan = run(stubs, "match.sh", ID, "rehearsal", forbid_aws=True).stderr
    assert "task livestream-petra, model caps $0.00" in plan
    assert "litellm-key" not in plan and "zeroad-push" not in plan
    assert "agent-env run" in plan and "--task livestream-petra" in plan


def test_a_model_match_can_play_another_task_with_box_time_from_its_cap(stubs):
    plan = run(stubs, "match.sh", ID, "model", "--task", "livestream-4", forbid_aws=True).stderr
    assert "task livestream-4, model caps $5.40 (match cap $9), about 0.5 h of box time" in plan
    assert "--task livestream-4" in plan and "litellm-key" in plan
    assert run(stubs, "match.sh", ID, "rehearsal", "--task", "livestream-4", forbid_aws=True).returncode != 0
    assert run(stubs, "match.sh", ID, "model", "--task", "../x", forbid_aws=True).returncode != 0


def test_a_match_that_would_pass_the_budget_is_refused(stubs):
    stubs["live_env"].write_text(stubs["live_env"].read_text() + "LIVE_BUDGET_USD=10\n")
    out = run(stubs, "match.sh", ID, "model", forbid_aws=True)
    assert out.returncode != 0
    assert "REFUSED: $10.32 would pass the $10 budget" in out.stderr and "systemd-run" not in out.stderr


def test_a_model_match_whose_caps_pass_the_match_cap_is_refused(stubs):
    stubs["live_env"].write_text(stubs["live_env"].read_text() + "LIVE_MATCH_CAP_USD=5\n")
    out = run(stubs, "match.sh", ID, "model", forbid_aws=True)
    assert out.returncode != 0 and "pass the match cap $5" in out.stderr
    assert run(stubs, "match.sh", ID, "rehearsal", forbid_aws=True).returncode == 0  # $0: no match cap to pass


def test_the_budget_counts_the_ledger_the_box_hours_so_far_and_the_match_ahead(stubs):
    state = stubs["tmp"] / "state"
    state.mkdir()
    (state / "spend.jsonl").write_text(
        '{"what": "livestream a", "usd_est": 7.5}\n{"what": "livestream b", "usd_est": 1}\n'
    )
    now = 1_800_000_000
    clock = {"ZEROAD_NOW_EPOCH": str(now), "ZEROAD_BOOT_EPOCH": str(now - 2 * 3600)}
    args = ("check", "--planned-usd", "9", "--planned-hours", "1")
    ok = run_box(stubs, "budget.sh", *args, "--budget", "25", **clock)
    assert ok.returncode == 0, ok.stderr
    assert "model $8.50 (2 matches) + box $2.64 (2.0 h at $1.32/h) + this match $9.00 caps + $1.32 box" in ok.stdout
    assert "= $21.46 of $25" in ok.stdout
    refused = run_box(stubs, "budget.sh", *args, "--budget", "21", **clock)
    assert refused.returncode == 1 and "REFUSED: $21.46 would pass the $21 budget" in refused.stdout
    assert run_box(stubs, "budget.sh", "check", "--planned-usd", "-1", "--budget", "21", **clock).returncode != 0


def test_put_secret_sends_the_value_in_a_json_body_on_stdin_never_on_a_command_line(stubs):
    stubs["live_env"].write_text(stubs["live_env"].read_text() + "LIVE_AWS_ACCOUNT=123456789012\n")
    rules = [["sts&&get-caller-identity", "123456789012\n", 0], ["put-parameter", "1\n", 0]]
    out = run(stubs, "put-secret.sh", "litellm-key", stdin=SECRET + "\n", forbid_aws=True)
    assert out.returncode == 0 and "Add --yes" in out.stderr and not stubs["log"].exists()  # without --yes: nothing
    out = run(
        stubs, "put-secret.sh", "litellm-key", "--yes", stdin=SECRET + "\n", extra={"STUB_AWS_RULES": json.dumps(rules)}
    )
    assert out.returncode == 0, out.stderr
    calls = aws_calls(stubs)
    assert calls[0]["argv"][:2] == ["sts", "get-caller-identity"]  # the account check comes first
    put = calls[1]
    assert put["argv"][:3] == ["ssm", "put-parameter", "--cli-input-json"] and put["argv"][3].startswith("file://")
    assert not Path(put["argv"][3][len("file://") :]).exists()  # the body's temporary file is gone
    assert json.loads(put["stdin"]) == {
        "Name": "/zeroad-live/litellm-key",
        "Type": "SecureString",
        "Overwrite": True,
        "Value": SECRET,
    }
    assert all(SECRET not in " ".join(c["argv"]) for c in calls)
    assert SECRET not in out.stdout + out.stderr


def test_put_secret_refuses_an_rtmp_server_that_is_not_an_rtmp_url_and_the_wrong_account(stubs):
    out = run(stubs, "put-secret.sh", "rtmp-server", "--yes", stdin="https://x.example\n")
    assert out.returncode != 0 and "nothing stored" in out.stderr and not stubs["log"].exists()
    stubs["live_env"].write_text(stubs["live_env"].read_text() + "LIVE_AWS_ACCOUNT=123456789012\n")
    rules = [["sts&&get-caller-identity", "999999999999\n", 0]]
    out = run(stubs, "put-secret.sh", "rtmp-key", "--yes", stdin="k\n", extra={"STUB_AWS_RULES": json.dumps(rules)})
    assert out.returncode != 0 and "not account 123456789012" in out.stderr
    assert [c["argv"][1] for c in aws_calls(stubs)] == ["get-caller-identity"]


def test_with_model_puts_the_key_only_in_the_commands_environment(stubs):
    rules = [["ssm&&get-parameter&&/zeroad-live/litellm-key&&--with-decryption", SECRET + "\n", 0]]
    check = 'test "$LITELLM_API_KEY" = "$0" && test "$LITELLM_BASE_URL" = https://litellm.example && echo key-ok'
    out = run_box(stubs, "with-model.sh", "sh", "-c", check, SECRET, STUB_AWS_RULES=json.dumps(rules))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "key-ok" and SECRET not in out.stderr
    failing = run_box(stubs, "with-model.sh", "true", STUB_AWS_RULES=json.dumps([["get-parameter", "", 255]]))
    assert failing.returncode != 0 and "put-secret.sh litellm-key" in failing.stderr


@pytest.fixture
def ffmpeg_stub(stubs):
    """An ffmpeg that records its arguments, says something with the URL on stderr (as ffmpeg's errors do), and then
    either exits or (STUB_FFMPEG_WAIT) waits for a signal and records which one it got."""
    argv_file = stubs["tmp"] / "ffmpeg-argv"
    # Python, like ffmpeg, installs its own SIGINT handler: a background job of a non-interactive bash starts with
    # SIGINT ignored, and a bash stub could not trap it (ffmpeg's term_init resets it).
    write_exe(
        stubs["bin"] / "ffmpeg",
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, signal, sys, time
            open("{argv_file}", "w").write("\\n".join(sys.argv[1:]))
            sys.stderr.write(f"[flv] Connection to {{sys.argv[-1]}} failed\\n")
            sys.stderr.flush()
            if not os.environ.get("STUB_FFMPEG_WAIT"):
                sys.exit(1)
            def got(sig, _):
                open("{argv_file}.signal", "w").write(signal.Signals(sig).name)
                sys.exit(0)
            signal.signal(signal.SIGINT, got)
            signal.signal(signal.SIGTERM, got)
            open("{argv_file}.ready", "w").close()
            while True:
                time.sleep(0.1)
            """
        ),
    )
    return argv_file


RTMP_RULES = json.dumps(
    [
        ["get-parameter&&/zeroad-live/rtmp-server", "rtmps://ingest.example:443/x/\n", 0],
        ["get-parameter&&/zeroad-live/rtmp-key", "streamkey-SECRET\n", 0],
    ]
)


def test_the_push_reads_the_target_from_ssm_and_never_logs_it(stubs, ffmpeg_stub):
    out = run_box(stubs, "push.sh", "run", STUB_AWS_RULES=RTMP_RULES, ZEROAD_PUSH_ONCE="1")
    assert out.returncode == 0, out.stderr
    argv = ffmpeg_stub.read_text().split("\n")
    assert argv[argv.index("-i") + 1] == "rtmp://127.0.0.1:1935/zeroad" and "copy" in argv
    assert "rtmps://ingest.example:443/x/streamkey-SECRET" in argv
    log = (stubs["tmp"] / "log" / "push.log").read_text()
    assert "Connection to <rtmp url> failed" in log
    assert "streamkey-SECRET" not in log + out.stdout + out.stderr
    assert all("--with-decryption" in c["argv"] for c in aws_calls(stubs))


def test_the_push_stop_ends_ffmpeg_with_sigint_so_the_stream_closes_cleanly(stubs, ffmpeg_stub):
    proc = subprocess.Popen(
        ["bash", str(BOX / "push.sh"), "run"],
        env={**box_env(stubs, STUB_AWS_RULES=RTMP_RULES, STUB_FFMPEG_WAIT="1")},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = Path(f"{ffmpeg_stub}.ready")
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.1)
    assert ready.exists()
    proc.send_signal(signal.SIGTERM)  # what systemctl stop zeroad-push sends (KillMode=mixed: push.sh only)
    proc.wait(timeout=10)
    assert Path(f"{ffmpeg_stub}.signal").read_text().strip() == "SIGINT"
    assert (stubs["tmp"] / "log" / "push.log").read_text().rstrip().endswith("push: stopped")


def test_the_push_without_its_ssm_parameters_does_not_start(stubs):
    out = run_box(stubs, "push.sh", "start", STUB_AWS_RULES=json.dumps([["get-parameter", "", 255]]))
    assert out.returncode != 0 and "store it with live/put-secret.sh" in out.stderr


def run_shim(stubs, *args):
    real = write_exe(stubs["tmp"] / "real-docker", '#!/bin/sh\nprintf "%s\\n" "$@"\n')
    return subprocess.run(
        ["bash", str(BOX / "docker-shim.sh"), *args],
        env={**stubs["env"], "ZEROAD_REAL_DOCKER": str(real), "ZEROAD_SHIM_LOG": str(stubs["tmp"] / "shim.log")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


def test_the_docker_shim_gives_only_the_live_env_container_the_gpu_the_display_and_the_relay(stubs):
    live = run_shim(stubs, "run", "-d", "--name", "env-1", "-e", "A=1", "localhost:5000/mcp-server-zeroad-live:v3")
    assert live[:3] == ["run", "--gpus", "all"]
    for flag in [
        "/tmp/.X11-unix:/tmp/.X11-unix",
        "ZEROAD_DISPLAY=:0",
        "ZEROAD_ENCODER=nvenc",
        "ZEROAD_RTMP_URL=rtmp://host.docker.internal:1935/zeroad",
        "NVIDIA_DRIVER_CAPABILITIES=all",
    ]:
        assert flag in live
    assert live[-6:] == ["-d", "--name", "env-1", "-e", "A=1", "localhost:5000/mcp-server-zeroad-live:v3"]
    plain = ["create", "--name", "env-2", "localhost:5000/mcp-server-zeroad:v1"]
    assert run_shim(stubs, *plain) == plain
    agent = ["run", "-d", "-e", "NOTE=mcp-server-zeroad-live", "a2a-agent-zeroad-commander"]
    assert run_shim(stubs, *agent) == agent
    assert run_shim(stubs, "ps", "-a") == ["ps", "-a"]


WRONG_TAGS = ["running zeroad-live-onair someone-elses\n", "running other-box zeroad-live\n"]


@pytest.mark.parametrize("tags", WRONG_TAGS, ids=["wrong-purpose", "wrong-name"])
def test_teardown_refuses_a_box_whose_name_or_purpose_tag_is_not_ours(stubs, tags):
    out = run(
        stubs, "teardown.sh", ID, "--yes", extra={"STUB_AWS_RULES": json.dumps([["describe-instances", tags, 0]])}
    )
    assert out.returncode != 0 and "not terminating" in out.stderr
    assert not [c for c in aws_calls(stubs) if "terminate-instances" in c["argv"]]


def test_teardown_terminates_a_box_with_both_tags(stubs):
    rules = [["describe-instances", "running zeroad-live-onair zeroad-live\n", 0], ["terminate", "shutting-down\n", 0]]
    out = run(stubs, "teardown.sh", ID, "--yes", extra={"STUB_AWS_RULES": json.dumps(rules)})
    assert out.returncode == 0, out.stderr
    assert [c["argv"][1] for c in aws_calls(stubs)] == ["describe-instances", "terminate-instances", "wait"]
    assert "aws ssm delete-parameters" in out.stderr


@pytest.mark.parametrize("tags", WRONG_TAGS, ids=["wrong-purpose", "wrong-name"])
def test_stop_box_refuses_a_box_whose_name_or_purpose_tag_is_not_ours(stubs, tags):
    out = run(
        stubs, "stop-box.sh", ID, "--yes", extra={"STUB_AWS_RULES": json.dumps([["describe-instances", tags, 0]])}
    )
    assert out.returncode != 0 and "not stopping" in out.stderr
    assert not [c for c in aws_calls(stubs) if "stop-instances" in c["argv"]]


def test_a_box_script_runs_through_ssm_with_its_arguments_and_its_output_comes_back(stubs):
    line = '{"status": "ok"}'
    rules = [
        ["send-command", "cmd-1\n", 0],
        ["get-command-invocation&&--query Status", "Success\n", 0],
        ["get-command-invocation", json.dumps({"StandardOutputContent": line + "\n", "StandardErrorContent": ""}), 0],
    ]
    out = run(stubs, "health.sh", ID, extra={"STUB_AWS_RULES": json.dumps(rules)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == line
    send = next(c["argv"] for c in aws_calls(stubs) if "send-command" in c["argv"])
    assert send[send.index("--instance-ids") + 1] == ID
    command = json.loads(send[send.index("--parameters") + 1])["commands"][0]
    script = base64.b64decode(re.search(r"echo (\S+) \| base64", command).group(1)).decode()
    assert script.startswith("set --\nexport ZEROAD_LIVE_HOME=/opt/zeroad-live\n")
    assert "health.sh: one JSON line" in script


def test_a_failed_ssm_command_fails_the_script(stubs):
    rules = [
        ["send-command", "cmd-1\n", 0],
        ["get-command-invocation&&--query Status", "Failed\n", 0],
        ["get-command-invocation", json.dumps({"StandardOutputContent": "", "StandardErrorContent": "boom"}), 0],
    ]
    out = run(stubs, "repair.sh", ID, "--yes", extra={"STUB_AWS_RULES": json.dumps(rules)})
    assert out.returncode != 0 and "boom" in out.stderr and "ended Failed" in out.stderr


def test_health_flags_a_match_whose_stream_does_not_reach_the_relay(stubs):
    units = {"zeroad-x": "active", "zeroad-relay": "active", "zeroad-push": "inactive", "zeroad-match": "active"}
    write_exe(
        stubs["bin"] / "systemctl",
        "#!/bin/sh\ncase $2 in\n"
        + "".join(f"  {u}) echo {s}; [ {s} = active ] ;;\n" for u, s in units.items())
        + "  *) echo inactive; exit 3 ;;\nesac\n",
    )
    write_exe(stubs["bin"] / "xdpyinfo", "#!/bin/sh\nexit 0\n")
    write_exe(stubs["bin"] / "docker", "#!/bin/sh\necho 'localhost:5000/mcp-server-zeroad-live:v3 Up 3 minutes'\n")
    write_exe(stubs["bin"] / "curl", '#!/bin/sh\necho \'{"ready": false, "bytesReceived": 0}\'\n')
    write_exe(stubs["bin"] / "nvidia-smi", "#!/bin/sh\necho '41, 12, 3100, 55'\n")
    write_exe(stubs["bin"] / "busctl", "#!/bin/sh\necho 't 0'\n")
    state = stubs["tmp"] / "state"
    state.mkdir()
    (state / "push.wanted").touch()
    out = run_box(stubs, "health.sh")
    assert out.returncode == 0, out.stderr
    health = json.loads(out.stdout)
    assert health["status"] == "warn"
    assert health["problems"] == [
        "push wanted but not running",
        "match running but nothing publishes to the relay",
        "match running but the recording is not growing",
    ]
    assert health["gpu"] == {"gpu_pct": "41", "nvenc_pct": "12", "mem_mb": "3100", "temp_c": "55"}
    assert health["env_container"].startswith("localhost:5000/mcp-server-zeroad-live")


def test_the_runbook_names_only_scripts_that_exist_and_marks_each_go_ahead_step():
    doc = (ROOT / "docs" / "livestream.md").read_text()
    named = set(re.findall(r"live/[\w/.-]+\.sh", doc))
    assert len(named) >= 12
    for path in named:
        assert (ROOT / path).is_file(), path
    for fact in ["$1.32", "$9", "g6.4xlarge", "--yes", "put-secret.sh", "SSM", "teardown.sh"]:
        assert fact in doc
    assert doc.count("GO-AHEAD") >= 4
    assert "hidepid=2" in doc  # the push URL is on ffmpeg's command line; only root may read other argv
    assert "Only root and SSM sessions" not in doc


def test_no_box_script_reads_a_secret_into_a_file_or_a_log():
    for path in [*LIVE.glob("*.sh"), *BOX.glob("*.sh")]:
        text = path.read_text()
        for line in text.splitlines():
            if "--with-decryption" in line and not line.lstrip().startswith("#"):
                assert ">" not in line.split("--with-decryption")[1].split(")")[0], f"{path.name}: {line}"


# The box scripts below run for real (DRY_RUN=0) against stubs, so that their conditions are exercised: a dry run
# prints every branch whatever the state of the box.

STUB_SYSTEMCTL = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    units = json.loads(os.environ.get("STUB_UNITS", "{{}}"))
    if sys.argv[1] == "is-active":
        state = units.get(sys.argv[2], "inactive")
        print(state)
        sys.exit(0 if state == "active" else 3)
    with open(os.environ["STUB_SYSTEMCTL_LOG"], "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
    """
)


@pytest.fixture
def box_stubs(stubs):
    """systemctl, systemd-run, docker and xhost stubs on PATH; systemctl answers is-active from STUB_UNITS and logs
    every other call, as does systemd-run."""
    write_exe(stubs["bin"] / "systemctl", STUB_SYSTEMCTL.format(python=sys.executable))
    calls = stubs["tmp"] / "systemctl.log"
    write_exe(stubs["bin"] / "systemd-run", f'#!/bin/sh\necho "systemd-run $*" >> "{calls}"\n')
    write_exe(stubs["bin"] / "docker", '#!/bin/sh\n[ -z "$STUB_DOCKER_PS" ] || echo "$STUB_DOCKER_PS"\n')
    write_exe(stubs["bin"] / "xhost", "#!/bin/sh\nexit 0\n")
    (stubs["tmp"] / "state").mkdir()
    stubs["env"].update(STUB_SYSTEMCTL_LOG=str(calls), ZEROAD_DISPLAY_WAIT_S="0", ZEROAD_MIN_FREE_GB="0")
    return stubs


def systemctl_calls(stubs) -> list[str]:
    log = stubs["tmp"] / "systemctl.log"
    return log.read_text().splitlines() if log.exists() else []


ALL_UP = {"zeroad-x": "active", "zeroad-relay": "active", "zeroad-push": "inactive", "zeroad-match": "inactive"}


def start_match(stubs, *args, units=ALL_UP, **extra):
    clock = {"ZEROAD_NOW_EPOCH": "1800000000", "ZEROAD_BOOT_EPOCH": "1800000000"}
    return run_box(
        stubs, "match.sh", "start", *args, "--budget", "20", DRY_RUN="0", STUB_UNITS=json.dumps(units), **clock, **extra
    )


def test_a_second_match_is_refused_while_one_is_running(box_stubs):
    out = start_match(box_stubs, "rehearsal", "--name", "r2", units={**ALL_UP, "zeroad-match": "active"})
    assert out.returncode != 0 and "a match is running" in out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_a_match_is_refused_while_an_env_container_of_an_earlier_match_is_still_up(box_stubs):
    out = start_match(box_stubs, "rehearsal", "--name", "r2", STUB_DOCKER_PS="localhost:5000/mcp-server-zeroad-live:v1")
    assert out.returncode != 0 and "live env container is still up" in out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_a_match_without_push_stops_a_push_left_on_by_an_earlier_match(box_stubs):
    wanted = box_stubs["tmp"] / "state" / "push.wanted"
    wanted.touch()
    out = start_match(box_stubs, "rehearsal", "--name", "r1", units={**ALL_UP, "zeroad-push": "active"})
    assert out.returncode == 0, out.stderr
    assert "a push was still on" in out.stdout and "push: off" in out.stdout
    assert not wanted.exists()
    calls = systemctl_calls(box_stubs)
    assert "stop zeroad-push" in calls and not [c for c in calls if c.startswith("start zeroad-push")]
    assert calls.index("stop zeroad-push") < next(i for i, c in enumerate(calls) if c.startswith("systemd-run"))


def test_a_match_without_push_leaves_no_push_running_when_none_was_on(box_stubs):
    out = start_match(box_stubs, "rehearsal", "--name", "r1")
    assert out.returncode == 0, out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if "zeroad-push" in c]


def test_push_start_marks_the_push_wanted_and_stop_unmarks_it(box_stubs):
    rules = json.dumps([["get-parameter", "/zeroad-live/x\n", 0]])
    wanted = box_stubs["tmp"] / "state" / "push.wanted"
    assert run_box(box_stubs, "push.sh", "start", DRY_RUN="0", STUB_AWS_RULES=rules).returncode == 0
    assert wanted.exists()
    assert run_box(box_stubs, "push.sh", "stop", DRY_RUN="0").returncode == 0
    assert not wanted.exists()  # else the half-hourly repair would start the public push again
    assert systemctl_calls(box_stubs) == ["start zeroad-push", "stop zeroad-push"]


@pytest.mark.parametrize(
    ("units", "wanted", "expected"),
    [
        (ALL_UP, False, []),
        ({**ALL_UP, "zeroad-x": "inactive"}, False, ["restart zeroad-x"]),
        ({**ALL_UP, "zeroad-relay": "failed"}, False, ["restart zeroad-relay"]),
        (ALL_UP, True, ["start zeroad-push"]),
        ({**ALL_UP, "zeroad-push": "active"}, True, []),
    ],
    ids=["all-up", "display-down", "relay-down", "push-wanted-but-down", "push-up"],
)
def test_repair_restarts_only_what_is_down(box_stubs, units, wanted, expected):
    if wanted:
        (box_stubs["tmp"] / "state" / "push.wanted").touch()
    out = run_box(box_stubs, "repair.sh", DRY_RUN="0", STUB_UNITS=json.dumps(units))
    assert out.returncode == 0, out.stderr
    assert systemctl_calls(box_stubs) == expected
    assert ("nothing to do" in out.stdout) == (not expected)


def test_repair_frees_disk_by_deleting_the_oldest_segments_never_the_newest(box_stubs):
    rec = box_stubs["tmp"] / "state" / "rec"
    rec.mkdir()
    for i, name in enumerate(["a.mp4", "b.mp4", "c.mp4"]):
        (rec / name).write_text(name)
        os.utime(rec / name, (1_000 + i, 1_000 + i))
    out = run_box(box_stubs, "repair.sh", DRY_RUN="0", STUB_UNITS=json.dumps(ALL_UP), ZEROAD_MIN_FREE_GB="999999")
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in rec.iterdir()) == ["c.mp4"]
    assert "no old segment left to delete" in out.stdout


@pytest.fixture
def box_home(box_stubs):
    """A copy of the box scripts with stub uv, aws and systemctl in its bin: run-match.sh puts only $BOX_HOME/bin and
    the system folders on PATH. The uv stub logs every call; `agent-env run` writes .ready, then (STUB_AE_WAIT) waits
    for a signal, logs it and exits 143 as agent-env does after its teardown, or (STUB_AE_RESULTS) writes a results
    file."""
    import shutil

    home = box_stubs["tmp"] / "home"
    shutil.copytree(BOX, home / "box")
    bin_dir = home / "bin"
    bin_dir.mkdir()
    for name in ("aws", "systemctl"):
        shutil.copy(box_stubs["bin"] / name, bin_dir / name)
    uv_log = box_stubs["tmp"] / "uv.log"
    write_exe(
        bin_dir / "uv",
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, pathlib, signal, sys, time
            log = open("{uv_log}", "a", buffering=1)
            argv = sys.argv[1:]
            log.write(" ".join(argv) + "\\n")
            if "agent-env" in argv:
                if os.environ.get("STUB_AE_RESULTS"):
                    out = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "runs" / "m-results.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text('{{"spend": 1}}')
                if os.environ.get("STUB_AE_WAIT"):
                    def got(sig, _):
                        log.write("agent-env got " + signal.Signals(sig).name + ", tearing down\\n")
                        time.sleep(0.3)
                        sys.exit(143)
                    signal.signal(signal.SIGTERM, got)
                    open("{uv_log}.ready", "w").close()
                    while True:
                        time.sleep(0.05)
            elif "results" in argv:
                print("1.25")
            """
        ),
    )
    return {**box_stubs, "uv_log": uv_log, "home": home}


def home_env(box_home, **extra) -> dict:
    return box_env(box_home, ZEROAD_LIVE_HOME=str(box_home["home"]), DRY_RUN="0", **extra)


def booked(box_home) -> list[str]:
    calls = box_home["uv_log"].read_text().splitlines()
    return [c.split("--usd ")[1] for c in calls if "spend add" in c]


def test_stopping_a_match_reaches_agent_env_and_still_books_its_spend(box_home):
    proc = subprocess.Popen(
        ["bash", str(box_home["tmp"] / "home" / "box" / "run-match.sh"), "m1", "model", "livestream", "9.00"],
        env=home_env(box_home, STUB_AE_WAIT="1", STUB_AWS_RULES=json.dumps([["get-parameter", "k\n", 0]])),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = Path(f"{box_home['uv_log']}.ready")
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.1)
    assert ready.exists()
    proc.send_signal(signal.SIGTERM)  # what systemctl stop zeroad-match sends (KillMode=mixed: run-match.sh only)
    out, err = proc.communicate(timeout=20)
    assert "agent-env got SIGTERM, tearing down" in box_home["uv_log"].read_text(), err
    assert booked(box_home) == ["9.00"]  # no results after a stop: the full caps
    assert "match m1 ended (exit 143), spend $9.00" in out


@pytest.mark.parametrize(
    ("kind", "cap", "extra", "usd"),
    [
        ("model", "9.00", {}, "9.00"),
        ("rehearsal", "0.00", {}, "0"),
        ("model", "9.00", {"STUB_AE_RESULTS": "1"}, "1.25"),
    ],
    ids=["model-no-results", "rehearsal-no-results", "model-with-results"],
)
def test_a_finished_match_books_its_results_or_its_full_caps(box_home, kind, cap, extra, usd):
    # No $STATE/agentenv folder at the start: the results lookup must not end the script before the ledger line.
    rules = json.dumps([["get-parameter", "k\n", 0]])
    script = box_home["tmp"] / "home" / "box" / "run-match.sh"
    out = subprocess.run(
        ["bash", str(script), "m2", kind, "livestream", cap],
        env=home_env(box_home, STUB_AWS_RULES=rules, **extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert booked(box_home) == [usd]
    assert f"match m2 ended (exit 0), spend ${usd}" in out.stdout


def test_a_finished_match_ends_the_push_it_was_pushed_with(box_home):
    (box_home["tmp"] / "state" / "push.wanted").touch()
    out = subprocess.run(
        ["bash", str(box_home["tmp"] / "home" / "box" / "run-match.sh"), "m3", "rehearsal", "livestream-petra", "0"],
        env=home_env(box_home, STUB_UNITS=json.dumps({**ALL_UP, "zeroad-push": "active"})),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert systemctl_calls(box_home) == ["stop zeroad-push"]
    assert not (box_home["tmp"] / "state" / "push.wanted").exists()


def test_the_budget_keeps_the_box_hours_of_earlier_boots(box_stubs):
    now = 1_800_000_000
    tick = run_box(box_stubs, "budget.sh", "tick", ZEROAD_NOW_EPOCH=str(now), ZEROAD_BOOT_EPOCH=str(now - 10 * 3600))
    assert tick.returncode == 0, tick.stderr
    # stop-box after 10 h, then start again an hour later: a new boot, just up. Since boot alone it would be $10.32.
    later = {"ZEROAD_NOW_EPOCH": str(now + 3600), "ZEROAD_BOOT_EPOCH": str(now + 3600)}
    out = run_box(
        box_stubs, "budget.sh", "check", "--planned-usd", "9", "--planned-hours", "1", "--budget", "23", **later
    )
    assert out.returncode == 1, out.stdout
    assert "box $13.20 (10.0 h at $1.32/h, 2 boots)" in out.stdout
    assert "REFUSED: $23.52 would pass the $23 budget" in out.stdout


@pytest.mark.parametrize("budget", ["nan", "NaN", "inf", "-inf"])
def test_a_budget_that_is_not_a_finite_number_is_refused(stubs, budget):
    out = run_box(stubs, "budget.sh", "check", "--planned-usd", "9", "--planned-hours", "1", "--budget", budget)
    assert out.returncode != 0 and "REFUSED" not in out.stdout


def test_box_bootstrap_hides_other_users_command_lines_before_anything_is_installed(stubs):
    out = run_box(stubs, "bootstrap.sh", "https://example.invalid/payload", "https://example.invalid/repo", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    plan = out.stdout.replace("\\", "")  # the dry run prints with printf %q
    assert "remount,hidepid=2 /proc" in plan  # the push's RTMP URL is on ffmpeg's argv
    assert "hidepid=2 0 0" in plan  # kept across reboots
    assert plan.index("hidepid=2") < plan.index("apt-get")


def start_pilot(stubs, *args, units=None, **extra):
    write_exe(stubs["bin"] / "curl", "#!/bin/sh\nexit 0\n")  # vLLM's health
    clock = {"ZEROAD_NOW_EPOCH": "1800000000", "ZEROAD_BOOT_EPOCH": "1800000000"}
    return run_box(
        stubs, "pilot.sh", "start", *args, "--budget", "150", DRY_RUN="0", STUB_UNITS=json.dumps(units or {}), **clock,
        **extra,
    )  # fmt: skip


def test_the_pilot_starts_its_round_loop_in_its_own_unit_with_the_frontier_probe_unless_told_not_to(box_stubs):
    out = start_pilot(box_stubs, "--baseline")
    assert out.returncode == 0, out.stderr
    [call] = [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]
    assert "--unit zeroad-pilot" in call and "KillMode=control-group" in call
    assert call.split("pilot-run.sh ")[1] == "1 --budget 150 --baseline"
    out = start_pilot(box_stubs, "--no-frontier", "--rounds", "3")
    assert out.returncode == 0, out.stderr
    call = [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")][-1]
    assert call.split("pilot-run.sh ")[1] == "0 --budget 150 --frontier  --rounds 3"


def test_a_second_pilot_is_refused_while_one_runs_and_a_bad_number_before_anything(box_stubs):
    out = start_pilot(box_stubs, units={"zeroad-pilot": "active"})
    assert out.returncode != 0 and "the pilot is running" in out.stderr
    out = start_pilot(box_stubs, "--rounds", "many")
    assert out.returncode != 0 and "--rounds needs a number" in out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_a_pilot_that_would_pass_the_budget_is_refused(box_stubs):
    ledger = box_stubs["tmp"] / "state" / "spend.jsonl"
    ledger.write_text(json.dumps({"what": "pilot r00-p-sonnet", "usd_est": 149.9}) + "\n")
    out = start_pilot(box_stubs)
    assert out.returncode != 0 and "REFUSED" in out.stdout
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_the_pilot_run_puts_agent_envs_endpoint_on_vllm_without_the_frontier_and_on_litellm_with_it(stubs):
    out = run_box(stubs, "pilot-run.sh", "0", "--budget", "150", "--baseline", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    assert "agentenv_zeroad.pilot.run --home" in out.stdout and "with-model" not in out.stdout
    out = run_box(stubs, "pilot-run.sh", "1", "--budget", "150", DRY_RUN="1")
    assert "aws ssm get-parameter --name /zeroad-live/litellm-key" in out.stdout and "never printed" in out.stdout


def test_a_one_gpu_pilot_box_trains_on_the_gpu_vllm_serves_on(stubs):
    etc = stubs["tmp"] / "etc"
    etc.mkdir()
    (etc / "pilot.conf").write_text("PILOT_TRAIN_GPUS=0\nPILOT_SHARE_GPU=1\n")
    out = run_box(stubs, "pilot-run.sh", "0", "--budget", "150", DRY_RUN="1", ZEROAD_PILOT_ETC=str(etc))
    assert out.returncode == 0, out.stderr
    assert "--train-gpus 0 --share-gpu" in out.stdout.replace("\\", "")


def test_the_pilot_bootstrap_binds_vllm_to_the_docker_bridge_behind_a_key_and_splits_the_gpus(stubs):
    out = run_box(stubs, "pilot-bootstrap.sh", "https://payload", "https://repo", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    text = out.stdout.replace("\\", "")  # the plan's printf %q escapes commas and quotes
    assert "-p 172.17.0.1:8000:8000" in text and "--env-file" in text and "VLLM_API_KEY=<random>" in text
    assert "device=0,1" in text and "--tensor-parallel-size 2" in text
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True" in text
    assert "zeroad setup --source" in text and "--live" not in text and "zeroad-trainer" in text
    assert "zeroad-boxtime.timer" in text and "zeroad-x" not in text


# ---- the training box --------------------------------------------------------------------------------------------
TRAIN_ENV = LIVE / "train.env.example"
TRAIN_CONFIG = LIVE / "train-run.example.toml"


def train_stubs(stubs):
    env_file = stubs["tmp"] / "train.env"
    env_file.write_text(
        TRAIN_ENV.read_text()
        + "LIVE_INSTANCE_PROFILE=zeroad-box\nLIVE_BUCKET=bucket-1\nLIVE_AWS_ACCOUNT=123456789012\n"
        + "LIVE_SUBNETS='subnet-a subnet-b'\nLIVE_SECURITY_GROUP=sg-1\n"
    )
    stubs["env"]["LIVE_ENV"] = str(env_file)
    return stubs


def test_the_training_env_example_is_a_g6e_box_with_its_own_rate_and_a_stop_not_terminate_shutdown(stubs):
    out = run(train_stubs(stubs), "launch.sh", "--max-hours", "12", forbid_aws=True)
    plan = out.stderr
    assert out.returncode == 0, plan
    assert "one of [g6e.12xlarge g6e.24xlarge], about $10.49/h" in plan and "powers off (stop) after 12 h" in plan
    assert "--instance-type g6e.12xlarge" in plan and "--instance-initiated-shutdown-behavior stop" in plan
    assert "at most about $125.88" in plan
    assert "Key=purpose,Value=zeroad-train" in plan and "Key=Name,Value=zeroad-train" in plan


def test_the_training_env_example_matches_the_run_example_and_leaves_two_hours_after_the_run():
    env = dict(line.split("=", 1) for line in TRAIN_ENV.read_text().splitlines() if re.match(r"^[A-Z_]+=", line))
    from agentenv_zeroad.train.config import check_run, load_config

    cfg = load_config(TRAIN_CONFIG)
    assert (cfg.games.interface, cfg.games.score_rule) == ("v2", "net_worth") and check_run(cfg) == []
    assert float(env["LIVE_BOX_USD_PER_HOUR"]) == cfg.box_usd_per_hour
    assert int(env["LIVE_MAX_HOURS"]) == cfg.duration_hours + 2
    assert env["LIVE_REQUIRE_RATE"] == "1" and env["LIVE_SHUTDOWN_BEHAVIOR"] == "stop"


def test_launch_keeps_terminate_as_the_default_and_the_stop_option_overrides_the_env(stubs):
    assert "--instance-initiated-shutdown-behavior terminate" in run(stubs, "launch.sh", forbid_aws=True).stderr
    env = {"LIVE_SHUTDOWN_BEHAVIOR": "stop"}
    assert "behavior stop" in run(stubs, "launch.sh", extra=env, forbid_aws=True).stderr
    out = run(stubs, "launch.sh", "--shutdown-behavior", "terminate", extra=env, forbid_aws=True)
    assert "behavior terminate" in out.stderr


def test_a_training_env_without_its_rate_is_refused_instead_of_priced_at_the_stream_boxs(stubs):
    text = "".join(
        line + "\n" for line in TRAIN_ENV.read_text().splitlines() if not line.startswith("LIVE_BOX_USD_PER_HOUR")
    )
    (stubs["tmp"] / "train.env").write_text(text)
    out = run(stubs, "launch.sh", extra={"LIVE_ENV": str(stubs["tmp"] / "train.env")}, forbid_aws=True)
    assert out.returncode != 0 and "set LIVE_BOX_USD_PER_HOUR" in out.stderr and "no default rate" in out.stderr
    out = run(stubs, "launch.sh", forbid_aws=True)  # the stream box's env is still fine without one
    assert out.returncode == 0


def test_the_box_budget_guard_has_no_default_rate(stubs):
    out = run_box(
        stubs, "budget.sh", "show", ZEROAD_LIVE_CONF=str(stubs["tmp"] / "none.conf"), ZEROAD_BOX_USD_PER_HOUR=""
    )
    assert out.returncode != 0 and "ZEROAD_BOX_USD_PER_HOUR is not set" in out.stderr and "no default" in out.stderr


def test_the_train_iam_role_reads_and_writes_the_run_prefix_and_not_the_pools(stubs):
    out = run(train_stubs(stubs), "train-iam.sh", "create", ID, forbid_aws=True)
    plan = out.stderr
    assert out.returncode == 0, plan
    assert "zeroad-train-box" in plan and "/run/*" in plan and "/pool/" not in plan
    assert "--policy-name zeroad-pilot-run" in plan and "iam add-role-to-instance-profile" in plan
    assert "delete-role" not in plan
    assert "live/pilot-iam.sh delete" in plan and "--train --yes" in plan
    pilot = run(stubs, "pilot-iam.sh", "create", ID, forbid_aws=True).stderr
    assert "zeroad-pilot-box" in pilot and "zeroad-pilot-pool" in pilot and "/run/" not in pilot
    out = run(train_stubs(stubs), "train-iam.sh", "delete", ID, forbid_aws=True)
    assert "delete-role --role-name zeroad-train-box" in out.stderr


def test_the_training_bootstrap_serves_two_bf16_vllm_instances_on_gpu_0_and_1_and_leaves_2_and_3(stubs):
    out = run_box(stubs, "pilot-bootstrap.sh", "https://payload", "https://repo", "--train", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    text = out.stdout.replace("\\", "")
    runs = [line for line in text.splitlines() if line.startswith("+ docker run -d --name zeroad-vllm")]
    assert [re.search(r"--name (\S+)", r).group(1) for r in runs] == ["zeroad-vllm-0", "zeroad-vllm-1"]
    for n, line in enumerate(runs):
        assert (
            f'--gpus "device={n}"' in line and "--tensor-parallel-size 1" in line and "--restart unless-stopped" in line
        )
        assert f"-p 172.17.0.1:800{n}:8000 -p 127.0.0.1:800{n}:8000" in line
        flags = (
            "--dtype bfloat16",
            "--generation-config vllm",
            "--enable-lora",
            "--max-lora-rank 32",
            "--max-loras 8",
            "--max-cpu-loras 32",
            "--tool-call-parser hermes",
            "--enable-prefix-caching",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True",
            ":/train:ro",
        )
        for flag in flags:
            assert flag in line, flag
        assert "--quantization" not in line and "/adapters" not in line
    assert "800 8001" not in text and "vllm.urls 8000 8001" in text
    assert "PILOT_TRAIN_GPUS=%s" in text and "2,3 0" in text.replace("\\,", ",")
    assert "train.smoke" in text and "pilot.smoke" not in text
    assert "zeroad-trainer" in text
    assert (
        "--max-model-len 8192 " in "".join(runs) and "12288" not in text
    )  # the learner's max_len: play nothing longer
    lines = text.splitlines()
    first_run = next(i for i, line in enumerate(lines) if line.startswith("+ docker run -d --name zeroad-vllm-0"))
    assert any(
        line.startswith("+ docker rm -f zeroad-vllm ") or line == "+ docker rm -f zeroad-vllm"
        for line in lines[:first_run]
    )


def test_the_pilot_bootstrap_still_serves_one_tp2_instance_and_does_not_mount_the_train_folder(stubs):
    out = run_box(stubs, "pilot-bootstrap.sh", "https://payload", "https://repo", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    text = out.stdout.replace("\\", "")
    runs = [line for line in text.splitlines() if line.startswith("+ docker run -d")]
    assert len(runs) == 1 and "--name zeroad-vllm " in runs[0] and "--max-loras 4 --max-cpu-loras 16" in runs[0]
    assert "/adapters:ro" in runs[0] and ":/train" not in runs[0] and "--generation-config" not in runs[0]
    assert "then pilot.smoke" in text and "train.smoke" not in text
    assert "--max-model-len 12288 " in runs[0] and "docker rm -f zeroad-vllm " not in text
    out = run_box(stubs, "pilot-bootstrap.sh", "https://payload", "https://repo", "--bogus", DRY_RUN="1")
    assert out.returncode == 2


def test_the_laptop_start_sends_the_config_then_starts_the_run_with_the_budget(stubs):
    env_file = stubs["tmp"] / "train.env"
    env_file.write_text(
        TRAIN_ENV.read_text()
        + "LIVE_INSTANCE_PROFILE=zeroad-box\nLIVE_BUCKET=bucket-1\nLIVE_AWS_ACCOUNT=123456789012\n"
    )
    stubs["env"]["LIVE_ENV"] = str(env_file)
    out = run(stubs, "train.sh", ID, "start", str(TRAIN_CONFIG), "--play-only", forbid_aws=True)
    plan = out.stderr + out.stdout
    assert out.returncode == 0, plan
    assert plan.count("aws ssm send-command") == 2
    assert "train.sh put-config" in plan and "train.sh start --play-only --budget 150" in plan
    sent = [base64.b64decode(b).decode() for b in re.findall(r"[A-Za-z0-9+/=]{200,}", plan)]
    assert any('run_id = "run1"' in text for text in sent)


def test_the_box_train_run_starts_the_learner_on_the_training_gpus_and_then_the_orchestrator(stubs):
    state = stubs["tmp"] / "state"
    out = run_box(stubs, "train-run.sh", str(TRAIN_CONFIG), "--init", f"{state}/train/start", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    text = out.stdout.replace("\\", "")
    [learner] = [line for line in text.splitlines() if "docker run -d --name zeroad-learner" in line]
    assert '--gpus "device=2,3"' in learner and "torchrun --nproc_per_node 2 /app/learner.py" in learner
    assert (
        f"--root {state}/train/RUN_ID" in learner
        and "--known-tools" in learner
        and f"--init {state}/train/start" in learner
    )
    assert "agentenv_zeroad.train.orchestrate run --config" in text and "--play-only" not in text
    out = run_box(stubs, "train-run.sh", str(TRAIN_CONFIG), "--play-only", DRY_RUN="1")
    assert (
        "zeroad-learner" not in out.stdout and "orchestrate run --config" in out.stdout and "--play-only" in out.stdout
    )


def start_train(stubs, *args, units=None, rate=1.32, hours=1, **extra):
    write_exe(stubs["bin"] / "curl", "#!/bin/sh\nexit 0\n")  # vLLM's health
    etc = stubs["tmp"] / "etc"
    etc.mkdir(exist_ok=True)
    (etc / "vllm.urls").write_text("http://172.17.0.1:8000\nhttp://172.17.0.1:8001\n")
    (stubs["tmp"] / "state" / "train").mkdir(parents=True, exist_ok=True)
    state = stubs["tmp"] / "state"
    (state / "train" / "config.toml").write_text(
        f'run_id = "run1"\nroot = "{state}/train/run1"\nbudget_usd = 100\nbox_usd_per_hour = {rate}\n'
        f"duration_hours = {hours}\n"
        'servers = { urls = ["http://172.17.0.1:8000"], adapter_dir = "/train/run1/adapters" }\n'
    )
    clock = {"ZEROAD_NOW_EPOCH": "1800000000", "ZEROAD_BOOT_EPOCH": "1800000000"}
    return run_box(
        stubs, "train.sh", "start", *args, "--budget", "150", DRY_RUN="0", STUB_UNITS=json.dumps(units or {}),
        ZEROAD_PILOT_ETC=str(etc), **clock, **extra,
    )  # fmt: skip


def test_the_run_starts_in_its_own_unit_and_a_second_is_refused(box_stubs):
    out = start_train(box_stubs, "--play-only")
    assert out.returncode == 0, out.stderr
    [call] = [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]
    assert "--unit zeroad-train" in call and "KillMode=control-group" in call
    assert "TimeoutStopSec=2400" in call  # 3 final uploads of 10 minutes (sync.SYNC_TIMEOUT_S) and the learner's grace
    assert "After=network-online.target docker.service" in call and "Wants=network-online.target docker.service" in call
    assert call.split("train-run.sh ")[1].endswith("config.toml --play-only")
    out = start_train(box_stubs, units={"zeroad-train": "active"})
    assert out.returncode != 0 and "a run is going" in out.stderr
    assert len([c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]) == 1


def test_a_run_that_would_pass_the_budget_or_names_a_path_outside_the_train_folder_is_refused(box_stubs):
    ledger = box_stubs["tmp"] / "state" / "spend.jsonl"
    ledger.write_text(json.dumps({"what": "panel", "usd_est": 149.9}) + "\n")
    out = start_train(box_stubs)
    assert out.returncode != 0 and "REFUSED" in out.stdout
    ledger.unlink()
    out = start_train(box_stubs, "--init", "/etc/passwd")
    assert out.returncode != 0 and "needs a path under" in out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_the_box_config_upload_refuses_what_is_not_base64(stubs):
    out = run_box(stubs, "train.sh", "put-config", "not base64!", DRY_RUN="1")
    assert out.returncode != 0 and "needs base64" in out.stderr


# ---- review fixes: rates per type, the power-off on every boot, the run's start checks -------------------------------
def launch_rules(*extra) -> str:
    return json.dumps(
        [
            ["sts&&get-caller-identity", "123456789012\n", 0],
            ["describe-instances", "None\n", 0],
            ["ssm&&get-parameter", "ami-1\n", 0],
            *extra,
            ["describe-instance-information", "Online\n", 0],
        ]
    )


def test_a_launch_that_falls_back_to_the_bigger_type_is_tagged_with_that_types_rate(stubs):
    rules = launch_rules(
        ["run-instances&&g6e.12xlarge", "", 254], ["run-instances&&g6e.24xlarge", "i-00000000000000002\n", 0]
    )
    out = run(train_stubs(stubs), "launch.sh", "--yes", extra={"STUB_AWS_RULES": rules})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "i-00000000000000002"
    runs = [c["argv"] for c in aws_calls(stubs) if "run-instances" in c["argv"]]
    assert [a[a.index("--instance-type") + 1] for a in runs][-1] == "g6e.24xlarge"
    tags = runs[-1][runs[-1].index("--tag-specifications") + 1]
    assert "Key=zeroad-box-usd-per-hour,Value=15.07" in tags and "Key=zeroad-instance-type,Value=g6e.24xlarge" in tags
    assert "Key=zeroad-box-usd-per-hour,Value=10.49" in runs[0][runs[0].index("--tag-specifications") + 1]
    assert "g6e.24xlarge costs $15.07/h, not the $10.49/h" in out.stderr and "must be 15.07" in out.stderr


def test_a_fallback_type_with_no_rate_is_refused_before_anything_is_launched(stubs):
    train_stubs(stubs)
    env_file = Path(stubs["env"]["LIVE_ENV"])
    env_file.write_text(
        re.sub(r"^LIVE_GPU_RATES=.*\n", 'LIVE_GPU_RATES="g6e.12xlarge=10.49"\n', env_file.read_text(), flags=re.M)
    )
    for args in ((), ("--yes",)):
        out = run(stubs, "launch.sh", *args, extra={"STUB_AWS_RULES": launch_rules()}, forbid_aws=not args)
        assert out.returncode != 0 and "no rate for the fallback type g6e.24xlarge" in out.stderr
    assert not any("run-instances" in c["argv"] for c in aws_calls(stubs))


def test_a_stream_box_without_rates_still_falls_back_at_its_default_rate_and_warns(stubs):
    env = {"LIVE_GPU_TYPES": "g6.4xlarge gr6.4xlarge"}  # the stream settings: no LIVE_GPU_RATES, no LIVE_REQUIRE_RATE
    rules = launch_rules(
        ["run-instances&&g6.4xlarge", "", 254], ["run-instances&&gr6.4xlarge", "i-00000000000000003\n", 0]
    )
    out = run(stubs, "launch.sh", "--yes", extra={"STUB_AWS_RULES": rules, **env})
    assert out.returncode == 0, out.stderr
    runs = [c["argv"] for c in aws_calls(stubs) if "run-instances" in c["argv"]]
    tags = runs[-1][runs[-1].index("--tag-specifications") + 1]
    assert (
        "Key=zeroad-box-usd-per-hour,Value=1.32" in tags and "no rate for gr6.4xlarge in LIVE_GPU_RATES" in out.stderr
    )


def test_the_power_off_is_armed_on_every_boot_from_the_launch_deadline(stubs, tmp_path):
    out = run(stubs, "launch.sh", "--max-hours", "12", forbid_aws=True)
    script = base64.b64decode(re.search(r"--user-data (\S+)", out.stderr).group(1)).decode()
    assert script.startswith("#cloud-boothook\n")  # cloud-init runs a boothook at every boot, user-data scripts once
    deadline = int(re.search(r"^DEADLINE=(\d+)$", script, re.M).group(1))
    assert abs(deadline - (time.time() + 12 * 3600)) < 120

    def minutes_scheduled(boot_epoch: int) -> int:
        bin_dir = tmp_path / f"bin{boot_epoch}"
        bin_dir.mkdir()
        write_exe(bin_dir / "date", f"#!/bin/sh\necho {boot_epoch}\n")
        write_exe(bin_dir / "shutdown", '#!/bin/sh\necho "$@" > "$STUB_SHUTDOWN"\n')
        log = tmp_path / f"shutdown{boot_epoch}"
        # DEADLINE is computed from the launch's `date`; the boot's `date` is the stub's
        body = script.replace(re.search(r"^DEADLINE=.*$", script, re.M).group(0), f"DEADLINE={deadline}")
        subprocess.run(
            ["bash", "-c", body], env={"PATH": f"{bin_dir}:/usr/bin:/bin", "STUB_SHUTDOWN": str(log)}, check=True
        )
        return int(log.read_text().split()[1].lstrip("+"))

    assert minutes_scheduled(deadline - 12 * 3600) == 720  # first boot
    assert minutes_scheduled(deadline - 5 * 3600) == 300  # started again after a stop: the remaining hours, not 12
    assert minutes_scheduled(deadline + 3600) == 30  # started after the deadline: a half hour to fetch the results


def test_the_box_config_carries_the_rate_the_cap_and_the_free_disk_floor(stubs):
    out = subprocess.run(
        ["bash", "-c", f'source "{LIVE}/lib/common.sh"; RATE=15.07; LIVE_MATCH_CAP_USD=5; AWS_REGION=x; box_conf'],
        capture_output=True, text=True, check=True, env={**os.environ, "LIVE_ENV": str(stubs["live_env"])},
    )  # fmt: skip
    assert "ZEROAD_BOX_USD_PER_HOUR=15.07" in out.stdout and "ZEROAD_MATCH_CAP_USD=5" in out.stdout


def test_a_run_whose_config_rate_is_not_the_boxs_is_refused(box_stubs):
    out = start_train(box_stubs, rate=10.49)  # the box's config says 1.32
    assert out.returncode != 0 and "is not this box's rate" in out.stderr
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]


def test_a_run_whose_whole_duration_would_pass_the_budget_is_refused_at_the_start(box_stubs):
    out = start_train(box_stubs, hours=200)  # 200 h at $1.32 is $264 against a $150 budget; nothing is spent yet
    assert out.returncode != 0 and "REFUSED" in out.stdout and "$264" in out.stdout
    assert not [c for c in systemctl_calls(box_stubs) if c.startswith("systemd-run")]
    assert start_train(box_stubs, hours=10).returncode == 0


def test_the_box_train_run_reads_the_frontier_key_from_ssm_only_when_a_panel_needs_it(stubs):
    state = stubs["tmp"] / "state"
    panel = stubs["tmp"] / "panel.jsonl"
    seat = {
        "kind": "llm",
        "civ": "pers",
        "model": "anthropic/claude-opus-5-5",
        "base_url": "http://l",
        "api_key_env": "LITELLM_API_KEY",
    }
    panel.write_text(json.dumps({"game_id": "p0", "seats": [{"kind": "bot", "civ": "rome"}, seat]}))
    config = stubs["tmp"] / "run.toml"
    base = f'run_id = "r"\nroot = "{state}/train/r"\nbudget_usd = 100\nbox_usd_per_hour = 10\nduration_hours = 1\n'
    base += 'servers = { urls = ["http://x:8000"], adapter_dir = "/train/r/adapters" }\n'
    config.write_text(base + f'[[panels]]\nname = "end"\nat_hours = 1\nspecs = "{panel}"\n')
    out = run_box(stubs, "train-run.sh", str(config), "--play-only", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    assert "aws ssm get-parameter --name /zeroad-live/litellm-key" in out.stdout and "never printed" in out.stdout
    assert "orchestrate run --config" in out.stdout
    config.write_text(base)
    out = run_box(stubs, "train-run.sh", str(config), "--play-only", DRY_RUN="1")
    assert "litellm-key" not in out.stdout and "with-model" not in out.stdout


def test_the_train_run_stops_the_learner_container_when_it_is_stopped_too():
    text = (BOX / "train-run.sh").read_text()
    assert "trap ':' TERM" in text and "trap stop_learner EXIT" in text
    assert text.count("docker stop -t 60 zeroad-learner") == 1  # in the trap's function, not after the orchestrator
