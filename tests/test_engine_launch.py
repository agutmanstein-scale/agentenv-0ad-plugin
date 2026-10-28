import asyncio
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentenv_zeroad import engine as engine_module
from agentenv_zeroad.engine import (
    DockerLauncher,
    Engine,
    EngineError,
    EngineOptions,
    LocalLauncher,
    free_port,
    launch_engine,
    pyrogenesis_args,
)


def test_nonvisual_args_include_the_0_28_autostart_workaround():
    assert pyrogenesis_args(EngineOptions(port=6123)) == [
        "--rl-interface=127.0.0.1:6123",
        "-autostart-nonvisual",
        "-autostart=random/mainland",
        "-autostart-size=128",
        "-mod=public",
    ]


def test_visual_args_need_a_display_and_can_ask_for_realtime():
    with pytest.raises(ValueError, match="display"):
        pyrogenesis_args(EngineOptions(visual=True))
    args = pyrogenesis_args(EngineOptions(visual=True, display=":99", realtime=True, extra_args=("-xres=1280",)))
    assert args == ["--rl-interface=127.0.0.1:6000", "-mod=public", "-rl-realtime", "-xres=1280"]


def test_docker_launcher_runs_image_with_host_network_uid_home_and_init():
    launcher = DockerLauncher(EngineOptions(port=6001), name="zeroad-engine-test")
    cmd = launcher.command()
    assert cmd[:4] == ["docker", "run", "-d", "-t"]
    for flag in (
        ["--init"],
        ["--network", "host"],
        ["-u", "1000:1000"],
        ["-e", "HOME=/tmp"],
        ["-e", "XDG_RUNTIME_DIR=/tmp"],
        ["--name", "zeroad-engine-test"],
    ):
        joined = " ".join(cmd)
        assert " ".join(flag) in joined
    image_at = cmd.index("zeroad-engine:0.28")
    assert cmd[image_at + 1] == "/usr/games/pyrogenesis"
    assert cmd[image_at + 2 :] == pyrogenesis_args(EngineOptions(port=6001))


def test_docker_launcher_passes_display_for_visual_mode():
    cmd = DockerLauncher(EngineOptions(visual=True, display=":99")).command()
    assert "DISPLAY=:99" in cmd


def test_docker_launcher_names_are_unique():
    assert DockerLauncher(EngineOptions()).name != DockerLauncher(EngineOptions()).name


def fake_pyrogenesis(tmp_path, body: str) -> str:
    path = tmp_path / "pyrogenesis"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_local_launcher_starts_and_kills_a_process_that_ignores_sigterm(tmp_path):
    # Nonvisual 0.28 hangs on shutdown; stop() must still end it.
    binary = fake_pyrogenesis(tmp_path, "trap '' TERM\nwhile :; do sleep 0.05; done\n")

    async def go():
        launcher = LocalLauncher(EngineOptions(), binary=binary, stop_timeout=0.3)
        assert launcher.command()[0] == binary
        await launcher.start()
        assert await launcher.alive()
        pid = launcher._proc.pid
        await launcher.stop()
        assert not await launcher.alive()
        return pid

    pid = asyncio.run(go())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_launch_engine_reports_an_engine_that_exits_before_opening_its_port(tmp_path, monkeypatch):
    fake_pyrogenesis(tmp_path, "exit 1\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    async def go():
        async with launch_engine(EngineOptions(port=1), mode="local", ready_timeout=10):
            pass

    with pytest.raises(EngineError, match="exited"):
        asyncio.run(go())


def test_local_launcher_passes_the_rl_interface_args_to_the_process(tmp_path):
    argv_file = tmp_path / "argv"
    binary = fake_pyrogenesis(tmp_path, f'printf "%s\\n" "$@" > {argv_file}\nwhile :; do sleep 0.05; done\n')
    options = EngineOptions(port=6123)

    async def go():
        launcher = LocalLauncher(options, binary=binary, stop_timeout=0.3)
        assert launcher.command() == [binary, *pyrogenesis_args(options)]
        await launcher.start()
        for _ in range(100):
            if argv_file.exists() and argv_file.read_text():
                break
            await asyncio.sleep(0.02)
        await launcher.stop()

    asyncio.run(go())
    assert argv_file.read_text().split("\n")[:-1] == pyrogenesis_args(options)


def record_docker(monkeypatch, answer=(0, "")):
    calls: list[list[str]] = []

    async def fake_run(cmd):
        calls.append(cmd)
        return answer

    monkeypatch.setattr(engine_module, "_run", fake_run)
    return calls


@pytest.mark.parametrize(
    ("answer", "alive"), [((0, "true\n"), True), ((0, "false\n"), False), ((1, "Error: No such object"), False)]
)
def test_docker_launcher_is_alive_only_while_its_container_runs(monkeypatch, answer, alive):
    calls = record_docker(monkeypatch, answer)
    launcher = DockerLauncher(EngineOptions(), name="zeroad-engine-x")
    assert asyncio.run(launcher.alive()) is alive
    assert calls == [["docker", "inspect", "-f", "{{.State.Running}}", "zeroad-engine-x"]]


def test_docker_launcher_stop_force_removes_its_container(monkeypatch):
    calls = record_docker(monkeypatch)
    asyncio.run(DockerLauncher(EngineOptions(), name="zeroad-engine-x").stop())
    assert calls == [["docker", "rm", "-f", "zeroad-engine-x"]]


class _Answer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


class StubLauncher:
    """Stands in for DockerLauncher: `start` opens an HTTP port, `stop` closes it; every call is recorded."""

    instances: list["StubLauncher"] = []

    def __init__(self, options, image=None, *, dies=False):
        self.options, self.dies, self.calls, self._server = options, dies, [], None
        StubLauncher.instances.append(self)

    async def start(self):
        self.calls.append("start")
        self._server = HTTPServer(("127.0.0.1", self.options.port), _Answer)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    async def alive(self):
        return not self.dies

    async def stop(self):
        self.calls.append("stop")
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


@pytest.fixture
def stub_launcher(monkeypatch):
    StubLauncher.instances = []
    monkeypatch.setattr(engine_module, "DockerLauncher", StubLauncher)
    return StubLauncher


def test_launch_engine_stops_the_engine_on_normal_exit(stub_launcher):
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), ready_timeout=10) as engine:
            assert isinstance(engine, Engine)

    asyncio.run(go())
    assert stub_launcher.instances[0].calls == ["start", "stop"]


def test_launch_engine_stops_the_engine_when_the_body_raises(stub_launcher):
    async def go():
        async with launch_engine(EngineOptions(port=free_port()), ready_timeout=10):
            raise KeyError("match crashed")

    with pytest.raises(KeyError):
        asyncio.run(go())
    assert stub_launcher.instances[0].calls == ["start", "stop"]


def test_launch_engine_refuses_a_port_another_server_already_answers_on(stub_launcher):
    port = free_port()
    foreign = HTTPServer(("127.0.0.1", port), _Answer)
    threading.Thread(target=foreign.serve_forever, daemon=True).start()

    async def go():
        async with launch_engine(EngineOptions(port=port), ready_timeout=10):
            pass

    try:
        with pytest.raises(EngineError, match="already in use"):
            asyncio.run(go())
    finally:
        foreign.shutdown()
        foreign.server_close()
    assert stub_launcher.instances == []  # nothing was started


def test_launch_engine_reports_an_engine_that_died_once_its_port_answered(stub_launcher, monkeypatch):
    monkeypatch.setattr(engine_module, "DockerLauncher", lambda options, image=None: StubLauncher(options, dies=True))

    async def go():
        async with launch_engine(EngineOptions(port=free_port()), ready_timeout=10):
            pass

    with pytest.raises(EngineError, match="exited"):
        asyncio.run(go())
    assert StubLauncher.instances[0].calls == ["start", "stop"]


def test_the_local_launcher_copies_the_newest_replay_out(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    launcher, dest = LocalLauncher(EngineOptions()), tmp_path / "out.txt"
    assert asyncio.run(launcher.copy_replay(dest)) is None and not dest.exists()
    root = tmp_path / ".local/share/0ad/replays/0.28.0"
    for i, name in enumerate(["2026-10-25_0001", "2026-10-25_0002"]):
        (root / name).mkdir(parents=True)
        (root / name / "commands.txt").write_text(f"start {name}\n")
        os.utime(root / name / "commands.txt", (1_000_000 + i, 1_000_000 + i))
    os.utime(root / "2026-10-25_0001/commands.txt", (2_000_000, 2_000_000))  # newest by mtime, not by name
    assert asyncio.run(launcher.copy_replay(dest)) == dest
    assert dest.read_text() == "start 2026-10-25_0001\n"
