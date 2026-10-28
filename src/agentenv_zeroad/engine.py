"""Drive 0 A.D. 0.28 through its built-in RL interface, and start/stop the engine process.

One HTTP connection drives every empire: `/step` takes `<player>;<command json>` lines for any player and advances
one 200 ms sim turn (spec §4.1). The env's clock is the only caller of `step`; brains never touch the engine.

Launch quirks measured in the spike (docs/spike-results.md) are encoded here, not left to callers:
- nonvisual 0.28 quits at startup unless an `-autostart` map is given, so a throwaway mainland map is loaded and
  `/reset` replaces it;
- nonvisual 0.28 hangs on shutdown, so stopping kills the process instead of waiting for a clean exit;
- `/reset` is posted with `?playerID=-1` as in the spike, because 0.28 reads `playerID` into a 1-byte buffer and
  leaves it uninitialised when there is no query string;
- `/templates` answers one XML line per template it found and silently drops unknown names, so the reply can only be
  matched to the request by position when every name was found (RLInterface.cpp `GetTemplates`).

Engine containers (the `docker` launcher, used on a dev machine and by the engine tests) are labelled with the pid,
start time and host of the process that started them, and `reap_orphans` removes those whose owner is gone, before
every docker start and at the start of each test session: a runner killed mid-test no longer leaves engines running.

`FakeEngine` has the same async interface for unit tests: deterministic state, no process, records every step. Like
the real engine, whose startup autostart game is already running, it accepts `step` and `evaluate` before any reset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

log = logging.getLogger(__name__)

TURN_MS = 200
DEFAULT_IMAGE = "zeroad-engine:0.28"
DOCKER_PYROGENESIS = "/usr/games/pyrogenesis"
SIM_TIME_JS = "Engine.QueryInterface(SYSTEM_ENTITY, IID_Timer).GetTime()"
# `TemplateExists` looks the file up without loading it, so an absent name logs nothing; `/templates` (GetTemplate) logs
# "Failed to load entity template" for each, which a rendered match shows on screen.
EXISTS_JS = (
    "(function (names) {{ const t = Engine.QueryInterface(SYSTEM_ENTITY, IID_TemplateManager); "
    "return names.filter(n => t.TemplateExists(n)); }})({names})"
)

Command = Mapping[str, Any]
Commands = Mapping[int, Sequence[Command]] | Iterable[tuple[int, Command]]


class EngineError(RuntimeError):
    """The engine answered with an error (e.g. "Game not running") or not at all."""


class GameEngine(Protocol):
    """What the clock and later layers need from an engine; `Engine` and `FakeEngine` both satisfy it."""

    async def reset(self, scenario: Mapping[str, Any] | str, *, save_replay: bool = False) -> dict[str, Any]: ...

    async def step(self, commands: Commands = ()) -> dict[str, Any]: ...

    async def evaluate(self, code: str) -> Any: ...

    async def templates(self, names: Sequence[str]) -> dict[str, str]:
        """XML for each named template; names the engine doesn't know are left out of the result."""
        ...

    async def existing_templates(self, names: Sequence[str]) -> list[str]:
        """The names among `names` that are templates, asked without loading them (so nothing is logged)."""
        ...


def format_commands(commands: Commands) -> str:
    """Encode commands as the newline-separated `<player>;<json>` body `/step` expects."""
    pairs = commands.items() if isinstance(commands, Mapping) else None
    lines: list[str] = []
    if pairs is not None:
        for player, cmds in pairs:
            lines.extend(f"{int(player)};{json.dumps(cmd, separators=(',', ':'))}" for cmd in cmds)
    else:
        lines.extend(f"{int(player)};{json.dumps(cmd, separators=(',', ':'))}" for player, cmd in commands)
    return "\n".join(lines)


class Engine:
    """Async client for one engine's RL interface."""

    def __init__(
        self, url: str = "http://127.0.0.1:6000", *, timeout: float = 120.0, client: httpx.AsyncClient | None = None
    ):
        self.url = url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def reset(self, scenario: Mapping[str, Any] | str, *, save_replay: bool = False) -> dict[str, Any]:
        body = scenario if isinstance(scenario, str) else json.dumps(scenario)
        path = "reset?playerID=-1" + ("&saveReplay=1" if save_replay else "")
        return json.loads(await self._post(path, body))

    async def step(self, commands: Commands = ()) -> dict[str, Any]:
        return json.loads(await self._post("step", format_commands(commands)))

    async def evaluate(self, code: str) -> Any:
        return json.loads(await self._post("evaluate", code))

    async def templates(self, names: Sequence[str]) -> dict[str, str]:
        if not names:
            return {}
        found = await self._template_lines(names)
        if len(found) == len(names):
            return dict(zip(names, found, strict=True))
        # Some name is unknown and the engine dropped it, so positions no longer line up: ask one name at a time.
        result: dict[str, str] = {}
        for name in names:
            single = await self._template_lines([name])
            if single:
                result[name] = single[0]
        return result

    async def _template_lines(self, names: Sequence[str]) -> list[str]:
        """The reply's lines: one XML string per template found (each ends in a newline)."""
        lines = (await self._post("templates", "\n".join(names))).split("\n")
        if lines[-1] == "":
            lines.pop()
        return lines

    async def existing_templates(self, names: Sequence[str]) -> list[str]:
        if not names:
            return []
        return list(await self.evaluate(EXISTS_JS.format(names=json.dumps(list(names)))) or [])

    async def sim_time_ms(self) -> int:
        return int(await self.evaluate(SIM_TIME_JS))

    async def answers(self) -> bool:
        """Whether anything answers HTTP on this engine's address right now."""
        try:
            await self._client.get(f"{self.url}/", timeout=2.0)
        except httpx.TransportError:
            return False
        return True

    async def wait_ready(self, timeout: float = 120.0, *, alive: Callable[[], Awaitable[bool]] | None = None) -> None:
        """Wait until the RL interface answers HTTP at all (any status means the port is open)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if await self.answers():
                return
            if alive is not None and not await alive():
                raise EngineError("engine process exited before the RL interface opened")
            if loop.time() >= deadline:
                raise EngineError(f"RL interface at {self.url} did not open within {timeout:.0f}s")
            await asyncio.sleep(0.5)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, body: str) -> str:
        try:
            response = await self._client.post(f"{self.url}/{path}", content=body.encode())
        except httpx.TransportError as exc:
            raise EngineError(f"POST /{path} failed: {exc!r}") from exc
        if response.status_code != 200:
            raise EngineError(f"POST /{path} -> {response.status_code}: {response.text.strip()}")
        return response.text


@dataclass(frozen=True)
class EngineOptions:
    """How to start pyrogenesis. `visual` needs a `display`; `realtime` needs the patched engine
    (`zeroad-engine:0.28-rt`, patches/rl-realtime.patch): the frame loop then advances the game in real time once the
    first `/step` arrives, and `/step` only posts commands and returns the state."""

    port: int = 6000
    host: str = "127.0.0.1"
    visual: bool = False
    display: str | None = None
    realtime: bool = False
    extra_args: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def pyrogenesis_args(options: EngineOptions) -> list[str]:
    """Command-line arguments for pyrogenesis (without the binary)."""
    if options.visual and not options.display:
        raise ValueError("visual mode needs a display (e.g. ':99')")
    args = [f"--rl-interface={options.host}:{options.port}"]
    if not options.visual:
        args += ["-autostart-nonvisual", "-autostart=random/mainland", "-autostart-size=128"]
    args.append("-mod=public")
    if options.realtime:
        args.append("-rl-realtime")
    args += options.extra_args
    return args


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Launcher(Protocol):
    async def start(self) -> None: ...

    async def alive(self) -> bool: ...

    async def stop(self) -> None: ...

    async def copy_replay(self, dest: Path) -> Path | None: ...


@dataclass
class LocalLauncher:
    """Runs pyrogenesis from PATH (inside the env container)."""

    options: EngineOptions
    binary: str | None = None
    stop_timeout: float = 5.0
    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)

    def command(self) -> list[str]:
        binary = self.binary or shutil.which("pyrogenesis") or DOCKER_PYROGENESIS
        return [binary, *pyrogenesis_args(self.options)]

    async def start(self) -> None:
        env = dict(os.environ)
        if self.options.display:
            env["DISPLAY"] = self.options.display
        self._proc = await asyncio.create_subprocess_exec(
            *self.command(), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    async def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), self.stop_timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()

    async def copy_replay(self, dest: Path) -> Path | None:
        """Copy the newest replay's commands.txt from this user's 0 A.D. data directory."""
        root = Path(os.environ.get("HOME", "~")).expanduser() / ".local/share/0ad/replays"
        found = sorted(root.glob("**/commands.txt"), key=lambda p: p.stat().st_mtime)
        if not found:
            return None
        shutil.copyfile(found[-1], dest)
        return dest


@dataclass
class DockerLauncher:
    """Runs the engine image with host networking, so the host talks to it on `options.port`.

    Every container carries labels naming the process that started it (`OWNER_LABELS`), so `reap_orphans` can remove
    the containers whose owner died without stopping them (a test runner killed mid-test).
    """

    options: EngineOptions
    image: str = DEFAULT_IMAGE
    name: str = field(default_factory=lambda: f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:8]}")
    xvfb: str | None = None
    """A screen size (e.g. "1440x1080"): run visual mode on an Xvfb inside the container instead of the host's
    display (the engine tests for the stream engine; the host needs no X server)."""
    mounts: tuple[tuple[str, str], ...] = ()
    """(host path, container path) pairs mounted read-only, e.g. a mod into /usr/share/games/0ad/mods/."""

    def command(self) -> list[str]:
        cmd = ["docker", "run", "-d", "-t", "--init", "--name", self.name, "-u", "1000:1000"]
        for key, value in owner_labels().items():
            cmd += ["--label", f"{key}={value}"]
        cmd += ["-e", "HOME=/tmp", "-e", "XDG_RUNTIME_DIR=/tmp", "--network", "host"]
        for host, inside in self.mounts:
            cmd += ["-v", f"{host}:{inside}:ro"]
        if self.xvfb:
            options = replace(self.options, display=XVFB_DISPLAY)
            engine = shlex.join([DOCKER_PYROGENESIS, *pyrogenesis_args(options)])
            script = f"Xvfb {XVFB_DISPLAY} -screen 0 {self.xvfb}x24 -nolisten tcp & sleep 1; exec {engine}"
            return [*cmd, "-e", f"DISPLAY={XVFB_DISPLAY}", self.image, "sh", "-c", script]
        if self.options.display:
            cmd += ["-e", f"DISPLAY={self.options.display}", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro"]
        return [*cmd, self.image, DOCKER_PYROGENESIS, *pyrogenesis_args(self.options)]

    async def start(self) -> None:
        await reap_orphans()
        code, out = await _run(self.command())
        if code != 0:
            raise EngineError(f"docker run failed ({code}): {out.strip()}")

    async def alive(self) -> bool:
        code, out = await _run(["docker", "inspect", "-f", "{{.State.Running}}", self.name])
        return code == 0 and out.strip() == "true"

    async def stop(self) -> None:
        await _run(["docker", "rm", "-f", self.name])

    async def copy_replay(self, dest: Path) -> Path | None:
        """Copy the newest replay's commands.txt out of the container (HOME=/tmp there)."""
        code, out = await _run(["docker", "exec", self.name, "sh", "-c", NEWEST_REPLAY_SH.format(home="/tmp")])
        found = out.strip().splitlines()
        if code != 0 or not found:
            return None
        code, out = await _run(["docker", "cp", f"{self.name}:{found[-1]}", str(dest)])
        return dest if code == 0 and dest.exists() else None


XVFB_DISPLAY = ":99"
NEWEST_REPLAY_SH = "ls -t $(find {home}/.local/share/0ad/replays -name commands.txt 2>/dev/null) 2>/dev/null | head -1"
CONTAINER_PREFIX = "zeroad-engine-"
LABEL_PID, LABEL_START, LABEL_HOST = "zeroad.owner-pid", "zeroad.owner-start", "zeroad.owner-host"


def _process_start(pid: int) -> str | None:
    """The process's start time in clock ticks since boot (/proc/<pid>/stat field 22), or None if it doesn't exist.

    Together with the pid it names one process: a reused pid has another start time.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def owner_labels() -> dict[str, str]:
    pid = os.getpid()
    return {LABEL_PID: str(pid), LABEL_START: _process_start(pid) or "", LABEL_HOST: socket.gethostname()}


def owner_alive(pid: str, start: str) -> bool:
    if not pid.isdigit():
        return True  # not ours to judge
    now = _process_start(int(pid))
    return now is not None and (not start or now == start)


async def reap_orphans() -> list[str]:
    """Remove this host's engine containers whose owner process is gone; returns their names.

    Only containers named `zeroad-engine-*` that carry the owner labels are considered, so containers started by
    anything else (including unlabeled ones from older code) are never touched. Where the owner check can't be made
    (no /proc, so even this process would look dead), nothing is reaped.
    """
    if _process_start(os.getpid()) is None:
        log.warning("not reaping engine containers: process start times can't be read on this host")
        return []
    fmt = f'{{{{.Names}}}}\t{{{{.Label "{LABEL_PID}"}}}}\t{{{{.Label "{LABEL_START}"}}}}\t{{{{.Label "{LABEL_HOST}"}}}}'
    code, out = await _run(["docker", "ps", "-a", "--filter", f"label={LABEL_PID}", "--format", fmt])
    if code != 0:
        return []
    host = socket.gethostname()
    reaped = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, pid, start, owner_host = parts
        if not name.startswith(CONTAINER_PREFIX) or owner_host != host or owner_alive(pid, start):
            continue
        await _run(["docker", "rm", "-f", name])
        log.warning("removed engine container %s: its owner process %s is gone", name, pid)
        reaped.append(name)
    return reaped


async def _run(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


@dataclass
class RunningEngine:
    """A started engine: its client and launcher. `close()` stops both; `copy_replay` copies the replay out."""

    engine: Engine
    launcher: Launcher

    async def close(self) -> None:
        try:
            await self.engine.aclose()
        finally:
            await self.launcher.stop()

    async def copy_replay(self, dest: Path) -> Path | None:
        return await self.launcher.copy_replay(dest)


async def start_engine(
    options: EngineOptions | None = None,
    *,
    mode: Literal["local", "docker"] = "docker",
    image: str = DEFAULT_IMAGE,
    ready_timeout: float = 120.0,
    docker: Mapping[str, Any] | None = None,
) -> RunningEngine:
    """Start an engine and wait for its port; the caller must `close()` it.

    The port must be free before the start, and our process must still be alive once the port answers; otherwise the
    `Engine` could be talking to some other server (a leftover engine on host networking) while ours has died.
    """
    options = options or EngineOptions()
    engine = Engine(options.url)
    if await engine.answers():
        await engine.aclose()
        raise EngineError(f"{options.url} is already in use by another server; pick a free port")
    launcher: Launcher = (
        LocalLauncher(options) if mode == "local" else DockerLauncher(options, image=image, **dict(docker or {}))
    )
    running = RunningEngine(engine, launcher)
    try:
        await launcher.start()
        await engine.wait_ready(ready_timeout, alive=launcher.alive)
        if not await launcher.alive():
            raise EngineError("engine process exited although its RL interface port answered")
    except BaseException:
        await running.close()
        raise
    return running


@asynccontextmanager
async def launch_engine(
    options: EngineOptions | None = None,
    *,
    mode: Literal["local", "docker"] = "docker",
    image: str = DEFAULT_IMAGE,
    ready_timeout: float = 120.0,
    docker: Mapping[str, Any] | None = None,
) -> AsyncIterator[Engine]:
    """Start an engine, wait for its port, yield a connected `Engine`, and always stop it on exit."""
    running = await start_engine(options, mode=mode, image=image, ready_timeout=ready_timeout, docker=docker)
    try:
        yield running.engine
    finally:
        await running.close()


class FakeEngine:
    """In-memory stand-in for `Engine`: deterministic state, records every reset and step, never touches a network.

    State mimics the shape of the real `/step` result where later layers read it (`timeElapsed`, `players`,
    `entities`) and advances `timeElapsed` by 200 ms per step. `step_delay` (seconds, awaited through `sleep`) lets
    clock tests simulate slow steps on a virtual clock.
    """

    def __init__(
        self,
        *,
        players: int = 2,
        step_delay: float | Callable[[int], float] = 0.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        templates: Mapping[str, str] | None = None,
        evaluations: Mapping[str, Any] | None = None,
    ):
        self.players = players
        self.step_delay = step_delay
        self._sleep = sleep
        self._templates = dict(templates or {})
        self._evaluations = dict(evaluations or {})
        self.scenario: dict[str, Any] | None = None
        self.time_ms = 0
        self.steps: list[list[tuple[int, dict[str, Any]]]] = []
        self.evaluated: list[str] = []
        self.template_requests: list[list[str]] = []
        self.resets = 0

    async def reset(self, scenario: Mapping[str, Any] | str, *, save_replay: bool = False) -> dict[str, Any]:
        self.scenario = json.loads(scenario) if isinstance(scenario, str) else dict(scenario)
        player_data = self.scenario.get("settings", {}).get("PlayerData")
        if player_data:
            self.players = len(player_data)
        self.time_ms = 0
        self.steps = []
        self.resets += 1
        return self.state()

    async def step(self, commands: Commands = ()) -> dict[str, Any]:
        batch = [(int(p), json.loads(c)) for p, c in (line.split(";", 1) for line in _lines(commands))]
        delay = self.step_delay(len(self.steps)) if callable(self.step_delay) else self.step_delay
        if delay:
            await self._sleep(delay)
        self.steps.append(batch)
        self.time_ms += TURN_MS
        return self.state()

    async def evaluate(self, code: str) -> Any:
        self.evaluated.append(code)
        if code == SIM_TIME_JS:
            return self.time_ms
        return self._evaluations.get(code)

    async def templates(self, names: Sequence[str]) -> dict[str, str]:
        self.template_requests.append(list(names))
        return {name: self._templates[name] for name in names if name in self._templates}

    async def existing_templates(self, names: Sequence[str]) -> list[str]:
        return [name for name in names if name in self._templates]

    async def sim_time_ms(self) -> int:
        return self.time_ms

    def state(self) -> dict[str, Any]:
        players = [{"name": "Gaia", "state": "active"}]
        players += [{"name": f"Player {i}", "state": "active"} for i in range(1, self.players + 1)]
        return {"timeElapsed": self.time_ms, "players": players, "entities": {}, "mapSize": 128}


def _lines(commands: Commands) -> list[str]:
    body = format_commands(commands)
    return body.split("\n") if body else []
