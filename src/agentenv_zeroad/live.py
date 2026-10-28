"""The live view (spec §4.8): the overlay's data, and the broadcast stack that renders, frames and records a match.

A match whose lobby sets `render: true` is played on the patched stream engine (`zeroad-engine:0.28-rt`, image
`mcp-server-zeroad-live`), in visual mode with `-rl-realtime`, as the observer (player -1, no fog). One 1920x1080 X
display holds both halves of the picture:

    +-------------------------------------+----------+
    | 0 A.D., 1440x1080, observer view,   | overlay  |
    | camera moved by the director mod    | 480x1080 |
    | (mod/zeroad-director)               | /live?stream (Chrome)
    +-------------------------------------+----------+

and ffmpeg records the display to an MP4 (libx264 here; `ZEROAD_ENCODER=nvenc` for the GPU box). The display is an
Xvfb the env starts itself unless `ZEROAD_DISPLAY` names one (the GPU box's Xorg, U7). An RTMP leg exists only when
the lobby sets `rtmp: true` *and* the env's own environment holds `ZEROAD_RTMP_URL`; the URL (it carries the stream
key) is never logged, written or put on a command line anyone prints.

The overlay (`GET /live`, `GET /live/state.json`) shows, per empire, the seat's name and model, civ, colour, resources,
population, army, the latest actions from the seat's action record as "who: line" ("Luna: pulled the army home to defend
the base", "autopilot (Luna held): ordered the advance to town", "strategist: army(...)") and `say` line, and a live
"thinking… 12 s" timer: a brain's turn starts when it reads its brief once the match has started (or begins a reported
wait) and ends when it reports the model call (`brain_calls`), ends the wait, or next polls its status; it never runs
once the match is over. It also shows the match clock and a scoreboard. It carries no keys, endpoints or prompts. It
does show every empire without fog, as the stream does, so its data goes only to the stream client: the env's own Chrome
on loopback with the rendered match's stream token (server.py), never to an agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentenv_zeroad.scenario import DEFAULT_COLORS
from agentenv_zeroad.state import clock_text

if TYPE_CHECKING:
    from agentenv_zeroad.match import LiveMatch

log = logging.getLogger(__name__)

FRAME = (1920, 1080)
GAME = (1440, 1080)
PANEL = (FRAME[0] - GAME[0], FRAME[1])
SOFTWARE_SCALE = 2 / 3
DIRECTOR_MOD = "zeroad-director"
GAME_WINDOW_CLASS = "pyrogenesis"
ENGINE_LOG = ".local/state/0ad/log/mainlog.html"  # under the engine user's HOME (the local launcher's)
X11_DIR = Path("/tmp/.X11-unix")
RECENT_ORDERS = 4
CIV_NAMES = {
    "athen": "Athenians",
    "brit": "Britons",
    "cart": "Carthaginians",
    "gaul": "Gauls",
    "germ": "Germans",
    "han": "Han",
    "iber": "Iberians",
    "kush": "Kushites",
    "mace": "Macedonians",
    "maur": "Mauryas",
    "pers": "Persians",
    "ptol": "Ptolemies",
    "rome": "Romans",
    "sele": "Seleucids",
    "spart": "Spartans",
}
# Software GL (llvmpipe) renders the full-quality scene at a few frames a second; these settings keep the picture
# and drop the costly passes (shadows, water reflections and refractions, post-processing, particles, fog).
LOW_GRAPHICS = (
    "shadows:false",
    "watereffects:false",
    "waterfancyeffects:false",
    "waterrealdepth:false",
    "waterreflection:false",
    "waterrefraction:false",
    "postproc:false",
    "particles:false",
    "fog:false",
    "silhouettes:false",
    "showsky:false",
)
Encoder = Literal["x264", "nvenc"]


# ---- the overlay ---------------------------------------------------------------------------------------------------
def color_hex(player: int) -> str:
    r, g, b = DEFAULT_COLORS[(player - 1) % len(DEFAULT_COLORS)]
    return f"#{r:02x}{g:02x}{b:02x}"


def overlay_state(live: LiveMatch) -> dict[str, Any]:
    """`GET /live/state.json`: the match clock, each empire's panel, and the scoreboard. Never holds a secret."""
    now = live.now()
    snap = live.snapshot
    seats = []
    for slot, seat in live.seats.items():
        plan = seat.plan
        me = snap.players.get(plan.player) if snap is not None else None
        book = seat.executor.book if seat.executor is not None else None
        said = book.said[-1] if book is not None and book.said else None
        score = live.scores.get(slot, {})
        seats.append(
            {
                "slot": slot,
                "player": plan.player,
                "name": plan.name,
                "model": seat.brain_model or _model_label(plan.kind),
                "kind": plan.kind,
                "reflex": plan.reflex,
                "civ": plan.civ,
                "civ_name": CIV_NAMES.get(plan.civ, plan.civ),
                "color": color_hex(plan.player),
                "state": live.states.get(slot, "active"),
                "outcome": (live.outcomes or {}).get(slot),
                "score": round(score.get("total", 0.0)),
                "resources": dict(me.resources) if me is not None else None,
                "pop": me.pop if me is not None else None,
                "pop_cap": me.pop_cap if me is not None else None,
                "army": me.army if me is not None else None,
                "phase": me.phase if me is not None else None,
                "posture": book.summary()["army"] if book is not None else None,
                "economy": dict(book.split) if book is not None else None,
                "recent": [
                    {
                        "at": clock_text(a.game_ms),
                        "who": seat.label(a),
                        "by": seat.who(a),
                        "line": a.line or a.what,
                        "status": a.status,
                    }
                    for a in list(seat.actions)[-RECENT_ORDERS:]
                ][::-1],
                "say": {"at": clock_text(said[0]), "text": said[1]} if said else None,
                "thinking_s": round(now - seat.thinking_since, 1)
                if seat.thinking_since is not None and not live.over
                else None,
                "last_call_s": round(seat.brain_calls[-1].seconds, 1) if seat.brain_calls else None,
                "calls": len(seat.brain_calls),
            }
        )
    board = sorted(seats, key=lambda s: (-s["score"], s["player"]))
    return {
        "clock": live.game_clock,
        "game_ms": live.time_ms,
        "cap": clock_text(int(live.config.time_cap_s * 1000)),
        "time_cap_ms": int(live.config.time_cap_s * 1000),
        "status": "over" if live.over else "playing" if live.started else "starting",
        "detail": live.detail(),
        "seats": seats,
        "scoreboard": [{"slot": s["slot"], "name": s["name"], "score": s["score"], "color": s["color"]} for s in board],
    }


def _model_label(kind: str) -> str:
    return {"petra": "Petra (0 A.D. AI)", "bot": "scripted bot"}.get(kind, "agent")


def overlay_html() -> str:
    return resources.files("agentenv_zeroad.viewer").joinpath("overlay.html").read_text()


# ---- the broadcast stack -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """The X display's layout at `scale` of 1920x1080: the game on the left, the overlay's panel on the right. The
    recording is always 1920x1080; under software GL the display is smaller (2/3: 1280x720) and ffmpeg scales it up,
    since llvmpipe's frame rate falls with the pixel count. Chrome draws the 480-px overlay at the same scale."""

    scale: float = 1.0

    def _px(self, n: int) -> int:
        return int(round(n * self.scale / 2)) * 2

    @property
    def frame(self) -> tuple[int, int]:
        return self._px(FRAME[0]), self._px(FRAME[1])

    @property
    def game(self) -> tuple[int, int]:
        return self._px(GAME[0]), self.frame[1]

    @property
    def panel(self) -> tuple[int, int]:
        return self.frame[0] - self.game[0], self.frame[1]


FULL = Layout()


def engine_args(*, low_graphics: bool = True, game: tuple[int, int] = GAME) -> tuple[str, ...]:
    """pyrogenesis arguments for the stream: the director mod on top of public, a window filling the game's part of
    the display, no sound device, and low graphics under software GL."""
    # pauseonfocusloss: the overlay's window takes the X focus, which would pause a single-player game.
    args = [f"-mod={DIRECTOR_MOD}", "-nosound", f"-xres={game[0]}", f"-yres={game[1]}", "-conf=windowed:true"]
    args.append("-conf=pauseonfocusloss:false")
    if low_graphics:
        args += [f"-conf={setting}" for setting in LOW_GRAPHICS]
    return tuple(args)


def xvfb_command(display: str, frame: tuple[int, int] = FRAME) -> list[str]:
    return ["Xvfb", display, "-screen", "0", f"{frame[0]}x{frame[1]}x24", "-nolisten", "tcp", "-ac"]


def window_move_command(game: tuple[int, int] = GAME, window_class: str = GAME_WINDOW_CLASS) -> list[str]:
    """SDL centres the game's window on the screen; this sizes it to the game's part of the display and moves it to
    the left edge, beside the overlay's panel. The size is set here because 0 A.D. caps a window at the output's mode,
    and the GPU box's Xorg keeps its output at the driver's 1024x768 (live/box/display.sh)."""
    w, h = (str(n) for n in game)
    search = ["xdotool", "search", "--sync", "--class", window_class]
    return [*search, "windowsize", "%@", w, h, "windowmove", "%@", "0", "0"]


def window_focus_command(window_class: str = GAME_WINDOW_CLASS) -> list[str]:
    """The X input focus back on the game: 0 A.D. moves its camera only while its window has the focus (the
    director's moves are ignored otherwise), and the overlay's window takes it when it opens. No window manager runs,
    so nothing else moves the focus."""
    return ["xdotool", "search", "--class", window_class, "windowfocus", "%@"]


def chrome_binary() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    return None


def chrome_command(binary: str, url: str, profile: Path, layout: Layout = FULL) -> list[str]:
    """Chrome as a bare window showing the overlay in the panel beside the game."""
    return [
        binary,
        "--no-sandbox",
        "--test-type",  # no "unsupported command-line flag" bar for --no-sandbox
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--noerrdialogs",
        "--disable-infobars",
        "--disable-features=Translate,MediaRouter",
        "--hide-scrollbars",
        f"--force-device-scale-factor={layout.scale:.4g}",
        f"--user-data-dir={profile}",
        # in device-independent pixels, which Chrome multiplies by the scale factor
        f"--window-position={GAME[0]},0",
        f"--window-size={PANEL[0]},{PANEL[1]}",
        f"--app={url}",
    ]


def ffmpeg_command(
    display: str,
    record: Path,
    *,
    fps: int = 30,
    encoder: Encoder = "x264",
    bitrate: str = "6000k",
    rtmp_url: str | None = None,
    frame: tuple[int, int] = FRAME,
) -> list[str]:
    """ffmpeg grabbing the whole display into an MP4, and, only when an RTMP URL is given, to that server too (the
    tee muxer keeps the recording going if the RTMP leg fails)."""
    rate = kbps(bitrate)
    bitrate = f"{rate}k"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    cmd += ["-thread_queue_size", "512", "-f", "x11grab", "-draw_mouse", "0", "-framerate", str(fps)]
    cmd += ["-video_size", f"{frame[0]}x{frame[1]}", "-i", f"{display}.0+0,0"]
    if frame != FRAME:
        cmd += ["-vf", f"scale={FRAME[0]}:{FRAME[1]}:flags=bicubic"]
    if encoder == "nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "ll", "-rc", "cbr"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency"]
    cmd += ["-pix_fmt", "yuv420p", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", f"{2 * rate}k"]
    cmd += ["-g", str(2 * fps), "-bf", "0", "-r", str(fps)]
    if rtmp_url:
        legs = f"[f=mp4:movflags=+faststart]{record}|[f=flv:onfail=ignore]{rtmp_url}"
        return [*cmd, "-flags", "+global_header", "-f", "tee", "-map", "0:v", legs]
    return [*cmd, "-movflags", "+faststart", "-f", "mp4", str(record)]


def kbps(bitrate: str) -> int:
    """A bitrate in ffmpeg's notation ("6000k", "6000K", "6M", "6000000") as kbit/s."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", bitrate)
    if m is not None:
        value, unit = float(m.group(1)), m.group(2).lower()
        result = int(value * 1000) if unit == "m" else int(value) if unit == "k" else int(value / 1000)
        if result >= 1:  # checked on the kbit/s ffmpeg gets: "0.5k" would become "0k"
            return result
    raise ValueError(f"ZEROAD_BITRATE must be a bitrate of at least 1k, like 6000k or 6M, not {bitrate!r}")


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<rtmp url>")
    return re.sub(r"rtmps?://\S+", "<rtmp url>", text)


def _positive(env: Mapping[str, str], name: str, default: str, kind: type[int] | type[float]) -> float:
    """A setting ffmpeg or the layout cannot use unless it is above 0, checked before anything starts."""
    raw = env.get(name, default)
    try:
        value = kind(raw)
    except ValueError:
        value = kind(0)
    if value <= 0:
        raise ValueError(f"{name} must be a number above 0, not {raw!r}")
    return value


@dataclass
class BroadcastSettings:
    """Where and how a rendered match is broadcast; `from_env` reads the env's own environment."""

    display: str | None = None
    encoder: Encoder = "x264"
    fps: int = 30
    bitrate: str = "6000k"
    low_graphics: bool = True
    scale: float = 1.0
    rtmp_url: str | None = field(default=None, repr=False)

    @property
    def layout(self) -> Layout:
        return Layout(self.scale)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, *, rtmp: bool = False) -> BroadcastSettings:
        env = os.environ if environ is None else environ
        encoder = env.get("ZEROAD_ENCODER", "x264")
        if encoder not in ("x264", "nvenc"):
            raise ValueError(f"ZEROAD_ENCODER must be x264 or nvenc, not {encoder!r}")
        bitrate = f"{kbps(env.get('ZEROAD_BITRATE', '6000k'))}k"  # checked here, before anything has started
        url = env.get("ZEROAD_RTMP_URL") if rtmp else None
        if rtmp and not url:
            raise ValueError("rtmp: true needs ZEROAD_RTMP_URL in the env's environment (it is never a lobby setting)")
        return cls(
            display=env.get("ZEROAD_DISPLAY") or None,
            encoder=encoder,  # type: ignore[arg-type]
            fps=int(_positive(env, "ZEROAD_FPS", "30", int)),
            bitrate=bitrate,
            low_graphics=env.get("ZEROAD_GRAPHICS", "low" if encoder == "x264" else "high") == "low",
            scale=_positive(env, "ZEROAD_SCALE", str(SOFTWARE_SCALE if encoder == "x264" else 1.0), float),
            rtmp_url=url,
        )


class Broadcast:
    """The X display, the overlay window and the recorder of one rendered match.

    `open_display()` before the engine starts (it draws on the display), `start(overlay_url)` once the match is set
    up, `stop()` at the end: ffmpeg is asked to quit (`q`), so the MP4 is finished, then Chrome and Xvfb are stopped.
    """

    def __init__(
        self,
        settings: BroadcastSettings,
        out_dir: Path,
        *,
        overlay_draw_s: float = 3.0,
        ffmpeg_check_s: float = 2.0,
    ):
        self.settings = settings
        self.overlay_draw_s = overlay_draw_s
        self.ffmpeg_check_s = ffmpeg_check_s
        self.opened_at: float | None = None
        self.out_dir = out_dir
        self.record_path = out_dir / "recording.mp4"
        self.log_path = out_dir / "broadcast.log"
        self.display: str | None = settings.display
        self._xvfb: asyncio.subprocess.Process | None = None
        self._chrome: asyncio.subprocess.Process | None = None
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._profile: Path | None = None
        self._log_task: asyncio.Task[None] | None = None

    async def open_display(self) -> str:
        self.opened_at = time.time()
        if self.display is None:
            display = f":{_free_display()}"
            self._xvfb = await asyncio.create_subprocess_exec(
                *xvfb_command(display, self.settings.layout.frame), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            socket = X11_DIR / f"X{display[1:]}"
            for _ in range(100):
                if socket.exists():
                    break
                if self._xvfb.returncode is not None:
                    raise RuntimeError(f"Xvfb {display} exited with {self._xvfb.returncode}")
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError(f"Xvfb {display} did not start")
            self.display = display
        return self.display

    def engine_args(self) -> tuple[str, ...]:
        return engine_args(low_graphics=self.settings.low_graphics, game=self.settings.layout.game)

    async def start(self, overlay_url: str) -> None:
        """Show the overlay beside the game and start recording."""
        assert self.display is not None, "open_display() first"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "DISPLAY": self.display}
        await self._xdotool(window_move_command(self.settings.layout.game), env, "move the game's window", sync=True)
        s = self.settings
        chrome = chrome_binary()
        if chrome is None:
            log.warning("no Chrome or Chromium: the recording has no overlay")
        else:
            self._profile = Path(tempfile.mkdtemp(prefix="zeroad-chrome-"))
            self._chrome = await asyncio.create_subprocess_exec(
                *chrome_command(chrome, overlay_url, self._profile, s.layout),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(self.overlay_draw_s)  # let the page draw once before the first recorded frame
        await self._xdotool(window_focus_command(), env, "focus the game's window")
        cmd = ffmpeg_command(
            self.display,
            self.record_path,
            fps=s.fps,
            encoder=s.encoder,
            bitrate=s.bitrate,
            rtmp_url=s.rtmp_url,
            frame=s.layout.frame,
        )
        self._ffmpeg = await asyncio.create_subprocess_exec(
            *cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._log_task = asyncio.create_task(self._relay(self._ffmpeg))
        # ffmpeg fails at once on a bad display, encoder or setting; a match played on without it records nothing.
        await asyncio.sleep(self.ffmpeg_check_s)
        if self._ffmpeg.returncode is not None:
            raise RuntimeError(f"ffmpeg exited with {self._ffmpeg.returncode} at start; see {self.log_path}")
        rtmp = " and streaming over RTMP" if s.rtmp_url else ""
        log.info("recording %s %s to 1920x1080, %d fps (%s)%s", self.display, s.layout.frame, s.fps, s.encoder, rtmp)

    async def _xdotool(self, cmd: list[str], env: dict[str, str], what: str, *, sync: bool = False) -> None:
        if shutil.which("xdotool") is None:
            log.warning("no xdotool: could not %s", what)
            return
        proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if await asyncio.wait_for(proc.wait(), 20 if sync else 5):
                log.warning("could not %s", what)
        except TimeoutError:
            proc.kill()
            log.warning("could not %s: the game's window was not found", what)

    async def _relay(self, proc: asyncio.subprocess.Process) -> None:
        """ffmpeg's messages into broadcast.log, with the RTMP URL redacted."""
        assert proc.stderr is not None
        secrets = [self.settings.rtmp_url or ""]
        with self.log_path.open("a") as out:
            async for line in proc.stderr:
                out.write(redact(line.decode(errors="replace"), secrets))
                out.flush()

    @property
    def recording(self) -> bool:
        return self._ffmpeg is not None and self._ffmpeg.returncode is None

    async def stop(self) -> Path | None:
        """Finish the recording and stop everything this broadcast started; the MP4's path, if one was written."""
        ffmpeg, self._ffmpeg = self._ffmpeg, None
        if ffmpeg is not None and ffmpeg.returncode is None:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, AttributeError):
                ffmpeg.stdin.write(b"q")  # type: ignore[union-attr]
                await ffmpeg.stdin.drain()  # type: ignore[union-attr]
            await _wait_or_kill(ffmpeg, 30)
        if self._log_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._log_task, 5)
        for proc in (self._chrome, self._xvfb):
            if proc is not None and proc.returncode is None:
                proc.terminate()
                await _wait_or_kill(proc, 10)
        self._chrome = self._xvfb = None
        if self._profile is not None:
            shutil.rmtree(self._profile, ignore_errors=True)
            self._profile = None
        self._keep_engine_log()
        if self.record_path.exists() and self.record_path.stat().st_size > 0:
            return self.record_path
        return None

    def _keep_engine_log(self) -> None:
        """Copy the engine's log out, only if this match's engine wrote it: an older log left by an earlier engine (or
        an engine run in its own container, whose log is not here) is not this match's."""
        engine_log = Path(os.environ.get("HOME", "~")).expanduser() / ENGINE_LOG
        with contextlib.suppress(OSError):
            if self.opened_at is not None and engine_log.stat().st_mtime >= self.opened_at - 1:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(engine_log, self.out_dir / "engine-log.html")

    @property
    def engine_log(self) -> Path | None:
        """The engine's own log, copied out at the end (the director's cuts and frame rate are in it)."""
        path = self.out_dir / "engine-log.html"
        return path if path.is_file() else None


async def _wait_or_kill(proc: asyncio.subprocess.Process, timeout: float) -> None:
    try:
        await asyncio.wait_for(proc.wait(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()


def _free_display(start: int = 90) -> int:
    for n in range(start, start + 100):
        if not (X11_DIR / f"X{n}").exists() and not Path(f"/tmp/.X{n}-lock").exists():
            return n
    raise RuntimeError("no free X display number")
