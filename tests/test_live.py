"""The live view (U6): a rendered lobby's broadcast lifecycle, the overlay's data and page, the thinking timer, and the
recorder's commands (display, Chrome, ffmpeg with RTMP only behind its flag)."""

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from agentenv_game import GameError, SlotRequest

from agentenv_zeroad import grading, live
from agentenv_zeroad.engine import DockerLauncher, EngineOptions
from agentenv_zeroad.live import (
    FRAME,
    GAME,
    BroadcastSettings,
    chrome_command,
    engine_args,
    ffmpeg_command,
    overlay_html,
    overlay_state,
    redact,
)
from agentenv_zeroad.server import ZeroADEnv, overlay_url
from tests.served import served
from tests.test_server import FAST, call
from tests.zeroad_fakes import Handle, placer

pytestmark = pytest.mark.anyio

SECRET = "sk-live-overlay-test-key"


class FakeBroadcast:
    """Records the broadcast's lifecycle; `stop` writes the recording like ffmpeg would."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.display: str | None = None
        self.events: list[str] = []
        self.overlay: str | None = None

    async def open_display(self) -> str:
        self.events.append("display")
        self.display = ":91"
        return self.display

    def engine_args(self) -> tuple[str, ...]:
        return engine_args()

    async def start(self, url: str) -> None:
        self.events.append("start")
        self.overlay = url

    async def stop(self) -> Path | None:
        self.events.append("stop")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "recording.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        (self.out_dir / "engine-log.html").write_text("<p>zeroad-director: 11.0 fps</p>")
        return path

    @property
    def engine_log(self) -> Path | None:
        path = self.out_dir / "engine-log.html"
        return path if path.exists() else None


@pytest.fixture
async def rendered(tmp_path):
    started: dict = {"plain": 0, "rendered": []}
    broadcasts: list[FakeBroadcast] = []

    async def start():
        started["plain"] += 1
        return Handle()

    async def start_rendered(broadcast):
        started["rendered"].append(broadcast.display)
        return Handle()

    def broadcast_for(settings, out_dir):
        broadcasts.append(FakeBroadcast(out_dir))
        return broadcasts[-1]

    env = ZeroADEnv(
        start,
        tmp_path,
        tick_s=0.05,
        match_options={"placer": placer()},
        start_rendered=start_rendered,
        broadcast_for=broadcast_for,
    )
    env.create_app()
    yield env, started, broadcasts
    if env.live is not None:
        await env.live.stop()


async def bot_vs_petra(env, **settings):
    env.new_lobby({"speed": 1.0, "time_cap_minutes": 60, **settings})
    env.fill_slot(SlotRequest(player_id="1", player_kind="ai", game_settings={"controller": "bot", "civ": "rome"}))
    env.fill_slot(SlotRequest(player_id="2", player_kind="ai", game_settings={"civ": "pers"}))
    await env.close_lobby()


async def test_a_rendered_lobby_plays_on_the_stream_engine_and_keeps_the_recording(rendered):
    env, started, broadcasts = rendered
    await bot_vs_petra(env, render=True, time_cap_minutes=0.02)
    assert started == {"plain": 0, "rendered": [":91"]}
    (b,) = broadcasts
    assert b.events == ["display", "start"] and b.overlay == overlay_url(env.stream_token)
    assert b.overlay.startswith("http://127.0.0.1:") and b.overlay.endswith(f"/live?stream&token={env.stream_token}")
    await asyncio.wait_for(env.finish_match(), 10)
    assert b.events == ["display", "start", "stop"]
    listed = await env.list_match_files()
    assert [f["kind"] for f in listed["files"]] == ["results", "replay", "recording", "engine_log"]
    assert listed["files"][2]["name"].endswith("recording.mp4")
    results = json.loads(env.live.files["results"].read_text())
    assert results["rendered"] is True and results["clock"]["realtime"] is True


async def test_a_plain_lobby_starts_no_broadcast_and_keeps_no_recording(rendered):
    env, started, broadcasts = rendered
    await bot_vs_petra(env, time_cap_minutes=0.02)
    await asyncio.wait_for(env.finish_match(), 10)
    assert started == {"plain": 1, "rendered": []} and broadcasts == []
    assert [f["kind"] for f in (await env.list_match_files())["files"]] == ["results", "replay"]
    assert json.loads(env.live.files["results"].read_text())["clock"]["realtime"] is False


async def test_the_plain_env_refuses_a_rendered_lobby_and_the_live_env_takes_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEROAD_LIVE", raising=False)
    plain = ZeroADEnv(lambda: _handle(), tmp_path)
    with pytest.raises(GameError, match="zeroad-live"):
        plain.new_lobby({"render": True, "speed": 1.0})
    plain.new_lobby({"speed": 1.0})
    monkeypatch.setenv("ZEROAD_LIVE", "1")
    ZeroADEnv(lambda: _handle(), tmp_path).new_lobby({"render": True, "speed": 1.0})


async def test_a_rendered_match_runs_in_real_time_and_rtmp_is_off_unless_asked_with_a_url(rendered, monkeypatch):
    env, _, _ = rendered
    with pytest.raises(GameError, match="speed 1.0"):
        env.new_lobby({"render": True, "speed": 2.0})
    with pytest.raises(GameError, match="rtmp needs render"):
        env.new_lobby({"rtmp": True})
    monkeypatch.delenv("ZEROAD_RTMP_URL", raising=False)
    with pytest.raises(GameError, match="ZEROAD_RTMP_URL"):
        env.new_lobby({"render": True, "rtmp": True})


async def test_a_broadcast_that_fails_to_start_stops_the_display_and_the_engine(tmp_path):
    handles: list[Handle] = []
    fake: list[FakeBroadcast] = []

    class Failing(FakeBroadcast):
        async def start(self, url: str) -> None:
            raise RuntimeError("no X")

    async def start_rendered(broadcast):
        handles.append(Handle())
        return handles[-1]

    def broadcast_for(settings, out_dir):
        fake.append(Failing(out_dir))
        return fake[-1]

    env = ZeroADEnv(
        lambda: None, tmp_path, match_options={"placer": placer()}, start_rendered=start_rendered,
        broadcast_for=broadcast_for,
    )  # fmt: skip
    env.create_app()
    with pytest.raises(Exception, match="no X"):
        await bot_vs_petra(env, render=True)
    assert fake[0].events == ["display", "stop"] and handles[0].closed == 1


async def agent_vs_petra(env, **settings):
    env.new_lobby({**FAST, **settings})
    slot = {"civ": "rome", "label": "Opus 5.5", "reflex": "luna"}
    env.fill_slot(SlotRequest(player_id="1", player_kind="agent", player_name="opus", game_settings=slot))
    env.fill_slot(SlotRequest(player_id="2", player_kind="ai", game_settings={"civ": "pers"}))
    await env.close_lobby()


def rendered_env(tmp_path, **match_options) -> ZeroADEnv:
    async def start_rendered(broadcast):
        return Handle()

    return ZeroADEnv(
        lambda: _handle(),
        tmp_path,
        tick_s=0.05,
        match_options={"placer": placer(), **match_options},
        start_rendered=start_rendered,
        broadcast_for=lambda settings, out_dir: FakeBroadcast(out_dir),
    )


async def test_the_overlay_shows_each_empire_and_a_thinking_timer_from_brief_to_reported_call(tmp_path):
    clock = [100.0]
    env = rendered_env(tmp_path, now=lambda: clock[0])
    async with served(env) as url, httpx.AsyncClient() as http:
        await agent_vs_petra(env, render=True, speed=1.0)
        page = await http.get(f"{url}/live?stream")
        assert page.status_code == 200 and "0 A.D. Live Arena" in page.text and "state.json" in page.text
        await call(env, "1", "get_brief")  # marks the seat ready, as both brains' first call does
        await call(env, "1", "get_brief")
        await call(env, "1", "train", role="cavalry", count=5)
        await call(env, "1", "say", text="For Rome!")
        env.live.set_reflex_credentials("1", "https://llm.example", SECRET)
        clock[0] += 12.4
        state = (await http.get(f"{url}/live/state.json", params={"token": env.stream_token})).json()
        opus, petra = state["seats"]
        assert state["status"] in ("playing", "starting") and state["cap"] == "60:00"
        assert opus["name"] == "Opus 5.5" and opus["civ_name"] == "Romans" and opus["color"] == "#961414"
        assert opus["thinking_s"] == 12.4 and opus["reflex"] == "luna"
        assert opus["recent"][0]["line"] == "train(role='cavalry', count=5)" and opus["recent"][0]["who"] == "brain"
        assert opus["say"]["text"] == "For Rome!"
        assert set(opus["resources"]) == {"food", "wood", "stone", "metal"} and opus["pop"] is not None
        assert petra["model"].startswith("Petra") and petra["thinking_s"] is None
        assert [s["slot"] for s in state["scoreboard"]] == sorted(
            ["1", "2"], key=lambda s: -state["seats"][int(s) - 1]["score"]
        )
        env.live.record_brain_call("1", grading.BrainCall(seconds=12.4), model="anthropic/claude-opus-5-5")
        state = (await http.get(f"{url}/live/state.json", params={"token": env.stream_token})).json()
        opus = state["seats"][0]
        assert (
            opus["thinking_s"] is None and opus["last_call_s"] == 12.4 and opus["model"] == "anthropic/claude-opus-5-5"
        )
        raw = (await http.get(f"{url}/live/state.json", params={"token": env.stream_token})).text
        assert SECRET not in raw and "llm.example" not in raw
    await env.live.stop()


def _handle():
    async def make():
        return Handle()

    return make()


async def test_a_wait_begun_by_a_brain_also_starts_the_timer(tmp_path):
    clock = [5.0]
    env = ZeroADEnv(
        lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer(), "now": lambda: clock[0]}
    )
    env.create_app()
    await agent_vs_petra(env)
    env.live.record_waiting("1", "begin")
    clock[0] += 3
    assert overlay_state(env.live)["seats"][0]["thinking_s"] == 3.0
    await env.live.stop()


async def test_the_overlay_has_no_match_before_a_lobby_closes(tmp_path):
    env = ZeroADEnv(lambda: _handle(), tmp_path)
    async with served(env) as url, httpx.AsyncClient() as http:
        assert (await http.get(f"{url}/live/state.json")).json()["status"] == "no_match"


def test_the_overlay_page_polls_its_state_and_escapes_what_models_write():
    html = overlay_html()
    assert '"/state.json"' in html and "esc(seat.say.text)" in html and "thinking…" in html


def test_ffmpeg_records_the_whole_display_to_mp4_and_streams_only_when_given_a_url(tmp_path):
    record = tmp_path / "recording.mp4"
    cmd = ffmpeg_command(":91", record)
    assert cmd[cmd.index("-f") + 1] == "x11grab" and cmd[cmd.index("-video_size") + 1] == "1920x1080"
    assert cmd[cmd.index("-i") + 1] == ":91.0+0,0" and "libx264" in cmd and cmd[-1] == str(record)
    assert not any("rtmp" in part or "tee" == part for part in cmd)
    nvenc = ffmpeg_command(":0", record, encoder="nvenc")
    assert "h264_nvenc" in nvenc and "libx264" not in nvenc
    url = "rtmp://live.example/app/" + SECRET
    tee = ffmpeg_command(":91", record, rtmp_url=url)
    assert tee[tee.index("-f", tee.index("-i")) + 1] == "tee" and tee[-1].endswith(f"[f=flv:onfail=ignore]{url}")
    assert f"[f=mp4:movflags=+faststart]{record}" in tee[-1]
    assert SECRET not in redact(f"Error opening {url}: refused", [url])


def test_broadcast_settings_come_from_the_envs_own_environment():
    s = BroadcastSettings.from_env({"ZEROAD_DISPLAY": ":0", "ZEROAD_ENCODER": "nvenc"})
    assert (s.display, s.encoder, s.low_graphics, s.rtmp_url) == (":0", "nvenc", False, None)
    assert BroadcastSettings.from_env({}).low_graphics is True
    assert BroadcastSettings.from_env({"ZEROAD_RTMP_URL": "rtmp://x/k"}).rtmp_url is None  # off without the flag
    assert BroadcastSettings.from_env({"ZEROAD_RTMP_URL": "rtmp://x/k"}, rtmp=True).rtmp_url == "rtmp://x/k"
    assert "rtmp://x/k" not in repr(BroadcastSettings.from_env({"ZEROAD_RTMP_URL": "rtmp://x/k"}, rtmp=True))
    with pytest.raises(ValueError, match="ZEROAD_RTMP_URL"):
        BroadcastSettings.from_env({}, rtmp=True)
    with pytest.raises(ValueError, match="x264 or nvenc"):
        BroadcastSettings.from_env({"ZEROAD_ENCODER": "vp9"})


def test_a_bitrate_in_ffmpegs_own_notation_is_taken_and_a_bad_one_fails_before_anything_starts(tmp_path):
    assert BroadcastSettings.from_env({"ZEROAD_BITRATE": "6M"}).bitrate == "6000k"
    assert BroadcastSettings.from_env({"ZEROAD_BITRATE": "4500K"}).bitrate == "4500k"
    cmd = ffmpeg_command(":91", tmp_path / "r.mp4", bitrate="6M")
    assert cmd[cmd.index("-b:v") + 1] == "6000k" and cmd[cmd.index("-bufsize") + 1] == "12000k"
    for bad in ("fast", "0.5k", "0.0001M", "999", "0"):
        with pytest.raises(ValueError, match="ZEROAD_BITRATE"):
            BroadcastSettings.from_env({"ZEROAD_BITRATE": bad})
    assert BroadcastSettings.from_env({"ZEROAD_BITRATE": "1000"}).bitrate == "1k"


def test_a_frame_rate_or_scale_that_cannot_record_fails_before_anything_starts():
    for bad in ({"ZEROAD_FPS": "0"}, {"ZEROAD_FPS": "-5"}, {"ZEROAD_FPS": "fast"}, {"ZEROAD_SCALE": "0"}):
        with pytest.raises(ValueError, match="ZEROAD_FPS|ZEROAD_SCALE"):
            BroadcastSettings.from_env(bad)


def test_the_game_and_the_overlay_split_one_1920x1080_display(tmp_path):
    args = engine_args()
    assert "-mod=zeroad-director" in args and f"-xres={GAME[0]}" in args and f"-yres={GAME[1]}" in args
    assert "-conf=shadows:false" in args and "-conf=shadows:false" not in engine_args(low_graphics=False)
    for a in (args, engine_args(low_graphics=False)):
        # the overlay's window takes the X focus, which would pause the game without pauseonfocusloss:false
        assert {"-conf=pauseonfocusloss:false", "-nosound", "-conf=windowed:true"} <= set(a)
    chrome = chrome_command("/usr/bin/google-chrome", "http://127.0.0.1:18765/live?stream", tmp_path)
    assert f"--window-position={GAME[0]},0" in chrome and f"--window-size={FRAME[0] - GAME[0]},{FRAME[1]}" in chrome
    assert chrome[-1] == "--app=http://127.0.0.1:18765/live?stream"


def test_under_software_gl_the_display_is_two_thirds_size_and_scaled_up_to_1080p(tmp_path):
    s = BroadcastSettings.from_env({})
    assert s.layout.frame == (1280, 720) and s.layout.game == (960, 720) and s.layout.panel == (320, 720)
    cmd = ffmpeg_command(":91", tmp_path / "r.mp4", frame=s.layout.frame)
    assert cmd[cmd.index("-video_size") + 1] == "1280x720" and "scale=1920:1080:flags=bicubic" in cmd
    chrome = chrome_command("chrome", "http://x/live?stream", tmp_path, s.layout)
    # Chrome takes its window's place and size in device-independent pixels: 1440,0 and 480x1080 at 2/3 scale
    assert "--force-device-scale-factor=0.6667" in chrome and "--window-position=1440,0" in chrome
    assert "--window-size=480,1080" in chrome
    assert BroadcastSettings.from_env({"ZEROAD_ENCODER": "nvenc"}).layout.frame == FRAME


def test_the_docker_launcher_can_run_the_stream_engine_on_its_own_xvfb_with_the_mod_mounted():
    options = EngineOptions(port=6123, visual=True, realtime=True, extra_args=("-mod=zeroad-director",))
    cmd = DockerLauncher(
        options, image="zeroad-engine:0.28-rt", xvfb="1440x1080", mounts=(("/m", "/mods/d"),)
    ).command()
    assert "/m:/mods/d:ro" in cmd and "DISPLAY=:99" in cmd and cmd[-3:-1] == ["sh", "-c"]
    script = cmd[-1]
    assert script.startswith("Xvfb :99 -screen 0 1440x1080x24") and "-rl-realtime" in script
    assert "-mod=public -rl-realtime -mod=zeroad-director" in script


async def test_the_thinking_timer_skips_the_ready_brief_and_stops_when_the_brain_stops_or_the_match_ends(tmp_path):
    clock = [5.0]
    env = ZeroADEnv(
        lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer(), "now": lambda: clock[0]}
    )
    env.create_app()
    await agent_vs_petra(env)

    def thinking():
        return overlay_state(env.live)["seats"][0]["thinking_s"]

    await call(env, "1", "get_brief")  # the brain's first call marks the seat ready; it starts no turn
    clock[0] += 90
    assert thinking() is None
    # A turn read its brief, then the spend cap stopped the brain (no call reported); the brain goes back to polling.
    await call(env, "1", "get_brief")
    clock[0] += 4
    assert thinking() == 4.0
    env.live.status("1")
    clock[0] += 600
    assert thinking() is None
    # A wait that ends without a reported call.
    await call(env, "1", "get_brief")
    env.live.record_waiting("1", "begin")
    clock[0] += 3
    env.live.record_waiting("1", "end")
    clock[0] += 600
    assert thinking() is None
    # A turn still open when the match ends.
    await call(env, "1", "get_brief")
    await env.live.stop()
    clock[0] += 900
    assert overlay_state(env.live)["status"] == "over" and thinking() is None


async def test_an_unrendered_match_serves_no_overlay_data_since_its_players_play_under_fog(tmp_path):
    env = ZeroADEnv(lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer()})
    async with served(env) as url, httpx.AsyncClient() as http:
        await agent_vs_petra(env)
        state = (await http.get(f"{url}/live/state.json")).json()
        assert state == {"status": "no_match", "seats": [], "scoreboard": []}
        assert overlay_state(env.live)["seats"][1]["resources"]  # the data exists; it is only not served
    await env.live.stop()


async def test_overlay_data_goes_only_to_the_stream_client_and_an_agent_side_request_is_refused(tmp_path):
    env = rendered_env(tmp_path)
    async with served(env) as url, httpx.AsyncClient() as http:
        await agent_vs_petra(env, render=True, speed=1.0)
        token = env.stream_token
        assert token and len(token) >= 24
        # The stream client: the env's own Chrome on loopback, with the token from the URL the env opened.
        ok = await http.get(f"{url}/live/state.json", params={"token": token})
        assert ok.status_code == 200 and ok.json()["seats"][1]["resources"]
        # An agent, at its seat's address: refused even if it had the token.
        for path in ("/players/1/live/state.json", "/players/2/live/state.json", "/spectators/live/state.json"):
            refused = await http.get(f"{url}{path}", params={"token": token})
            assert refused.status_code == 403 and "resources" not in refused.text, path
        # Without the token, or with a wrong one, at the env's own address: refused.
        wrong = token[:-1] + ("A" if token[-1] != "A" else "B")
        for params in ({}, {"token": ""}, {"token": wrong}):
            refused = await http.get(f"{url}/live/state.json", params=params)
            assert refused.status_code == 403 and "resources" not in refused.text, params
        # The token reaches no agent: not in a tool result, the brief, the seat's status or the env card.
        texts = [await call(env, "1", "get_brief"), await call(env, "1", "army", action="defend", target="home")]
        texts.append((await http.get(f"{url}/players/1/status")).text)
        texts.append((await http.get(f"{url}/players/1/.well-known/agent-env.json")).text)
        assert all(token not in t for t in texts)
    await env.live.stop()


def test_the_stream_client_is_refused_from_a_non_loopback_address():
    from starlette.testclient import TestClient

    env = ZeroADEnv(lambda: _handle(), Path("/tmp"))
    env.live = type("RenderedMatch", (), {"broadcast": object()})()
    env.stream_token = "t" * 32
    client = TestClient(env.create_app().streamable_http_app())  # its client address is "testclient"
    response = client.get("/live/state.json", params={"token": env.stream_token})
    assert response.status_code == 403


async def test_a_scripted_bots_orders_show_among_the_latest_orders_marked_as_the_bots(tmp_path):
    env = ZeroADEnv(lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer()})
    env.create_app()
    await bot_vs_petra(env)
    for _ in range(100):
        if env.live.seats["1"].bot.log:
            break
        await asyncio.sleep(0.05)
    assert env.live.seats["1"].bot.log
    recent = overlay_state(env.live)["seats"][0]["recent"]
    assert recent and {o["by"] for o in recent} == {"bot"} and {o["who"] for o in recent} == {"bot"}
    assert any(o["line"].startswith("build(structure='storehouse'") for o in recent)
    await env.live.stop()


async def test_the_render_check_reads_the_validated_lobby_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEROAD_LIVE", raising=False)
    plain = ZeroADEnv(lambda: _handle(), tmp_path)
    for bad in (["render"], "render", 5):
        with pytest.raises(GameError, match="game_settings is an object"):
            plain.new_lobby(bad)
    assert plain.new_lobby({"render": "false", "speed": 1.0}).game_settings["render"] is False


async def test_the_reflexs_actions_show_among_the_latest_as_luna_lines_beside_the_strategists_orders(tmp_path):
    from agentenv_zeroad.reflex import ReflexAction

    env = ZeroADEnv(lambda: _handle(), tmp_path, tick_s=0.05, match_options={"placer": placer()})
    env.create_app()
    await agent_vs_petra(env)
    await call(env, "1", "train", role="cavalry", count=5)
    env.live.record_brain_call("1", grading.BrainCall(seconds=3.0), model="anthropic/claude-opus-5-5")
    reflex = env.live.seats["1"].reflex
    reflex.on_action(ReflexAction(0, "counter:cavalry", "queued more cavalry", "train(role='cavalry', count=2)", "ok"))
    recent = overlay_state(env.live)["seats"][0]["recent"]
    assert [(o["who"], o["line"]) for o in recent] == [
        ("Luna", "queued more cavalry"),
        ("strategist", "train(role='cavalry', count=5)"),
    ]
    assert "esc(o.who)" in overlay_html() and "esc(o.line)" in overlay_html()
    await env.live.stop()


def test_the_overlay_page_passes_its_own_token_on_to_the_state_it_polls():
    html = overlay_html()
    source = next(line for line in html.splitlines() if line.startswith("const source ="))
    assert '"/state.json") + "?token=" + encodeURIComponent(params.get("token")' in source
    assert "fetch(source" in html


def test_the_overlay_compacts_its_panels_for_more_than_two_seats_so_four_fit_the_stream():
    html = overlay_html()
    assert 'classList.toggle("compact", last.seats.length > 2)' in html
    assert "body.compact .orders div:nth-child(n+3) { display: none; }" in html
    assert "body.stream .seat { flex: 0 1 auto; min-height: 0; overflow: hidden; }" in html


# ---- the real Broadcast, on fake processes ------------------------------------------------------------------------
class FakeProc:
    """A process the broadcast starts: xdotool exits at once; Xvfb and Chrome run until terminated; ffmpeg finishes
    its MP4 only when it reads `q` on stdin."""

    def __init__(
        self,
        argv: tuple[str, ...],
        events: list[str],
        tmp: Path,
        *,
        xvfb_fails: bool = False,
        ffmpeg_fails: bool = False,
    ):
        self.argv, self.events, self.tmp = argv, events, tmp
        self.name = Path(argv[0]).name
        self.returncode: int | None = None
        self._done = asyncio.Event()
        self.stdin = self
        self.stderr = self._lines()
        self.got_q = False
        if self.name == "xdotool" or (self.name == "ffmpeg" and ffmpeg_fails):
            self._exit(0 if self.name == "xdotool" else 1)
        elif self.name == "Xvfb":
            if xvfb_fails:
                self._exit(1)
            else:
                (live.X11_DIR / f"X{argv[1][1:]}").touch()

    def _exit(self, code: int) -> None:
        self.returncode = code
        self._done.set()

    async def _lines(self):
        yield b"ffmpeg: frame=1\n"

    def write(self, data: bytes) -> None:
        self.events.append(f"{self.name} stdin {data.decode()}")
        if data == b"q":
            self.got_q = True
            (self.tmp / "out" / "recording.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom moov")
            self._exit(0)

    async def drain(self) -> None:
        pass

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.events.append(f"{self.name} terminate")
        self._exit(-15)

    def kill(self) -> None:
        self.events.append(f"{self.name} kill")
        self._exit(-9)


@pytest.fixture
def fake_processes(tmp_path, monkeypatch):
    events: list[str] = []
    procs: list[FakeProc] = []
    options = {"xvfb_fails": False, "ffmpeg_fails": False}
    (tmp_path / "x11").mkdir()
    monkeypatch.setattr(live, "X11_DIR", tmp_path / "x11")
    monkeypatch.setattr(live.shutil, "which", lambda name: f"/usr/bin/{name}")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    async def exec_(*argv, **kwargs):
        proc = FakeProc(argv, events, tmp_path, **options)
        what = " ".join(argv[:5]) if proc.name == "xdotool" else proc.name
        events.append(what if proc.name != "google-chrome" else "chrome " + argv[-1])
        procs.append(proc)
        return proc

    monkeypatch.setattr(live.asyncio, "create_subprocess_exec", exec_)
    return events, procs, options, home


async def test_the_broadcast_places_the_windows_records_and_finishes_the_mp4_before_stopping_the_rest(
    tmp_path, fake_processes
):
    events, procs, _, home = fake_processes
    b = live.Broadcast(BroadcastSettings(), tmp_path / "out", overlay_draw_s=0, ffmpeg_check_s=0)
    display = await b.open_display()
    assert display.startswith(":") and events == ["Xvfb"]
    assert procs[0].argv[1:5] == (display, "-screen", "0", "1920x1080x24")
    # the engine starts here and writes its log
    (home / live.ENGINE_LOG).parent.mkdir(parents=True)
    (home / live.ENGINE_LOG).write_text("<p>zeroad-director: cut to base at 1,2</p>")
    await b.start("http://127.0.0.1:18765/live?stream")
    assert events == [
        "Xvfb",
        "xdotool search --sync --class pyrogenesis",
        "chrome --app=http://127.0.0.1:18765/live?stream",
        "xdotool search --class pyrogenesis windowfocus",
        "ffmpeg",
    ]
    move, focus = procs[1].argv, procs[3].argv
    assert move[0] == "xdotool" and "windowmove" in move and move[-2:] == ("0", "0")
    size = move.index("windowsize")
    assert move[size + 2 : size + 4] == ("1440", "1080")  # 0 A.D. alone caps its window at the output's mode
    assert focus[0] == "xdotool" and "windowfocus" in focus  # the game's camera moves only while it has the focus
    ffmpeg = procs[4]
    assert ffmpeg.argv[ffmpeg.argv.index("-f") + 1] == "x11grab" and ffmpeg.argv[-1] == str(b.record_path)
    assert b.recording

    path = await b.stop()
    assert events[5:] == ["ffmpeg stdin q", "google-chrome terminate", "Xvfb terminate"]
    assert ffmpeg.got_q and "ffmpeg kill" not in events
    assert path == b.record_path and path.stat().st_size > 0
    assert b.engine_log is not None and "cut to base" in b.engine_log.read_text()
    assert "frame=1" in b.log_path.read_text()


async def test_a_broadcast_returns_no_recording_when_ffmpeg_wrote_none(tmp_path, fake_processes):
    b = live.Broadcast(BroadcastSettings(display=":0"), tmp_path / "out", overlay_draw_s=0)
    await b.open_display()
    assert await b.stop() is None and b.engine_log is None


async def test_a_broadcast_whose_ffmpeg_exits_at_once_fails_to_start(tmp_path, fake_processes):
    _, _, options, _ = fake_processes
    options["ffmpeg_fails"] = True
    b = live.Broadcast(BroadcastSettings(), tmp_path / "out", overlay_draw_s=0, ffmpeg_check_s=0.05)
    await b.open_display()
    with pytest.raises(RuntimeError, match="ffmpeg exited with 1"):
        await b.start("http://127.0.0.1:18765/live?stream")


async def test_opening_the_display_fails_when_xvfb_exits(tmp_path, fake_processes):
    _, _, options, _ = fake_processes
    options["xvfb_fails"] = True
    b = live.Broadcast(BroadcastSettings(), tmp_path / "out")
    with pytest.raises(RuntimeError, match="exited with 1"):
        await b.open_display()


async def test_an_engine_log_older_than_the_broadcast_is_not_kept_as_this_matchs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    old = home / live.ENGINE_LOG
    old.parent.mkdir(parents=True)
    old.write_text("<p>a previous match's log</p>")
    os.utime(old, (0, 0))
    b = live.Broadcast(BroadcastSettings(display=":0"), tmp_path / "out")
    await b.open_display()
    await b.stop()
    assert b.engine_log is None
    old.write_text("<p>this match's log</p>")
    await b.stop()
    assert b.engine_log is not None and "this match" in b.engine_log.read_text()
