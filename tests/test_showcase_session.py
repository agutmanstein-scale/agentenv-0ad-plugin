"""The showcase session runner on scripted engines and fake models: the rotation, the stop rules, no results and
dropped opponents, restarts, SIGTERM, pacing, the broadcast and the verify command.

The session's own clock is fake (`World.t`: each engine start takes `GAME_S`, an intermission its length), the games'
pace is tiny except where a test reads it, and the failure watch's second is a millisecond, so a test takes seconds."""

import asyncio
import json
import os
import signal
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentenv_zeroad.showcase import __main__ as main_mod
from agentenv_zeroad.showcase import session as session_mod
from agentenv_zeroad.showcase.__main__ import (
    _check,
    budget_ok,
    check_with_retries,
    exit_code,
    game_problems,
    main,
    unwritable,
)
from agentenv_zeroad.showcase.results import GameRow, Session
from agentenv_zeroad.showcase.session import SessionDeps, SessionOptions, run_session
from agentenv_zeroad.showcase.specs import ShowcaseConfig
from agentenv_zeroad.train.panels import position
from agentenv_zeroad.train.spec import SeatSpec
from tests.fake_models import FakeStrategist
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

ROOT = Path(__file__).resolve().parents[1]
CFG = ShowcaseConfig.load(ROOT / "live" / "showcase" / "showcase.toml")
FIXTURE = Path(__file__).parent / "fixtures" / "showcase_results_frontier.json"
ROTATION = ("opus", "astra", "petra-3")
GAME_S = 100.0
FAST_PACE_S = 0.001


class RelayStub:
    """mediamtx's API on loopback: `GET /v3/paths/get/zeroad` says whether the path has a stream."""

    def __init__(self) -> None:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = json.dumps({"name": "zeroad", "ready": stub.ready}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                return None

        self.ready = True
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v3/paths/get/zeroad"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


RELAY = RelayStub()


@pytest.fixture(autouse=True)
def relay_up() -> None:
    RELAY.ready = True


class CountingModel(FakeStrategist):
    """A fake model that counts calls and failed calls as `StrategistModel.sampler_stats` does."""

    def __init__(self, *, fail: Exception | None = None, on_call: Any = None):
        super().__init__(fail=fail)
        self.counts = {"calls": 0, "failed_calls": 0}
        self.on_call = on_call

    async def call(self, system: str, user: str, tools: list[Any]):
        if self.on_call is not None:
            self.on_call()
        try:
            reply = await super().call(system, user, tools)
        except Exception:
            self.counts["failed_calls"] += 1
            raise
        self.counts["calls"] += 1
        return reply

    def sampler_stats(self) -> dict[str, int]:
        return dict(self.counts)


class DyingEngine(ScriptedEngine):
    """Steps `lives` times, then the engine is gone."""

    def __init__(self, lives: int):
        super().__init__()
        self.lives = lives

    async def step_raw(self, commands=()) -> None:
        if self.raw_steps >= self.lives:
            raise ConnectionError("the engine went away")
        await super().step_raw(commands)


class World:
    """Everything a test needs to see of one session run."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cap_minutes: float = 1.0):
        self.out = tmp_path / "session"
        self.handles: list[Handle] = []
        self.seen: list[tuple[Any, str]] = []
        self.budget_calls: list[tuple[float, float]] = []
        self.failing: set[str] = set()
        self.dying: dict[int, int] = {}
        self.budget = True
        self.left_s = 1e9
        self.preflight: list[str] = []
        self.preflights = 0
        self.skipped: list[set[str]] = []
        self.policy_fails = False
        self.game_checks: list[str] = []
        self.game_problems: dict[str, dict[str, str]] = {}
        self.spend: list[tuple[str, float]] = []
        self.on_call: Any = None
        self.cap_minutes = cap_minutes
        self.t = 0.0
        real_game_at, real_play = session_mod.game_at, session_mod.play_game

        def game_at(cfg, panel, j, opponent, base_url):
            spec = real_game_at(cfg, panel, j, opponent, base_url)
            seats = tuple(replace(s, civ=civ) for s, civ in zip(spec.seats, ("rome", "pers"), strict=True))
            return replace(spec, seats=seats, time_cap_minutes=self.cap_minutes)

        async def play(*args, **kwargs):
            return await real_play(*args, match_options={"placer": placer()}, **kwargs)

        real_used = session_mod.used_seeds
        monkeypatch.setattr(session_mod, "used_seeds", lambda configs, sessions=(): real_used([], sessions))
        monkeypatch.setattr(session_mod, "run_configs", lambda rules: [rules])
        monkeypatch.setattr(session_mod, "game_at", game_at)
        monkeypatch.setattr(session_mod, "play_game", play)

    def opts(self, **over: Any) -> SessionOptions:
        args: dict[str, Any] = {
            "out": self.out,
            "rotation": ROTATION,
            "intermission_s": 5.0,
            "hours": 6.0,
            "pace_s": FAST_PACE_S,
            "relay_status_url": RELAY.url,
        }
        return SessionOptions(**{**args, **over})

    def deps(self, *, sleep: Any = None) -> SessionDeps:
        async def fake_sleep(seconds: float) -> None:
            if seconds == 5.0:  # an intermission passes on the session's clock
                self.t += seconds
                await asyncio.sleep(0)
            else:  # the failure watch's second
                await asyncio.sleep(0.001)

        async def start_engine() -> Handle:
            self.t += GAME_S
            engine = DyingEngine(self.dying[len(self.handles)]) if len(self.handles) in self.dying else None
            self.handles.append(Handle(engine))
            return self.handles[-1]

        async def budget_ok(usd: float, hours: float) -> bool:
            self.budget_calls.append((usd, hours))
            return self.budget

        async def preflight(skip) -> list[str]:
            self.preflights += 1
            self.skipped.append(set(skip))
            return self.preflight

        async def check_game(opponent: str) -> dict[str, str]:
            self.game_checks.append(opponent)
            return self.game_problems.get(opponent, {})

        def model_for(seat, slot):
            self.seen.append((seat, slot))
            if seat.trainable:
                return CountingModel(
                    on_call=self.on_call, fail=RuntimeError("vllm down") if self.policy_fails else None
                )
            failing = (seat.name or "") in self.failing
            return CountingModel(fail=RuntimeError("proxy down") if failing else None)

        return SessionDeps(
            start_engine=start_engine,
            budget_ok=budget_ok,
            poweroff_left_s=lambda: self.left_s,
            model_for=model_for,
            preflight=preflight,
            record_spend=lambda what, usd: self.spend.append((what, usd)),
            now=lambda: self.t,
            sleep=sleep or fake_sleep,
            check_game=check_game,
        )

    def run(self, opts: SessionOptions | None = None, broadcast: Any = None) -> Session:
        opts = opts or self.opts()
        return asyncio.run(run_session(opts, CFG, self.deps(), broadcast))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def names(session: Session) -> list[str]:
    return [r.opponent for r in session.rows]


def test_session_plays_rotation_in_order(world: World) -> None:
    session = world.run(world.opts(max_games=4))
    assert names(session) == ["Opus", "Astra", "Petra-3", "Opus"], (session.ended, session.ended_why)
    assert [r.index for r in session.rows] == [0, 1, 2, 3]
    assert len({r.seed for r in session.rows}) == 4
    assert all(r.result in ("win", "loss", "draw") for r in session.rows), [r.reason for r in session.rows]
    assert [h.closed for h in world.handles] == [1, 1, 1, 1]


def test_policy_plays_the_pinned_adapter_and_teachers_play_the_13_3s_settings(world: World) -> None:
    world.run(world.opts(max_games=2, policy_url="http://vllm.invalid:8000"))
    policy = [s for s, _ in world.seen if s.trainable]
    teachers = [s for s, _ in world.seen if s.kind == "llm" and not s.trainable]
    assert {(s.model, s.base_url) for s in policy} == {("local/policy-v1", "http://vllm.invalid:8000")}
    assert [(s.name, s.model) for s in teachers] == [
        ("opus", "anthropic/claude-opus-5-5"),
        ("astra", "openai/gpt-6-astra"),
    ]
    saved = json.loads((world.out / "session.json").read_text())
    assert len(saved["rows"]) == 2 and saved["ended"] == "max_games"


def test_session_ends_after_hours_at_a_game_boundary(world: World) -> None:
    session = world.run(world.opts(hours=0.05))  # 180 s: the second game ends at 205 s, so there is no third
    assert session.ended == "hours"
    assert len(session.rows) == 2
    assert all(r.result in ("win", "loss", "draw") for r in session.rows)


def test_max_games(world: World) -> None:
    session = world.run(world.opts(max_games=2))
    assert len(session.rows) == 2 and session.ended == "max_games"


def test_budget_refusal_ends_session_before_a_game(world: World) -> None:
    world.budget = False
    session = world.run()
    assert session.rows == [] and world.handles == []
    assert session.ended == "budget"
    assert world.budget_calls == [(9.0, 0.75)]


def test_petra_games_plan_no_model_spend(world: World) -> None:
    world.run(world.opts(rotation=("petra-3",), max_games=1))
    assert world.budget_calls == [(0.0, 0.75)]


def test_poweroff_margin_ends_session(world: World) -> None:
    world.left_s = 2699.0
    session = world.run()
    assert session.rows == [] and world.handles == []
    assert session.ended == "poweroff"


def test_engine_crash_is_no_result_and_rotation_moves_on(world: World) -> None:
    world.dying = {0: 20}
    session = world.run(world.opts(max_games=2))
    first, second = session.rows
    assert (first.opponent, first.result, first.blame) == ("Opus", "no_result", "engine")
    assert first.reason == "engine failed"
    assert second.opponent == "Astra" and second.result in ("win", "loss", "draw")
    assert first.seed != second.seed


def test_start_engine_failure_is_no_result_and_three_in_a_row_end_the_session(world: World) -> None:
    async def broken() -> Handle:
        raise OSError("no display with the secret name")

    deps = replace(world.deps(), start_engine=broken)
    session = asyncio.run(run_session(world.opts(), CFG, deps, None))
    assert [r.result for r in session.rows] == ["no_result"] * 3
    assert all(r.reason == "engine failed (OSError)" for r in session.rows)
    assert session.ended == "engine"
    assert "secret" not in json.dumps([r.to_json() for r in session.rows])


def test_five_failures_in_a_row_end_game_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"opus"}
    session = world.run(world.opts(max_games=1))
    (row,) = session.rows
    assert (row.result, row.blame) == ("no_result", "opponent")
    assert row.reason == "Opus, 5 failed calls in a row"
    results = json.loads(next((world.out / "games").glob("*/results.json")).read_text())
    assert results["showcase"]["ended_early"]["failures"] == 5
    assert results["game_time_s"] < 180


def test_dropped_opponent_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}
    session = world.run(world.opts(max_games=8))
    assert names(session) == ["Opus", "Astra", "Petra-3", "Opus", "Astra", "Petra-3", "Opus", "Petra-3"]
    assert [r.result for r in session.rows if r.opponent == "Astra"] == ["no_result", "no_result"]
    assert session.dropped() == {"Astra"}


def test_everyone_dropped_ends_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}
    session = world.run(world.opts(rotation=("astra",), max_games=5))
    assert names(session) == ["Astra", "Astra"] and session.ended == "opponents"


def test_restart_continues_from_next_index(world: World) -> None:
    first = world.run(world.opts(max_games=2))
    again = world.run(world.opts(max_games=2))
    assert [r.index for r in again.rows] == [0, 1, 2, 3]
    assert names(again) == ["Opus", "Astra", "Petra-3", "Opus"]
    assert [r.seed for r in again.rows[:2]] == [r.seed for r in first.rows]
    assert len({r.seed for r in again.rows}) == 4
    assert again.ended == "max_games"


def test_preflight_failure_refuses_to_start(world: World) -> None:
    world.preflight = ["anthropic/claude-opus-5-5: HTTP 401"]
    session = world.run()
    assert session.rows == [] and world.handles == []
    assert session.ended == "preflight" and "claude-opus-5-5" in (session.ended_why or "")


def test_sigterm_records_stopped_by_hand(world: World) -> None:
    before = signal.getsignal(signal.SIGTERM)
    world.on_call = lambda: os.kill(os.getpid(), signal.SIGTERM)
    session = world.run(world.opts())
    (row,) = session.rows
    assert (row.result, row.reason, row.blame) == ("no_result", "stopped by hand", None)
    assert session.ended == "stopped"
    assert json.loads((world.out / "session.json").read_text())["ended"] == "stopped"
    assert [h.closed for h in world.handles] == [1]
    assert signal.getsignal(signal.SIGTERM) == before


def test_games_are_paced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=0.05)  # 15 steps at the default pace, 3 s
    opts = SessionOptions(out=world.out, rotation=ROTATION, intermission_s=5.0, max_games=1)
    assert opts.pace_s == 0.2
    world.run(opts)
    (path,) = (world.out / "games").glob("*/results.json")
    clock = json.loads(path.read_text())["clock"]
    assert clock["pace_s"] == 0.2 and clock["paced_wait_s"] > 2


class FakeBroadcast:
    record_path = Path("/nonexistent/recording.mp4")

    def __init__(self, *, start_fails_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.url = ""
        self.recording = False
        self.overlay_alive = False
        self.starts = 0
        self.start_fails_after = start_fails_after

    async def open_display(self) -> str:
        self.calls.append("open")
        return ":0"

    async def start(self, url: str) -> None:
        self.calls.append("start")
        self.url = url
        self.starts += 1
        if self.start_fails_after is not None and self.starts > self.start_fails_after:
            raise RuntimeError("ffmpeg exited with 1 at start")
        self.recording = self.overlay_alive = True

    async def attach_game(self) -> None:
        self.calls.append("attach")

    async def stop(self) -> None:
        self.calls.append("stop")
        self.recording = self.overlay_alive = False


class FakeOverlay:
    last: "FakeOverlay"

    def __init__(self, state, token, host="127.0.0.1", port=0):
        self.state, self.token, self.running = state, token, False
        FakeOverlay.last = self

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


def test_broadcast_starts_once_and_the_overlay_shows_the_intermission(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)
    broadcast = FakeBroadcast()
    shots: list[dict[str, Any]] = []

    async def sleep(seconds: float) -> None:
        if seconds == 5.0:  # the intermission
            shots.append(FakeOverlay.last.state())
            world.t += seconds
        await asyncio.sleep(0.001)

    deps = world.deps(sleep=sleep)
    asyncio.run(run_session(world.opts(max_games=2), CFG, deps, broadcast))  # type: ignore[arg-type]
    assert broadcast.calls == ["open", "start", "attach", "stop"]
    token = FakeOverlay.last.token
    assert len(token) >= 24 and f"token={token}" in broadcast.url and broadcast.url.startswith("http://127.0.0.1:")
    assert FakeOverlay.last.running is False
    first = shots[0]
    headline = first["intermission"]["last"]["headline"]
    assert "net_worth" not in headline
    assert first["phase"] == "intermission" and first["intermission"]["next"] == "Astra"
    assert first["series"]["opponent"] == "Opus" and first["series"]["game"] == 1
    assert [row["opponent"] for row in first["session"]["table"]] == ["Opus", "Astra", "Petra-3"]
    assert shots[1]["intermission"]["next"] == "end of session"


def test_run_configs_are_the_loadable_ones() -> None:
    found = {p.name for p in session_mod.run_configs(CFG.rules_config)}
    assert "run1-frontier.toml" in found and "baselines.toml" in found
    assert not found & {"sequence.toml", "sft-panels.toml"}


def _session_with(directory: Path, results: dict[str, Any]) -> None:
    row = GameRow(
        index=0,
        game_id=results["spec"]["game_id"],
        opponent="Opus",
        seed=1,
        policy_slot="1",
        result="win",
        how="conquest",
        reason=None,
        failed={"policy": 0, "opponent": 0},
        decisions={"policy": 1, "opponent": 1},
        played_at=0.0,
    )
    session = Session(directory / "session.json")
    session.add(row)
    session.save()
    game = directory / "games" / row.game_id
    game.mkdir(parents=True)
    (game / "results.json").write_text(json.dumps(results))


def test_verify_cli_exits_0_on_a_clean_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _session_with(tmp_path, json.loads(FIXTURE.read_text()))
    assert main(["verify", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_verify_cli_exits_1_on_a_problem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = json.loads(FIXTURE.read_text())
    applied = next(d for d in results["lockstep"]["decisions"] if d.get("applied_ms") is not None)
    applied["applied_ms"] += 200
    _session_with(tmp_path, results)
    assert main(["verify", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert results["spec"]["game_id"] in out and "landed 5200 ms later, not 5000" in out


def test_every_rotation_model_is_checked_before_game_1_and_only_that_games_seats_before_later_ones(
    world: World,
) -> None:
    world.run(world.opts(max_games=3))
    assert world.preflights == 1
    assert world.game_checks == ["astra", "petra-3"]


def test_preflight_refuses_a_200_without_the_model_and_a_bad_key() -> None:
    seat = SeatSpec(kind="llm", civ="rome", model="local/policy-v1", base_url="http://v.invalid/v1", api_key_env="K")

    def check(handler):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                return await _check(seat, {"K": "secret-key"}, http)

        return asyncio.run(go())

    served = lambda ids: lambda request: httpx.Response(200, json={"data": [{"id": i} for i in ids]})  # noqa: E731
    assert check(served(["local/policy-v1"])) is None
    assert check(lambda request: httpx.Response(401)) == "local/policy-v1: HTTP 401"
    assert "secret-key" not in (check(lambda request: httpx.Response(500)) or "")


def test_a_model_the_list_omits_passes_on_one_token_chat_completion_that_answers_200() -> None:
    seat = SeatSpec(kind="llm", civ="rome", model="openai/gpt-6-astra", base_url="http://v.invalid/v1", api_key_env="K")
    seen: list[httpx.Request] = []

    def check(completion: int, listing: Any = None):
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "GET":
                return listing or httpx.Response(200, json={"data": [{"id": "other"}]})
            return httpx.Response(completion, json={})

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                return await _check(seat, {"K": "secret-key"}, http)

        return asyncio.run(go())

    assert check(200) is None
    post = next(r for r in seen if r.method == "POST")
    assert json.loads(post.content)["max_tokens"] == 1 and json.loads(post.content)["model"] == "openai/gpt-6-astra"
    assert post.url.path.endswith("/v1/chat/completions")
    for answer in (check(401), check(503, httpx.Response(200, text="<html>"))):
        assert answer is not None and "secret-key" not in answer and "v.invalid" not in answer
    assert check(401) == "openai/gpt-6-astra: HTTP 401"


def test_two_policy_no_results_in_a_row_end_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.policy_fails = True
    session = world.run(world.opts(max_games=6))
    assert [(r.result, r.blame) for r in session.rows] == [("no_result", "policy")] * 2
    assert session.ended == "policy"


def test_an_interrupted_game_is_a_no_result_and_its_start_is_not_replayed(world: World) -> None:
    async def interrupted() -> None:
        task = asyncio.ensure_future(run_session(world.opts(), CFG, world.deps(), None))
        world.on_call = lambda: asyncio.get_running_loop().call_soon(task.cancel)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupted())
    saved = json.loads((world.out / "session.json").read_text())
    assert [(r["result"], r["reason"], r["blame"]) for r in saved["rows"]] == [
        ("no_result", "the runner failed", "engine")
    ]
    assert saved["ended"] == "crashed"
    assert len(world.spend) == 1 and world.spend[0][1] == 9.0  # booked at its caps
    world.on_call = None
    again = world.run(world.opts(max_games=1))
    first, second = again.rows
    assert (first.result, first.blame) == ("no_result", "engine")
    assert second.seed != first.seed and second.opponent == "Astra"


def test_a_start_with_a_game_folder_is_not_replayed(world: World) -> None:
    first = world.run(world.opts(max_games=1))
    (world.out / "session.json").unlink()
    again = world.run(world.opts(max_games=1))
    assert again.rows[0].seed != first.rows[0].seed


def test_a_sibling_sessions_seed_is_skipped(world: World) -> None:
    seed = position(CFG.run_config(), "held_out", CFG.first_index).seed
    sibling = world.out.parent / "earlier"
    sibling.mkdir(parents=True)
    (sibling / "session.json").write_text(json.dumps({"rows": [{"seed": seed, "game_id": "x"}], "ended": None}))
    session = world.run(world.opts(max_games=2))
    assert seed not in {r.seed for r in session.rows}


def test_spend_is_booked_once_per_game(world: World) -> None:
    session = world.run(world.opts(max_games=2))
    assert [what for what, _ in world.spend] == [r.game_id for r in session.rows]
    results = json.loads((world.out / "games" / session.rows[0].game_id / "results.json").read_text())
    assert world.spend[0][1] == results["spend_usd_est"]


def test_an_engine_failed_game_is_booked_at_its_caps(world: World) -> None:
    async def broken() -> Handle:
        raise OSError("no display")

    deps = replace(world.deps(), start_engine=broken)
    asyncio.run(run_session(world.opts(max_games=2, rotation=("opus", "petra-3")), CFG, deps, None))
    assert [usd for _, usd in world.spend] == [9.0, 0.0]


def test_the_overlay_carries_a_no_result_and_the_dropped_opponents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}
    monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)
    shots: list[dict[str, Any]] = []

    async def sleep(seconds: float) -> None:
        if seconds == 5.0:
            shots.append(FakeOverlay.last.state())
            world.t += seconds
        await asyncio.sleep(0.001)

    deps = world.deps(sleep=sleep)
    asyncio.run(run_session(world.opts(rotation=("astra",), max_games=2), CFG, deps, FakeBroadcast()))  # type: ignore[arg-type]
    last = shots[0]["intermission"]["last"]
    assert (last["result"], last["reason"]) == ("no_result", "Astra, 5 failed calls in a row")
    assert shots[0]["session"]["dropped"] == []
    assert len(shots[1]["session"]["dropped"]) == 1 and "Astra" in shots[1]["session"]["dropped"][0]


def test_run_refuses_without_a_spend_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ZEROAD_SPEND_LEDGER", raising=False)
    assert main(["run", "--out", str(tmp_path / "s")]) == 2
    assert "ZEROAD_SPEND_LEDGER" in capsys.readouterr().err


def test_a_natural_end_is_never_recorded_as_a_preflight_failure(world: World) -> None:
    world.preflight = ["anthropic/claude-opus-5-5: HTTP 401"]
    session = world.run(world.opts(max_games=0))
    assert session.ended == "max_games" and world.preflights == 0
    world.preflight = []
    session = world.run(world.opts(max_games=1))
    assert session.ended == "max_games" and world.preflights == 1


def test_preflight_skips_dropped_opponents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}
    world.run(world.opts(max_games=6))
    assert world.skipped == [set()]  # the strict check runs once, before game 1; astra's later games are checked alone
    assert world.game_checks.count("astra") == 2  # then it is dropped and never checked again


def test_sigterm_interrupts_a_preflight_in_progress(world: World) -> None:
    async def hangs(skip) -> list[str]:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.Event().wait()
        return []

    deps = replace(world.deps(), preflight=hangs)
    session = asyncio.run(run_session(world.opts(), CFG, deps, None))
    assert session.rows == [] and session.ended == "stopped"


def test_sigint_is_a_stop_by_hand_too(world: World) -> None:
    before = signal.getsignal(signal.SIGINT)
    world.on_call = lambda: os.kill(os.getpid(), signal.SIGINT)
    session = world.run(world.opts())
    (row,) = session.rows
    assert (row.result, row.reason) == ("no_result", "stopped by hand") and session.ended == "stopped"
    assert signal.getsignal(signal.SIGINT) == before


def test_a_stop_during_the_broadcast_start_still_stops_the_broadcast(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)

    class Hanging(FakeBroadcast):
        async def start(self, url: str) -> None:
            self.calls.append("start")
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.Event().wait()

    broadcast = Hanging()
    session = world.run(world.opts(), broadcast)
    assert broadcast.calls == ["open", "start", "stop"]
    assert session.ended == "stopped" and session.rows[0].reason == "stopped by hand"


def test_preflight_retries_each_model_three_times_ten_seconds_apart() -> None:
    seat = SeatSpec(kind="llm", civ="rome", model="m", base_url="http://v.invalid/v1", api_key_env="K")
    answers = iter([503, 503, 200])
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def handler(request):
        code = next(answers) if request.method == "GET" else 503
        return httpx.Response(code, json={"data": [{"id": "m"}]})

    async def go(h):
        async with httpx.AsyncClient(transport=httpx.MockTransport(h)) as http:
            return await check_with_retries(seat, {"K": "k"}, http, sleep)

    assert asyncio.run(go(handler)) is None and sleeps == [10.0, 10.0]
    sleeps.clear()
    assert asyncio.run(go(lambda request: httpx.Response(503))) == "m: HTTP 503" and sleeps == [10.0, 10.0]


def test_a_later_games_seats_are_tried_five_times_fifteen_seconds_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    seat = SeatSpec(kind="llm", civ="rome", model="m", base_url="http://v.invalid/v1", api_key_env="K")
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
            return await check_with_retries(seat, {"K": "k"}, http, sleep, tries=5, retry_s=15.0)

    assert asyncio.run(go()) == "m: HTTP 503" and sleeps == [15.0] * 4


def test_game_problems_names_the_unreachable_role_and_leaves_petra_unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    opts = SessionOptions(out=Path("unused"), rotation=ROTATION, policy_url="http://vllm.invalid:8000")
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        if "vllm.invalid" in str(request.url):
            return httpx.Response(200, json={"data": [{"id": "local/policy-v1"}]})
        return httpx.Response(503)

    async def go(opponent: str):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await game_problems(CFG, opts, http, opponent, sleep)

    for var in ("LITELLM_API_KEY", "VLLM_API_KEY"):
        monkeypatch.setenv(var, "k")
    found = asyncio.run(go("opus"))
    assert list(found) == ["opponent"] and "HTTP 503" in found["opponent"] and sleeps == [15.0] * 4
    assert asyncio.run(go("petra-3")) == {}


def test_budget_check_is_called_without_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    class Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_exec(*args, **kwargs):
        seen.append(args)
        return Proc()

    monkeypatch.setenv("LIVE_BUDGET_USD", "60")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert asyncio.run(budget_ok(9.0, 0.75)) is True
    args = seen[0]
    assert args[0].endswith("box/budget.sh") and args[1:3] == ("check", "--no-record")
    assert "--planned-usd" in args and "--budget" in args


def test_unwritable_ledger_or_out_is_refused_with_a_message(tmp_path: Path) -> None:
    assert unwritable(tmp_path / "ledger.jsonl", tmp_path / "out") is None
    assert "spend ledger" in unwritable(tmp_path / "nope" / "ledger.jsonl", tmp_path / "out")
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert "output folder" in unwritable(tmp_path / "ledger.jsonl", blocker / "out")


def test_a_pending_row_is_a_verify_problem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    session = Session(tmp_path / "session.json")
    session.add(
        GameRow(
            0,
            "g-pending",
            "Opus",
            1,
            "1",
            "pending",
            None,
            None,
            {"policy": 0, "opponent": 0},
            {"policy": 0, "opponent": 0},
            0.0,
        )
    )
    session.save()
    assert main(["verify", str(tmp_path)]) == 1
    assert "g-pending: the game never finished (pending)" in capsys.readouterr().out


def test_an_unreachable_opponent_is_a_no_result_that_books_no_spend_and_the_rotation_moves_on(world: World) -> None:
    world.game_problems = {"astra": {"opponent": "openai/gpt-6-astra: no answer (ConnectTimeout)"}}
    session = world.run(world.opts(max_games=3))
    opus, astra, petra = session.rows
    assert (astra.opponent, astra.result, astra.reason, astra.blame) == (
        "Astra",
        "no_result",
        "Astra unreachable",
        "opponent",
    )
    assert astra.problems == [] and astra.seed not in {opus.seed, petra.seed}
    assert opus.result in ("win", "loss", "draw") and petra.opponent == "Petra-3"
    assert dict(world.spend)[astra.game_id] == 0.0 and len(world.handles) == 2  # no engine for it
    assert "openai" not in json.dumps([r.to_json() for r in session.rows])


def test_an_opponent_unreachable_twice_is_dropped(world: World) -> None:
    world.game_problems = {"astra": {"opponent": "x"}}
    session = world.run(world.opts(max_games=8))
    assert session.dropped() == {"Astra"}
    assert names(session).count("Astra") == 2


def test_an_unreachable_policy_ends_the_session_as_policy(world: World) -> None:
    world.game_problems = {"astra": {"policy": "local/policy-v1: no answer (ConnectError)"}}
    session = world.run(world.opts(max_games=3))
    assert [r.opponent for r in session.rows] == ["Opus"] and session.ended == "policy"
    assert "policy" in (session.ended_why or "")


def test_exit_code_is_0_only_when_a_session_ran_and_ended_normally() -> None:
    def session(ended: str | None, rows: list[GameRow]) -> Session:
        s = Session(Path("unused.json"))
        s.rows, s.ended = rows, ended
        return s

    def row(**over: Any) -> GameRow:
        base: dict[str, Any] = {
            "index": 0, "game_id": "g", "opponent": "Opus", "seed": 1, "policy_slot": "1", "result": "win",
            "how": "conquest", "reason": None, "failed": {}, "decisions": {}, "played_at": 0.0,
        }  # fmt: skip
        return GameRow(**{**base, **over})

    for refusal in ("preflight", "budget", "poweroff", "starts", "opponents"):
        assert exit_code(session(refusal, [])) == 1, refusal
        assert exit_code(session(refusal, [row(carried_from="20261119-100000")])) == 1, refusal  # none of its own
        assert exit_code(session(refusal, [row()])) == 0, refusal  # a per-game refusal after a game
    for normal in ("hours", "max_games", "stopped"):
        assert exit_code(session(normal, [])) == 0, normal


def test_the_runner_exits_non_zero_on_a_refusal_before_game_1_and_zero_after_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world: World
) -> None:
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("ZEROAD_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("LIVE_BUDGET_USD", "60")
    out = tmp_path / "run"
    plan = {"session": None}

    async def fake_run_session(opts: Any, cfg: Any, deps: Any, broadcast: Any) -> Session:
        return plan["session"]

    monkeypatch.setattr(main_mod, "run_session", fake_run_session)
    argv = ["run", "--out", str(out), "--config", str(ROOT / "live" / "showcase" / "showcase.toml")]
    plan["session"] = world.run(world.opts(max_games=0, hours=6.0))
    plan["session"].ended = "preflight"
    assert main(argv) == 1
    plan["session"] = world.run(world.opts(max_games=1))
    assert main(argv) == 0


def test_a_crash_raises_and_leaves_no_pending_row(world: World) -> None:
    async def explode(skip) -> list[str]:
        raise RuntimeError("boom")

    deps = replace(world.deps(), preflight=explode)
    with pytest.raises(RuntimeError):
        asyncio.run(run_session(world.opts(), CFG, deps, None))
    saved = json.loads((world.out / "session.json").read_text())
    assert saved["ended"] == "crashed" and saved["rows"] == []


def _earlier_session_with_a_pending_row(world: World) -> Path:
    earlier = world.run(world.opts(max_games=2))
    path = world.out / "session.json"
    data = json.loads(path.read_text())
    data["rows"][-1]["result"] = "pending"  # a SIGKILL leaves the claimed row as it was
    data["rows"][-1].update(how=None, reason=None, blame=None)
    path.write_text(json.dumps(data))
    assert len(earlier.rows) == 2
    return world.out


def test_continue_starts_the_table_from_the_earlier_sessions_rows_and_marks_them_carried(world: World) -> None:
    folder = _earlier_session_with_a_pending_row(world)
    later = world.out.parent / "later"
    session = world.run(world.opts(out=later, max_games=1, continue_from=folder))
    carried, mine = session.rows[:2], session.rows[2:]
    assert [r.carried_from for r in carried] == [folder.name] * 2 and [r.carried_from for r in mine] == [None]
    assert carried[0].result in ("win", "loss", "draw")
    assert (carried[1].result, carried[1].reason, carried[1].blame) == (
        "no_result",
        "the box or the runner failed",
        "engine",
    )
    assert [r.index for r in session.rows] == [0, 1, 2] and len({r.seed for r in session.rows}) == 3
    assert mine[0].opponent == "Petra-3"  # the rotation goes on after the carried games
    saved = json.loads((later / "session.json").read_text())
    assert [r["carried_from"] for r in saved["rows"]] == [folder.name, folder.name, None]
    table = {t["opponent"]: t for t in session.table()}
    assert table["Opus"]["w"] + table["Opus"]["l"] + table["Opus"]["d"] == 1 and table["Astra"]["nr"] == 1


def test_without_continue_a_new_session_starts_at_0_0(world: World) -> None:
    _earlier_session_with_a_pending_row(world)
    session = world.run(world.opts(out=world.out.parent / "later", max_games=1))
    assert len(session.rows) == 1 and session.rows[0].carried_from is None
    assert session.rows[0].opponent == "Opus"
    assert sum(t["w"] + t["l"] + t["d"] + t["nr"] for t in session.table()) == 1


def test_continue_needs_the_earlier_sessions_file(world: World) -> None:
    with pytest.raises(ValueError, match="cannot continue"):
        world.run(world.opts(out=world.out.parent / "later", continue_from=world.out.parent / "never-was"))


def _boundary_broadcast(world: World, monkeypatch: pytest.MonkeyPatch, broadcast: FakeBroadcast, *, break_it: Any):
    """Run 2 games on `broadcast`; `break_it(broadcast)` runs in the first intermission."""
    monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)

    async def sleep(seconds: float) -> None:
        if seconds == 5.0:
            if broadcast.calls.count("attach") + broadcast.starts == 1:
                break_it(broadcast)
            world.t += seconds
        await asyncio.sleep(0.001)

    return asyncio.run(run_session(world.opts(max_games=2), CFG, world.deps(sleep=sleep), broadcast))  # type: ignore[arg-type]


def test_a_dead_ffmpeg_is_restarted_at_the_game_boundary(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    broadcast = FakeBroadcast()
    session = _boundary_broadcast(world, monkeypatch, broadcast, break_it=lambda b: setattr(b, "recording", False))
    assert broadcast.calls == ["open", "start", "stop", "open", "start", "attach", "stop"]
    assert len(session.rows) == 2 and session.ended == "max_games"


def test_a_dead_chrome_or_a_relay_without_the_stream_is_restarted_too(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    chrome = FakeBroadcast()
    _boundary_broadcast(world, monkeypatch, chrome, break_it=lambda b: setattr(b, "overlay_alive", False))
    assert chrome.calls.count("start") == 2
    relay = FakeBroadcast()

    def relay_drops(b: FakeBroadcast) -> None:
        RELAY.ready = False
        original = b.start

        async def start(url: str) -> None:
            await original(url)
            RELAY.ready = True  # the restarted ffmpeg publishes again

        b.start = start  # type: ignore[method-assign]

    _boundary_broadcast(world, monkeypatch, relay, break_it=relay_drops)
    assert relay.calls.count("start") == 2


def test_a_failed_restart_ends_the_session_as_broadcast(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    broadcast = FakeBroadcast(start_fails_after=1)
    session = _boundary_broadcast(world, monkeypatch, broadcast, break_it=lambda b: setattr(b, "recording", False))
    assert session.ended == "broadcast" and len(session.rows) == 1
    assert len(world.handles) == 1  # no second engine started
    assert broadcast.calls[-1] == "stop"


def test_checks_that_still_fail_after_the_restart_end_the_session_as_broadcast(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    broadcast = FakeBroadcast()

    def relay_never_comes_back(b: FakeBroadcast) -> None:
        RELAY.ready = False

    session = _boundary_broadcast(world, monkeypatch, broadcast, break_it=relay_never_comes_back)
    assert session.ended == "broadcast" and broadcast.calls.count("start") == 2


def test_the_hold_line_names_the_seat_not_the_model_id(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)
    seen: list[Any] = []
    world.on_call = lambda: seen.append(FakeOverlay.last.state()["hold"])
    world.run(world.opts(max_games=1), FakeBroadcast())
    for hold in (h for h in seen if h):
        assert hold["name"] in ("Qwen3-8B + RL", "Opus 5.5")


def test_verify_does_not_count_a_game_with_no_decision_as_a_problem_when_the_engine_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = Session(tmp_path / "session.json")
    for blame, n in (("engine", 0), ("opponent", 1)):
        session.add(
            GameRow(n, f"g{n}", "Opus", n, "1", "no_result", None, "x", {}, {}, 0.0,
                    problems=["no decision was applied"], blame=blame)  # type: ignore[arg-type]
        )  # fmt: skip
    session.save()
    assert main(["verify", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines() == ["g1: no decision was applied"]


def test_the_intermission_card_words_how_a_game_was_won_never_the_raw_rule() -> None:
    def card(how: str) -> dict[str, Any]:
        row = GameRow(0, "g", "Opus", 1, "1", "win", how, None, {}, {}, 0.0)  # type: ignore[arg-type]
        return session_mod._Runner._card(None, row, "Astra", [])  # type: ignore[arg-type]

    assert card("conquest")["last"]["how"] == "conquest"
    assert card("net_worth")["last"]["how"] == "ahead on net worth at 30:00"
