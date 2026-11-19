"""The showcase session runner on scripted engines and fake models: the rotation, the stop rules, no results and
dropped opponents, restarts, SIGTERM, pacing, the broadcast and the verify command.

The session's own clock is fake (`World.t`: each engine start takes `GAME_S`, an intermission its length), the games'
pace is tiny except where a test reads it, and the failure watch's second is a millisecond, so a test takes seconds."""

import asyncio
import json
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agentenv_zeroad.showcase import session as session_mod
from agentenv_zeroad.showcase.__main__ import main
from agentenv_zeroad.showcase.results import GameRow, Session
from agentenv_zeroad.showcase.session import SessionDeps, SessionOptions, run_session
from agentenv_zeroad.showcase.specs import ShowcaseConfig
from tests.fake_models import FakeStrategist
from tests.zeroad_fakes import Handle, ScriptedEngine, placer

ROOT = Path(__file__).resolve().parents[1]
CFG = ShowcaseConfig.load(ROOT / "live" / "showcase" / "showcase.toml")
FIXTURE = Path(__file__).parent / "fixtures" / "showcase_results_frontier.json"
ROTATION = ("opus", "astra", "petra-3")
GAME_S = 100.0
FAST_PACE_S = 0.001


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

        monkeypatch.setattr(session_mod, "used_seeds", lambda configs, sessions=(): set())
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

        async def preflight() -> list[str]:
            return self.preflight

        def model_for(seat, slot):
            self.seen.append((seat, slot))
            if seat.trainable:
                return CountingModel(on_call=self.on_call)
            failing = (seat.name or "") in self.failing
            return CountingModel(fail=RuntimeError("proxy down") if failing else None)

        return SessionDeps(
            start_engine=start_engine,
            budget_ok=budget_ok,
            poweroff_left_s=lambda: self.left_s,
            model_for=model_for,
            preflight=preflight,
            now=lambda: self.t,
            sleep=sleep or fake_sleep,
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


def test_replay_mode_is_not_built_yet(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(NotImplementedError, match="replay mode is not built yet"):
        world.run(world.opts(mode="replay"))
    assert main(["run", "--out", str(world.out), "--mode", "replay"]) == 2
    assert "replay mode is not built yet" in capsys.readouterr().err


class FakeBroadcast:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.url = ""

    async def open_display(self) -> str:
        self.calls.append("open")
        return ":0"

    async def start(self, url: str) -> None:
        self.calls.append("start")
        self.url = url

    async def attach_game(self) -> None:
        self.calls.append("attach")

    async def stop(self) -> None:
        self.calls.append("stop")


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
