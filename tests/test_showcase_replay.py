"""Replay mode (A2): the timeline the overlay follows, the engine arguments, and the session that plays games headless
and shows their replays in order, adding each game to the table only when its replay ends.

The session tests reuse the session runner's fake world; the replay engine is a fake process and the replay's clock a
fake that only advances once the game one ahead has been played, so the order of events does not depend on timing."""

import asyncio
import itertools
import json
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agentenv_zeroad.engine import pyrogenesis_args
from agentenv_zeroad.live import DIRECTOR_MOD
from agentenv_zeroad.showcase import replay as replay_mod
from agentenv_zeroad.showcase import session as session_mod
from agentenv_zeroad.showcase.overlay import showcase_state
from agentenv_zeroad.showcase.replay import (
    ReplayTimeline,
    WallReplayClock,
    et_clock,
    replay_args,
    start_headless_engine,
)
from agentenv_zeroad.showcase.replay_session import _Shown
from agentenv_zeroad.showcase.results import Session
from agentenv_zeroad.showcase.session import run_session
from tests.test_showcase_session import CFG, FakeBroadcast, FakeOverlay, World
from tests.zeroad_fakes import Handle


def _results() -> dict[str, Any]:
    def row(t: int, total: int, pop: int, state: str = "active") -> dict[str, Any]:
        return {"state": state, "total": total, "pop": pop, "pop_cap": 50, "army": pop // 2, "phase": "village"}

    def turn(began: int, *tools: str) -> dict[str, Any]:
        orders = [{"tool": t, "arguments": {"n": i}, "status": "accepted"} for i, t in enumerate(tools)]
        return {"began_ms": began, "orders": orders}

    return {
        "game_time_s": 25.0,
        "time_cap_s": 1800,
        "lockstep": {
            "latency_ms": 5000,
            "decisions": [
                {"slot": "1", "frozen_ms": 0, "applied_ms": 5000, "missed": False},
                {"slot": "1", "frozen_ms": 10_000, "applied_ms": None, "missed": True},
                {"slot": "1", "frozen_ms": 20_000, "applied_ms": 25_000, "missed": False},
            ],
        },
        "seats": {
            "1": {
                "player": 1,
                "kind": "agent",
                "civ": "rome",
                "name": "Agent 1",
                "model": "local/policy-v1",
                "outcome": "won",
                "turns": [turn(0, "train"), turn(10_000, "build"), turn(20_000, "army", "say")],
            },
            "2": {"player": 2, "kind": "petra", "civ": "pers", "name": "Petra", "model": None, "outcome": "lost"},
        },
        "timeline": [
            {"t_ms": 0, "players": {"1": row(0, 10, 5), "2": row(0, 12, 5)}},
            {"t_ms": 10_000, "players": {"1": row(10_000, 40, 9), "2": row(10_000, 30, 8)}},
            {"t_ms": 20_000, "players": {"1": row(20_000, 70, 12), "2": row(20_000, 35, 8, "defeated")}},
        ],
    }


def test_timeline_state_at_matches_results() -> None:
    timeline = ReplayTimeline(_results())
    mid = timeline.state_at(15_000)
    one, two = mid["seats"]
    assert (one["score"], one["pop"], one["pop_cap"], one["army"], one["phase"]) == (40, 9, 50, 4, "village")
    assert (two["score"], two["state"], two["kind"], two["model"]) == (30, "active", "petra", "Petra (0 A.D. AI)")
    assert one["model"] == "local/policy-v1" and one["civ_name"] == "Romans" and one["color"].startswith("#")
    assert [s["slot"] for s in mid["scoreboard"]] == ["1", "2"] and mid["scoreboard"][0]["score"] == 40
    assert (mid["clock"], mid["cap"], mid["status"], one["outcome"]) == ("00:15", "30:00", "playing", None)
    assert one["thinking_s"] is None and mid["seats"][0]["say"] is None
    end = timeline.state_at(25_000)
    assert end["status"] == "over" and [s["outcome"] for s in end["seats"]] == ["won", "lost"]
    assert end["seats"][1]["state"] == "defeated" and end["seats"][1]["score"] == 35
    assert timeline.state_at(0)["seats"][0]["score"] == 10


def test_a_result_is_not_in_the_state_before_the_games_end() -> None:
    timeline = ReplayTimeline(_results())
    assert all(s["outcome"] is None for s in timeline.state_at(24_999)["seats"])


def test_recent_orders_never_from_the_future() -> None:
    timeline = ReplayTimeline(_results())

    def lines(game_ms: int) -> list[str]:
        return [o["line"] for o in timeline.state_at(game_ms)["seats"][0]["recent"]]

    assert lines(4_999) == [] and lines(5_000) == ['train({"n":0})']
    assert lines(24_999) == ['train({"n":0})']  # the missed decision's order never went out
    assert lines(25_000) == ['say({"n":1})', 'army({"n":0})', 'train({"n":0})']
    assert [o["at"] for o in timeline.state_at(25_000)["seats"][0]["recent"]] == ["00:25", "00:25", "00:05"]
    for ms in range(0, 30_000, 500):  # whatever the time, every line's own time is not after it
        for o in timeline.state_at(ms)["seats"][0]["recent"]:
            minutes, seconds = o["at"].split(":")
            assert int(minutes) * 60 + int(seconds) <= ms // 1000


def test_a_timeline_of_a_game_with_no_results_to_read_is_empty_not_an_error() -> None:
    state = ReplayTimeline({"status": "failed"}).state_at(0)
    assert state["seats"] == [] and state["scoreboard"] == [] and state["status"] == "over"


def test_the_wall_clock_counts_from_the_first_frame() -> None:
    now = [100.0]
    clock = WallReplayClock(lambda: now[0])
    assert clock.game_ms() == 0
    now[0] = 130.0
    assert clock.game_ms() == 0  # not started: the engine is still opening
    clock.start()
    now[0] = 142.5
    assert clock.game_ms() == 12_500
    clock.start(at=140.0)  # the first frame was seen before the session got round to starting the clock
    assert clock.game_ms() == 2_500


def test_header_says_replay_and_time_played() -> None:
    winter = 1_795_115_100.0  # 2026-11-19 19:05 UTC, EST
    summer = 1_782_929_100.0  # 2026-07-01 18:05 UTC, EDT
    assert et_clock(winter) == "14:05" and et_clock(summer) == "14:05"
    table = Session(Path("unused.json"))
    state = showcase_state(
        table,
        None,
        phase="playing",
        opponent="Opus",
        game_no=1,
        round_no=1,
        hold=None,
        mode="replay",
        played_at=et_clock(winter),
        intermission=None,
        base=ReplayTimeline(_results()).state_at(1000),
    )
    assert state["replay"] == {"played_at": "14:05 ET"} and state["hold"] is None
    assert state["series"]["clock"] == "00:01" and len(state["seats"]) == 2


def test_replay_args_include_director_mod_and_replay_path(tmp_path: Path) -> None:
    commands = tmp_path / "games" / "g" / "commands.txt"
    args = replay_args(commands)
    assert f"-mod={DIRECTOR_MOD}" in args and "-mod=public" in args
    assert args.index("-mod=public") < args.index(f"-mod={DIRECTOR_MOD}")
    assert f"-replay-visual={commands}" in args
    assert "-xres=1440" in args and "-yres=1080" in args
    assert not [a for a in args if "rl-" in a]  # no RL interface, no pacing, no realtime


def test_headless_games_use_the_training_image_unpaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []

    async def fake_start(options, **kwargs):
        seen.append((options, kwargs))
        return object()

    monkeypatch.setattr(replay_mod, "start_engine", fake_start)
    asyncio.run(start_headless_engine(tmp_path / "home"))
    ((options, kwargs),) = seen
    assert (options.visual, options.paced, options.realtime) == (False, False, False)
    assert kwargs["no_state_step"] is True and kwargs["mode"] == "local" and kwargs["home"] == tmp_path / "home"
    args = pyrogenesis_args(options)
    assert "-autostart-nonvisual" in args and "-rl-paced" not in args and "-rl-realtime" not in args


# ---- the session -----------------------------------------------------------------------------------------------------
class FakeReplay:
    def __init__(self, commands: Path, log: list[str], *, exits: bool = False) -> None:
        self.commands, self.log, self.exits = commands, log, exits

    def exited(self) -> bool:
        return self.exits

    async def first_frame(self, timeout_s: float) -> float | None:
        return 0.0

    async def stop(self) -> None:
        self.log.append(self.commands.parent.name)


class HeldClock:
    """Starts at 0 and, once started, advances 5 s a read unless `hold()` says the screen has to wait."""

    def __init__(self, hold: Any) -> None:
        self.hold, self.ms, self.started, self.at = hold, 0, False, None

    def start(self, at: float | None = None) -> None:
        self.started, self.at = True, at

    def game_ms(self) -> int:
        if self.started and not self.hold():
            self.ms += 5_000
        return self.ms


class Rig:
    """A replay-mode session on the fake world: samples the overlay at every wait, logs what is shown and played."""

    def __init__(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(session_mod, "OverlayServer", FakeOverlay)
        self.world = world
        self.stopped: list[str] = []  # game ids, as each replay's engine is stopped
        self.shown_starts: list[str] = []
        self.samples: list[dict[str, Any]] = []
        self.paces: list[Any] = []
        self.clocks = 0
        self.total = 0
        self.limit = 3
        self.render_error: Exception | None = None
        self.early_exit = False
        real_play = session_mod.play_game

        async def play(*args: Any, **kwargs: Any):
            self.paces.append(kwargs.get("pace_s", "missing"))
            return await real_play(*args, **kwargs)

        monkeypatch.setattr(session_mod, "play_game", play)

    def finished(self) -> int:
        saved = json.loads((self.world.out / "session.json").read_text())["rows"]
        return sum(1 for r in saved if r["result"] != "pending")

    def sample(self) -> None:
        state = FakeOverlay.last.state()
        table = state["session"]["table"]
        self.samples.append(
            {
                "phase": state["phase"],
                "shown": sum(r["w"] + r["l"] + r["d"] + r["nr"] for r in table),
                "stopped": len(self.stopped),
                "started": len(self.world.handles),
                "finished": self.finished(),
                "replay": state["replay"],
                "hold": state["hold"],
                "clock": state["series"]["clock"],
                "next": (state["intermission"] or {}).get("next"),
            }
        )

    def deps(self) -> session_mod.SessionDeps:
        world = self.world
        base = world.deps()

        async def sleep(seconds: float) -> None:
            self.sample()
            if seconds == 5.0:  # an intermission passes on the session's clock
                world.t += seconds
            await asyncio.sleep(0.001)

        async def never() -> Any:
            raise AssertionError("replay mode must not start the paced engine")

        async def start_headless() -> Handle:
            self.total += 1
            return await base.start_engine()

        async def start_replay(commands: Path) -> FakeReplay:
            self.shown_starts.append(commands.parent.name)
            if self.render_error is not None:
                raise self.render_error
            return FakeReplay(commands, self.stopped, exits=self.early_exit)

        def make_clock() -> HeldClock:
            k = self.clocks
            self.clocks += 1
            return HeldClock(lambda: self.finished() < min(k + 2, self.limit))

        return replace(
            base,
            sleep=sleep,
            start_engine=never,
            start_headless=start_headless,
            start_replay=start_replay,
            replay_clock=make_clock,
        )

    def run(self, *, broadcast: Any = None, deps: Any = None, **over: Any) -> Session:
        opts = self.world.opts(mode="replay", **{"max_games": 3, **over})
        self.limit = opts.max_games or 3
        return asyncio.run(run_session(opts, CFG, deps or self.deps(), broadcast or FakeBroadcast()))


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(World(tmp_path, monkeypatch, cap_minutes=1.0), monkeypatch)


def test_replay_mode_shows_every_game_in_order_played(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}  # the second game is a no result: it is shown like the others
    rig = Rig(world, monkeypatch)
    session = rig.run()
    assert [r.opponent for r in session.rows] == ["Opus", "Astra", "Petra-3"]
    assert session.rows[1].result == "no_result"
    ids = [r.game_id for r in session.rows]
    assert rig.shown_starts == ids and rig.stopped == ids
    assert session.ended == "max_games"
    assert (rig.samples[-1]["phase"], rig.samples[-1]["shown"]) == ("intermission", 3)


def test_table_updates_only_when_replay_ends(rig: Rig) -> None:
    session = rig.run()
    assert len(session.rows) == 3
    assert rig.samples
    for s in rig.samples:
        assert s["shown"] == s["stopped"], s  # a result reaches the table when its picture has ended, not before
        assert s["started"] - s["stopped"] <= 2, s  # at most one game played ahead of the one on screen
    ahead = [s for s in rig.samples if s["phase"] == "playing" and s["finished"] > s["shown"] + 1]
    assert ahead, "the next game was never played while a replay was on screen"


def test_the_overlay_follows_the_replays_game_time_and_says_replay(rig: Rig) -> None:
    rig.run()
    playing = [s for s in rig.samples if s["phase"] == "playing"]
    assert playing and all(s["hold"] is None for s in playing)
    assert all(s["replay"] and s["replay"]["played_at"].endswith(" ET") for s in playing)
    assert any(s["clock"] not in ("00:00",) for s in playing)


def test_headless_games_are_unpaced_and_never_on_the_paced_engine(rig: Rig) -> None:
    rig.run()
    assert rig.paces == [None, None, None]
    assert rig.total == 3


def test_a_game_with_no_replay_is_still_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path, monkeypatch)
    rig = Rig(world, monkeypatch)
    base = rig.deps

    def deps() -> session_mod.SessionDeps:
        d = base()

        async def start_headless() -> Handle:
            rig.total += 1
            engine = await world.deps().start_engine()
            engine.replay = False
            return engine

        return replace(d, start_headless=start_headless)

    rig.deps = deps  # type: ignore[method-assign]
    session = rig.run()
    assert len(session.rows) == 3 and rig.shown_starts == []
    assert rig.samples[-1]["shown"] == 3


def test_sigterm_in_replay_mode_stops_the_game_in_play(rig: Rig) -> None:
    before = signal.getsignal(signal.SIGTERM)
    rig.world.on_call = lambda: os.kill(os.getpid(), signal.SIGTERM)
    session = rig.run()
    (row,) = session.rows
    assert (row.result, row.reason) == ("no_result", "stopped by hand") and session.ended == "stopped"
    assert signal.getsignal(signal.SIGTERM) == before
    assert [h.closed for h in rig.world.handles] == [1]


def test_refusals_are_checked_before_each_headless_game(rig: Rig) -> None:
    rig.world.budget = False
    session = rig.run()
    assert session.rows == [] and rig.world.handles == [] and session.ended == "budget"
    assert rig.world.budget_calls == [(9.0, pytest.approx(0.75 + (1800 + 5) / 3600))]


def test_the_pending_row_is_claimed_before_the_headless_engine_starts(rig: Rig) -> None:
    seen: list[list[str]] = []
    base = rig.deps

    def deps() -> session_mod.SessionDeps:
        d = base()
        start = d.start_headless

        async def watched() -> Any:
            saved = json.loads((rig.world.out / "session.json").read_text())["rows"]
            seen.append([r["result"] for r in saved])
            assert start is not None
            return await start()

        return replace(d, start_headless=watched)

    rig.deps = deps  # type: ignore[method-assign]
    rig.run()
    assert seen[0] == ["pending"] and seen[1][-1] == "pending"


def test_replay_mode_needs_its_engines(rig: Rig) -> None:
    deps = replace(rig.deps(), start_headless=None)
    with pytest.raises(ValueError, match="start_headless"):
        asyncio.run(run_session(rig.world.opts(mode="replay"), CFG, deps, None))


def test_the_intermission_names_the_next_game_and_the_last_one_ends_the_session(rig: Rig) -> None:
    rig.run()
    nexts = [s["next"] for s in rig.samples if s["phase"] == "intermission"]
    assert list(dict.fromkeys(nexts)) == ["Astra", "Petra-3", "end of session"]


def test_the_next_game_named_never_gives_away_a_game_played_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(tmp_path, monkeypatch, cap_minutes=3.0)
    world.failing = {"astra"}  # game 4 is Astra's second no result in a row: she is dropped, but not yet on screen
    rig = Rig(world, monkeypatch)
    session = rig.run(rotation=("opus", "astra"), max_games=4)
    assert [r.opponent for r in session.rows] == ["Opus", "Astra", "Opus", "Astra"]
    assert session.dropped() == {"Astra"}
    nexts = [k for k, _ in itertools.groupby(s["next"] for s in rig.samples if s["phase"] == "intermission")]
    assert nexts == ["Astra", "Opus", "Astra", "end of session"]


def test_a_failing_ledger_write_in_replay_mode_propagates_as_in_live_mode(rig: Rig) -> None:
    def boom(what: str, usd: float) -> None:
        raise OSError("ledger full")

    with pytest.raises(OSError, match="ledger full"):
        rig.run(deps=replace(rig.deps(), record_spend=boom))


def test_the_clock_starts_when_the_first_frame_was_seen_even_if_the_broadcast_starts_later(rig: Rig) -> None:
    events: list[str] = []
    clocks: list[HeldClock] = []

    class Broadcast(FakeBroadcast):
        async def start(self, url: str) -> None:
            for _ in range(5):  # the window, Chrome and ffmpeg take seconds; the engine's map loads meanwhile
                await asyncio.sleep(0.01)
            events.append("window")

    class Early(FakeReplay):
        async def first_frame(self, timeout_s: float) -> float | None:
            events.append("first_frame")
            return 101.0  # seen long before the broadcast was up

    def make_clock() -> HeldClock:
        clocks.append(HeldClock(lambda: False))
        return clocks[-1]

    base = rig.deps()

    async def start_replay(commands: Path) -> Early:
        return Early(commands, rig.stopped)

    deps = replace(base, start_replay=start_replay, replay_clock=make_clock)
    rig.run(deps=deps, broadcast=Broadcast(), max_games=1)
    assert events[:2] == ["first_frame", "window"]  # the watch runs while the broadcast starts
    assert clocks[0].at == 101.0


def test_the_first_frame_watch_is_cancelled_when_the_broadcast_fails_to_start(rig: Rig) -> None:
    cancelled: list[bool] = []

    class Broadcast(FakeBroadcast):
        async def start(self, url: str) -> None:
            await asyncio.sleep(0.01)
            raise OSError("no display")

    class Waiting(FakeReplay):
        async def first_frame(self, timeout_s: float) -> float | None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return None

    base = rig.deps()

    async def start_replay(commands: Path) -> Waiting:
        return Waiting(commands, rig.stopped)

    rig.run(deps=replace(base, start_replay=start_replay), broadcast=Broadcast(), max_games=1)
    assert cancelled


def test_a_missing_first_frame_starts_the_clock_anyway_with_slack(rig: Rig, caplog: pytest.LogCaptureFixture) -> None:
    class Never(FakeReplay):
        async def first_frame(self, timeout_s: float) -> float | None:
            return None

    reads: list[int] = []

    class Clock(HeldClock):
        def game_ms(self) -> int:
            ms = super().game_ms()
            reads.append(ms)
            return ms

    base = rig.deps()

    async def start_replay(commands: Path) -> Never:
        return Never(commands, rig.stopped)

    deps = replace(base, start_replay=start_replay, replay_clock=lambda: Clock(lambda: False))
    with caplog.at_level("WARNING"):
        rig.run(deps=deps, max_games=1)
    assert "no first frame" in caplog.text
    # a 60 s game: the end, the tail and the slack for the missing first frame
    assert max(reads) >= 60_000 + 2 * int(replay_mod.REPLAY_TAIL_S * 1000)


@pytest.mark.parametrize("how", ["start fails", "early exit", "no replay file"])
def test_three_failing_replays_end_the_session_as_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    world = World(tmp_path, monkeypatch)
    rig = Rig(world, monkeypatch)
    if how == "start fails":
        rig.render_error = OSError("no display")
    elif how == "early exit":
        rig.early_exit = True
    base = rig.deps

    def deps() -> session_mod.SessionDeps:
        d = base()
        if how != "no replay file":
            return d

        async def start_headless() -> Handle:
            engine = await world.deps().start_engine()
            engine.replay = False
            return engine

        return replace(d, start_headless=start_headless)

    rig.deps = deps  # type: ignore[method-assign]
    session = rig.run(max_games=8)
    assert session.ended == "render" and "3 replays in a row" in (session.ended_why or "")
    assert 3 <= len(session.rows) <= 4


def test_a_whole_replay_resets_the_render_failures(rig: Rig) -> None:
    fails = iter([True, True, False, True, True, False])
    base = rig.deps()

    async def start_replay(commands: Path) -> FakeReplay:
        if next(fails):
            raise OSError("no display")
        return FakeReplay(commands, rig.stopped)

    session = rig.run(deps=replace(base, start_replay=start_replay), max_games=6)
    assert session.ended == "max_games" and len(session.rows) == 6


def test_a_deadline_live_mode_admits_is_refused_in_replay_mode(rig: Rig) -> None:
    rig.world.left_s = 3000.0  # above the 2700 s margin, below it plus a game cap and an intermission
    session = rig.run()
    assert session.rows == [] and session.ended == "poweroff"
    assert rig.world.budget_calls == []


def test_the_budget_check_plans_the_replay_owed_too(rig: Rig) -> None:
    rig.run(max_games=1)
    assert rig.world.budget_calls == [(9.0, pytest.approx(0.75 + (1800 + 5) / 3600))]


def test_the_shown_rows_can_never_write_the_session_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _Shown(tmp_path / "session.json").save()
    assert not (tmp_path / "session.json").exists()
