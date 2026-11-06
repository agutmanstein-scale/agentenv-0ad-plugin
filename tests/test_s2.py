"""Box day 1 and Run 1 as one sequence: the gate rules, the throughput probe, the panels a config can ask for, the
orchestrator's stop-when-done, and the sequencer's chaining (a gate that passes starts the next step, one that fails
stops with its reason, a cap or a budget refusal stops too), all against fakes: no box, container, vLLM or network."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from agentenv_zeroad.llm import ToolSpec
from agentenv_zeroad.train import gates as G
from agentenv_zeroad.train import probe as P
from agentenv_zeroad.train import s2 as S
from agentenv_zeroad.train import stats
from agentenv_zeroad.train.calibration import panel_specs
from agentenv_zeroad.train.config import RunConfig, check_run
from agentenv_zeroad.train.distill import BriefSupply
from agentenv_zeroad.train.panels import frontier_specs, position, schedule
from tests.fake_models import FakeLiteLLM
from tests.test_train_orchestrate import HOSTS, config, drive, make
from tests.train_fakes import results, timeline, turns_every

ROOT = Path(__file__).resolve().parents[1]
LIVE_S2 = ROOT / "live" / "s2"


# ---- the gate rules -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "op", "threshold", "passed"),
    [(3.0, ">=", 3.0, True), (2.99, ">=", 3.0, False), (2.0, "<=", 2.0, True), (0.05, "<", 0.05, False),
     (0.049, "<", 0.05, True), (None, ">=", 0.0, False), (float("nan"), "<", 1.0, False)],
)  # fmt: skip
def test_a_gate_compares_its_value_with_its_threshold_and_a_missing_value_never_passes(value, op, threshold, passed):
    assert G.Gate("g", value, op, threshold).passed is passed


def test_throughput_is_the_best_per_gpu_rate_over_the_levels_and_the_knee_is_where_it_flattens():
    by_level = {16: {"a": 2.0, "b": 2.4}, 32: {"a": 3.0, "b": 3.4}, 48: {"a": 3.3, "b": 3.5}, 64: {"a": 3.0, "b": 3.2}}
    rates = G.per_gpu_rates(by_level)
    assert rates == pytest.approx({16: 2.2, 32: 3.2, 48: 3.4, 64: 3.1})
    assert G.knee(rates) == 32  # 3.2 is 94% of the 3.4 best; 16 is 65%
    gate = G.throughput_gate(rates)
    assert gate.passed and gate.value == pytest.approx(3.4) and "best at 48" in gate.note and "knee at 32" in gate.note
    assert not G.throughput_gate({16: 1.0, 32: 2.9}).passed and not G.throughput_gate({}).passed


def test_the_learner_and_agreement_gates_use_the_written_thresholds():
    assert G.learner_gate(1.9).passed and G.learner_gate(2.0).passed and not G.learner_gate(2.1).passed
    assert G.agreement_gate(0.049).passed and not G.agreement_gate(0.05).passed and not G.agreement_gate(None).passed


def test_mann_whitney_is_one_sided_in_the_direction_of_x_and_handles_ties_and_empty_arms():
    high, low = [5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 5]
    up, down = G.mann_whitney_greater(high, low), G.mann_whitney_greater(low, high)
    assert up["p"] < 0.01 < 0.99 < down["p"] and up["u"] == 35.0  # the 5 ties two 5s, half a win each
    assert G.mann_whitney_greater([3, 3, 3], [3, 3, 3])["p"] == 1.0
    assert G.mann_whitney_greater([], [1])["p"] is None


def test_the_positive_control_needs_a_higher_median_than_the_frozen_arm_a_floor_and_no_trip():
    policy, control = [8.0, 9.0, 10.0, 7.0] * 10, [2.0, 3.0, 4.0, 1.0] * 10
    gates, detail = G.positive_control_gates(policy, control, trips=0)
    assert G.all_passed(gates) and detail["medians"] == {"policy": 8.5, "control": 2.5}
    assert not G.all_passed(G.positive_control_gates(policy, control, trips=1)[0])  # any trip fails the control
    flat = G.positive_control_gates([2.0, 3.0] * 20, [2.0, 3.0] * 20, trips=0)[0]
    assert [g.passed for g in flat] == [True, False, False, True]  # it did not beat v0, and its median is under 6
    assert not G.positive_control_gates(policy[:5], control, trips=0)[0][0].passed  # too few games in an arm


def panel_game(opponent: str, margin: float, *, version: int = 0, i: int = 0):
    return stats.Game(version, 0.0, opponent, 1.0 if margin > 0 else 0.0, margin)


def test_run_one_starts_from_the_chosen_epoch_only_if_it_beats_v0_by_the_written_rule():
    opponents = ["petra-1", "petra-2", "petra-3"] * 20
    base = [panel_game(o, -100.0 + (i % 7) * 10) for i, o in enumerate(opponents)]
    better = [panel_game(o, 400.0 + (i % 7) * 10, version=2) for i, o in enumerate(opponents)]
    same = [panel_game(o, -100.0 + (i % 7) * 10, version=2) for i, o in enumerate(opponents)]
    chosen = {"chosen": "epoch2"}
    win = G.init_decision(chosen, base, better)
    assert win["init"] == "epoch2" and win["test"]["p"] < 0.05 and "beats v0" in win["reason"]
    tie = G.init_decision(chosen, base, same)
    assert tie["init"] is None and "does not beat v0" in tie["reason"]
    none = G.init_decision({"chosen": None}, base, [])
    assert none["init"] is None and "entropy floor" in none["reason"]
    nothing = G.init_decision(chosen, [], better)
    assert nothing["init"] is None and "could not be run" in nothing["reason"]


# ---- the throughput probe ---------------------------------------------------------------------------------------
PROMPTS = [
    P.PromptItem("system", f"brief {i}", [ToolSpec("hold", "d", {"type": "object", "properties": {}})])
    for i in range(3)
]


@pytest.mark.anyio
async def test_the_probe_holds_each_level_per_server_counts_rates_and_keeps_aligned_rows():
    fake = FakeLiteLLM(token_ids=True, delay_s=0.01)
    async with fake.http() as http:
        results_, rows = await P.run_probe(
            ["http://vllm0", "http://vllm1"], "k", PROMPTS, levels=(2, 4), window_s=0.3, keep=5, http=http
        )
    assert [(r.url, r.level) for r in results_] == [
        ("http://vllm0", 2), ("http://vllm1", 2), ("http://vllm0", 4), ("http://vllm1", 4)
    ]  # fmt: skip
    assert all(r.completed > 0 and r.failed == 0 for r in results_)
    assert len(rows) == 5 and rows[0]["prompt_token_ids"] == [1, 2, 3] and rows[0]["token_ids"] == [7, 8]
    assert len(rows[0]["logprobs"]) == len(rows[0]["token_ids"]) and len({r["game_id"] for r in rows}) == 5
    summary = P.summarise(results_)
    assert set(summary["per_gpu"]) == {"2", "4"} and summary["knee"] in (2, 4)
    assert all(s["decisions_per_s"] > 0 for s in summary["servers"])


@pytest.mark.anyio
async def test_a_probe_with_failing_calls_counts_them_and_keeps_no_rows(tmp_path):
    fake = FakeLiteLLM(token_ids=True, fail=500)
    async with fake.http() as http:
        results_, rows = await P.run_probe(["http://vllm0"], "k", PROMPTS, levels=(2,), window_s=0.1, keep=5, http=http)
    assert results_[0].completed == 0 and results_[0].failed > 0 and rows == []
    summary = P.write_outputs(tmp_path / "e1", results_, rows)
    assert summary["rows_kept"] == 0 and (tmp_path / "e1" / "throughput.json").is_file()
    assert (tmp_path / "e1" / "probe-rows.jsonl").read_text() == ""


# ---- the panels ------------------------------------------------------------------------------------------------
def cfg_for(tmp_path: Path, panels: list[dict], **over) -> RunConfig:
    held = {"civ_pairs": [["rome", "pers"]], "maps": [], "seed_ranges": [[900000, 900500]]}
    return config(tmp_path / "run", panels=panels, held_out=held, games={"score_rule": "net_worth"}, **over)


def test_the_opponent_schedule_is_balanced_in_every_prefix_and_its_positions_do_not_depend_on_the_panels_size(tmp_path):
    mix = {"petra-1": 0.3125, "petra-2": 0.3125, "petra-3": 0.25, "bot": 0.0625, "base": 0.0625}
    full = schedule(mix, 384)
    assert Counter(full) == {"petra-1": 120, "petra-2": 120, "petra-3": 96, "bot": 24, "base": 24}
    assert schedule(mix, 192) == full[:192]
    for n in (16, 48, 100):
        for opponent, share in mix.items():
            assert abs(Counter(full[:n])[opponent] - share * n) <= 1
    cfg = cfg_for(tmp_path, [{"name": "a", "kind": "heldout", "at_hours": 0, "games": 384}])
    assert [position(cfg, "held_out", j) for j in range(5)] == [position(cfg, "held_out", j) for j in range(5)]
    assert position(cfg, "held_out", 3) != position(cfg, "held_out", 4)


def test_a_panel_with_a_start_plays_the_later_games_of_the_same_sequence_so_two_panels_make_one_larger(tmp_path):
    from agentenv_zeroad.train.panels import heldout_specs

    def specs(**kw):
        panel = {"name": "p", "kind": "heldout", "at_hours": 0, **kw}
        cfg = cfg_for(tmp_path, [panel])
        return heldout_specs(cfg, cfg.panels[0])

    whole, first, rest = specs(games=384), specs(games=192), specs(games=192, start=192)
    key = lambda g: (g.game_id.split("-", 2)[1:], g.seed, g.ai_seed, g.map, [s.kind for s in g.seats])  # noqa: E731
    assert [key(g) for g in first + rest] == [key(g) for g in whole]
    assert rest[0].game_id.startswith("p-0192-") and rest[-1].game_id.startswith("p-0383-")
    assert Counter(g.game_id.split("-", 2)[2] for g in first + rest) == Counter(
        g.game_id.split("-", 2)[2] for g in whole
    )


@pytest.mark.parametrize(
    ("panel", "message"),
    [({"kind": "heldout", "games": 4, "start": -1}, "start is not negative"),
     ({"kind": "frontier", "teachers": [{"name": "a", "model": "m", "games": 1}], "start": 4}, "is for held-out")],
)  # fmt: skip
def test_a_panel_start_is_for_heldout_panels_and_not_negative(tmp_path, panel, message):
    with pytest.raises(ValueError, match=message):
        cfg_for(tmp_path, [{"name": "p", "at_hours": 0, **panel}])


def test_a_heldout_panel_uses_only_held_out_pairs_and_seeds_and_pins_its_policy_seat(tmp_path):
    cfg = cfg_for(tmp_path, [{"name": "heldout", "kind": "heldout", "at_hours": 0, "games": 384}])
    specs = panel_specs(cfg, cfg.panels[0])
    assert len(specs) == 384 and len({s.game_id for s in specs}) == 384
    for spec in specs:
        assert 900000 <= spec.seed < 900500 and spec.map == "mainland"
        policy = [s for s in spec.seats if s.trainable]
        assert len(policy) == 1 and policy[0].model == "local/qwen-base" and policy[0].interface == "v2"
        assert {s.civ for s in spec.seats} == {"rome", "pers"} and spec.time_cap_minutes == 30
    sides = Counter(next(i for i, s in enumerate(spec.seats) if s.trainable) for spec in specs)
    assert sides == {0: 192, 1: 192}
    assert {s.seats[1 - next(i for i, x in enumerate(s.seats) if x.trainable)].kind for s in specs} == {
        "petra",
        "bot",
        "llm",
    }  # the non-Petra slice is the scripted bot and the frozen base model


def test_the_v1_subset_plays_the_same_positions_on_the_old_interface_with_its_own_score_rule(tmp_path):
    panels = [
        {"name": "heldout", "kind": "heldout", "at_hours": 0, "games": 48},
        {"name": "v1", "kind": "heldout", "at_hours": 0, "games": 48, "interface": "v1",
         "opponents": {"petra-1": 1, "petra-2": 1, "petra-3": 1}},
    ]  # fmt: skip
    cfg = cfg_for(tmp_path, panels)
    v2, v1 = panel_specs(cfg, cfg.panels[0]), panel_specs(cfg, cfg.panels[1])
    assert [(s.map, s.seed, s.ai_seed) for s in v1] == [(s.map, s.seed, s.ai_seed) for s in v2]
    assert all(s.score_rule is None and next(x for x in s.seats if x.trainable).interface == "v1" for s in v1)
    assert all(s.score_rule == "net_worth" for s in v2)
    assert Counter(next(x for x in s.seats if not x.trainable).difficulty for s in v1) == {1: 16, 2: 16, 3: 16}


def test_a_frontier_panel_gives_each_teacher_its_games_on_shared_positions_at_the_same_latency(tmp_path):
    teachers = [
        {"name": "opus", "model": "anthropic/claude-opus-5-5", "games": 8},
        {"name": "sonnet", "model": "anthropic/claude-sonnet-5-5", "games": 8},
        {"name": "astra", "model": "openai/gpt-6-astra", "games": 4},
    ]
    cfg = cfg_for(tmp_path, [{"name": "frontier", "kind": "frontier", "at_hours": 0, "teachers": teachers}])
    specs = frontier_specs(cfg, cfg.panels[0])
    by_teacher = Counter(next(s.name for s in spec.seats if not s.trainable) for spec in specs)
    assert by_teacher == {"opus": 8, "sonnet": 8, "astra": 4} and len({s.game_id for s in specs}) == 20
    opus = [s for s in specs if "opus" in s.game_id]
    sonnet = [s for s in specs if "sonnet" in s.game_id]
    assert [(s.seed, s.map) for s in opus] == [(s.seed, s.map) for s in sonnet]
    assert {s.latency_ms for s in specs} == {cfg.games.latency_ms}
    assert all(next(x for x in s.seats if x.trainable).model == "local/qwen-base" for s in specs)
    assert check_run(cfg) == ["LITELLM_API_KEY"]  # the box runs it under with-model.sh


@pytest.mark.parametrize(
    ("panel", "message"),
    [({"kind": "heldout", "games": 0}, "games is at least 1"),
     ({"kind": "heldout", "games": 4, "opponents": {"petra-9": 1}}, "opponents are weights"),
     ({"kind": "heldout", "games": 4, "positions": "nowhere"}, "positions must be"),
     ({"kind": "frontier", "teachers": [{"name": "o", "model": "m"}]}, "each with games"),
     ({"kind": "frontier", "teachers": []}, "needs teachers")],
)  # fmt: skip
def test_a_malformed_generated_panel_is_refused(tmp_path, panel, message):
    with pytest.raises(ValueError, match=message):
        cfg_for(tmp_path, [{"name": "p", "at_hours": 0, **panel}])


def test_held_out_seed_ranges_keep_every_range_seed_out_of_training(tmp_path):
    cfg = cfg_for(tmp_path, [{"name": "p", "kind": "heldout", "at_hours": 0, "games": 4}], seed=3)
    assert cfg.held_out.is_held_seed(900123) and not cfg.held_out.is_held_seed(899999)
    from agentenv_zeroad.train.mix import GameFactory

    factory = GameFactory(cfg)
    assert not any(cfg.held_out.is_held_seed(s) for _ in range(300) for s in [factory.next(0, HOSTS[0]).seed])
    with pytest.raises(ValueError, match="empty"):
        config(tmp_path / "x", held_out={"seed_ranges": [[5, 5]]})


# ---- the orchestrator's end condition ------------------------------------------------------------------------------
def briefs_results(spec_dict: dict) -> dict:
    """A finished v2 game the policy lost to Petra-2: every logged turn is a usable (losing) brief."""
    out = results(
        spec_dict["game_id"], outcomes=("lost", "won"), net_worth=(1000.0, 5000.0),
        turns={"1": turns_every(25, 10, model="local/qwen-base")},
        opponent={"kind": "petra", "civ": "pers", "difficulty": 2, "behaviour": "balanced"},
        model_1="local/qwen-base",
    )  # fmt: skip
    out["seats"]["1"]["reference"] = "cost tables"
    out["spec"]["game_id"] = spec_dict["game_id"]
    return out


def install_results(t) -> None:
    async def play(spec, out_dir, model_for):
        from agentenv_zeroad.train.runner import GameOutcome

        folder = out_dir / spec.game_id
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(briefs_results(spec.to_json())))
        return GameOutcome(spec.game_id, "finished", "time_cap", {"1": "lost", "2": "won"}, 600.0, 7.0)

    t.orch.play = play


def ticking(t) -> bool:
    """The run's clock moves a second each time it is asked, so every interval of the loop comes due."""
    t.clock.t += 1
    return t.orch.stop_reason is not None


def heldout_panel(games: int) -> dict:
    return {"name": "h", "kind": "heldout", "at_hours": 0, "games": games}


@pytest.mark.anyio
async def test_a_panels_only_run_starts_no_training_game_and_completes_when_its_panels_are_played(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, learner=False, concurrency=2, panels=[heldout_panel(4)], panels_only=True,
             stop_when_done=True, held_out={"seed_ranges": [[900000, 900500]]})  # fmt: skip
    install_results(t)
    await t.orch.refresh_serving()
    await drive(t.orch, lambda: ticking(t))
    assert t.orch.stop_reason == "complete"
    assert t.orch.counts["panel_finished"] == 4 and t.orch.counts["train_finished"] == 0
    assert not list((t.root / "games").glob("*")) and (t.root / "panels" / "h" / "specs.jsonl").is_file()
    summary = json.loads((t.root / "panels" / "h" / "summary.json").read_text())
    assert summary["games"] == 4 and summary["finished"] == 4


@pytest.mark.anyio
async def test_a_baseline_run_keeps_playing_until_the_games_hold_the_briefs_it_was_asked_for(tmp_path, monkeypatch):
    # one 10-minute game holds 24 usable turns, 8 of which the per-game cap lets through: 30 briefs need 4 games
    t = make(tmp_path, monkeypatch, learner=False, concurrency=1, panels=[heldout_panel(2)], stop_when_done=True,
             brief_target=30, brief_per_game=8, held_out={"seed_ranges": [[900000, 900500]]})  # fmt: skip
    install_results(t)
    await t.orch.refresh_serving()
    await drive(t.orch, lambda: ticking(t))
    assert t.orch.stop_reason == "complete"
    assert t.orch.counts["panel_finished"] == 2 and t.orch.counts["train_finished"] >= 4
    assert t.orch.supply.capacity()["selectable"] >= 30


def test_the_brief_supply_follows_the_selection_rule_under_the_per_game_cap():
    from agentenv_zeroad.train.distill import Brief

    def brief(g, stratum):
        return Brief(g, "1", 0, "u", (1,), stratum)

    supply = BriefSupply(10, {"losing": 0.6, "late": 0.2, "rejected": 0.2}, per_game=3)
    for g in "abc":
        supply.add(g, [brief(g, "losing")] * 6)
    cap = supply.capacity()
    assert cap["selectable"] == 9 and cap["candidates"] == 18 and cap["strata"]["losing"] == 9  # 3 a game, all losing
    assert cap["shares_met"] is False  # no late or rejected turns to fill their 20% each
    supply.add("d", [brief("d", "late")] * 4 + [brief("d", "rejected")] * 4)
    assert supply.capacity()["selectable"] == 10 and supply.capacity()["shares_met"] is False  # d's cap is 3 a game


# ---- the sequencer -----------------------------------------------------------------------------------------------
def make_config_dir(tmp_path: Path) -> Path:
    """live/s2 with its roots moved under tmp_path/train so that nothing needs /var/lib."""
    out = tmp_path / "cfg"
    shutil.copytree(LIVE_S2, out)
    for path in out.glob("*.toml"):
        text = path.read_text().replace("/var/lib/zeroad-live/train/", f"{tmp_path}/train/")
        if path.name != "sequence.toml":
            text = re.sub(r"^(run_id = .*)$", rf'\1\ntrain_root = "{tmp_path}/train"', text, count=1, flags=re.M)
        path.write_text(text)
    return out


class FakeRunner:
    """Records every command; `effects` maps a substring of the command to what it does (and its exit code)."""

    def __init__(self, effects=None):
        self.calls: list[list[str]] = []
        self.effects = effects or {}
        self.timeouts: list[float] = []

    def run(self, argv, *, timeout_s, label, grace_s=0):
        self.calls.append(list(argv))
        self.timeouts.append(timeout_s)
        line = " ".join(argv)
        for needle, effect in self.effects.items():
            if needle in line:
                return effect(argv) if callable(effect) else effect
        return 0

    def lines(self) -> list[str]:
        return [" ".join(c) for c in self.calls]


def make_ctx(tmp_path, runner, *, refuse=None, clock=None, **env):
    cdir = make_config_dir(tmp_path)
    settings = S.load_settings(cdir / "sequence.toml")
    t = clock or [1_800_000_000.0]
    now = lambda: t[0]  # noqa: E731
    ctx = S.Ctx(settings, S.ConfigDir(cdir, settings), tmp_path / "box", runner, 350.0, S.Record(settings.root, now),
                now=now, python="python", env={"LITELLM_BASE_URL": "https://litellm.example", **env})  # fmt: skip
    ctx.budget_ok = lambda hours, usd=0.0: refuse(hours, usd) if refuse else None  # type: ignore[method-assign]
    return ctx, t


def step_stubs(monkeypatch, verdicts: dict[str, bool], order: list[str]):
    """Replace every step by one that notes it ran and passes or fails as told."""

    def stub(name):
        def run(ctx):
            order.append(name)
            ok = verdicts.get(name, True)
            return S.Outcome([G.Gate(f"{name} gate", 1.0 if ok else 0.0, ">=", 1.0)], {"ran": name})

        return run

    for name in S.STEPS:
        monkeypatch.setitem(S.STEP_FUNCS, name, stub(name))


def test_the_example_config_directory_passes_its_own_check_and_prices_the_day():
    numbers = S.check_dir(LIVE_S2, 450)
    assert numbers["frontier"] is True and numbers["rate"] == 10.49 and numbers["run1_hours"] == 10
    assert 19 <= numbers["expected_hours"] <= 20 and numbers["cap_hours"] > numbers["expected_hours"]
    cost = (
        numbers["expected_hours"] * numbers["rate"] + 46 + 15 + 63
    )  # box + two frontier panels + calibration + labels
    assert cost < 450


@pytest.mark.parametrize(
    ("file", "old", "new", "message"),
    [("e2.toml", "positive_control = true", "positive_control = false", "positive_control must be true"),
     ("run1.toml", "shutdown = false", "shutdown = true", "shutdown must be false"),
     ("run1.toml", "box_usd_per_hour = 10.49", "box_usd_per_hour = 15.07", "different box rates"),
     ("e1.toml", "budget_usd = 450", "budget_usd = 500", "above the stage budget"),
     ("run1.toml", "duration_hours = 10", "duration_hours = 12", "past its cap_hours"),
     ("baselines.toml", 'uri = "s3://BUCKET/zeroad-train/run/s2-baselines"', "", "s3.uri is required"),
     ("sft-panels.toml", 's2/sft"', 's2/other"', "share one root"),
     ("sequence.toml", 'heldout_panel = "heldout"\n', 'heldout_panel = "nope"\n', "has no such panel"),
     ("sequence.toml", "[e1]  ", "[e1]\nbogus = 1  ", "unknown \\[e1\\] keys"),
     ("sequence.toml", "heldout_games = 384", "heldout_games = 192", "panel_games must be under heldout_games"),
     ("sequence.toml", "heldout_games = 384", "heldout_games = 300", "is not the baselines' held-out panel's 384")],
)  # fmt: skip
def test_a_config_directory_that_would_break_the_sequence_is_refused(tmp_path, file, old, new, message):
    cdir = make_config_dir(tmp_path)
    path = cdir / file
    assert old in path.read_text(), old
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        S.check_dir(cdir, 450)


def test_every_step_is_recorded_with_its_start_and_end_and_the_next_starts_only_if_the_gate_passed(
    tmp_path, monkeypatch
):
    order: list[str] = []
    step_stubs(monkeypatch, {}, order)
    ctx, clock = make_ctx(tmp_path, FakeRunner())
    assert S.sequence(ctx) == ("complete", None)
    assert order == list(S.STEPS)
    steps = json.loads((ctx.s2 / "gates.json").read_text())["steps"]
    assert list(steps) == list(S.STEPS) and all(
        v["status"] == "passed" and v["ended"] is not None for v in steps.values()
    )
    assert steps["e1"]["gates"][0] == {
        "name": "e1 gate",
        "value": 1.0,
        "op": ">=",
        "threshold": 1.0,
        "note": "",
        "passed": True,
    }
    assert "e1: start (cap 0.75 h)" in (ctx.s2 / "progress.log").read_text()


def test_the_baselines_come_before_e2_so_e2_can_be_given_the_tripwire_baseline():
    steps = list(S.STEPS)
    assert steps.index("baselines") < steps.index("tripwire") < steps.index("e2") < steps.index("run1")
    assert steps.index("e1") < steps.index("baselines")


@pytest.mark.parametrize(
    "failing", ["e1", "baselines", "calibration", "tripwire", "e2", "labels", "sft", "sft_panels", "run1"]
)
def test_a_failed_gate_stops_the_sequence_with_its_reason_and_starts_nothing_after_it(tmp_path, monkeypatch, failing):
    order: list[str] = []
    step_stubs(monkeypatch, {failing: False}, order)
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    verdict, reason = S.sequence(ctx)
    assert verdict == "stopped" and reason.startswith(f"{failing}: FAIL {failing} gate")
    assert order == list(S.STEPS)[: list(S.STEPS).index(failing) + 1]
    ctx.record.finish(verdict, reason)
    data = json.loads((ctx.s2 / "gates.json").read_text())
    assert data["verdict"] == "stopped" and data["steps"][failing]["status"] == "failed"
    assert json.loads((ctx.s2 / "status.json").read_text())["reason"] == reason


def test_a_step_that_crashes_or_raises_a_failure_stops_with_the_exception_as_its_reason(tmp_path, monkeypatch):
    order: list[str] = []
    step_stubs(monkeypatch, {}, order)

    def boom(ctx):
        raise RuntimeError("the pool is gone")

    monkeypatch.setitem(S.STEP_FUNCS, "e2", boom)
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    verdict, reason = S.sequence(ctx)
    assert (
        verdict == "stopped"
        and "RuntimeError: the pool is gone" in reason
        and order == ["e1", "baselines", "calibration", "tripwire"]
    )

    def capped(ctx):
        raise S.StepFailed("the step's cap ran out")

    monkeypatch.setitem(S.STEP_FUNCS, "e2", capped)
    ctx, _ = make_ctx(tmp_path / "b", FakeRunner())
    assert S.sequence(ctx)[1] == "e2: the step's cap ran out"


def test_the_budget_guard_is_asked_before_every_step_with_its_hours_and_its_frontier_dollars(tmp_path, monkeypatch):
    order: list[str] = []
    step_stubs(monkeypatch, {}, order)
    asked: list[tuple[float, float]] = []

    def refuse(hours, usd):
        asked.append((hours, usd))

    ctx, _ = make_ctx(tmp_path, FakeRunner(), refuse=refuse)
    S.sequence(ctx)
    assert asked == [
        (0.75, 0),
        (3.5, 45.0),
        (0.1, 0),
        (0.5, 0),
        (2.6, 0),
        (1.0, 68.0),
        (1.25, 0),
        (3.0, 0),
        (0.1, 0),
        (11.0, 30.0),
    ]


def test_a_budget_refusal_stops_the_sequence_before_the_step_with_the_guards_words(tmp_path, monkeypatch):
    order: list[str] = []
    step_stubs(monkeypatch, {}, order)
    ctx, _ = make_ctx(
        tmp_path, FakeRunner(), refuse=lambda h, u: "REFUSED: $340.00 would pass the $350 budget" if h > 5 else None
    )
    verdict, reason = S.sequence(ctx)
    assert verdict == "stopped" and order == list(S.STEPS)[:-1]
    assert reason == "run1: the budget guard refuses it: REFUSED: $340.00 would pass the $350 budget"


def test_a_resumed_sequence_skips_the_steps_that_passed_and_retries_the_one_that_failed(tmp_path, monkeypatch):
    order: list[str] = []
    step_stubs(monkeypatch, {"sft": False}, order)
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    assert S.sequence(ctx)[0] == "stopped" and order[-1] == "sft"
    order.clear()
    step_stubs(monkeypatch, {}, order)
    again = S.Ctx(
        ctx.settings, ctx.configs, ctx.box_home, ctx.runner, 350.0, S.Record(ctx.s2, ctx.now), now=ctx.now, env=ctx.env
    )
    again.budget_ok = lambda hours, usd=0.0: None  # type: ignore[method-assign]
    assert S.sequence(again) == ("complete", None)
    assert order == ["sft", "sft_panels", "init", "run1"]


def test_a_step_cap_is_enforced_as_a_timeout_on_each_command_inside_it(tmp_path):
    runner = FakeRunner({"smoke": 0, "probe": 124})
    ctx, clock = make_ctx(tmp_path, runner)
    ctx.step, ctx.step_started, ctx.step_cap_h = "e1", clock[0], 0.75
    assert ctx.run(["x"], "first") == 0 and runner.timeouts == [0.75 * 3600]
    clock[0] += 0.5 * 3600
    ctx.run(["y"], "second")
    assert runner.timeouts[-1] == pytest.approx(0.25 * 3600)
    clock[0] += 0.3 * 3600
    with pytest.raises(S.StepFailed, match="cap ran out before third"):
        ctx.run(["z"], "third")
    ctx.step_started = clock[0]
    runner.effects = {"stuck": 124}
    with pytest.raises(S.StepFailed, match="still running at the step's 0.75 h cap"):
        ctx.run(["stuck"], "long")


# ---- the steps, one by one ---------------------------------------------------------------------------------------
def e1_effects(ctx, *, rate: float, rows: int = 48, gap: float = 0.02, seconds: float = 1.2, smoke_rc=0):
    root = ctx.cfg("e1").root

    def probe(argv):
        root.mkdir(parents=True, exist_ok=True)
        (root / "throughput.json").write_text(
            json.dumps({"per_gpu": {"16": rate / 2, "32": rate}, "knee": 32, "rows_kept": rows})
        )
        return 0

    def learner(argv):
        (root / "probe.json").write_text(json.dumps({
            "agreement": {"mean_abs_gap": gap}, "timing": {"seconds_per_decision": seconds},
        }))  # fmt: skip
        return 0

    return {"train.smoke": smoke_rc, "train.probe": probe, "train-run.sh": learner}


def run_step(ctx, name):
    ctx.step, ctx.step_started, ctx.step_cap_h = name, ctx.now(), S.step_hours(ctx.settings, name)
    return S.STEP_FUNCS[name](ctx)


def test_e1_passes_on_good_numbers_and_runs_smoke_per_server_then_the_probe_then_the_learner(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    runner.effects = e1_effects(ctx, rate=3.6)
    out = run_step(ctx, "e1")
    assert out.passed and [g.name for g in out.gates][0].startswith("log-prob alignment")
    lines = runner.lines()
    assert sum("train.smoke" in line for line in lines) == len(ctx.cfg("e1").servers.urls) == 2
    assert any("train.probe" in line and "--levels 16,32,48,64" in line and "--window-s 60" in line for line in lines)
    learner = [line for line in lines if "train-run.sh" in line]
    assert len(learner) == 1 and "e1.toml --probe" in learner[0] and learner[0].endswith("probe-rows.jsonl")
    assert out.detail["learner"]["timing"]["seconds_per_decision"] == 1.2


@pytest.mark.parametrize(
    ("override", "failed_gate"),
    [
        ({"rate": 2.5}, "decisions per second per GPU"),
        ({"rate": 3.6, "seconds": 2.5}, "learner seconds per trained decision"),
        ({"rate": 3.6, "gap": 0.2}, "log-prob gap"),
    ],
)
def test_e1_fails_on_each_gate_that_misses_and_names_it(tmp_path, override, failed_gate):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    runner.effects = e1_effects(ctx, **override)
    out = run_step(ctx, "e1")
    assert not out.passed and failed_gate in out.why()


def test_e1_stops_at_a_misaligned_server_and_when_the_probe_kept_too_few_decisions(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    runner.effects = e1_effects(ctx, rate=4, smoke_rc=1)
    out = run_step(ctx, "e1")
    assert not out.passed and not any("train.probe" in line for line in runner.lines())
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path / "b", runner)
    runner.effects = e1_effects(ctx, rate=4, rows=10)
    out = run_step(ctx, "e1")
    assert "kept 10 sampled decisions" in out.why() and not any("train-run.sh" in line for line in runner.lines())


def game_with_army(folder: Path, game_id: str, army: float, version: int, mtime: int):
    model = "local/qwen-base" if version == 0 else f"local/policy-v{version}"
    out = results(game_id, model_1=model, turns={"1": [{"began_ms": 0, "model": model}]},
                  rows=timeline(12, army=(army, 5)))  # fmt: skip
    path = folder / game_id / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(out))
    import os

    os.utime(path, (mtime, mtime))


def e2_world(ctx, *, policy_army: float, control_army: float, trips: int = 0, reason: str = "duration", n: int = 40):
    cfg = ctx.cfg("e2")

    def run(argv):
        for i in range(n):
            game_with_army(cfg.root / "games", f"p{i}", policy_army + (i % 5) * 0.3, 3, 1000 + i)
            game_with_army(cfg.root / "panels" / "control-v0", f"c{i}", control_army + (i % 5) * 0.3, 0, 1000 + i)
        (cfg.root / "orchestrator").mkdir(parents=True, exist_ok=True)
        (cfg.root / "orchestrator" / "final.json").write_text(json.dumps({"reason": reason}))
        records = [
            {"version": v, "tripwires": {"trips": [{"message": "x"}] if v < trips else []}} for v in range(1, 12)
        ]
        (cfg.root / "health.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        return 0

    return {"train-run.sh": run}


def install_baseline(ctx, records: int = 12) -> Path:
    """What the tripwire step leaves: ROOT/s2/baseline.json and the records it was made from in the baselines root."""
    ctx.s2.mkdir(parents=True, exist_ok=True)
    (ctx.s2 / "baseline.json").write_text(json.dumps({"tokens_p50": {"mean": 50.0, "sd": 2.0}}))
    root = ctx.cfg("baselines").root
    root.mkdir(parents=True, exist_ok=True)
    (root / "baseline-records.jsonl").write_text("".join(json.dumps({"entropy": 1.0}) + "\n" for _ in range(records)))
    return ctx.s2 / "baseline.json"


def test_e2_and_run_one_refuse_to_start_without_a_tripwire_baseline_of_enough_records(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    with pytest.raises(S.StepFailed, match="no tripwire baseline.*is missing with 0 of 10 records"):
        run_step(ctx, "e2")
    install_baseline(ctx, records=9)
    with pytest.raises(S.StepFailed, match="exists with 9 of 10 records"):
        run_step(ctx, "run1")
    assert runner.calls == []  # nothing was started


def test_e2_passes_when_the_trained_policy_out_armies_the_frozen_arm_and_fails_on_a_flat_curve_or_a_trip(tmp_path):
    ok = FakeRunner()
    ctx, _ = make_ctx(tmp_path, ok)
    ok.effects = e2_world(ctx, policy_army=9, control_army=3)
    install_baseline(ctx)
    out = run_step(ctx, "e2")
    assert out.passed, out.why()
    assert out.detail["medians"]["policy"] > 8 and out.detail["n_policy"] == 40 == out.detail["n_control"]
    assert len(ok.calls) == 1 and "e2.toml" in ok.calls[0][2]
    assert ok.calls[0][3:] == [
        "--baseline",
        str(ctx.s2 / "baseline.json"),
    ]  # an RL run: a learner with the tripwire baseline
    flat = FakeRunner()
    ctx, _ = make_ctx(tmp_path / "flat", flat)
    install_baseline(ctx)
    flat.effects = e2_world(ctx, policy_army=3, control_army=3)
    out = run_step(ctx, "e2")
    assert not out.passed and "Mann-Whitney" in out.why() and "median army" in out.why()
    tripped = FakeRunner()
    ctx, _ = make_ctx(tmp_path / "trip", tripped)
    install_baseline(ctx)
    tripped.effects = e2_world(ctx, policy_army=9, control_army=3, trips=2)
    assert "tripwire trips" in run_step(ctx, "e2").why()
    crashed = FakeRunner()
    ctx, _ = make_ctx(tmp_path / "crash", crashed)
    install_baseline(ctx)
    crashed.effects = e2_world(ctx, policy_army=9, control_army=3, reason="tripwire: grad norm")
    assert "the run ended normally" in run_step(ctx, "e2").why()


def test_e2_counts_only_trained_games_for_the_policy_and_only_v0s_for_the_control(tmp_path):
    folder = tmp_path / "games"
    game_with_army(folder, "a", 9, 3, 1)
    game_with_army(folder, "b", 2, 0, 2)
    trained, base = S.armies(folder, 10, trained_only=True), S.armies(folder, 10, trained_only=False)
    assert len(trained) == 1 and len(base) == 1 and trained[0] > base[0]
    assert S.armies(folder, 10, trained_only=True, last=0) == trained


def test_baselines_pass_when_every_panel_finished_and_the_games_hold_the_briefs_and_fail_otherwise(
    tmp_path, monkeypatch
):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    cfg = ctx.cfg("baselines")

    def play(argv, finished=None, reason="complete"):
        for panel in cfg.panels:
            folder = cfg.root / "panels" / panel.name
            folder.mkdir(parents=True, exist_ok=True)
            n = {"teacher-calibration": 16, "frontier": 20, "heldout": 384, "heldout-v1": 48}[panel.name]
            (folder / "summary.json").write_text(
                json.dumps({"games": n, "finished": n if finished is None else finished.get(panel.name, n)})
            )
        (cfg.root / "orchestrator").mkdir(parents=True, exist_ok=True)
        (cfg.root / "orchestrator" / "final.json").write_text(json.dumps({"reason": reason}))
        return 0

    monkeypatch.setattr(S, "brief_supply", lambda *a: {"selectable": 3000, "candidates": 9000, "shares_met": True})
    runner.effects = {"train-run.sh": play}
    out = run_step(ctx, "baselines")
    assert out.passed and runner.calls[0][-1] == "--play-only" and "baselines.toml" in runner.calls[0][2]
    runner.effects = {"train-run.sh": lambda argv: play(argv, finished={"frontier": 12})}
    out = run_step(ctx, "baselines")
    assert not out.passed and "panel frontier" in out.why()
    monkeypatch.setattr(S, "brief_supply", lambda *a: {"selectable": 1200, "candidates": 1200, "shares_met": False})
    runner.effects = {"train-run.sh": play}
    assert "distillation briefs available" in run_step(ctx, "baselines").why()


def baseline_games(ctx, n: int) -> Path:
    """n finished ordinary v0 games in the baselines root, as the baseline run leaves them."""
    games = ctx.cfg("baselines").root / "games"
    for i in range(n):
        folder = games / f"g{i:04d}"
        folder.mkdir(parents=True)
        out = briefs_results({"game_id": f"g{i:04d}"})
        (folder / "results.json").write_text(json.dumps(out))
    return games


def learner_writes_baseline(ctx, records: int):
    root = ctx.cfg("baselines").root

    def run(argv):
        (root / "baseline-records.jsonl").write_text(
            "".join(json.dumps({"entropy": 1.0}) + "\n" for _ in range(records))
        )
        (root / "baseline.json").write_text(json.dumps({"entropy": {"mean": 1.0, "sd": 0.1}}))
        return 0

    return {"train-run.sh": run}


def test_the_tripwire_step_credits_v0s_ordinary_games_runs_the_stats_only_learner_and_keeps_its_baseline(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    ctx.settings.tables["tripwire"]["min_records"] = 1  # 30 games of about 10 decisions hold one batch of 256
    baseline_games(ctx, 30)
    rows_file = ctx.cfg("baselines").root / "baseline-rows.jsonl"
    runner.effects = learner_writes_baseline(ctx, records=12)
    out = run_step(ctx, "tripwire")
    assert out.passed, out.why()
    [call] = runner.calls
    assert call[2].endswith("baselines.toml") and call[3:] == ["--stats-only", str(rows_file)]
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
    assert len(rows) >= 256 and {r["version"] for r in rows} == {0} and out.detail["rows"]["rows"] == len(rows)
    assert json.loads((ctx.s2 / "baseline.json").read_text()) == {"entropy": {"mean": 1.0, "sd": 0.1}}
    assert out.detail["records"] == 12


def test_the_tripwire_step_stops_when_v0_has_too_few_decisions_or_the_learner_makes_too_few_records(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    baseline_games(ctx, 30)  # about 300 decisions: not the 2,560 that ten records need
    out = run_step(ctx, "tripwire")
    assert not out.passed and "v0 decisions for the tripwire baseline" in out.why() and runner.calls == []
    assert not (ctx.s2 / "baseline.json").exists()


def test_the_tripwire_step_fails_a_learner_that_wrote_fewer_records_than_the_minimum(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    ctx.settings.tables["tripwire"]["min_records"] = 1
    baseline_games(ctx, 30)
    runner.effects = learner_writes_baseline(ctx, records=0)
    out = run_step(ctx, "tripwire")
    assert not out.passed and "health records of v0" in out.why()


def test_the_calibration_picks_the_teacher_with_the_best_win_rate_against_petra_3_and_fails_with_none(tmp_path):
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    folder = ctx.cfg("baselines").root / "panels" / "teacher-calibration"
    for i, (teacher, outcome) in enumerate([("astra", "won"), ("astra", "lost"), ("opus", "won"), ("opus", "won")]):
        lost = "lost" if outcome == "won" else "won"
        out = results(f"{teacher}-{i}", outcomes=(outcome, lost), trainable=(False, False),
                      opponent={"kind": "petra", "civ": "pers", "difficulty": 3})  # fmt: skip
        out["spec"]["seats"][0] = {"kind": "llm", "civ": "rome", "model": f"x/{teacher}", "name": teacher,
                                   "base_url": "u", "api_key_env": "K"}  # fmt: skip
        out["spec"]["seats"][1] = {"kind": "petra", "civ": "pers", "difficulty": 3}
        (folder / f"{teacher}-{i}").mkdir(parents=True)
        (folder / f"{teacher}-{i}" / "results.json").write_text(json.dumps(out))
    done = run_step(ctx, "calibration")
    assert done.passed and done.detail["teacher"] == "opus"
    empty, _ = make_ctx(tmp_path / "none", FakeRunner())
    assert not run_step(empty, "calibration").passed


def test_the_labels_run_the_distill_cli_under_with_model_into_the_sft_inbox_with_the_spend_caps(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    out_dir = ctx.s2 / "distill"

    def distill(argv):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sft.jsonl").write_text('{"x": 1}\n' * 2900)
        (out_dir / "distill.json").write_text("{}")
        return 0

    runner.effects = {"train.distill": distill}
    out = run_step(ctx, "labels")
    assert out.passed
    line = runner.lines()[0]
    assert line.startswith(f"{ctx.box_home}/box/with-model.sh python -m agentenv_zeroad.train.distill")
    for part in ("--cap-usd 60.0", "--second-cap-usd 8.0", "--n 3000", "--api-key-env LITELLM_API_KEY",
                 "--base-url https://litellm.example", "--per-game 8", "--tokenizer Qwen/Qwen3-8B"):  # fmt: skip
        assert part in line
    assert f"--calibration {ctx.cfg('baselines').root}/panels/teacher-calibration" in line
    assert f"--inbox {ctx.cfg('sft').root}/inbox" in line and f"--ledger {ctx.cfg('baselines').ledger_path}" in line
    short = FakeRunner(
        {"train.distill": lambda argv: (out_dir.joinpath("sft.jsonl").write_text('{"x": 1}\n' * 100), 0)[1]}
    )
    ctx.runner = short
    assert "labelled briefs" in run_step(ctx, "labels").why()
    ctx.env = {}
    with pytest.raises(S.StepFailed, match="LITELLM_BASE_URL"):
        run_step(ctx, "labels")


def test_the_sft_step_runs_the_learner_only_mode_and_wants_every_epoch_and_the_done_file(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    root = ctx.cfg("sft").root

    def learner(argv, epochs=2, done=True):
        for n in range(1, epochs + 1):
            (root / "epochs" / f"epoch{n}").mkdir(parents=True, exist_ok=True)
            (root / "epochs" / f"epoch{n}" / "manifest.json").write_text(json.dumps({"epoch": n, "version": 10 * n}))
        if done:
            (root / "sft-done.json").write_text("{}")
        return 0

    runner.effects = {"train-run.sh": learner}
    out = run_step(ctx, "sft")
    assert out.passed and runner.calls[0][-4:] == ["--sft", "2", "--lr", "0.0001"] and "sft.toml" in runner.calls[0][2]
    assert S.epoch_versions(root) == {1: 10, 2: 20}
    (root / "sft-done.json").unlink()
    runner.effects = {"train-run.sh": lambda argv: learner(argv, epochs=1, done=False)}
    out = run_step(ctx, "sft")
    assert not out.passed and "did not finish" in out.why()


def test_the_epoch_panels_are_rendered_one_per_epoch_pinned_to_its_version_and_played_play_only(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    root = ctx.cfg("sft").root
    for n, version in ((1, 10), (2, 20)):
        (root / "epochs" / f"epoch{n}").mkdir(parents=True)
        (root / "epochs" / f"epoch{n}" / "manifest.json").write_text(json.dumps({"epoch": n, "version": version}))

    def play(argv):
        for n in (1, 2):
            folder = root / "panels" / f"sft-epoch{n}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "summary.json").write_text(json.dumps({"games": 192, "finished": 192}))
        (root / "orchestrator").mkdir(parents=True, exist_ok=True)
        (root / "orchestrator" / "final.json").write_text(json.dumps({"reason": "complete"}))
        return 0

    runner.effects = {"train-run.sh": play}
    out = run_step(ctx, "sft_panels")
    assert out.passed and out.detail["versions"] == {1: 10, 2: 20}
    rendered = ctx.s2 / "generated" / "sft-panels.toml"
    cfg = S.load_config(rendered)
    assert [(p.name, p.version, p.games, p.kind) for p in cfg.panels] == [
        ("sft-epoch1", 10, 192, "heldout"), ("sft-epoch2", 20, 192, "heldout")
    ]  # fmt: skip
    assert cfg.panels_only and cfg.stop_when_done
    assert runner.calls[0][-1] == "--play-only" and str(rendered) in runner.calls[0]


def test_the_chosen_epoch_then_plays_the_remaining_held_out_configs_under_its_own_panel(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    root = ctx.cfg("sft").root
    init_world(ctx, chosen_margin=600.0)  # epoch 2 (version 20) wins the first 30 games each

    def play(argv):
        rest = argv[2].endswith("sft-rest.toml")
        for name, n in (("sft-rest", 192),) if rest else (("sft-epoch1", 192), ("sft-epoch2", 192)):
            folder = root / "panels" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "summary.json").write_text(json.dumps({"games": n, "finished": n}))
        (root / "orchestrator").mkdir(parents=True, exist_ok=True)
        (root / "orchestrator" / "final.json").write_text(json.dumps({"reason": "complete"}))
        return 0

    runner.effects = {"train-run.sh": play}
    out = run_step(ctx, "sft_panels")
    assert out.passed, out.why()
    assert out.detail["selection"]["chosen"] == "epoch2" and len(runner.calls) == 2
    assert [c[2].rsplit("/", 1)[1] for c in runner.calls] == ["sft-panels.toml", "sft-rest.toml"]
    assert all(c[-1] == "--play-only" for c in runner.calls)
    cfg = S.load_config(ctx.s2 / "generated" / "sft-rest.toml")
    [panel] = cfg.panels
    assert (panel.name, panel.version, panel.start, panel.games, panel.kind) == ("sft-rest", 20, 192, 192, "heldout")
    assert any("sft-rest" in g.name for g in out.gates)
    first, total = 192, 384
    assert first + panel.games == total  # together: v0's 384 configs


def test_no_epoch_that_passes_the_selection_means_no_second_panel_is_played(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    init_world(ctx, chosen_margin=600.0, entropy=0.5)  # under the entropy floor: nothing is chosen
    root = ctx.cfg("sft").root

    def play(argv):
        for name in ("sft-epoch1", "sft-epoch2"):
            (root / "panels" / name).mkdir(parents=True, exist_ok=True)
            (root / "panels" / name / "summary.json").write_text(json.dumps({"games": 192, "finished": 192}))
        (root / "orchestrator").mkdir(parents=True, exist_ok=True)
        (root / "orchestrator" / "final.json").write_text(json.dumps({"reason": "complete"}))
        return 0

    runner.effects = {"train-run.sh": play}
    out = run_step(ctx, "sft_panels")
    assert out.passed and len(runner.calls) == 1 and out.detail["selection"]["chosen"] is None


def write_panel(folder: Path, name: str, version: int, margins: list[tuple[str, float]], start: int = 0):
    model = "local/qwen-base" if version == 0 else f"local/policy-v{version}"
    for i, (opponent, margin) in enumerate(margins, start):
        game_id = f"{name}-{i:04d}-{opponent}"
        petra = {"kind": "petra", "civ": "pers", "difficulty": int(opponent[-1])}
        out = results(game_id, outcomes=("won", "lost") if margin > 0 else ("lost", "won"),
                      net_worth=(max(margin, 0.0) + 1000, max(-margin, 0.0) + 1000), opponent=petra, model_1=model,
                      turns={"1": [{"began_ms": 0, "model": model}]})  # fmt: skip
        (folder / game_id).mkdir(parents=True)
        (folder / game_id / "results.json").write_text(json.dumps(out))


def init_world(ctx, *, chosen_margin: float, entropy: float = 0.9, rest: bool = False):
    sft = ctx.cfg("sft")
    for n, version in ((1, 10), (2, 20)):
        folder = sft.root / "epochs" / f"epoch{n}"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"epoch": n, "version": version, "entropy_ratio": entropy}))
    opponents = ["petra-1", "petra-2", "petra-3"] * 14
    base = [(o, -200.0 + (i % 6) * 40) for i, o in enumerate(opponents)]
    write_panel(ctx.cfg("baselines").root / "panels" / "heldout", "heldout", 0, base)
    write_panel(
        sft.root / "panels" / "sft-epoch1", "sft-epoch1", 10, [(o, m + chosen_margin / 6) for o, m in base[:30]]
    )
    write_panel(sft.root / "panels" / "sft-epoch2", "sft-epoch2", 20, [(o, m + chosen_margin) for o, m in base[:30]])
    if rest:  # the chosen epoch's other configs: games 30 .. 41 of the 42 that v0 played
        write_panel(
            sft.root / "panels" / "sft-rest", "sft-rest", 20, [(o, m + chosen_margin) for o, m in base[30:]], start=30
        )


def test_the_init_step_picks_the_adapter_that_beats_v0_on_the_same_configs_and_otherwise_the_base_model(tmp_path):
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    init_world(ctx, chosen_margin=600.0)
    out = run_step(ctx, "init")
    assert out.passed and out.detail["selection"]["chosen"] == "epoch2"
    decision = out.detail["decision"]
    assert decision["init"] == "epoch2" and out.detail["init_path"].endswith("epochs/epoch2")
    assert decision["test"]["n_other"] == 30 and decision["test"]["n_base"] == 30  # v0's games on the same configs only
    full, _ = make_ctx(tmp_path / "full", FakeRunner())
    init_world(full, chosen_margin=600.0, rest=True)
    test = run_step(full, "init").detail["decision"]["test"]
    assert test["n_other"] == 42 == test["n_base"]  # the first panel and the rest: all of v0's configs, paired
    flat, _ = make_ctx(tmp_path / "flat", FakeRunner())
    init_world(flat, chosen_margin=0.0)
    out = run_step(flat, "init")
    assert out.passed and out.detail["init_path"] is None and "does not beat v0" in out.detail["decision"]["reason"]
    collapsed, _ = make_ctx(tmp_path / "ent", FakeRunner())
    init_world(collapsed, chosen_margin=600.0, entropy=0.5)
    out = run_step(collapsed, "init")
    assert out.detail["init_path"] is None and out.detail["selection"]["chosen"] is None


def test_run_one_starts_from_the_chosen_adapter_with_it_as_its_anchor_or_from_the_base_model(tmp_path):
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    baseline = install_baseline(ctx)
    final = ctx.cfg("run1").root / "orchestrator"
    final.mkdir(parents=True)
    final.joinpath("final.json").write_text(json.dumps({"reason": "duration"}))
    init = str(ctx.cfg("sft").root / "epochs" / "epoch2")
    ctx.record.data["steps"]["init"] = {"detail": {"init_path": init}}
    out = run_step(ctx, "run1")
    assert out.passed and runner.calls[0][3:] == ["--baseline", str(baseline), "--init", init, "--anchor", init]
    ctx.record.data["steps"]["init"] = {"detail": {"init_path": None}}
    run_step(ctx, "run1")
    assert runner.calls[1][2].endswith("run1.toml") and runner.calls[1][3:] == ["--baseline", str(baseline)]
    ctx.record.data["steps"]["init"] = {"detail": {"init_path": str(ctx.cfg("e2").root / "adapters" / "v9")}}
    with pytest.raises(S.StepFailed, match="no E2 adapter may carry over"):
        run_step(ctx, "run1")
    final.joinpath("final.json").write_text(json.dumps({"reason": "tripwire: grad norm"}))
    ctx.record.data["steps"]["init"] = {"detail": {"init_path": None}}
    assert not run_step(ctx, "run1").passed  # two trips end the run early: that is a stop, not a pass


# ---- the end of the day ------------------------------------------------------------------------------------------
def test_the_end_uploads_the_report_and_files_then_powers_the_box_off_unless_the_upload_failed_or_it_was_stopped(
    tmp_path, monkeypatch
):
    order: list[str] = []
    step_stubs(monkeypatch, {"e2": False}, order)
    runner = FakeRunner()
    ctx, _ = make_ctx(tmp_path, runner)
    monkeypatch.setattr(S, "final_report", lambda ctx: None)
    off: list[bool] = []
    assert S.run(ctx, shutdown=lambda: off.append(True)) == 1
    assert off == [True] and json.loads((ctx.s2 / "gates.json").read_text())["verdict"] == "stopped"
    sync = [c for c in runner.lines() if "aws s3 sync" in c]
    assert len(sync) == 2 and sync[0].startswith(f"aws s3 sync {ctx.s2} s3://BUCKET/zeroad-train/run/s2/s2 ")
    assert "--include gates.json" in sync[0] and "--include s2-report.json" in sync[0]
    assert sync[1].endswith("run/s2/s2/sft-epochs --only-show-errors") and "/epochs " in sync[1]
    failing = FakeRunner({"aws s3 sync": 1})
    ctx2, _ = make_ctx(tmp_path / "up", failing)
    off.clear()
    S.run(ctx2, shutdown=lambda: off.append(True))
    assert off == [] and "NOT powering off" in (ctx2.s2 / "progress.log").read_text()
    assert sum("aws s3 sync" in c for c in failing.lines()) == 3  # three tries
    step_stubs(monkeypatch, {}, [])

    def stopped(ctx):
        raise KeyboardInterrupt

    monkeypatch.setitem(S.STEP_FUNCS, "e1", stopped)
    ctx3, _ = make_ctx(tmp_path / "hand", FakeRunner())
    off.clear()
    assert S.run(ctx3, shutdown=lambda: off.append(True)) == 1
    assert off == [] and "a stop by hand" in (ctx3.s2 / "progress.log").read_text()


def test_a_complete_day_ends_with_verdict_complete_and_no_reason(tmp_path, monkeypatch):
    step_stubs(monkeypatch, {}, [])
    monkeypatch.setattr(S, "final_report", lambda ctx: None)
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    assert S.run(ctx, shutdown=None) == 0
    data = json.loads((ctx.s2 / "gates.json").read_text())
    assert data["verdict"] == "complete" and data["reason"] is None


def test_the_status_command_prints_each_steps_verdict_and_its_failed_gates(tmp_path, monkeypatch, capsys):
    step_stubs(monkeypatch, {"e2": False}, [])
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    verdict, reason = S.sequence(ctx)
    ctx.record.finish(verdict, reason)
    assert S.main(["status", str(ctx.s2)]) == 0
    text = capsys.readouterr().out
    assert "verdict: stopped" in text and "e1: passed" in text and "e2: failed" in text and "FAIL e2 gate" in text


def test_the_runner_stops_a_command_at_its_timeout_with_sigterm_and_reports_124(tmp_path):
    runner = S.SubprocessRunner(tmp_path / "logs", env={"PATH": "/usr/bin:/bin"})
    assert runner.run(["sh", "-c", "exit 3"], timeout_s=5, label="quick") == 3
    script = f"trap 'echo got term >> {tmp_path}/term.txt; exit 0' TERM; sleep 30 & wait"
    assert runner.run(["sh", "-c", script], timeout_s=0.5, label="slow", grace_s=5) == 124
    assert (tmp_path / "term.txt").read_text().strip() == "got term"  # SIGTERM first, so a step can upload
    assert "$ sh -c" in (tmp_path / "logs" / "slow.log").read_text()


def test_the_tripwire_and_update_counters_read_the_learners_health_log(tmp_path):
    health = tmp_path / "health.jsonl"
    health.write_text("\n".join(json.dumps(r) for r in [
        {"version": 1, "tripwires": {"trips": []}}, {"version": 2, "tripwires": {"trips": [{"message": "m"}]}},
        {"event": "rollback", "restored": 1, "version": 3}, {"version": 4}]) + "\nnot json\n")  # fmt: skip
    assert S.trips_in(health) == 1 and S.updates_in(health) == 3
    assert S.trips_in(tmp_path / "missing") == 0


def test_the_final_report_links_the_panels_of_every_step_and_writes_s2_report_json(tmp_path):
    ctx, _ = make_ctx(tmp_path, FakeRunner())
    init_world(ctx, chosen_margin=600.0)
    run1 = ctx.cfg("run1").root
    write_panel(run1 / "panels" / "end-heldout", "end-heldout", 5, [("petra-2", 300.0)] * 6)
    write_panel(run1 / "games", "s2-run1-g000001", 5, [("petra-2", 100.0)] * 2)
    assert S.final_report(ctx) is None
    report = json.loads((ctx.s2 / "s2-report.json").read_text())
    assert report["panel"]["base_name"] == "v0-heldout" and report["panel"]["other_name"] == "run1-heldout"
    assert report["sft_selection"]["chosen"] == "epoch2" and report["schema"] == "v2"
    links = {p.name for p in (ctx.s2 / "report-panels").iterdir()}
    assert {"v0-heldout", "run1-heldout", "sft-epoch1", "sft-epoch2"} <= links
    broken, _ = make_ctx(tmp_path / "broken", FakeRunner())
    broken.cfg = lambda step, **kw: (_ for _ in ()).throw(RuntimeError("no config"))  # type: ignore[method-assign]
    assert "RuntimeError: no config" in S.final_report(broken)  # a report that cannot be built does not raise
