"""Teacher labels on hard states: the rolling win rates, the sampling quotas and caps, the rows, and the wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.train import distill
from agentenv_zeroad.train.config import HardStateConfig
from agentenv_zeroad.train.hardstate import HardStates
from tests.fake_models import KEY, FakeLiteLLM
from tests.test_train_calibration import calibration_game
from tests.test_train_orchestrate import config as run_config
from tests.test_train_orchestrate import drive, make
from tests.train_fakes import results, timeline, turn

pytestmark = pytest.mark.anyio

OPUS, ASTRA = "anthropic/claude-opus-5-5", "openai/gpt-6-astra"
TOOLS = [ToolSpec(n, f"{n} tool", {"type": "object"}) for n in ("army", "hold", "say")]
POLICIES = {"v2": distill.Policy("You command an empire.", tuple(TOOLS))}


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def game(game_id: str, outcome: str = "lost", *, difficulty: int = 3, turns: int = 6, behind: bool = True) -> dict:
    """A finished v2 game of the policy against Petra-`difficulty`, `turns` briefs long."""
    briefs = []
    for i in range(turns):
        t = turn(i * 60_000 + 1000)
        t["user"] = f"brief {i} of {game_id}"
        briefs.append(t)
    r = results(
        game_id,
        outcomes=(outcome, {"won": "lost", "lost": "won", "drawn": "drawn"}[outcome]),
        turns={"1": briefs},
        rows=timeline(minutes=30, net_worth=(50, 100) if behind else (100, 50)),
        opponent={"kind": "petra", "civ": "pers", "difficulty": difficulty, "behaviour": "balanced"},
    )
    r["seats"]["1"]["interface"] = "v2"
    r["seats"]["1"]["reference"] = "Costs for rome."
    return r


def cfg(**over) -> HardStateConfig:
    return HardStateConfig(**{"enabled": True, "window": 10, "min_games": 5, **over})


def states(tmp_path: Path, fake: FakeLiteLLM | None = None, **over) -> tuple[HardStates, FakeLiteLLM]:
    fake = fake or FakeLiteLLM(tool_calls=[("army", {"action": "attack"})])
    client = fake.http()

    def make_teacher(model, guard):
        return StrategistModel(model, "https://p", KEY, guard, http=client)

    hard = HardStates(
        cfg(**{"teacher": OPUS, **over}), root=tmp_path / "root", ledger=tmp_path / "spend.jsonl", policies=POLICIES,
        make_teacher=make_teacher, tokenizer=CharTokenizer, seed=1,
    )  # fmt: skip
    return hard, fake


def feed(hard: HardStates, outcomes: str, *, difficulty: int = 3, start: int = 0) -> list[bool]:
    word = {"W": "won", "L": "lost", "D": "drawn"}
    return [
        hard.observe(game(f"g{difficulty}-{start + i}", word[o], difficulty=difficulty), f"g{difficulty}-{start + i}")
        for i, o in enumerate(outcomes)
    ]


# ---- what is hard -----------------------------------------------------------------------------------------------
def test_an_opponent_is_hard_once_enough_games_are_in_the_window_and_the_rate_is_under_the_threshold(tmp_path):
    hard, _ = states(tmp_path)
    kept = feed(hard, "LLLL")
    assert not any(kept) and hard.rate("petra-3") is None and not hard.is_hard("petra-3")  # 4 games: too few
    assert feed(hard, "L", start=4) == [True] and hard.is_hard("petra-3") and hard.hard_opponents() == ["petra-3"]
    assert hard.rate("petra-3") == 0.0 and hard.rate("petra-2") is None


def test_the_threshold_is_strict_draws_count_half_and_the_window_rolls(tmp_path):
    hard, _ = states(tmp_path, window=20, min_games=20, threshold=0.05)
    feed(hard, "L" * 19 + "D")  # 0.5 / 20 = 2.5%
    assert hard.rate("petra-3") == pytest.approx(0.025) and hard.is_hard("petra-3")
    feed(hard, "W", start=20)  # the first loss falls out; one win in 20 is 5.0 + 2.5 = 7.5%
    assert hard.rate("petra-3") == pytest.approx(0.075) and not hard.is_hard("petra-3")
    only = states(tmp_path, window=20, min_games=20, threshold=0.05)[0]
    feed(only, "W" + "L" * 19)  # exactly 5%: not below the threshold
    assert only.rate("petra-3") == pytest.approx(0.05) and not only.is_hard("petra-3")


def test_each_opponent_has_its_own_window_and_only_hard_opponents_games_are_kept(tmp_path):
    hard, _ = states(tmp_path)
    assert not any(feed(hard, "WWWWW", difficulty=2))  # beating Petra-2: nothing to teach
    assert all(feed(hard, "LLLLL", difficulty=3)[-1:]) and hard.hard_opponents() == ["petra-3"]
    assert {b.opponent for b in hard.pool} == {"petra-3"} and len(hard.pool) == 6  # the fifth game: min_games reached


def test_games_that_did_not_finish_or_have_no_policy_seat_are_not_counted(tmp_path):
    hard, _ = states(tmp_path)
    bad = game("x", "lost")
    assert not hard.observe({**bad, "ended_by": "stopped"}, "x")
    assert not hard.observe({**bad, "status": "failed"}, "x") and not hard.windows
    assert not hard.observe(results("mirror", opponent={"kind": "llm"}, trainable=(False, False)), "m")


def test_the_waiting_pool_is_bounded_and_drops_the_oldest(tmp_path):
    hard, _ = states(tmp_path, max_pool=8)
    feed(hard, "L" * 6)
    assert len(hard.pool) == 8 and hard.pool[-1].game_id == "g3-5"


# ---- labelling --------------------------------------------------------------------------------------------------
async def test_a_pass_labels_a_batch_as_sft_rows_in_the_distill_format_under_the_policys_prompt(tmp_path):
    hard, fake = states(tmp_path, batch=4)
    feed(hard, "LLLLL")
    rows, info = await hard.label(0)
    assert len(rows) == 4 and info["asked"] == 4 and info["labelled"] == 4 and info["teacher"] == OPUS
    first = rows[0]
    assert first["sft"] is True and first["version"] == 0 and first["interface"] == "v2"
    assert first["opponent"] == "petra-3" and first["teacher"] == OPUS
    text = "".join(chr(i) for i in first["token_ids"])
    assert text.startswith("<tool_call>") and text.endswith("<|im_end|>") and '"army"' in text
    assert first["prompt_token_ids"] and first["game_id"].startswith("g3-")
    body = fake.bodies("/v1/chat/completions")[0]
    assert body["messages"][0]["content"] == "You command an empire.\n\nCosts for rome."
    assert [t["function"]["name"] for t in body["tools"]] == ["army", "hold", "say"]
    assert len(hard.pool) == 6 - 4  # the asked briefs left the pool


async def test_at_most_per_hour_decisions_are_asked_an_hour_and_a_new_hour_starts_afresh(tmp_path):
    hard, fake = states(tmp_path, per_hour=5, batch=3, per_game=8)
    feed(hard, "L" * 10)
    asked = [(await hard.label(0))[1] for _ in range(3)]
    assert [a["asked"] for a in asked] == [3, 2, 0]
    assert "already sampled this hour" in asked[2]["skipped"]
    assert len(fake.bodies("/v1/chat/completions")) == 5 and hard.quota(0) == 0
    assert (await hard.label(1))[1]["asked"] == 3 and hard.quota(1) == 2


async def test_the_hourly_dollar_cap_binds_the_guard_and_the_ledger_and_resets_with_the_hour(tmp_path):
    hard, fake = states(tmp_path, per_hour=1000, batch=20, cap_usd_per_hour=0.10)
    feed(hard, "L" * 10)
    waiting = len(hard.pool)
    rows, info = await hard.label(0)
    assert len(hard.pool) == waiting - info["asked"]  # briefs the cap left unasked are still waiting
    assert 0 < len(rows) < 20 and info["stopped"] and 0 < info["usd"] <= 0.10
    assert hard.usd[0] == pytest.approx(info["usd"]) and hard.usd_left(0) < 0.10
    again, second = await hard.label(0)
    assert hard.usd[0] <= 0.10 and (again == [] or second["usd"] <= 0.10 - info["usd"] + 1e-9)
    ledger = [json.loads(line) for line in (tmp_path / "spend.jsonl").read_text().splitlines()]
    assert ledger and sum(entry["usd_est"] for entry in ledger) <= 0.10 + 1e-3
    assert all(entry["what"] == "hard-state labels h0" for entry in ledger)
    next_hour, third = await hard.label(1)
    assert next_hour and third["usd"] <= 0.10 and hard.usd[1] == pytest.approx(third["usd"])


async def test_a_spent_hour_asks_nothing_and_says_why(tmp_path):
    hard, fake = states(tmp_path, cap_usd_per_hour=0.10)
    feed(hard, "L" * 6)
    hard.usd[0] = 0.10
    rows, info = await hard.label(0)
    assert rows == [] and "hourly cap" in info["skipped"] and not fake.requests


async def test_nothing_is_asked_with_no_hard_states_waiting_or_no_calibrated_teacher(tmp_path):
    hard, fake = states(tmp_path)
    assert (await hard.label(0))[1]["skipped"] == "no hard states waiting"
    auto, fake = states(tmp_path / "auto", teacher="auto")
    feed(auto, "L" * 6)
    assert (await auto.label(0))[1]["skipped"] == "no calibrated teacher yet" and not fake.requests


async def test_the_auto_teacher_is_the_calibration_panels_best_against_petra_3_unless_one_is_configured(tmp_path):
    folder = tmp_path / "root" / "panels" / "teacher-calibration"
    for i, (name, model, rung, outcome) in enumerate(
        [("astra", ASTRA, 3, "lost"), ("astra", ASTRA, 4, "won"), ("opus", OPUS, 3, "won"), ("opus", OPUS, 4, "lost")]
    ):
        calibration_game(folder, f"c{i}", name, rung, outcome, model)
    auto, fake = states(tmp_path, teacher="auto")
    assert auto.teacher_model() == OPUS
    feed(auto, "L" * 6)
    assert (await auto.label(0))[1]["teacher"] == OPUS
    assert states(tmp_path, teacher=ASTRA)[0].teacher_model() == ASTRA
    elsewhere = states(tmp_path / "x", teacher="auto", calibration=str(folder))[0]
    assert elsewhere.teacher_model() == OPUS  # S2's panel folder, copied or mounted anywhere


async def test_a_teacher_call_that_names_no_known_tool_makes_no_row_and_the_asked_brief_is_not_asked_again(tmp_path):
    hard, fake = states(tmp_path, fake=FakeLiteLLM(tool_calls=[("teleport", {})]), batch=3)
    feed(hard, "L" * 6)
    rows, info = await hard.label(0)
    assert rows == [] and info["asked"] == 3 and info["no_calls"] == 3
    assert len(hard.pool) == 12 - 3  # games 5 and 6, less the three asked: not asked again


def test_the_windows_and_the_hours_counters_survive_a_restart(tmp_path):
    hard, _ = states(tmp_path)
    feed(hard, "LLLLLW")
    hard.asked[3], hard.usd[3] = 40, 1.25
    again, _ = states(tmp_path / "again")
    again.load_state_dict(json.loads(json.dumps(hard.state_dict())))
    assert again.rate("petra-3") == hard.rate("petra-3") and list(again.windows["petra-3"]) == [0, 0, 0, 0, 0, 1]
    assert again.quota(3) == hard.cfg.per_hour - 40 and again.usd[3] == 1.25
    assert again.pool == []  # the waiting decisions are not saved: the games that finish refill it


# ---- the config -------------------------------------------------------------------------------------------------
def test_hard_states_are_off_by_default_with_the_documented_defaults_and_loaded_from_their_table(tmp_path):
    off = run_config(tmp_path / "run").hard_states
    assert not off.enabled and (off.window, off.threshold, off.per_hour, off.cap_usd_per_hour) == (40, 0.05, 200, 10.0)
    on = run_config(tmp_path / "run", hard_states={"enabled": True, "threshold": 0.1, "per_hour": 50}).hard_states
    assert on.enabled and on.threshold == 0.1 and on.per_hour == 50 and on.teacher == "auto"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"surprise": 1}, "unknown hard_states keys"),
        ({"min_games": 50}, "min_games"),
        ({"threshold": 2}, "threshold"),
        ({"per_hour": 0}, "at least 1"),
        ({"cap_usd_per_hour": 0}, "positive"),
    ],
)
def test_a_hard_state_config_that_could_not_run_is_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        run_config(tmp_path / "run", hard_states={"enabled": True, **change})


def test_enabling_it_needs_the_frontier_key_in_the_check(tmp_path):
    from agentenv_zeroad.train.config import check_run

    assert check_run(run_config(tmp_path / "a")) == []
    assert check_run(run_config(tmp_path / "b", hard_states={"enabled": True})) == ["LITELLM_API_KEY"]


# ---- in the orchestrator ----------------------------------------------------------------------------------------
class FakeHard:
    """What the orchestrator needs of HardStates: observe, label, status and state, with the labels scripted."""

    def __init__(self, rows_per_pass: int = 2):
        self.rows_per_pass, self.passes, self.seen, self.hours = rows_per_pass, 0, [], []

    def observe(self, results, name):
        self.seen.append(name)
        return len(self.seen) % 2 == 0

    async def label(self, hour):
        self.passes += 1
        self.hours.append(hour)
        rows = [{"game_id": f"h{self.passes}-{i}", "version": 0, "sft": True, "prompt_token_ids": [1],
                 "token_ids": [2]} for i in range(self.rows_per_pass)]  # fmt: skip
        return rows, {"asked": self.rows_per_pass, "labelled": len(rows)}

    def status(self, hour):
        return {"hard": ["petra-3"], "hour": hour}

    def state_dict(self):
        return {"seen": list(self.seen)}

    def load_state_dict(self, state):
        self.seen = list(state["seen"])


async def test_the_orchestrator_counts_training_games_and_writes_labels_to_the_inbox_as_sft_rows(tmp_path, monkeypatch):
    hard = FakeHard()
    t = make(tmp_path, monkeypatch, concurrency=2, hard_states={"enabled": True, "interval_s": 0.001})
    t.orch.hard = hard
    t.clock.t += 60
    await drive(t.orch, lambda: t.orch.counts["train_finished"] >= 2 and t.orch.counts["hard_rows"] >= 2)
    await t.orch.join_hard()
    assert len(hard.seen) >= 2 and t.orch.counts["hard_games"] >= 1
    inbox = sorted(p.name for p in (t.root / "inbox").glob("hard-*.jsonl"))
    assert inbox and inbox[0].startswith("hard-000-")
    row = json.loads((t.root / "inbox" / inbox[0]).read_text().splitlines()[0])
    assert row["sft"] is True and row["prompt_token_ids"] == [1]
    saved = json.loads((t.root / "orchestrator" / "hard.json").read_text())
    assert saved["seen"] == hard.seen
    t.orch.write_status(t.clock.t, t.orch.check_budget(t.clock.t))
    status = json.loads((t.root / "orchestrator" / "status.json").read_text())
    assert status["hard_states"]["hard"] == ["petra-3"]
    assert t.orch.counts["hard_asked"] == t.orch.counts["hard_rows"]


async def test_a_failing_labelling_pass_is_counted_and_the_run_goes_on(tmp_path, monkeypatch):
    hard = FakeHard()

    async def boom(hour):
        raise RuntimeError("teacher down")

    hard.label = boom
    t = make(tmp_path, monkeypatch, concurrency=1, hard_states={"enabled": True, "interval_s": 0.001})
    t.orch.hard = hard
    t.orch._start_hard_labels(0)
    await t.orch.join_hard()
    assert t.orch.counts["hard_errors"] == 1 and t.orch.stop_reason is None
    await asyncio.sleep(0)


def test_the_windows_come_back_after_an_orchestrator_restart(tmp_path, monkeypatch):
    first = FakeHard()
    t = make(tmp_path, monkeypatch, hard_states={"enabled": True})
    t.orch.hard = first
    first.seen = ["a", "b"]
    t.orch._write("hard.json", first.state_dict())
    second = FakeHard()
    t2 = make(tmp_path, monkeypatch, hard_states={"enabled": True})
    t2.orch.hard = second
    t2.orch._restore()
    assert second.seen == ["a", "b"]


def test_the_example_config_documents_both_sections_and_they_load_once_uncommented(tmp_path):
    from agentenv_zeroad.train.config import check_run, load_config

    text = (Path(__file__).resolve().parents[1] / "live" / "train-run.example.toml").read_text()
    assert "TEACHER CALIBRATION" in text and "HARD-STATE TEACHER LABELS" in text
    lines, live, seen = [], False, False
    for line in text.splitlines():
        seen = seen or "TEACHER CALIBRATION" in line  # the older commented [[panels]] example above is left alone
        live = (seen and line.startswith(("# [[panels]]", "# [hard_states]"))) or (
            live and line.startswith("# ") and "----" not in line
        )
        lines.append(line[2:] if live else line)  # the new commented blocks, uncommented
    path = tmp_path / "run.toml"
    path.write_text("\n".join(lines))
    cfg = load_config(path)
    panel = next(p for p in cfg.panels if p.kind == "teacher_calibration")
    assert [t.model for t in panel.teachers] == [
        "openai/gpt-6-astra",
        "anthropic/claude-opus-5-5",
    ] and panel.games == 10
    assert cfg.hard_states.enabled and (cfg.hard_states.per_hour, cfg.hard_states.cap_usd_per_hour) == (200, 10.0)
    assert check_run(cfg) == ["LITELLM_API_KEY"]
