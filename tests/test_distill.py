"""The distillation probe against fake teachers: which briefs, what labels, the spend cap, label noise, SFT rows."""

import asyncio
import json
import sys
import types

import pytest

from agentenv_zeroad import spend
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train import distill as D
from tests.fake_models import KEY, FakeLiteLLM
from tests.train_fakes import results, timeline, turn

pytestmark = pytest.mark.anyio

TOOLS = [ToolSpec(n, f"{n} tool", {"type": "object"}) for n in ("army", "hold", "set_economy", "say")]
SYSTEM = "You command an empire."
POLICIES = {"v2": D.Policy(SYSTEM, tuple(TOOLS))}


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def brief_turn(minute, *, status="accepted"):
    t = turn(int(minute * 60_000), status=status)
    t["user"] = f"brief at minute {minute}"
    return t


def save_game(
    games, game_id, minutes=(3, 9, 10, 25), *, behind=True, rejected_at=None, status="finished", interface="v2"
):
    nw = (50, 100) if behind else (100, 50)
    turns = [brief_turn(m, status="rejected" if m == rejected_at else "accepted") for m in minutes]
    r = results(game_id, turns={"1": turns}, rows=timeline(minutes=30, net_worth=nw), status=status)
    r["interface"] = interface
    (games / game_id).mkdir(parents=True)
    (games / game_id / "results.json").write_text(json.dumps(r))


def teacher(calls, *, cap=None, model="anthropic/claude-opus-5-5"):
    fake = FakeLiteLLM(tool_calls=calls)
    guard = SpendGuard(model, cap_usd=cap)
    return StrategistModel(model, "https://p", KEY, guard, http=fake.http()), fake


# --- choosing briefs ---


def test_a_turn_is_rejected_behind_late_or_other_in_that_order(tmp_path):
    save_game(tmp_path, "g1", minutes=(3, 9, 10, 25, 26), rejected_at=26)
    by = {b.began_ms // 60_000: b.stratum for b in D.candidates(tmp_path)}
    assert by == {3: "other", 9: "behind", 10: "behind", 25: "late", 26: "rejected"}


def test_ahead_at_ten_minutes_is_not_behind_and_unfinished_games_give_no_briefs(tmp_path):
    save_game(tmp_path, "ahead", behind=False)
    save_game(tmp_path, "failed", status="failed")
    pool = D.candidates(tmp_path)
    assert {b.game_id for b in pool} == {"ahead"} and "behind" not in {b.stratum for b in pool}


def test_games_played_under_another_interface_give_no_briefs(tmp_path):
    save_game(tmp_path, "old", interface="v1")
    save_game(tmp_path, "new")
    assert {b.game_id for b in D.candidates(tmp_path)} == {"new"}


def test_a_brief_keeps_the_brief_text_and_the_prompt_ids_the_policy_was_sampled_from(tmp_path):
    save_game(tmp_path, "g1")
    first = D.candidates(tmp_path)[0]
    assert first.user == "brief at minute 3" and first.prompt_token_ids == (1, 2, 3, 30) and first.slot == "1"
    assert first.key == "g1:1:180000"


def pool_of(counts, per=3):
    out = []
    for stratum, n in counts.items():
        out += [D.Brief(f"{stratum}{i // per}", "1", i, f"b{i}", (1,), stratum) for i in range(n)]
    return out


def test_selection_takes_a_third_from_each_stratum_and_caps_briefs_per_game():
    pool = pool_of({"rejected": 60, "behind": 60, "late": 60, "other": 60}, per=5)
    chosen = D.select(pool, 30, per_game=2)
    counts = {s: sum(b.stratum == s for b in chosen) for s in D.STRATA}
    assert counts == {"rejected": 10, "behind": 10, "late": 10, "other": 0}
    games = {}
    for b in chosen:
        games[b.game_id] = games.get(b.game_id, 0) + 1
    assert max(games.values()) == 2
    assert [b.key for b in D.select(pool, 30, per_game=2)] == [b.key for b in chosen]
    assert [b.key for b in D.select(pool, 30, per_game=2, seed=1)] != [b.key for b in chosen]


def test_a_short_stratum_hands_its_share_to_the_others_and_an_exhausted_pool_returns_fewer():
    pool = pool_of({"rejected": 2, "behind": 50, "late": 50, "other": 10})
    chosen = D.select(pool, 30, per_game=50)
    assert len(chosen) == 30 and sum(b.stratum == "rejected" for b in chosen) == 2
    assert len(D.select(pool_of({"late": 4}), 30)) == 4
    assert len(D.select(pool_of({"rejected": 1, "other": 40}), 20, per_game=50)) == 20


# --- labels ---


def test_a_completion_is_qwens_tool_call_format_for_each_call_then_the_end_of_turn_token():
    text = D.completion_text([("army", {"action": "attack"}), ("hold", {})])
    assert text == (
        '<tool_call>\n{"name": "army", "arguments": {"action": "attack"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "hold", "arguments": {}}\n</tool_call><|im_end|>'
    )


async def test_the_teacher_gets_the_policys_system_prompt_tools_and_brief_and_its_calls_become_sft_rows(tmp_path):
    save_game(tmp_path, "g1")
    briefs = D.candidates(tmp_path)
    model, fake = teacher([("army", {"action": "attack"}), ("say", {"text": "hi"})])
    labels, stats = await D.label(briefs, model, POLICIES)
    assert stats["labelled"] == 4 and stats["no_calls"] == 0 and stats["stopped"] is None
    body = fake.bodies("/v1/chat/completions")[0]
    assert body["messages"][0] == {"role": "system", "content": SYSTEM} and body["messages"][1]["content"].startswith(
        "brief"
    )
    assert [t["function"]["name"] for t in body["tools"]] == ["army", "hold", "set_economy", "say"]
    rows = D.sft_rows(labels, CharTokenizer(), "opus")
    assert (
        rows[0]["prompt_token_ids"] == [1, 2, 3, 30] and rows[0]["game_id"] == "g1:1:180000" and rows[0]["version"] == 0
    )
    text = "".join(chr(i) for i in rows[0]["token_ids"])
    assert text.startswith("<tool_call>") and text.endswith("<|im_end|>") and '"name": "say"' in text
    assert rows[0]["teacher"] == "opus" and rows[0]["stratum"] == "other" and rows[0]["interface"] == "v2"


async def test_calls_to_tools_the_policy_lacks_are_left_out_and_a_brief_with_none_left_is_dropped(tmp_path):
    save_game(tmp_path, "g1")
    briefs = D.candidates(tmp_path)
    mixed, _ = teacher([("teleport", {}), ("hold", {})])
    labels, _ = await D.label(briefs, mixed, POLICIES)
    assert all(item.calls == [("hold", {})] for item in labels)
    nothing, _ = teacher([("teleport", {})])
    labels, stats = await D.label(briefs, nothing, POLICIES)
    assert labels == [] and stats["no_calls"] == 4


async def test_labelling_stops_at_the_spend_cap_and_sends_nothing_past_it(tmp_path):
    for i in range(6):
        save_game(tmp_path, f"g{i}", minutes=(3,))
    briefs = D.candidates(tmp_path)
    model, fake = teacher([("hold", {})], cap=0.12)
    labels, stats = await D.label(briefs, model, POLICIES, concurrency=1)
    assert 0 < len(labels) < len(briefs) and stats["stopped"] and "cap" in stats["stopped"]
    assert len(fake.requests) == len(labels) == stats["asked"]
    assert stats["spend"]["usd_est"] <= 0.12 and stats["spend"]["cap_usd"] == 0.12


class Slow:
    """A teacher that takes a moment per call, so concurrent calls overlap before any of them is charged."""

    def __init__(self, inner):
        self.inner, self.model, self.guard, self.max_tokens = inner, inner.model, inner.guard, inner.max_tokens

    def request(self, system, user, tools):
        return self.inner.request(system, user, tools)

    async def call(self, system, user, tools):
        await asyncio.sleep(0.01)
        return await self.inner.call(system, user, tools)


async def test_calls_in_flight_together_never_pass_the_cap(tmp_path):
    for i in range(12):
        save_game(tmp_path, f"g{i}", minutes=(3,))
    briefs = D.candidates(tmp_path)
    probe, _ = teacher([("hold", {})])
    _, body = probe.request(SYSTEM, briefs[0].user, TOOLS)
    most = spend.bound(probe.model, body, probe.max_tokens)
    model, fake = teacher([("hold", {})], cap=2.5 * most)
    labels, stats = await D.label(briefs, Slow(model), POLICIES, concurrency=8)
    assert len(fake.requests) == 2 == len(labels) and stats["stopped"]


async def test_a_failed_call_is_counted_and_the_other_briefs_go_on(tmp_path):
    save_game(tmp_path, "g1")
    briefs = D.candidates(tmp_path)
    fake = FakeLiteLLM(fail=500)
    model = StrategistModel("anthropic/claude-opus-5-5", "https://p", KEY, SpendGuard("t"), http=fake.http())
    labels, stats = await D.label(briefs, model, POLICIES)
    assert labels == [] and stats["failed"] == 4 and "HTTPStatusError" in stats["errors"][0]


# --- label noise ---


def labelled(key_calls):
    return [
        D.Label(D.Brief("g", "1", i, "b", (1,), "other"), [(n, {}) for n in names]) for i, names in enumerate(key_calls)
    ]


def test_the_noise_check_is_the_mean_jaccard_of_tool_sets_over_briefs_both_teachers_labelled():
    a = labelled([["army", "hold"], ["army"], ["hold"], ["army"]])
    b = labelled([["army", "set_economy"], ["army"], ["say"]])
    result = D.noise_check(a, b)
    assert result["briefs"] == 3 and result["tool_set_jaccard"] == pytest.approx((1 / 3 + 1 + 0) / 3)
    assert result["identical_tool_sets"] == pytest.approx(1 / 3)
    assert D.noise_check(a, [])["tool_set_jaccard"] is None


def test_the_noise_check_also_reports_the_agreement_per_stratum():
    a = [D.Label(D.Brief("g", "1", i, "b", (1,), st), [(n, {}) for n in names]) for i, (st, names) in enumerate(
        [("rejected", ["army"]), ("rejected", ["hold"]), ("late", ["army"])])]  # fmt: skip
    b = [D.Label(item.brief, [("army", {})]) for item in a]
    by = D.noise_check(a, b)["by_stratum"]
    assert by["rejected"]["tool_set_jaccard"] == pytest.approx(0.5) and by["late"]["tool_set_jaccard"] == 1.0
    assert "behind" not in by


def test_the_check_sample_is_random_and_takes_an_equal_share_from_every_stratum():
    labels = [
        D.Label(D.Brief(f"{st}{i}", "1", i, "b", (1,), st), [("army", {})])
        for st in ("rejected", "behind", "late")
        for i in range(40)
    ]
    sample = D.check_sample(labels, 30)
    assert {s: sum(b.stratum == s for b in sample) for s in D.STRATA} == {
        "rejected": 10, "behind": 10, "late": 10, "other": 0
    }  # fmt: skip
    assert [b.key for b in sample] != [item.brief.key for item in labels[:30]]
    assert [b.key for b in D.check_sample(labels, 30)] == [b.key for b in sample]
    assert [b.key for b in D.check_sample(labels, 30, seed=1)] != [b.key for b in sample]
    assert len(D.check_sample(labels[:5], 30)) == 5


async def test_the_second_teachers_check_sample_spans_the_strata(tmp_path):
    save_game(tmp_path, "g1", minutes=(3, 9, 25, 26), rejected_at=26)
    save_game(tmp_path, "g2", minutes=(3, 9, 25, 26), rejected_at=26)
    briefs = D.candidates(tmp_path)
    first, _ = teacher([("army", {"action": "attack"})])
    second, _ = teacher([("army", {"action": "attack"})], model="openai/gpt-6-astra")
    _, report = await D.distill(briefs, first, second, POLICIES, CharTokenizer(), check=4)
    assert set(report["second_teacher"]["by_stratum"]) == {"rejected", "behind", "late", "other"}


async def test_the_second_teacher_labels_only_the_check_subset_and_the_report_has_the_noise(tmp_path):
    for i in range(5):
        save_game(tmp_path, f"g{i}", minutes=(3, 9))
    briefs = D.candidates(tmp_path)
    first, _ = teacher([("army", {"action": "attack"}), ("hold", {})])
    second, second_fake = teacher([("army", {"action": "attack"})], model="openai/gpt-6-astra")
    rows, report = await D.distill(briefs, first, second, POLICIES, CharTokenizer(), check=4)
    assert len(rows) == 10 and report["labelled"] == 10 and report["teacher"] == "anthropic/claude-opus-5-5"
    assert len(second_fake.requests) == 4 and report["second_teacher"]["briefs"] == 4
    assert report["second_teacher"]["tool_set_jaccard"] == pytest.approx(0.5)
    _, solo = await D.distill(briefs, first, None, POLICIES, CharTokenizer())
    assert "second_teacher" not in solo


# --- the policy the teacher sees ---


def test_the_default_policy_is_the_interfaces_prompt_and_its_order_tools_from_the_env(monkeypatch):
    from agentenv_zeroad import brains

    monkeypatch.setattr(
        D, "env_tool_specs", lambda: [*TOOLS, ToolSpec("status", "not an order tool", {"type": "object"})]
    )
    monkeypatch.setattr(brains, "ORDER_TOOLS_V2", ("army", "hold"), raising=False)
    monkeypatch.setattr(brains, "strategist_system", lambda interface: f"system {interface}", raising=False)
    policy = D.default_policy("v2")
    assert policy.system == "system v2" and [t.name for t in policy.tools] == ["army", "hold"]


def test_the_default_policy_falls_back_to_the_v1_prompt_where_the_v2_one_is_not_merged_yet(monkeypatch):
    from agentenv_zeroad import brains

    monkeypatch.delattr(brains, "strategist_system", raising=False)
    monkeypatch.delattr(brains, "ORDER_TOOLS_V2", raising=False)
    monkeypatch.setattr(D, "env_tool_specs", lambda: list(TOOLS))
    assert D.default_policy("v2").system == brains.STRATEGIST_SYSTEM


def test_the_env_tool_list_comes_from_the_class_without_a_running_server():
    names = {t.name for t in D.env_tool_specs()}
    assert {"army", "train", "get_brief"} <= names


async def test_each_brief_is_asked_under_its_own_interfaces_prompt_and_tools(tmp_path):
    save_game(tmp_path, "g1", minutes=(3,))
    briefs = D.candidates(tmp_path)
    other = D.Policy("a second prompt", TOOLS[:1])
    model, fake = teacher([("army", {})])
    await D.label(briefs, model, {"v2": other})
    body = fake.bodies("/v1/chat/completions")[0]
    assert body["messages"][0]["content"] == "a second prompt" and len(body["tools"]) == 1


# --- writing ---


def test_the_inbox_gets_one_file_per_epoch_reshuffled_and_never_a_half_written_one(tmp_path):
    rows = [{"game_id": f"g{i}", "version": 0, "prompt_token_ids": [1], "token_ids": [2, i]} for i in range(20)]
    files = D.write_inbox(rows, tmp_path / "inbox", epochs=3)
    assert [f.name for f in files] == ["sft-epoch00.jsonl", "sft-epoch01.jsonl", "sft-epoch02.jsonl"]
    assert not list((tmp_path / "inbox").glob("*.tmp"))
    orders = [[json.loads(line)["game_id"] for line in f.read_text().splitlines()] for f in files]
    assert all(sorted(o) == sorted(r["game_id"] for r in rows) for o in orders) and orders[0] != orders[1]


def test_the_command_line_writes_rows_a_report_and_the_ledger(tmp_path, monkeypatch):
    for i in range(3):
        save_game(tmp_path / "games", f"g{i}", minutes=(3, 9, 25))
    (tmp_path / "tools.json").write_text(json.dumps([t.__dict__ for t in TOOLS]))
    (tmp_path / "system.txt").write_text(SYSTEM)
    fake = FakeLiteLLM(tool_calls=[("army", {"action": "attack"})])
    client = fake.http()
    monkeypatch.setattr(D.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: CharTokenizer())),
    )
    monkeypatch.setenv("TEACHER_KEY", KEY)
    out, ledger = tmp_path / "out", tmp_path / "spend.jsonl"
    args = [
        "--games", str(tmp_path / "games"), "--out", str(out), "--tools", str(tmp_path / "tools.json"),
        "--system", str(tmp_path / "system.txt"), "--tokenizer", "Qwen/Qwen3-8B",
        "--teacher", "anthropic/claude-opus-5-5", "--base-url", "https://p",
        "--api-key-env", "TEACHER_KEY", "--cap-usd", "5", "--n", "6", "--inbox", str(tmp_path / "inbox"),
        "--epochs", "2", "--ledger", str(ledger),
    ]  # fmt: skip
    D.main(args)
    rows = [json.loads(line) for line in (out / "sft.jsonl").read_text().splitlines()]
    report = json.loads((out / "distill.json").read_text())
    assert len(rows) == 6 and report["labelled"] == 6 and report["pool"] == 9
    assert len(list((tmp_path / "inbox").glob("*.jsonl"))) == 2
    assert json.loads(ledger.read_text())["what"] == "distill"


def test_the_command_line_refuses_to_run_without_the_teachers_key(tmp_path, monkeypatch):
    save_game(tmp_path / "games", "g1")
    (tmp_path / "tools.json").write_text("[]")
    (tmp_path / "system.txt").write_text(SYSTEM)
    monkeypatch.delenv("TEACHER_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: CharTokenizer())),
    )
    args = [
        "--games", str(tmp_path / "games"), "--out", str(tmp_path / "o"), "--tools", str(tmp_path / "tools.json"),
        "--system", str(tmp_path / "system.txt"), "--tokenizer", "t", "--teacher", "m", "--base-url", "u",
        "--api-key-env", "TEACHER_KEY", "--cap-usd", "1",
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="TEACHER_KEY"):
        D.main(args)
