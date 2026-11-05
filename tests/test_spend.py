"""The spend guard stops a brain before a call that could pass its cap, prices conservatively, and the loop's ledger
check refuses a run that could pass the cumulative or the unit cap."""

import json

import pytest

from agentenv_zeroad import spend
from agentenv_zeroad.spend import SpendCapReached, SpendGuard


def test_prices_are_the_price_map_with_a_margin_and_unknown_models_are_priced_high():
    assert spend.price("anthropic/claude-opus-5-5") == (6.0, 30.0)
    assert spend.price("openai/gpt-6-luna") == pytest.approx((0.15, 0.75))
    assert spend.price("xai/grok-4.7") == spend.DEFAULT_PRICE
    assert spend.price("local/policy-r3") == (0.0, 0.0)  # served on the box's own GPUs
    assert spend.cost("anthropic/claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(9.0)


def test_the_guard_refuses_a_call_that_could_pass_the_cap_before_it_is_sent(tmp_path):
    guard = SpendGuard("test", cap_usd=0.10, log_path=tmp_path / "spend.jsonl")
    guard.check(0.04)
    guard.record("anthropic/claude-opus-5-5", 3000, 1000, most_usd=0.04)  # 0.018 + 0.03 = 0.048
    assert guard.spent_usd == pytest.approx(0.048)
    guard.check(0.05)
    with pytest.raises(SpendCapReached, match=r"past the \$0.1 cap"):
        guard.check(0.06)
    assert not guard.allows(0.0001)  # once stopped, stays stopped
    line = json.loads((tmp_path / "spend.jsonl").read_text().splitlines()[0])
    assert line == {
        "what": "test",
        "model": "anthropic/claude-opus-5-5",
        "input_tokens": 3000,
        "output_tokens": 1000,
        "usd_est": 0.048,
    }


def test_a_call_without_usage_is_charged_at_its_bound():
    guard = SpendGuard("test")
    request = {"input": "x" * 4000}
    most = spend.bound("openai/gpt-6-astra", request, 1000)
    assert most > spend.cost("openai/gpt-6-astra", 4000, 1000)
    assert guard.record("openai/gpt-6-astra", None, None, most_usd=most) == most
    assert guard.unpriced_calls == 1 and guard.summary()["usd_est"] == round(most, 6)


def test_the_ledger_check_refuses_a_run_past_the_cumulative_or_the_unit_cap(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text('{"what": "spike latency x", "usd_est": 0.7}\n{"what": "U5 setup-a", "usd_est": 4.0}\n\n')
    assert spend.ledger_total(ledger) == pytest.approx(4.7)
    assert spend.check_budget(ledger, planned_usd=5, cap_usd=25, unit_prefix="U5", unit_cap_usd=10) is None
    assert "unit" not in (spend.check_budget(ledger, planned_usd=21, cap_usd=25) or "unit")
    assert "U5" in spend.check_budget(ledger, planned_usd=6.5, cap_usd=25, unit_prefix="U5", unit_cap_usd=10)
    spend.append_ledger(ledger, "U5 setup-b", 1.23456)
    assert json.loads(ledger.read_text().splitlines()[-1]) == {"what": "U5 setup-b", "usd_est": 1.2346}
    assert spend.main(["check", "--ledger", str(ledger), "--planned", "30"]) == 1
    assert spend.main(["check", "--ledger", str(ledger), "--planned", "1"]) == 0


def test_a_matchs_spend_sums_every_seats_brain_and_reflex():
    results = {
        "seats": {
            "1": {"spend": {"brain": {"usd_est": 0.5}, "reflex": {"usd_est": 0.1}}},
            "2": {"spend": {"brain": {"usd_est": 0.25}}},
            "3": {"spend": None},
        }
    }
    assert spend.results_spend(results) == pytest.approx(0.85)
