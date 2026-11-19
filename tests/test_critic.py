"""The optional critic: features, the prequential gate, the blend, and state."""

import json
import math
import random
import statistics

import pytest

pytest.importorskip("sklearn")

from agentenv_zeroad.train import critic as C  # noqa: E402
from agentenv_zeroad.train.credit import Credit, CreditConfig  # noqa: E402
from tests.train_fakes import results, timeline, turn, turns_every  # noqa: E402

FAST = dict(min_games=60, eval_games=40, max_iter=25, min_samples_leaf=5, cv_folds=3, half_life_games=1000.0)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def synth(i: int, rng: random.Random, *, informative: bool = True):
    """A game whose winner is whoever had the bigger army: visible in the features, invisible to a running mean."""
    a1, a2 = rng.uniform(2, 20), rng.uniform(2, 20)
    won = a1 > a2 if informative else rng.random() < 0.5
    return results(
        f"c{i}",
        ended_by="conquest",
        outcomes=("won", "lost") if won else ("lost", "won"),
        rows=timeline(minutes=10, army=(a1, a2)),
        turns={"1": turns_every(60, minutes=10)},
    ), (1.0 if won else -1.0)


def feed(critic, n: int, *, informative: bool = True, seed: int = 0, start: int = 0):
    rng, b = random.Random(seed), 0.0
    for i in range(n):
        g, r = synth(start + i, rng, informative=informative)
        critic.observe(g, "1", r, b)
        b += (r - b) / 64


def probe(army_own: float, army_opp: float):
    return results("probe", rows=timeline(minutes=10, army=(army_own, army_opp)), turns={"1": [turn(300_000)]})


# --- features ---


def test_features_are_both_sides_stats_deltas_and_the_setup_without_exploration():
    g = results("f", rows=timeline(minutes=10, army=(20, 5), net_worth=(50, 10)), turns={"1": [turn(300_000)]})
    x = dict(zip(C.FEATURES, C.features(g, "1", 300_000), strict=True))
    assert len(C.features(g, "1", 300_000)) == len(C.FEATURES)
    assert not any("exploration" in name or name.endswith("total") for name in C.FEATURES)
    assert x["minute"] == 5.0 and x["own_army"] == pytest.approx(10.0) and x["opp_army"] == pytest.approx(2.5)
    assert x["own_army_delta"] == pytest.approx(20 * 60 / 600) and x["opp_net_worth"] == pytest.approx(3000.0)
    assert x["own_net_worth_delta"] == pytest.approx(50 * 60) and x["level"] == 2 and x["L"] == 5000
    assert x["own_phase"] == 0 and x["kind"] == C.KINDS["petra"] and x["behaviour"] == C.BEHAVIOURS["balanced"]
    assert x["civ"] != x["opponent_civ"] and not math.isnan(x["map"])


def test_missing_net_worth_is_nan_not_zero_and_the_first_minute_has_no_deltas_to_speak_of():
    g = results("f", rows=timeline(minutes=10, net_worth=None), turns={"1": [turn(0)]})
    x = dict(zip(C.FEATURES, C.features(g, "1", 0), strict=True))
    assert math.isnan(x["own_net_worth"]) and math.isnan(x["opp_net_worth_delta"])
    assert x["own_army_delta"] == 0.0


# --- the gate ---


def test_a_critic_with_real_signal_beats_the_running_mean_and_is_switched_in():
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    feed(critic, 110)
    events = [h["event"] for h in critic.history]
    assert events[:2] == ["candidate", "switch"] and critic.active is not None
    clock = critic.clock
    clock.now += 10_000
    assert critic.baseline(probe(20, 2), "1", 300_000, 0.0) > 0.5
    assert critic.baseline(probe(2, 20), "1", 300_000, 0.0) < -0.5


def test_the_candidate_is_judged_only_on_games_after_the_one_it_was_fit_on():
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    feed(critic, 60)
    assert critic.candidate is not None and critic.candidate.upto == 60 and critic.window["games"] == 0
    feed(critic, 39, start=60, seed=1)
    assert critic.window["games"] == 39 and critic.active is None
    feed(critic, 1, start=99, seed=2)
    assert critic.active is not None and critic.window["games"] == 0  # decided, and the next window opens fresh


def test_a_critic_fit_to_noise_never_replaces_the_baseline():
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    feed(critic, 200, informative=False, seed=3)
    assert critic.active is None
    assert {h["event"] for h in critic.history} <= {"screened_out", "candidate", "kept"}
    assert critic.baseline(probe(20, 2), "1", 300_000, 0.125) == 0.125


def test_a_candidate_must_beat_the_critic_in_use_as_well_as_the_mean():
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    feed(critic, 110)
    incumbent = critic.active
    critic.candidate = C.Fit(critic.active.model, 110)
    critic.window = {"games": 30, "rows": 300, "candidate": 30.0, "incumbent": 20.0, "mean": 90.0}  # beats mean only
    critic._decide()
    assert critic.active is incumbent and critic.history[-1]["event"] == "kept"
    critic.candidate = C.Fit(critic.active.model, 110)
    critic.window = {"games": 30, "rows": 300, "candidate": 10.0, "incumbent": 20.0, "mean": 90.0}
    critic._decide()
    assert critic.active.upto == 110 and critic.previous is incumbent and critic.history[-1]["event"] == "switch"
    critic.candidate = C.Fit(critic.active.model, 110)
    critic.window = {"games": 30, "rows": 300, "candidate": 95.0, "incumbent": 99.0, "mean": 90.0}  # beats critic only
    before = critic.active
    critic._decide()
    assert critic.active is before


def test_after_a_switch_the_baseline_moves_from_the_old_value_to_the_critics_over_the_blend_window():
    clock = Clock()
    critic = C.GBCritic(C.CriticConfig(**{**FAST, "blend_s": 900.0}), clock=clock)
    feed(critic, 110)
    g = probe(20, 2)
    assert critic.switched_at is not None
    clock.now = critic.switched_at
    assert critic.baseline(g, "1", 300_000, 0.2) == pytest.approx(0.2)  # a fresh switch is still the running mean
    clock.now = critic.switched_at + 450
    half = critic.baseline(g, "1", 300_000, 0.2)
    clock.now = critic.switched_at + 900
    full = critic.baseline(g, "1", 300_000, 0.2)
    assert half == pytest.approx((0.2 + full) / 2) and full > 0.5


def test_refits_weight_recent_games_more():
    critic = C.GBCritic(C.CriticConfig(**{**FAST, "half_life_games": 10.0}), clock=Clock())
    feed(critic, 30)
    _, _, w, _, group = critic._data(30)
    assert w[group == 30][0] == 1.0 and w[group == 20][0] == pytest.approx(0.5) and w[group == 10][0] == 0.25


def test_state_round_trips_through_json_and_the_models_predict_the_same():
    clock = Clock()
    a = C.GBCritic(C.CriticConfig(**FAST), clock=clock)
    feed(a, 80)
    b = C.GBCritic(C.CriticConfig(**FAST), clock=clock)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    clock.now += 10_000
    g = probe(15, 4)
    assert b.baseline(g, "1", 300_000, 0.0) == pytest.approx(a.baseline(g, "1", 300_000, 0.0))
    assert b.observed == a.observed and [h["event"] for h in b.history] == [h["event"] for h in a.history]
    with pytest.raises(ValueError, match="version"):
        b.load_state_dict({"version": 0})


def test_state_reloads_after_the_games_were_trimmed_past_the_games_a_fit_used():
    clock = Clock()
    cfg = C.CriticConfig(**{**FAST, "max_games": 30})
    a = C.GBCritic(cfg, clock=clock)
    feed(a, 130)
    assert a.active is not None and len(a.games) == 30 and a.active.upto < a.games[0].seq
    a.previous = C.Fit(a.active.model, 40)
    clock.now = a.switched_at
    b = C.GBCritic(cfg, clock=clock)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert (b.active.upto, b.previous.upto) == (a.active.upto, 40)
    g = probe(15, 4)
    for t in (0, 450, 900):
        clock.now = a.switched_at + t
        assert b.baseline(g, "1", 300_000, 0.1) == pytest.approx(a.baseline(g, "1", 300_000, 0.1))
        if t < 900:
            assert b.previous is not None


def test_the_previous_critic_is_dropped_once_the_blend_is_complete():
    clock = Clock()
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=clock)
    feed(critic, 110)
    critic.previous = C.Fit(critic.active.model, 60)
    clock.now = critic.switched_at + 100
    critic.baseline(probe(20, 2), "1", 300_000, 0.0)
    assert critic.previous is not None
    clock.now = critic.switched_at + 900
    critic.baseline(probe(20, 2), "1", 300_000, 0.0)
    assert critic.previous is None and critic.state_dict()["fits"]["previous"] is None


def test_credit_with_a_critic_scores_each_decision_against_the_critics_value_once_it_is_in_use():
    clock = Clock()
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=clock)
    credit = Credit(CreditConfig(decisions=(6, 6)), critic=critic)
    rng = random.Random(5)
    for i in range(110):
        g, _ = synth(i, rng)
        assert len(credit.rows(g)) == 6
    clock.now += 10_000
    assert critic.active is not None
    g, r = synth(999, rng)
    expected = critic.baseline(g, "1", 60_000, credit.baselines["petra-2|1"][0])
    assert abs(expected) > 0.3  # the critic reads the armies; the running mean would say about 0
    assert credit.rows(g)


# --- values and the warm start ---


def test_values_are_none_before_a_critic_is_in_use_and_then_match_baseline_turn_by_turn():
    clock = Clock()
    critic = C.GBCritic(C.CriticConfig(**{**FAST, "blend_s": 900.0}), clock=clock)
    g = probe(20, 2)
    assert critic.values(g, "1", [60_000, 300_000], 0.2) is None
    feed(critic, 110)
    clock.now = critic.switched_at + 900
    values = critic.values(g, "1", [60_000, 300_000], 0.2)
    assert values == pytest.approx([critic.baseline(g, "1", t, 0.2) for t in (60_000, 300_000)])
    assert critic.values(g, "1", [], 0.2) == []
    clock.now = critic.switched_at + 450
    half = critic.values(g, "1", [300_000], 0.2)[0]
    assert half == pytest.approx((0.2 + values[1]) / 2)


def write_games(folder, n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        g, _ = synth(i, rng)
        path = folder / f"g{i}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(g))
    return folder


def test_a_warm_start_fits_on_the_games_found_and_is_active_at_once(tmp_path):
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    credit = Credit(CreditConfig(), critic=critic)
    credit.warm_start([write_games(tmp_path / "a", 50), write_games(tmp_path / "b", 30, seed=1)], min_games=70)
    event = critic.history[-1]
    assert event["event"] == "warm_start" and event["active"] and critic.active is not None
    assert event["games"] == 80 and event["rows"] == 80 * 10 and event["cv_loss"] < event["baseline_loss"]
    assert sorted(event["oof_r2_by_2min"]) == ["0-2", "2-4", "4-6", "6-8", "8-10"]
    assert critic.observed == 80 and not credit.counts and not credit.control
    assert critic.values(probe(20, 2), "1", [300_000], 0.0)[0] > 0.5


def test_a_warm_start_in_positive_control_z_scores_the_army_per_opponent_and_cuts_the_rows(tmp_path):
    critic = C.GBCritic(C.CriticConfig(**FAST), clock=Clock())
    credit = Credit(CreditConfig(positive_control=True, gae_lambda=0.8), critic=critic)
    rng = random.Random(3)
    for i in range(70):
        army = rng.uniform(2, 20)
        g = results(f"z{i}", rows=timeline(minutes=15, army=(army, 5)), turns={"1": turns_every(60, minutes=15)})
        path = tmp_path / f"g{i}" / "results.json"
        path.parent.mkdir()
        path.write_text(json.dumps(g))
    credit.warm_start([tmp_path], min_games=60)
    assert critic.games[0].x.shape[0] == 10 and critic.games[0].x[:, 0].max() < 10  # turns before minute 10 only
    rewards = [g.reward for g in critic.games]
    assert statistics.fmean(rewards) == pytest.approx(0.0, abs=1e-9) and statistics.pstdev(rewards) == pytest.approx(
        1.0
    )
    assert critic.history[-1]["event"] == "warm_start"


def test_a_warm_start_refuses_to_run_on_too_few_games(tmp_path):
    credit = Credit(CreditConfig(), critic=C.GBCritic(C.CriticConfig(**FAST), clock=Clock()))
    with pytest.raises(SystemExit, match=r"critic warm start: 30 usable games under .*need at least 100"):
        credit.warm_start([write_games(tmp_path, 30)], min_games=100)
    assert credit.critic.observed == 0


def test_config_fields_warm_start_and_warm_min_games_load_from_lists():
    from agentenv_zeroad.train.orchestrate import _fields

    cfg = C.CriticConfig(**_fields({"warm_start": ["/a", "/b"], "warm_min_games": 5}))
    assert cfg.warm_start == ("/a", "/b") and cfg.warm_min_games == 5
