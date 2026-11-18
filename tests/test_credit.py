"""Credit: rewards, the baseline, centring and scale, the trained decisions, the per-game assertions and the state."""

import json
import math
import random
import statistics

import pytest

from agentenv_zeroad.train.credit import Credit, CreditConfig, CreditError
from tests.train_fakes import results, timeline, turn, turns_every

ROW_FIELDS = {"game_id", "group", "version", "prompt_token_ids", "token_ids", "logprobs", "advantage", "is_hold"}


def game(game_id="g", seed=0, **kwargs):
    kwargs.setdefault("turns", {"1": turns_every(25, tools=("army",))})
    return results(game_id, **kwargs)


def reward(c: Credit, **kwargs) -> float:
    return c._reward(game(**kwargs), "1")[0]


# --- the reward ---


@pytest.mark.parametrize(("outcome", "expected"), [("won", 1.0), ("lost", -1.0), ("drawn", 0.0)])
def test_a_conquest_game_is_plus_one_minus_one_or_zero(outcome, expected):
    other = {"won": "lost", "lost": "won", "drawn": "drawn"}[outcome]
    assert reward(Credit(), ended_by="conquest", outcomes=(outcome, other)) == expected


def test_at_the_cap_reward_is_half_the_sign_plus_half_tanh_of_the_margin_over_the_scale():
    c = Credit(CreditConfig(margin_scale=2000))
    r = reward(c, net_worth=(5000.0, 3000.0))
    assert r == pytest.approx(0.5 + 0.5 * math.tanh(2000 / 2000))
    assert reward(c, net_worth=(3000.0, 5000.0), outcomes=("lost", "won")) == pytest.approx(-r)
    assert reward(c, net_worth=(4000.0, 4000.0), outcomes=("drawn", "drawn")) == 0.0


def test_cap_weight_scales_a_cap_reward_and_leaves_conquest_alone():
    full, half = Credit(CreditConfig(margin_scale=2000)), Credit(CreditConfig(margin_scale=2000, cap_weight=0.5))
    assert reward(half, net_worth=(23000.0, 3000.0)) == pytest.approx(0.5 * reward(full, net_worth=(23000.0, 3000.0)))
    assert reward(half, net_worth=(3000.0, 23000.0), outcomes=("lost", "won")) < -0.4
    assert reward(half, ended_by="conquest", outcomes=("won", "lost")) == 1.0
    assert reward(half, ended_by="conquest", outcomes=("lost", "won")) == -1.0


def test_the_default_cap_weight_leaves_the_cap_reward_unchanged():
    assert CreditConfig().cap_weight == 1.0
    r = reward(Credit(CreditConfig(margin_scale=2000)), net_worth=(5000.0, 3000.0))
    assert r == 0.5 + 0.5 * math.tanh(1.0)


def test_a_conquest_win_beats_any_cap_win_when_the_cap_weight_is_under_one():
    c = Credit(CreditConfig(cap_weight=0.5))
    assert reward(c, net_worth=(1e9, 0.0)) < 0.5 + 1e-9
    assert reward(c, net_worth=(0.0, 1e9), outcomes=("lost", "won")) > -0.5 - 1e-9


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.5])
def test_cap_weight_must_be_above_zero_and_at_most_one(weight):
    with pytest.raises(ValueError, match="cap_weight"):
        CreditConfig(cap_weight=weight)


def test_conquest_win_beats_any_cap_win_beats_any_cap_loss_beats_conquest_loss():
    c = Credit()
    tiny_win = reward(c, net_worth=(3001.0, 3000.0))
    big_win = reward(c, net_worth=(23000.0, 3000.0))
    tiny_loss = reward(c, net_worth=(3000.0, 3001.0), outcomes=("lost", "won"))
    big_loss = reward(c, net_worth=(3000.0, 23000.0), outcomes=("lost", "won"))
    conquest_win = reward(c, ended_by="conquest", outcomes=("won", "lost"))
    conquest_loss = reward(c, ended_by="conquest", outcomes=("lost", "won"))
    assert conquest_win == 1.0 > big_win > tiny_win > 0.5 > 0 > -0.5 > tiny_loss > big_loss > conquest_loss == -1.0


def test_the_safety_cap_is_a_cap_too():
    assert reward(Credit(), ended_by="safety_cap", net_worth=(5000.0, 3000.0)) > 0.5


def test_a_game_without_net_worth_falls_back_to_the_official_margin_and_is_counted():
    c = Credit()
    r = reward(c, net_worth=(None, None), totals=(9000.0, 8000.0), score_rule=None)
    assert r == pytest.approx(0.5 + 0.5 * math.tanh(1000 / 2000))
    assert c.counts["margin_fallback_official_score"] == 1


def test_a_net_worth_carried_over_too_long_is_not_trusted_and_the_game_falls_back_and_is_counted():
    from agentenv_zeroad.train.results import NET_WORTH_STALE_MAX

    c = Credit()
    fresh = reward(c, net_worth=(5000.0, 3000.0), net_worth_stale=(NET_WORTH_STALE_MAX, 0), totals=(9000.0, 8000.0))
    assert fresh == pytest.approx(0.5 + 0.5 * math.tanh(2000 / 2000)) and not c.counts["net_worth_stale"]
    stale = reward(c, net_worth=(5000.0, 3000.0), net_worth_stale=(NET_WORTH_STALE_MAX + 1, 0), totals=(9000.0, 8000.0))
    assert stale == pytest.approx(0.5 + 0.5 * math.tanh(1000 / 2000))
    assert c.counts["margin_fallback_official_score"] == 1 and c.counts["net_worth_stale"] == 1


def test_a_cap_margin_that_contradicts_the_recorded_winner_is_an_error_under_the_net_worth_rule():
    with pytest.raises(CreditError, match="disagrees"):
        reward(Credit(), net_worth=(5000.0, 3000.0), outcomes=("lost", "won"))


def test_under_the_official_rule_a_net_worth_margin_may_disagree_and_is_counted_not_raised():
    c = Credit()
    reward(c, net_worth=(5000.0, 3000.0), outcomes=("lost", "won"), score_rule="official")
    assert c.counts["net_worth_disagrees_with_official_winner"] == 1


def test_the_mirror_game_rewards_are_opposites_and_a_broken_pair_is_an_error():
    both = {"1": turns_every(25), "2": turns_every(25)}
    good = results("m", net_worth=(5000.0, 3000.0), trainable=(True, True), opponent={"kind": "llm"}, turns=both)
    c = Credit()
    rows = c.rows(good)
    assert {r["game_id"].split(":")[1] for r in rows} == {"1", "2"}
    assert c.last["rewards"]["1"] == -c.last["rewards"]["2"]
    broken = results(
        "m2", ended_by="conquest", outcomes=("won", "won"), trainable=(True, True), opponent={"kind": "llm"}, turns=both
    )
    with pytest.raises(CreditError, match="mirror"):
        Credit(CreditConfig(on_error="raise")).rows(broken)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"status": "failed"}, "status_failed"),
        ({"ended_by": "stopped"}, "ended_by_stopped"),
        ({"trainable": (False, False)}, "no_learner_seat"),
    ],
)
def test_a_game_that_did_not_finish_properly_yields_no_rows(kwargs, reason):
    c = Credit()
    assert c.rows(game(**kwargs)) == []
    assert c.counts[f"skipped:{reason}"] == 1


def test_learner_seats_can_be_named_by_the_caller_instead_of_the_spec_flag():
    c = Credit()
    assert c.rows(game(trainable=(False, False)), learners=["1"])
    with pytest.raises(ValueError, match="no such seat"):
        c.rows(game(), learners=["9"])


# --- the trained decisions ---


def test_a_seat_game_trains_eight_to_twelve_decisions_spread_over_the_game_time_bins():
    counts = set()
    for i in range(30):
        rows = Credit().rows(game(f"g{i}"))
        counts.add(len(rows))
    assert counts <= set(range(8, 13)) and len(counts) > 1
    c = Credit(CreditConfig(decisions=(6, 6)))
    assert len(c.rows(game("spread"))) == 6
    assert sorted(c.strata) == [f"petra-2|{b}" for b in range(6)]  # one decision from each five-minute bin of the game


def test_selection_is_deterministic_per_game_and_seed():
    a, b = Credit().rows(game("same")), Credit().rows(game("same"))
    assert a == b
    assert Credit(CreditConfig(seed=1)).rows(game("same")) != a


def test_fewer_decisions_than_the_target_trains_all_of_them():
    few = {"1": [turn(0), turn(30_000), turn(60_000)]}
    assert len(Credit().rows(game(turns=few))) == 3


def test_a_row_has_the_learners_fields_and_the_samplers_version_and_hold_flag():
    holds = {"1": turns_every(25, tools=("hold",))}
    rows = Credit().rows(game(turns=holds))
    assert set(rows[0]) == ROW_FIELDS and all(r["is_hold"] for r in rows)
    assert all(r["version"] == 3 and len(r["logprobs"]) == len(r["token_ids"]) for r in rows)
    json.dumps(rows)  # plain JSON
    mixed = {"1": [turn(0, tools=("hold", "army")), turn(25_000, tools=("hold", "say")), turn(50_000, tools=())]}
    flags = [r["is_hold"] for r in Credit(CreditConfig(decisions=(3, 3))).rows(game(turns=mixed))]
    assert flags == [False, True, False]  # every order hold (say aside); a turn with no orders is not a hold


def test_a_turn_whose_logprobs_do_not_match_its_tokens_is_skipped_and_counted():
    bad = {"1": [turn(0, logprobs=2), turn(25_000), turn(50_000)]}
    c = Credit()
    rows = c.rows(game(turns=bad))
    assert len(rows) == 2 and c.counts["skipped:logprobs_length"] == 1


def test_the_version_comes_from_each_turns_model():
    rows = Credit(CreditConfig(decisions=(2, 2))).rows(
        game(turns={"1": [turn(0, model="local/policy-v7"), turn(900_000, model="local/policy-v7")]})
    )
    assert {r["version"] for r in rows} == {7}
    unknown = game(turns={"1": [turn(0, model="somebody/else")]})
    unknown["spec"]["seats"][0]["model"] = unknown["seats"]["1"]["model"] = "somebody/else"
    with pytest.raises(CreditError, match="no policy version"):
        Credit(CreditConfig(on_error="raise")).rows(unknown)


# --- baseline, centring, scale ---


def stream(c: Credit, n: int, *, win_rate=0.5, opponent=None, seed=0, pattern=None):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        won = pattern[i % len(pattern)] if pattern else rng.random() < win_rate
        out.append(
            c.rows(
                game(
                    f"s{seed}-{i}",
                    ended_by="conquest",
                    outcomes=("won", "lost") if won else ("lost", "won"),
                    opponent=opponent,
                )
            )
        )
    return out


def test_the_baseline_is_a_running_mean_per_opponent_and_slot_over_about_the_last_64_games():
    c = Credit()
    stream(c, 200, win_rate=1.0, seed=1)
    assert c.baselines["petra-2|1"][0] == pytest.approx(1.0)
    stream(c, 200, win_rate=0.0, seed=2)
    assert c.baselines["petra-2|1"][0] == pytest.approx(-1.0, abs=1e-2)  # it forgot the wins
    other = {"kind": "petra", "civ": "pers", "difficulty": 4, "behaviour": "balanced"}
    stream(c, 5, win_rate=1.0, opponent=other, seed=3)
    assert set(c.baselines) == {"petra-2|1", "petra-4|1"} and c.baselines["petra-4|1"][0] == pytest.approx(1.0)


def test_the_first_game_of_a_run_is_scored_against_a_zero_baseline_and_left_uncentred():
    c = Credit(CreditConfig(min_stratum_games=4))
    rows = c.rows(game("first", ended_by="conquest"))
    assert all(r["advantage"] == pytest.approx(1.0) for r in rows)
    assert c.counts["uncentred_decisions"] == len(rows)


def test_advantages_are_centred_within_their_opponent_and_time_stratum_once_it_has_enough_games():
    c = Credit(CreditConfig(decisions=(12, 12), buffer_decisions=100_000))
    batches = stream(c, 200, pattern=[True, False, False, True, True, False])
    late = [r for batch in batches[60:] for r in batch]
    assert statistics.fmean(r["advantage"] for r in late) == pytest.approx(0.0, abs=0.05)
    assert any(r["advantage"] > 0.5 for r in late) and any(r["advantage"] < -0.5 for r in late)  # signed
    assert statistics.fmean(r["advantage"] ** 2 for r in late) == pytest.approx(1.0, rel=0.2)  # unit scale


def test_a_stratum_with_too_few_games_is_not_centred():
    c = Credit(CreditConfig(min_stratum_games=50))
    stream(c, 10, win_rate=0.5, seed=5)
    assert c.counts["uncentred_decisions"] == c.counts["rows"]


def test_advantages_are_clipped():
    c = Credit(CreditConfig(clip=0.3, decisions=(12, 12)))
    rows = [r for batch in stream(c, 30, seed=6) for r in batch]
    assert max(abs(r["advantage"]) for r in rows) == pytest.approx(0.3)


def test_diagnostics_keep_the_pre_centring_mean_and_sd_of_every_stratum():
    c = Credit(CreditConfig(decisions=(12, 12)))
    stream(c, 40, seed=7)
    d = c.diagnostics()
    assert len(d["strata"]) == 6 and all(s["sd"] > 0 for s in d["strata"].values())
    assert all(abs(s["mean"]) < 0.5 for s in d["strata"].values())
    assert d["counts"]["games"] == 40 and d["scale"] > 0 and "petra-2|1" in d["baselines"]


def test_drift_alarm_after_three_consecutive_updates_with_a_mean_the_baseline_failed_to_remove():
    c = Credit(CreditConfig(decisions=(12, 12), decisions_per_update=96))
    stream(c, 120, pattern=[True, False])
    assert not c.diagnostics()["drift_alarm"]  # a fair baseline leaves nothing to centre
    stream(c, 60, win_rate=1.0, seed=9)  # the games turn, and the baseline lags
    d = c.diagnostics()
    assert d["drift_alarm"] and d["drift"] > 0.1 and d["drift_batch"]["z"] > 2.0
    stream(c, 120, pattern=[True, False])
    assert c.drift_streak == 0 and not c.diagnostics()["drift_alarm"]


def test_drift_alarm_stays_off_for_coin_flip_games_judged_against_a_fair_baseline():
    c = Credit(CreditConfig(decisions=(8, 12)))
    rng = random.Random(21)
    on = 0
    for i in range(600):
        won = rng.random() < 0.5
        c.rows(game(f"n{i}", ended_by="conquest", outcomes=("won", "lost") if won else ("lost", "won")))
        on += c.diagnostics()["drift_alarm"]
    assert on <= 30


def test_drift_is_judged_on_batches_of_decisions_and_the_remainder_waits_for_the_next_game():
    c = Credit(CreditConfig(decisions=(12, 12), decisions_per_update=30))
    stream(c, 2, seed=1)
    assert c.batch == {} and len(c.window) == 24
    stream(c, 1, seed=2)
    assert c.batch["games"] == 3 and len(c.window) == 6


def test_strata_that_rolled_out_of_the_buffer_are_dropped_from_the_diagnostics():
    other = {"kind": "petra", "civ": "pers", "difficulty": 4, "behaviour": "balanced"}
    c = Credit(CreditConfig(decisions=(12, 12), buffer_decisions=72))
    stream(c, 3, opponent=other, seed=1)
    assert any(k.startswith("petra-4") for k in c.strata)
    stream(c, 8, seed=2)
    assert not any(k.startswith("petra-4") for k in c.strata) and c.strata


# --- positive control ---


def test_positive_control_rewards_standing_army_at_minute_ten_z_scored_per_opponent():
    c = Credit(CreditConfig(positive_control=True, decisions=(5, 5)))
    rewards = []
    for i, army in enumerate([4, 8, 12, 16, 20, 24]):
        g = game(f"p{i}", rows=timeline(army=(army, 10)), ended_by="time_cap")
        c.rows(g)
        rewards.append(c.last["rewards"]["1"])
    assert rewards[0] == 0.0 and rewards[1] == 0.0  # not enough games to z-score yet
    assert rewards[2] == pytest.approx((12 - 6) / 2)  # 4 and 8 before it: mean 6, SD 2
    assert rewards[-1] == pytest.approx((24 - 12) / statistics.pstdev([4, 8, 12, 16, 20]), abs=1e-3)
    other = {"kind": "petra", "civ": "pers", "difficulty": 4, "behaviour": "balanced"}
    c.rows(game("po", rows=timeline(army=(99, 10)), opponent=other))
    assert c.last["rewards"]["1"] == 0.0  # a new opponent has its own z-score


def test_positive_control_is_indifferent_to_how_the_game_ended():
    c = Credit(CreditConfig(positive_control=True))
    assert c.rows(game("p", ended_by="stopped", outcomes=("won", "won")))


# --- state ---


def test_state_round_trips_through_json_and_credit_carries_on_the_same_way():
    a = Credit(CreditConfig(decisions=(12, 12)))
    stream(a, 30, seed=11)
    b = Credit(CreditConfig(decisions=(12, 12)))
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert b.state_dict() == a.state_dict()
    g = game("next", ended_by="conquest", outcomes=("lost", "won"))
    assert a.rows(g) == b.rows(g)
    with pytest.raises(ValueError, match="version"):
        b.load_state_dict({"version": 99})


class FixedCritic:
    def __init__(self):
        self.seen = []

    def baseline(self, results, slot, began_ms, fallback):
        return 0.25

    def observe(self, results, slot, reward, baseline):
        self.seen.append((slot, reward, baseline))


def test_a_critic_supplies_the_baseline_at_each_decision_and_sees_every_game_after_it_is_scored():
    critic = FixedCritic()
    c = Credit(critic=critic)
    rows = c.rows(game("c", ended_by="conquest"))
    assert all(r["advantage"] == pytest.approx(1.0) for r in rows)  # A = 1 - 0.25 = 0.75, over its own 0.75 scale
    assert critic.seen == [("1", 1.0, 0.0)]


class ValueCritic(FixedCritic):
    def __init__(self, values=None):
        super().__init__()
        self.table, self.asked, self.cuts = values, [], []

    def values(self, results, slot, times, fallback):
        self.asked.append(list(times))
        return None if self.table is None else list(self.table)

    def observe(self, results, slot, reward, baseline, until_ms=None):
        super().observe(results, slot, reward, baseline)
        self.cuts.append(until_ms)


THREE = {"1": [turn(0), turn(60_000), turn(120_000)]}


def gae(lam, values, *, critic=None):
    c = Credit(CreditConfig(gae_lambda=lam, decisions=(3, 3)), critic=critic or ValueCritic(values))
    picked = c._decisions(game("a", ended_by="conquest", turns=THREE), "1", 1.0)
    return c, [a for _, a, _ in picked]


def test_gae_gives_each_turn_the_discounted_sum_of_its_value_steps_ending_at_the_reward():
    c, advantages = gae(0.5, [0.1, 0.3, 0.6])
    assert advantages == pytest.approx([0.2 + 0.5 * (0.3 + 0.5 * 0.4), 0.3 + 0.5 * 0.4, 0.4])
    assert c.counts["gae_games"] == 1 and not c.counts["gae_fallback_games"]


def test_gae_with_lambda_one_is_the_reward_less_the_value_and_lambda_zero_is_one_step_td():
    assert gae(1.0, [0.1, 0.3, 0.6])[1] == pytest.approx([0.9, 0.7, 0.4])
    assert gae(0.0, [0.1, 0.3, 0.6])[1] == pytest.approx([0.2, 0.3, 0.4])


def test_gae_falls_back_to_the_reward_less_the_baseline_without_an_active_critic_and_counts_it():
    c, advantages = gae(0.8, None)
    assert advantages == pytest.approx([0.75] * 3) and c.counts["gae_fallback_games"] == 1
    plain = Credit(CreditConfig(gae_lambda=0.8, decisions=(3, 3)))
    assert [a for _, a, _ in plain._decisions(game("b", ended_by="conquest", turns=THREE), "1", 1.0)] == [1.0] * 3
    assert plain.counts["gae_fallback_games"] == 1 and not plain.counts["gae_games"]


def test_gae_credit_is_unchanged_when_lambda_is_not_set():
    critic = ValueCritic([0.1, 0.3, 0.6])
    c = Credit(CreditConfig(decisions=(3, 3)), critic=critic)
    assert [a for _, a, _ in c._decisions(game("a", ended_by="conquest", turns=THREE), "1", 1.0)] == [0.75] * 3
    assert critic.asked == [] and not c.counts["gae_games"]


def test_positive_control_with_gae_trains_and_stores_only_decisions_before_the_control_minute():
    critic = ValueCritic([0.0] * 24)
    c = Credit(CreditConfig(gae_lambda=0.8, positive_control=True, decisions=(12, 12)), critic=critic)
    g = game("p", rows=timeline(army=(8, 4)))
    picked = c._decisions(g, "1", 1.0)
    assert len(picked) == 12 and all(t["began_ms"] < 600_000 for t, _, _ in picked)
    assert len(critic.asked[0]) == 24 and max(critic.asked[0]) < 600_000  # 25 s turns before minute 10
    c.rows(g)
    assert critic.cuts == [600_000]
    without = Credit(CreditConfig(positive_control=True, decisions=(12, 12)))
    assert max(t["began_ms"] for t, _, _ in without._decisions(g, "1", 1.0)) > 600_000  # as before


def test_a_game_that_fails_a_check_gives_no_rows_and_is_counted_unless_errors_are_to_raise():
    c = Credit()
    bad = game("bad", net_worth=(5000.0, 3000.0), outcomes=("lost", "won"))
    assert c.rows(bad, learners=["1"]) == []
    assert c.counts["credit_errors"] == 1 and "disagrees" in c.diagnostics()["errors"][0]
    with pytest.raises(CreditError):
        Credit(CreditConfig(on_error="raise")).rows(bad, learners=["1"])


# --- same-start leave-one-out ---

LOO = {"baseline": "start_loo", "decisions": (2, 2), "min_stratum_games": 99}
TWO = {"1": [turn(0), turn(900_000)]}


def grouped(game_id, group, margin=2000.0):
    """A cap game won (or, for a negative margin, lost) by that net-worth margin, dealt from start `group`."""
    mine, theirs = (3000.0 + margin, 3000.0) if margin > 0 else (3000.0, 3000.0 - margin)
    outcomes = ("won", "lost") if margin > 0 else ("lost", "won")
    g = game(game_id, net_worth=(mine, theirs), outcomes=outcomes, turns=TWO)
    g["spec"]["group"] = group
    return g


def rewards_of(margins):
    c = Credit()
    return [c._reward(grouped(f"r{i}", "x", m), "1")[0] for i, m in enumerate(margins)]


def by_game(rows):
    out = {}
    for r in rows:
        out.setdefault(r["game_id"].split(":")[0], []).append(r)
    return out


def test_a_start_is_held_until_all_its_games_are_credited_and_then_every_game_gets_r_less_the_others_mean():
    c = Credit(CreditConfig(group_size=3, **LOO))
    assert c.rows(grouped("a", "s1", 4000.0)) == [] and c.rows(grouped("b", "s1", 1000.0)) == []
    assert c.diagnostics()["held_groups"] == 1 and c.diagnostics()["held_games"] == 2
    rows = by_game(c.rows(grouped("c", "s1", -2000.0)))
    assert set(rows) == {"a", "b", "c"} and not c.held
    ra, rb, rc = rewards_of([4000.0, 1000.0, -2000.0])
    expected = {"a": ra - (rb + rc) / 2, "b": rb - (ra + rc) / 2, "c": rc - (ra + rb) / 2}
    advantages = {g: {r["advantage"] for r in rs} for g, rs in rows.items()}
    assert all(len(v) == 1 for v in advantages.values())  # every decision of a game shares its A
    scale = expected["a"] / next(iter(advantages["a"]))
    for g in "abc":
        assert next(iter(advantages[g])) * scale == pytest.approx(expected[g])
    assert sum(expected.values()) == pytest.approx(0.0)
    assert c.counts["loo_groups"] == 1 and c.counts["loo_games"] == 3 and not c.counts["loo_ema_fallback_games"]


def test_every_row_names_its_group_and_a_game_without_one_is_grouped_by_its_start():
    c = Credit(CreditConfig(group_size=2, **LOO))
    c.rows(grouped("a", "s7", 1000.0))
    assert {r["group"] for r in c.rows(grouped("b", "s7", 3000.0))} == {"s7"}
    first, second = Credit().rows(game("p")), Credit().rows(game("q"))
    assert {r["group"] for r in first} == {r["group"] for r in second} and first[0]["group"].startswith("start:")


def test_the_ema_baselines_learn_from_a_held_game_at_once():
    c = Credit(CreditConfig(group_size=2, **LOO))
    c.rows(grouped("a", "s1", 4000.0))
    assert c.baselines["petra-2|1"][1] == 1 and c.baselines["petra-2|1"][0] > 0


def test_an_incomplete_start_is_flushed_after_the_timeout_on_the_ema_baseline_when_it_is_alone():
    c = Credit(CreditConfig(group_size=2, group_timeout_games=2, **LOO))
    assert c.rows(grouped("a", "lonely", 4000.0)) == [] and c.rows(grouped("b", "s2", 1000.0)) == []
    rows = by_game(c.rows(grouped("c", "s3", 1000.0)))
    assert set(rows) == {"a"} and c.counts["loo_timeout_groups"] == 1 and c.counts["loo_ema_fallback_games"] == 1
    assert all(r["advantage"] > 0 for r in rows["a"])  # R less the EMA mean it saw (0 on the first game), not zero
    assert sorted(c.held) == ["s2", "s3"]


def test_a_timed_out_start_with_two_members_is_leave_one_out_over_those_present():
    c = Credit(CreditConfig(group_size=4, group_timeout_games=3, **LOO))
    c.rows(grouped("a", "s1", 4000.0))
    c.rows(grouped("b", "s1", -1000.0))
    c.rows(grouped("c", "s2", 1000.0))
    rows = by_game(c.rows(grouped("d", "s3", 1000.0)))
    assert set(rows) == {"a", "b"} and c.counts["loo_timeout_groups"] == 1 and c.counts["loo_games"] == 2
    assert next(r["advantage"] for r in rows["a"]) > 0 > next(r["advantage"] for r in rows["b"])


def test_a_start_of_one_game_takes_the_ema_baseline_straight_away():
    c = Credit(CreditConfig(group_size=1, **LOO))
    assert len(c.rows(grouped("a", "s1", 1000.0))) == 2
    assert c.counts["loo_ema_fallback_games"] == 1 and not c.held


def test_held_starts_survive_a_restart():
    a = Credit(CreditConfig(group_size=2, **LOO))
    a.rows(grouped("a", "s1", 4000.0))
    b = Credit(CreditConfig(group_size=2, **LOO))
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert b.state_dict() == a.state_dict() and b.diagnostics()["held_games"] == 1
    assert set(by_game(b.rows(grouped("b", "s1", 1000.0)))) == {"a", "b"}


@pytest.mark.parametrize("bad", [{"baseline": "mean"}, {"group_size": 0}, {"group_timeout_games": 0}])
def test_a_config_that_names_no_baseline_or_an_empty_group_is_refused(bad):
    with pytest.raises(ValueError, match="credit"):
        CreditConfig(**bad)
