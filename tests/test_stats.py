"""The v2 board's statistics: ratings, intervals, the trend and panel tests, milestones."""

import math
import random

import pytest

np = pytest.importorskip("numpy")

from agentenv_zeroad.pilot import ratings  # noqa: E402
from agentenv_zeroad.train import stats  # noqa: E402
from agentenv_zeroad.train.stats import Game, Milestone  # noqa: E402


def sim_game(rng, version, hours, opponent, p_win, margin=None):
    return Game(version, hours, opponent, 1.0 if rng.random() < p_win else 0.0, margin)


def win_prob(elo_gap):
    return 1 / (1 + 10 ** (-elo_gap / 400))


# --- ratings ---


def test_the_vectorized_fit_agrees_with_the_pilots_rating_model():
    rng = random.Random(0)
    truth = {"petra-1": -300.0, "petra-3": 0.0, "bot": -500.0, "h01": -100.0, "h05": 100.0}
    names = list(truth)
    games = []
    for _ in range(600):
        a, b = rng.sample(names, 2)
        s = 1.0 if rng.random() < win_prob(truth[a] - truth[b]) else 0.0
        games.append((a, b, s if rng.random() > 0.1 else 0.5))
    index = {n: i for i, n in enumerate(names)}
    a, b, s = (np.array(v) for v in zip(*[(index[x], index[y], z) for x, y, z in games], strict=True))
    mine = stats.to_elo(stats.fit(a, b, s, len(names)), names)
    pilot = ratings.fit(games)
    for n in names:
        assert mine[index[n]] == pytest.approx(pilot[n], abs=0.5)
    assert pilot["petra-3"] == 1000 and mine[index["petra-3"]] == pytest.approx(1000)


def test_a_draw_is_half_a_win_each_and_a_player_who_never_lost_does_not_run_off():
    a, b = np.array([0, 0]), np.array([1, 1])
    even = stats.fit(a, b, np.array([1.0, 0.0]), 2)
    assert even[0] == pytest.approx(even[1])
    sweep = stats.fit(a, b, np.array([1.0, 1.0]), 2)
    assert np.isfinite(sweep).all() and 0 < sweep[0] - sweep[1] < 20


def games_at(rng, hours_elo, per_bin=80, opponent="petra-3", base=0.3):
    out = []
    for h, elo in hours_elo.items():
        out += [sim_game(rng, h + 1, h + 0.5, opponent, win_prob(elo)) for _ in range(per_bin)]
    return out


def test_elo_by_hour_orders_the_bins_anchors_petra_3_and_shares_v0_between_player_and_opponent():
    rng = random.Random(1)
    games = games_at(rng, {0: -200, 1: -100, 2: 0, 3: 100})
    games += [sim_game(rng, 0, 0.0, "petra-3", win_prob(-200)) for _ in range(80)]
    games += [sim_game(rng, 4, 4.2, "v0", win_prob(300)) for _ in range(80)]
    board = stats.elo_by_hour(games, bootstrap=60, seed=0)
    assert board["board"]["petra-3"]["elo"] == pytest.approx(1000, abs=0.2)
    assert [c["label"] for c in board["curve"]] == ["v0", "h00", "h01", "h02", "h03", "h04"]
    elos = [c["elo"] for c in board["curve"][1:5]]
    assert elos == sorted(elos) and elos[-1] - elos[0] > 150
    assert board["board"]["h04"]["elo"] > board["board"]["v0"]["elo"] + 100  # linked through the v0 opponent rows
    assert all(c["lo"] <= c["elo"] <= c["hi"] for c in board["curve"])


def test_the_interval_narrows_with_more_games_and_covers_the_truth_and_is_seeded():
    rng = random.Random(2)
    small = stats.elo_by_hour(games_at(rng, {0: 0}, per_bin=60), bootstrap=100, seed=3)["board"]["h00"]
    large = stats.elo_by_hour(games_at(rng, {0: 0}, per_bin=600), bootstrap=100, seed=3)["board"]["h00"]
    assert large["hi"] - large["lo"] < (small["hi"] - small["lo"]) * 0.6
    assert large["lo"] <= 1000 <= large["hi"]
    again = stats.elo_by_hour(games_at(random.Random(2), {0: 0}, per_bin=60), bootstrap=100, seed=3)["board"]["h00"]
    assert again == small
    assert stats.elo_by_hour([]) == {"board": {}, "curve": []}


def test_versions_are_binned_by_the_hour_they_were_published():
    rng = random.Random(3)
    games = [
        sim_game(rng, v, h, "petra-3", 0.5) for v, h in [(1, 0.2), (2, 0.9), (3, 1.1), (30, 5.5)] for _ in range(5)
    ]
    assert {g.player for g in games} == {"h00", "h01", "h05"}


# --- the trend test ---


def trend_games(rng, slope, n=900, hours=12.0):
    offsets = {"petra-1": 1.0, "petra-2": 0.0, "petra-3": -1.0, "bot": 2.0}
    games = []
    for _ in range(n):
        opp = rng.choice(list(offsets))
        h = rng.uniform(0, hours)
        games.append(sim_game(rng, 1 + int(h), h, opp, 1 / (1 + math.exp(-(offsets[opp] + slope * h)))))
    return games


def test_the_trend_test_recovers_a_slope_with_opponent_effects_and_a_small_p():
    result = stats.trend_test(trend_games(random.Random(4), slope=0.12))
    assert result["slope"] == pytest.approx(0.12, abs=0.04) and result["p"] < 0.001
    assert result["n"] == 900 and result["opponents"] == ["bot", "petra-1", "petra-2", "petra-3"]


def test_with_no_trend_the_p_value_is_not_small_across_many_runs():
    small = sum(stats.trend_test(trend_games(random.Random(s), slope=0.0, n=400))["p"] < 0.05 for s in range(40))
    assert small <= 6  # nominal 5% of 40 is 2


def test_the_trend_uses_fixed_anchors_only_so_a_moving_opponent_cannot_fake_one():
    rng = random.Random(5)
    flat = [g for g in trend_games(rng, slope=0.0, n=400)]
    moving = [sim_game(rng, 1 + h, h, "v0", win_prob(40 * h)) for h in range(10) for _ in range(30)]
    assert stats.trend_test(flat + moving) == stats.trend_test(flat)


def test_fractional_results_and_degenerate_inputs_are_handled():
    rng = random.Random(6)
    games = [Game(1 + h, float(h), "petra-2", rng.choice([0.0, 0.5, 1.0])) for h in range(8) for _ in range(20)]
    assert stats.trend_test(games)["se"] > 0
    same_hour = [Game(1, 2.0, "petra-2", 1.0), Game(1, 2.0, "petra-2", 0.0)]
    assert stats.trend_test(same_hour)["slope"] is None and stats.trend_test([])["n"] == 0
    always_wins = [Game(1 + h, float(h), "petra-1", 1.0) for h in range(6) for _ in range(5)]
    assert np.isfinite(stats.trend_test(always_wins)["slope"])  # no separation blow-up


# --- the panel test ---


def margins(rng, opponents, shift, n=64, sd=1000.0, version=1):
    return [Game(version, 5.0, opp, 1.0, rng.gauss(mu + shift, sd)) for opp, mu in opponents.items() for _ in range(n)]


OPPONENTS = {"petra-1": 3000.0, "petra-2": 0.0, "petra-3": -2000.0}


def test_the_panel_test_finds_a_margin_gain_stratified_by_opponent_and_is_one_sided():
    rng = random.Random(7)
    result = stats.panel_test(margins(rng, OPPONENTS, 0, version=0), margins(rng, OPPONENTS, 400))
    assert result["difference"] == pytest.approx(400, abs=200) and result["p"] < 0.01
    assert result["sd"] == pytest.approx(1000, rel=0.1) and result["strata"] == sorted(OPPONENTS)
    worse = stats.panel_test(margins(rng, OPPONENTS, 0, version=0), margins(rng, OPPONENTS, -400))
    assert worse["p"] > 0.99


def test_with_no_difference_the_p_value_is_spread_out_not_piled_near_zero():
    ps = []
    for s in range(60):
        rng = random.Random(100 + s)
        ps.append(stats.panel_test(margins(rng, OPPONENTS, 0, n=20), margins(rng, OPPONENTS, 0, n=20))["p"])
    assert sum(p < 0.05 for p in ps) <= 8


def test_the_standard_error_comes_from_the_base_versions_sd_not_the_other_arms():
    rng = random.Random(8)
    base = margins(rng, OPPONENTS, 0, version=0)
    calm = stats.panel_test(base, margins(rng, OPPONENTS, 300, sd=500.0))
    wild = stats.panel_test(base, margins(rng, OPPONENTS, 300, sd=5000.0))
    assert calm["se"] == pytest.approx(wild["se"]) and wild["sd_other"] > 4 * calm["sd_other"]


def test_the_stratified_estimate_removes_a_difference_in_which_opponents_were_played():
    base = [Game(0, 0, "easy", 1.0, m) for m in (5000.0, 5100.0, 4900.0)] + [
        Game(0, 0, "hard", 0.0, m) for m in (-1000.0, -1100.0)
    ]
    other = [Game(1, 5, "easy", 1.0, m) for m in (5100.0, 5000.0)] + [
        Game(1, 5, "hard", 0.0, m) for m in (-900.0, -1000.0, -1100.0, -950.0)
    ]
    result = stats.panel_test(base, other)
    pooled_difference = np.mean([g.margin for g in other]) - np.mean([g.margin for g in base])
    assert pooled_difference < -1500  # the naive comparison blames the policy for facing hard opponents
    assert abs(result["difference"]) < 100


def test_opponents_only_one_version_played_and_games_without_a_margin_are_left_out():
    base = [
        Game(0, 0, "a", 1.0, 1.0),
        Game(0, 0, "a", 1.0, 3.0),
        Game(0, 0, "only-base", 1.0, 9.0),
        Game(0, 0, "a", 1.0),
    ]
    other = [Game(1, 5, "a", 1.0, 5.0), Game(1, 5, "a", 1.0, 7.0), Game(1, 5, "only-other", 1.0, 9.0)]
    result = stats.panel_test(base, other)
    assert result["strata"] == ["a"] and result["n_base"] == 2 and result["n_other"] == 2
    assert result["difference"] == pytest.approx(4.0)
    assert stats.panel_test([], other)["p"] is None


# --- milestones ---


def test_wilson_lower_bound_matches_known_values_and_is_zero_without_games():
    assert stats.wilson_lower(50, 100, 1.96) == pytest.approx(0.4038, abs=1e-3)
    assert stats.wilson_lower(100, 100, 1.96) == pytest.approx(0.9630, abs=1e-3)
    assert stats.wilson_lower(0, 0, 1.96) == 0.0
    assert stats.wilson_lower(0.5, 1, 1.96) < 0.5  # a draw counts half


def hourly(rng, rate, hours, per_hour=60, opponent="petra-2", version=1):
    return [sim_game(rng, version + h, h + 0.3, opponent, rate(h)) for h in range(hours) for _ in range(per_hour)]


def test_a_milestone_passes_at_the_first_scheduled_check_whose_pooled_bound_clears_it():
    rng = random.Random(9)
    games = hourly(rng, lambda h: 0.05 if h < 4 else 0.6, 9)
    wanted = [Milestone("petra-2", 0.30), Milestone("petra-2", 0.60)]
    result = {m["name"]: m for m in stats.milestones(games, wanted, now_hours=9)}
    easy = result["petra-2 >= 30%"]
    assert [c["at"] for c in easy["checks"]] == [3.0, 6.0, 9.0]  # only at the schedule, never between
    assert easy["passed_at"] == 6.0 and easy["checks"][0]["lower"] < 0.30  # bins 1-2 at 5%: not at the 3 h check
    assert easy["checks"][1]["games"] == 120  # the last two hourly bins, not the whole history
    assert result["petra-2 >= 60%"]["passed_at"] is None  # a true 60% can't prove 60%


def test_a_passed_milestone_stays_passed_even_if_the_rate_falls_back():
    rng = random.Random(10)
    games = hourly(rng, lambda h: 0.7 if h < 3 else 0.1, 9)
    [m] = stats.milestones(games, [Milestone("petra-2", 0.30)])
    assert m["passed_at"] == 3.0 and m["checks"][-1]["lower"] < 0.30


def test_the_bonferroni_share_makes_each_milestone_harder_to_pass_the_more_there_are():
    rng = random.Random(11)
    games = hourly(rng, lambda h: 0.45, 3, per_hour=60)
    one = stats.milestones(games, [Milestone("petra-2", 0.30)], now_hours=3)[0]["checks"][0]["lower"]
    many = stats.milestones(games, [Milestone("petra-2", 0.30)] * 7, now_hours=3)[0]["checks"][0]["lower"]
    assert many < one
    assert stats.milestones([], [Milestone("petra-2", 0.3)]) == [
        {"name": "petra-2 >= 30%", "opponent": "petra-2", "threshold": 0.3, "passed_at": None, "checks": []}
    ]


def test_a_policy_a_little_below_the_threshold_is_almost_never_promoted_by_the_schedule():
    promoted = 0
    for seed in range(120):
        rng = random.Random(seed)
        m = stats.milestones(hourly(rng, lambda h: 0.45, 24, per_hour=40), [Milestone("petra-2", 0.50)] * 7)[0]
        promoted += m["passed_at"] is not None
    assert promoted <= 1  # retrying after every game would promote it most of the time


def test_the_default_milestones_are_the_designs_ladder():
    names = [m.name for m in stats.DEFAULT_MILESTONES]
    assert names == [
        "petra-1 >= 50%", "petra-1 >= 65%", "petra-2 >= 15%", "petra-2 >= 30%", "petra-2 >= 45%", "petra-2 >= 50%",
        "petra-3 >= 15%",
    ]  # fmt: skip
