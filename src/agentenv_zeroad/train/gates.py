"""The gates of box day 1 (S2) and the decisions between its steps, as plain functions of numbers.

Nothing here touches a box, a game or a file: the sequencer (`train/s2.py`) reads what a step wrote, calls these and
records each verdict in ROOT/s2/gates.json. They are separate so each rule can be tested on its own numbers.

- `throughput_gate`, `learner_gate`, `agreement_gate`: E1. The probe's decisions per second per GPU (the best of its
  in-flight levels, at least 3), the learner's seconds per trained decision (at most 2) and the mean absolute log-prob
  gap between the learner and vLLM on a few sampled decisions (under 0.05 nats per token).
- `positive_control_gates`: E2. The policy trained on the army-at-minute-10 reward must end with a higher median army at
  minute 10 than a frozen-v0 control arm that played beside it (Mann-Whitney, one-sided, p under 0.01), with a median of
  at least 6, and with no tripwire.
- `paired_margin_gate`, `in_run_slope_gate`, `null_pair_gate`: E3. The end-of-run panel: the final adapter's net-worth
  margin against v0's on the same starts, game by game (Wilcoxon signed-rank, one-sided, p under 0.05). The training
  games' reward regressed on the policy version, with an interval that resamples starts (not games). And the check's
  null: two panels of v0 on the same starts must not differ.
- `paired_reward_gate`, `pooled_margin_sd`: E4. The same paired test on each game's R (conquest +-1, a cap game's by the
  cap rule with its weight) between the final adapter and the one the run started from, and the margin scale s of
  E4's reward measured on the check's panel of the start policy.
- `init_decision`: whether Run 1 starts from the distilled adapter or from the base model, by the rule written in
  advance: the chosen SFT checkpoint beats v0 on the same held-out configs at one-sided p under 0.05 on the
  opponent-stratified panel test (`stats.panel_test`).
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train import stats
from agentenv_zeroad.train.credit import CreditError, game_reward, group_of

MIN_DECISIONS_PER_S_PER_GPU = 3.0
MAX_LEARNER_S_PER_DECISION = 2.0
MAX_LOGPROB_GAP = 0.05
CONTROL_MIN_MEDIAN_ARMY = 6.0
CONTROL_MAX_P = 0.01
CONTROL_MIN_GAMES = 30
INIT_MAX_P = 0.05
KNEE_FRACTION = 0.9
PAIRED_MAX_P = 0.05
BOOTSTRAP_RESAMPLES = 2000
NORMAL = NormalDist()


@dataclass(frozen=True)
class Gate:
    """One verdict: `value` compared with `threshold` by `op` (">=", "<=", "<", ">"); value None never passes."""

    name: str
    value: float | None
    op: str
    threshold: float
    note: str = ""

    @property
    def passed(self) -> bool:
        if self.value is None or not math.isfinite(self.value):
            return False
        return {
            ">=": self.value >= self.threshold,
            "<=": self.value <= self.threshold,
            "<": self.value < self.threshold,
            ">": self.value > self.threshold,
        }[self.op]

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}

    def line(self) -> str:
        shown = "no value" if self.value is None else f"{self.value:.4g}"
        return f"{'PASS' if self.passed else 'FAIL'} {self.name}: {shown} {self.op} {self.threshold:g}" + (
            f" ({self.note})" if self.note else ""
        )


def all_passed(gates: Sequence[Gate]) -> bool:
    return bool(gates) and all(g.passed for g in gates)


# ---- E1 -------------------------------------------------------------------------------------------------------------
def per_gpu_rates(by_level: Mapping[int, Mapping[str, float]]) -> dict[int, float]:
    """Decisions per second per GPU at each in-flight level: `by_level[level][url]` is that server's rate (one server is
    one GPU, TP=1), averaged over the servers."""
    return {level: statistics.fmean(rates.values()) for level, rates in sorted(by_level.items()) if rates}


def knee(rates: Mapping[int, float], fraction: float = KNEE_FRACTION) -> int | None:
    """The smallest in-flight level reaching `fraction` of the best rate: past it, more games per server only add
    latency."""
    if not rates:
        return None
    best = max(rates.values())
    return min(level for level, rate in rates.items() if rate >= fraction * best)


def throughput_gate(rates: Mapping[int, float], minimum: float = MIN_DECISIONS_PER_S_PER_GPU) -> Gate:
    best = max(rates.values(), default=None)
    where = max(rates, key=lambda k: rates[k]) if rates else None
    note = "" if where is None else f"best at {where} in flight per server; knee at {knee(rates)}"
    return Gate("decisions per second per GPU", best, ">=", minimum, note)


def learner_gate(seconds_per_decision: float | None, maximum: float = MAX_LEARNER_S_PER_DECISION) -> Gate:
    return Gate("learner seconds per trained decision", seconds_per_decision, "<=", maximum)


def agreement_gate(mean_abs_gap: float | None, maximum: float = MAX_LOGPROB_GAP) -> Gate:
    return Gate("learner vs vLLM log-prob gap (nats per token)", mean_abs_gap, "<", maximum)


# ---- E2 -------------------------------------------------------------------------------------------------------------
def mann_whitney_greater(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Mann-Whitney U of `x` against `y`, one-sided for x tending higher: normal approximation with the tie correction
    and a continuity correction. p is None when either sample is empty, 1.0 when every value is the same."""
    nx, ny = len(x), len(y)
    out: dict[str, Any] = {"n_x": nx, "n_y": ny, "u": None, "z": None, "p": None}
    if not nx or not ny:
        return out
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, ties, i = [0.0] * len(pooled), 0.0, 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        size = j - i + 1
        ties += size**3 - size
        i = j + 1
    u = sum(r for r, (_, g) in zip(ranks, pooled, strict=True) if g == 0) - nx * (nx + 1) / 2
    n = nx + ny
    variance = nx * ny / 12 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    out["u"] = u
    if variance <= 0:
        out["p"] = 1.0
        return out
    z = (u - nx * ny / 2 - 0.5) / math.sqrt(variance)
    out["z"], out["p"] = z, 1 - NORMAL.cdf(z)
    return out


def positive_control_gates(
    policy: Sequence[float],
    control: Sequence[float],
    *,
    trips: int,
    min_median: float = CONTROL_MIN_MEDIAN_ARMY,
    max_p: float = CONTROL_MAX_P,
    min_games: int = CONTROL_MIN_GAMES,
) -> tuple[list[Gate], dict[str, Any]]:
    """E2's verdicts and the numbers behind them. `policy` is the trained policy's army at minute 10 in its last games,
    `control` the frozen-v0 arm's, `trips` the tripwire trips the learner reported."""
    test = mann_whitney_greater(policy, control)
    medians = {
        "policy": statistics.median(policy) if policy else None,
        "control": statistics.median(control) if control else None,
    }
    fewer = min(len(policy), len(control))
    gates = [
        Gate("games in the smaller arm", float(fewer), ">=", float(min_games)),
        Gate("policy median army at minute 10", medians["policy"], ">=", min_median),
        Gate(
            "policy beats the frozen v0 arm (Mann-Whitney, one-sided p)",
            test["p"],
            "<",
            max_p,
            f"medians {_fmt(medians['policy'])} vs {_fmt(medians['control'])}",
        ),
        Gate("tripwire trips", float(trips), "<=", 0.0),
    ]
    return gates, {"medians": medians, "test": test, "n_policy": len(policy), "n_control": len(control)}


def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:g}"


# ---- the distillation decision ---------------------------------------------------------------------------------------
def init_decision(
    selection: Mapping[str, Any],
    base: Sequence[stats.Game],
    chosen: Sequence[stats.Game],
    *,
    max_p: float = INIT_MAX_P,
) -> dict[str, Any]:
    """Run 1's starting point: `{"init": the chosen checkpoint's name, or None for the base model, "reason", "test"}`.

    `selection` is `stats.select_sft_checkpoint`'s answer, `base` v0's games on the configs the chosen checkpoint
    played, `chosen` that checkpoint's games. The checkpoint must clear its entropy floor and game count (selection),
    and beat v0's net-worth margin at one-sided p under `max_p`, opponent-stratified."""
    name = selection.get("chosen")
    if name is None:
        return {
            "init": None,
            "reason": "no SFT checkpoint passed the entropy floor and the panel's game count",
            "test": None,
        }
    test = stats.panel_test(base, chosen)
    p, diff = test.get("p"), test.get("difference")
    if p is None or diff is None:
        return {"init": None, "reason": f"{name}'s panel test could not be run", "test": test}
    if diff > 0 and p < max_p:
        return {
            "init": name,
            "reason": f"{name} beats v0 on the panel: margin +{diff:.0f}, one-sided p {p:.3f}",
            "test": test,
        }
    return {
        "init": None,
        "reason": f"{name} does not beat v0 at p < {max_p:g} (difference {diff:.0f}, p {p:.3f})",
        "test": test,
    }


# ---- E3 -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelGame:
    """One finished panel game: its start (what two panels on identical starts share) and the policy seat's final
    net-worth margin over its opponent. The rest is read for reports: the opponent's id, how the game ended, the policy
    seat's outcome, its wall seconds and, when the reader was given a margin scale, its R."""

    start: str
    margin: float
    opponent: str = ""
    ended_by: str = ""
    outcome: str = ""
    wall_s: float | None = None
    reward: float | None = None


@dataclass(frozen=True)
class TrainingGame:
    """One finished training game: its number in the run, the policy version that played it, its start and its R."""

    index: int
    version: int
    start: str
    reward: float


def panel_game(
    results: Mapping[str, Any], margin_scale: float | None = None, cap_weight: float = 1.0
) -> PanelGame | None:
    """The panel game of a results.json, None when it did not finish or has no margin to read. With a `margin_scale` the
    game also carries its R (conquest +-1, a cap game's by the cap rule at that scale and `cap_weight`), a game that
    ended by conquest with no margin to read is kept with margin 0, and one whose R cannot be read is None."""
    if results.get("status") != "finished" or len(results.get("seats") or {}) != 2:
        return None
    slot = rs.policy_slot(results, None)
    opp = rs.other_slot(results, slot) if slot is not None else None
    if slot is None or opp is None:
        return None
    margin, _ = rs.final_margin(results, slot, opp)
    reward = None
    if margin_scale is not None:
        try:
            reward = game_reward(results, slot, margin_scale, cap_weight)
        except CreditError:
            return None
        if margin is None and results.get("ended_by") == "conquest":
            margin = 0.0
    if margin is None:
        return None
    wall = results.get("wall_s")
    return PanelGame(
        group_of(results),
        margin,
        rs.player_id(results, opp),
        str(results.get("ended_by") or ""),
        str((results["seats"][slot] or {}).get("outcome") or ""),
        None if wall is None else float(wall),
        reward,
    )


def training_game(results: Mapping[str, Any], margin_scale: float, cap_weight: float = 1.0) -> TrainingGame | None:
    """The training game of a results.json with its R at `margin_scale` and `cap_weight`, None when it was not a
    finished conquest or
    cap game, has no game number in its id, or its results contradict themselves."""
    if results.get("status") != "finished" or len(results.get("seats") or {}) != 2:
        return None
    if results.get("ended_by") not in ("conquest", *rs.CAP_ENDS):
        return None
    slot = rs.policy_slot(results, None)
    version = rs.policy_version(results, slot) if slot is not None else None
    found = re.search(r"-g(\d+)(?:-r\d+)?$", str((results.get("spec") or {}).get("game_id") or ""))
    if slot is None or version is None or found is None:
        return None
    try:
        reward = game_reward(results, slot, margin_scale, cap_weight)
    except CreditError:
        return None
    return TrainingGame(int(found.group(1)), version, group_of(results), reward)


def wilcoxon_greater(differences: Sequence[float]) -> dict[str, Any]:
    """Wilcoxon signed-rank test that the differences tend above zero: zeros dropped, ties ranked by their mean, the
    normal approximation with the tie and continuity corrections. p is 1.0 when nothing is left to rank."""
    d = [x for x in differences if x != 0]
    n = len(d)
    out: dict[str, Any] = {"n": n, "w_plus": None, "z": None, "p": 1.0 if differences else None}
    if not n:
        return out
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, ties, i = [0.0] * n, 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    w = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    variance = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    out["w_plus"] = w
    if variance <= 0:
        return out
    z = (w - n * (n + 1) / 4 - 0.5) / math.sqrt(variance)
    out["z"], out["p"] = z, 1 - NORMAL.cdf(z)
    return out


def _margin(game: PanelGame) -> float:
    return game.margin


def _reward(game: PanelGame) -> float:
    if game.reward is None:
        raise ValueError(f"panel game {game.start} has no R: read it with a margin scale")
    return game.reward


def _by_start(games: Sequence[PanelGame], value: Callable[[PanelGame], float] = _margin) -> dict[str, float]:
    """A panel's value (the margin unless given) per start (the mean when a start was played more than once)."""
    grouped: dict[str, list[float]] = {}
    for g in games:
        grouped.setdefault(g.start, []).append(value(g))
    return {start: statistics.fmean(v) for start, v in grouped.items()}


def _interval(values: Sequence[float]) -> tuple[float, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[round(0.975 * (len(ordered) - 1))]


def paired_margin_test(
    final_games: Sequence[PanelGame],
    v0_games: Sequence[PanelGame],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    value: Callable[[PanelGame], float] = _margin,
) -> dict[str, Any]:
    """The numbers of a paired comparison: games paired by start, the paired differences in `value` of a game (its final
    net-worth margin unless given; final less v0) under a one-sided Wilcoxon signed-rank test, their median, and each
    arm's share of games ending ahead (net-worth margin above zero) with the difference of the shares and its
    bootstrap 95% interval over pairs."""
    a, b = _by_start(final_games, value), _by_start(v0_games, value)
    starts = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in starts]
    test = wilcoxon_greater(diffs)
    margins_a, margins_b = _by_start(final_games), _by_start(v0_games)
    ahead_a, ahead_b = [margins_a[s] > 0 for s in starts], [margins_b[s] > 0 for s in starts]
    n = len(starts)
    share_a = sum(ahead_a) / n if n else None
    share_b = sum(ahead_b) / n if n else None
    rng, boot = random.Random(seed), []
    for _ in range(resamples if n else 0):
        picks = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(ahead_a[i] for i in picks) / n - sum(ahead_b[i] for i in picks) / n)
    p_down = wilcoxon_greater([-x for x in diffs])["p"] if diffs else None
    return {
        "n_pairs": n,
        "unpaired": {"final": len(a) - n, "v0": len(b) - n},
        "median_difference": statistics.median(diffs) if diffs else None,
        "mean_difference": statistics.fmean(diffs) if diffs else None,
        "test": test,
        "p": test["p"] if n >= 2 else None,
        "p_two_sided": min(1.0, 2 * min(test["p"], p_down)) if n >= 2 and p_down is not None else None,
        "cap_win_share": {
            "final": share_a,
            "v0": share_b,
            "difference": None if share_a is None or share_b is None else share_a - share_b,
            "interval": _interval(boot),
        },
    }


def paired_margin_gate(
    final_games: Sequence[PanelGame],
    v0_games: Sequence[PanelGame],
    *,
    max_p: float = PAIRED_MAX_P,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    value: Callable[[PanelGame], float] = _margin,
    what: str = "net-worth margin",
    digits: int = 0,
) -> tuple[Gate, dict[str, Any]]:
    """E3's end-of-run verdict: the final adapter beats v0 on the same starts, by net-worth margin at the end of the
    game (or by `value` of a game, named `what`), one-sided Wilcoxon signed-rank p under `max_p`. The cap-win shares are
    reported, not gated."""
    detail = paired_margin_test(final_games, v0_games, resamples=resamples, seed=seed, value=value)
    median = detail["median_difference"]
    note = f"{detail['n_pairs']} pairs" + ("" if median is None else f", median paired difference {median:+.{digits}f}")
    name = f"final beats v0 on the paired {what} (Wilcoxon signed-rank, one-sided p)"
    return Gate(name, detail["p"], "<", max_p, note), detail


def paired_reward_gate(
    final_games: Sequence[PanelGame],
    start_games: Sequence[PanelGame],
    *,
    max_p: float = PAIRED_MAX_P,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[Gate, dict[str, Any]]:
    """E4's end-of-run verdict: the final adapter beats the adapter the run started from on the same starts, by R (the
    games must be read with a margin scale), one-sided Wilcoxon signed-rank p under `max_p`."""
    gate, detail = paired_margin_gate(
        final_games, start_games, max_p=max_p, resamples=resamples, seed=seed, value=_reward, what="R", digits=3
    )
    return Gate(gate.name.replace("final beats v0", "final beats the start"), gate.value, "<", max_p, gate.note), detail


def pooled_margin_sd(games: Sequence[PanelGame]) -> dict[str, Any]:
    """The within-opponent SD of the final net-worth margins of the games that ended at a cap, pooled over opponents
    (sum of squares about each opponent's mean over the degrees of freedom): the s of E4's tanh(m / s)."""
    by_opponent: dict[str, list[float]] = {}
    for g in games:
        if g.ended_by in rs.CAP_ENDS:
            by_opponent.setdefault(g.opponent, []).append(g.margin)
    n = sum(len(v) for v in by_opponent.values())
    squares = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in by_opponent.values())
    dof = n - len(by_opponent)
    return {
        "cap_games": n,
        "opponents": {k: len(v) for k, v in sorted(by_opponent.items())},
        "sd": math.sqrt(squares / dof) if dof > 0 else None,
    }


def null_pair_gate(
    a_games: Sequence[PanelGame],
    b_games: Sequence[PanelGame],
    *,
    max_p: float = PAIRED_MAX_P,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[Gate, dict[str, Any]]:
    """The check's null: two panels of v0 on identical starts through `paired_margin_gate`'s test must not reject. A
    test that rejects here would reject a good final adapter by chance or by a pairing that does not pair."""
    detail = paired_margin_test(a_games, b_games, resamples=resamples, seed=seed)
    note = f"{detail['n_pairs']} pairs; a rejection would make the end test meaningless"
    return Gate("pair-a vs pair-b (v0 twice) does not reject (one-sided p)", detail["p"], ">=", max_p, note), detail


Cluster = tuple[int, float, float, float, float]  # n, sum x, sum y, sum xx, sum xy


def _slope(clusters: Sequence[Cluster]) -> float | None:
    """The OLS slope of y on x from per-cluster sums; None when x does not vary."""
    n = sum(c[0] for c in clusters)
    if not n:
        return None
    sx, sy = sum(c[1] for c in clusters), sum(c[2] for c in clusters)
    sxx, sxy = sum(c[3] for c in clusters), sum(c[4] for c in clusters)
    spread = sxx - sx * sx / n
    return None if spread <= 1e-12 else (sxy - sx * sy / n) / spread


def in_run_slope(
    games: Sequence[TrainingGame], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> dict[str, Any]:
    """R regressed on the policy version over the training games (taken in game order), the slope's 95% interval from
    a bootstrap that resamples whole starts: the games of a start share their opponent, civs and map, so they are not
    independent observations."""
    ordered = sorted(games, key=lambda g: g.index)
    sums: dict[str, list[float]] = {}
    for g in ordered:
        c = sums.setdefault(g.start, [0, 0.0, 0.0, 0.0, 0.0])
        x, y = float(g.version), g.reward
        c[0] += 1
        c[1] += x
        c[2] += y
        c[3] += x * x
        c[4] += x * y
    clusters: list[Cluster] = [(int(c[0]), c[1], c[2], c[3], c[4]) for c in sums.values()]
    slope = _slope(clusters)
    rng, boot = random.Random(seed), []
    for _ in range(resamples if slope is not None else 0):
        value = _slope([clusters[rng.randrange(len(clusters))] for _ in clusters])
        if value is not None:
            boot.append(value)
    return {
        "games": len(ordered),
        "starts": len(clusters),
        "slope_per_version": slope,
        "slope_per_10_versions": None if slope is None else 10 * slope,
        "interval": _interval(boot),
        "mean_reward": statistics.fmean(g.reward for g in ordered) if ordered else None,
        "versions": [min(g.version for g in ordered), max(g.version for g in ordered)] if ordered else None,
    }


def in_run_slope_gate(
    games: Sequence[TrainingGame], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[Gate, dict[str, Any]]:
    """E3's in-run check: the training reward rises with the policy version, the interval's lower bound above zero."""
    detail = in_run_slope(games, resamples=resamples, seed=seed)
    low = None if detail["interval"] is None else detail["interval"][0]
    per_10 = detail["slope_per_10_versions"]
    note = f"{detail['games']} games in {detail['starts']} starts" + (
        "" if per_10 is None else f", {per_10:+.4f} per 10 versions"
    )
    name = "training reward slope on the policy version (lower bound of the 95% interval)"
    return Gate(name, low, ">", 0.0, note), detail
