"""The gates of box day 1 (S2) and the decisions between its steps, as plain functions of numbers.

Nothing here touches a box, a game or a file: the sequencer (`train/s2.py`) reads what a step wrote, calls these and
records each verdict in ROOT/s2/gates.json. They are separate so each rule can be tested on its own numbers.

- `throughput_gate`, `learner_gate`, `agreement_gate`: E1. The probe's decisions per second per GPU (the best of its
  in-flight levels, at least 3), the learner's seconds per trained decision (at most 2) and the mean absolute log-prob
  gap between the learner and vLLM on a few sampled decisions (under 0.05 nats per token).
- `positive_control_gates`: E2. The policy trained on the army-at-minute-10 reward must end with a higher median army at
  minute 10 than a frozen-v0 control arm that played beside it (Mann-Whitney, one-sided, p under 0.01), with a median of
  at least 6, and with no tripwire.
- `init_decision`: whether Run 1 starts from the distilled adapter or from the base model, by the rule written in
  advance: the chosen SFT checkpoint beats v0 on the same held-out configs at one-sided p under 0.05 on the
  opponent-stratified panel test (`stats.panel_test`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from agentenv_zeroad.train import stats

MIN_DECISIONS_PER_S_PER_GPU = 3.0
MAX_LEARNER_S_PER_DECISION = 2.0
MAX_LOGPROB_GAP = 0.05
CONTROL_MIN_MEDIAN_ARMY = 6.0
CONTROL_MAX_P = 0.01
CONTROL_MIN_GAMES = 30
INIT_MAX_P = 0.05
KNEE_FRACTION = 0.9
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
