"""What the v2 board is made of: ratings with intervals, a trend test, the panel test and milestones, in numpy.

Every function takes `Game` rows, one per finished game of a version-pinned policy, so the same code reads a live run
and a replay of one. The pilot's ratings (pilot/ratings.py) looped over every game in Python for 2,000 steps, too slow
to bootstrap; the fit here is the same model solved by Newton's method on arrays.

- `fit` and `elo_by_hour`: one Bradley-Terry model (the chance a beats b is 1 / (1 + 10^((Rb - Ra) / 400)), a draw a
  half win each, the pilot's weak prior), Petra-3 = 1000. A policy version is not a player: versions are binned by the
  training hour they were published in, so a bin has the games to be rated (about +-40-50 Elo an hour), and the base
  model, v0, is its own player so frozen-v0 opponents and the pre-training games rate it together. Intervals come from
  resampling games (a game is the unit; a bootstrap that resampled decisions would be far too tight).
- `trend_test`: result against training hours with an intercept per opponent, a logistic regression by IRLS, on games
  against fixed anchors only (a Petra level or the scripted bot: opponents whose strength does not move, so a rising
  line is the policy's). Draws count half, which makes the response fractional, so the SE is the sandwich one.
- `panel_test`: the pre-registered endpoint. The final net-worth margin of two versions on the same opponents, as an OLS
  with a fixed effect per opponent; the SE uses the SD of the first (base) version's margins within opponent,
  as the design asks ("the SE comes from S2's own SD"), and the p-value is one-sided for the second being better.
- `milestones`: "Petra-2 at 30%" and its siblings. At each scheduled check time a milestone passes if the one-sided
  97.5% lower bound (Wilson) of the win rate against its opponent, pooled over the last two completed hourly bins,
  clears its threshold; 97.5% is Bonferroni-corrected across the milestones. Checking only on a schedule is what keeps
  the error rate: retrying after every game would promote a 55% policy most of the time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from agentenv_zeroad.pilot.ratings import ANCHOR, ELO, PRIOR

FIXED_PREFIXES = ("petra-", "bot")
BASE = "v0"
CHECK_EVERY_HOURS = 3.0
POOL_HOURS = 2
ALPHA = 0.025  # the milestones' one-sided level, shared out by Bonferroni
NORMAL = NormalDist()


@dataclass(frozen=True)
class Game:
    version: int  # the policy version that played (0 is the base model)
    hours: float  # training hours when that version was published (0 for v0)
    opponent: str  # petra-3, bot, v0, a frontier model's name
    result: float  # the policy's score: 1 won, 0.5 drawn, 0 lost
    margin: float | None = None  # the policy's final net-worth margin (the official one when none was recorded)

    @property
    def fixed(self) -> bool:
        return self.opponent.startswith(FIXED_PREFIXES)

    @property
    def player(self) -> str:
        return BASE if self.version == 0 else f"h{int(self.hours):02d}"


# ---- ratings --------------------------------------------------------------------------------------------------
def fit(
    a: np.ndarray, b: np.ndarray, score: np.ndarray, players: int, *, prior: float = PRIOR, iterations: int = 60
) -> np.ndarray:
    """Natural-unit Bradley-Terry ratings of `players` players from games `a[i]` vs `b[i]` scored `score[i]` for a."""
    r = np.zeros(players)
    for _ in range(iterations):
        p = 1 / (1 + np.exp(r[b] - r[a]))
        resid, w = score - p, p * (1 - p)
        grad = np.bincount(a, resid, players) - np.bincount(b, resid, players) - prior * r
        info = np.diag(np.bincount(a, w, players) + np.bincount(b, w, players) + prior)
        np.add.at(info, (a, b), -w)
        np.add.at(info, (b, a), -w)
        step = np.linalg.solve(info, grad)
        r += step
        if np.abs(step).max() < 1e-9:
            break
    return r


def to_elo(r: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Ratings on the Elo scale with Petra-3 at 1000 (the mean at 1000 when it never played)."""
    zero = r[labels.index(ANCHOR)] if ANCHOR in labels else r.mean()
    return 1000 + (r - zero) * ELO


def elo_by_hour(
    games: Sequence[Game], *, bootstrap: int = 200, seed: int = 0, confidence: float = 0.95
) -> dict[str, Any]:
    """{"board": {player: {elo, lo, hi, games}}, "curve": [...]} over every game, with game-bootstrap intervals.

    The curve is the policy's hourly bins in order: label, hour, elo, lo, hi and the games in the bin."""
    if not games:
        return {"board": {}, "curve": []}
    labels = sorted({g.player for g in games} | {g.opponent for g in games})
    index = {p: i for i, p in enumerate(labels)}
    a = np.array([index[g.player] for g in games])
    b = np.array([index[g.opponent] for g in games])
    s = np.array([g.result for g in games])
    point = to_elo(fit(a, b, s, len(labels)), labels)
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(labels)))
    for k in range(bootstrap):
        pick = rng.integers(0, len(games), len(games))
        draws[k] = to_elo(fit(a[pick], b[pick], s[pick], len(labels)), labels)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(draws, [tail, 100 - tail], axis=0) if bootstrap else (point, point)
    played = np.bincount(a, minlength=len(labels)) + np.bincount(b, minlength=len(labels))
    board = {
        p: {"elo": round(float(point[i]), 1), "lo": round(float(lo[i]), 1), "hi": round(float(hi[i]), 1),
            "games": int(played[i])}
        for p, i in index.items()
    }  # fmt: skip
    bins = {g.player for g in games}
    curve = [{"label": p, "hour": 0 if p == BASE else int(p[1:]), **board[p]} for p in bins]
    return {"board": board, "curve": sorted(curve, key=lambda c: (c["hour"], c["label"] != BASE))}


# ---- the trend test ---------------------------------------------------------------------------------------------
def trend_test(games: Sequence[Game]) -> dict[str, Any]:
    """Logistic regression of result on training hours with an intercept per opponent, fixed-anchor games only.

    Returns the slope (log-odds per training hour), its sandwich SE and the one-sided p for a positive slope. With
    fewer than two distinct hours, or no games, the numbers are None: there is no trend to test."""
    anchored = [g for g in games if g.fixed]
    opponents = sorted({g.opponent for g in anchored})
    out: dict[str, Any] = {"n": len(anchored), "opponents": opponents, "slope": None, "se": None, "z": None, "p": None}
    if len({g.hours for g in anchored}) < 2:
        return out
    x = np.zeros((len(anchored), 1 + len(opponents)))
    for i, g in enumerate(anchored):
        x[i, 0] = g.hours
        x[i, 1 + opponents.index(g.opponent)] = 1.0
    y = np.array([g.result for g in anchored])
    beta, bread, resid = _logistic(x, y)
    meat = (x * resid[:, None] ** 2).T @ x
    cov = np.linalg.solve(bread, np.linalg.solve(bread, meat).T)
    se = float(math.sqrt(max(cov[0, 0], 0.0)))
    slope = float(beta[0])
    z = slope / se if se > 0 else None
    out |= {"slope": slope, "se": se, "z": z, "p": 1 - NORMAL.cdf(z) if z is not None else None}
    return out


def _logistic(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(beta, X'WX + ridge, y - mu) for a fractional-response logistic regression fit by Newton steps."""
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-x @ beta))
        w = np.maximum(mu * (1 - mu), 1e-9)
        bread = (x * w[:, None]).T @ x + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(bread, x.T @ (y - mu) - ridge * beta)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    mu = 1 / (1 + np.exp(-x @ beta))
    w = np.maximum(mu * (1 - mu), 1e-9)
    return beta, (x * w[:, None]).T @ x + ridge * np.eye(x.shape[1]), y - mu


# ---- the panel test ---------------------------------------------------------------------------------------------
def panel_test(base: Sequence[Game], other: Sequence[Game]) -> dict[str, Any]:
    """Did `other` finish with a better net-worth margin than `base` on the same opponents? One-sided.

    OLS of margin on an `other` indicator plus a fixed effect per opponent that both played (the rest cannot be
    compared). The SE is sigma x sqrt((X'X)^-1[0, 0]) with sigma the pooled within-opponent SD of `base`'s margins; the
    p-value is one-sided on the normal, and `sd_other` reports the other arm's SD so a mismatch is visible."""
    base = [g for g in base if g.margin is not None]
    other = [g for g in other if g.margin is not None]
    strata = sorted({g.opponent for g in base} & {g.opponent for g in other})
    base, other = ([g for g in arm if g.opponent in strata] for arm in (base, other))
    out: dict[str, Any] = {
        "strata": strata, "n_base": len(base), "n_other": len(other), "difference": None, "se": None, "sd": None,
        "sd_other": None, "z": None, "p": None,
    }  # fmt: skip
    sd = _within_sd(base)
    if not strata or sd is None or not other:
        return out
    games = [*base, *other]
    x = np.zeros((len(games), 1 + len(strata)))
    for i, g in enumerate(games):
        x[i, 0] = float(i >= len(base))
        x[i, 1 + strata.index(g.opponent)] = 1.0
    y = np.array([g.margin for g in games], dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    se = float(sd * math.sqrt(np.linalg.inv(x.T @ x)[0, 0]))
    diff = float(beta[0])
    z = diff / se if se > 0 else None
    out |= {
        "difference": diff, "se": se, "sd": sd, "sd_other": _within_sd(other), "z": z,
        "p": 1 - NORMAL.cdf(z) if z is not None else None,
    }  # fmt: skip
    return out


def _within_sd(games: Sequence[Game]) -> float | None:
    """Pooled within-opponent SD of the margins, or None when no opponent has two games."""
    by: dict[str, list[float]] = defaultdict(list)
    for g in games:
        if g.margin is not None:
            by[g.opponent].append(g.margin)
    dof = sum(len(v) - 1 for v in by.values() if v)
    if dof <= 0:
        return None
    ss = sum(float(np.sum((np.array(v) - np.mean(v)) ** 2)) for v in by.values() if v)
    return math.sqrt(ss / dof)


# ---- milestones -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Milestone:
    opponent: str
    threshold: float

    @property
    def name(self) -> str:
        return f"{self.opponent} >= {self.threshold:.0%}"


DEFAULT_MILESTONES = (
    Milestone("petra-1", 0.50),
    Milestone("petra-1", 0.65),
    Milestone("petra-2", 0.15),
    Milestone("petra-2", 0.30),
    Milestone("petra-2", 0.45),
    Milestone("petra-2", 0.50),
    Milestone("petra-3", 0.15),
)


def wilson_lower(wins: float, n: float, z: float) -> float:
    """The Wilson lower bound of a win rate (fractional wins allowed) at normal quantile z."""
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half)


def milestones(
    games: Sequence[Game],
    wanted: Sequence[Milestone] = DEFAULT_MILESTONES,
    *,
    now_hours: float | None = None,
    check_every_hours: float = CHECK_EVERY_HOURS,
    pool_hours: int = POOL_HOURS,
    alpha: float = ALPHA,
) -> list[dict[str, Any]]:
    """Each milestone's status at the schedule's check times up to `now_hours` (default: the last game's hour).

    A check at time t pools the games against the milestone's opponent whose version was published in the `pool_hours`
    hourly bins before t. It passes when the lower bound of the win rate at one-sided level alpha / len(wanted) (the
    Bonferroni share) is at or above the threshold; a milestone passed once stays passed, at the first such check."""
    horizon = now_hours if now_hours is not None else max((g.hours for g in games), default=0.0)
    z = NORMAL.inv_cdf(1 - alpha / max(len(wanted), 1))
    times = [k * check_every_hours for k in range(1, int(horizon // check_every_hours) + 1)]
    out = []
    for m in wanted:
        mine = [g for g in games if g.opponent == m.opponent]
        checks, passed_at = [], None
        for t in times:
            pooled = [g for g in mine if t - pool_hours <= int(g.hours) < t]
            wins, n = sum(g.result for g in pooled), len(pooled)
            lower = wilson_lower(wins, n, z)
            checks.append({"at": t, "games": n, "rate": wins / n if n else None, "lower": lower})
            if passed_at is None and n and lower >= m.threshold:
                passed_at = t
        out.append({"name": m.name, "opponent": m.opponent, "threshold": m.threshold, "passed_at": passed_at,
                    "checks": checks})  # fmt: skip
    return out


# ---- teacher calibration ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TeacherGame:
    teacher: str  # the seat's name (the config's teacher name, else its model)
    opponent: str  # petra-3, petra-4
    result: float  # the teacher's score: 1 won, 0.5 drawn, 0 lost
    model: str | None = None


CALIBRATION_RUNG = "petra-3"  # the teacher the distillation uses is the one that beats this rung most often
CALIBRATION_TIEBREAK = "petra-4"


def wilson_interval(wins: float, n: float, z: float = NORMAL.inv_cdf(0.975)) -> tuple[float, float]:
    """The two-sided Wilson interval of a win rate (fractional wins allowed); (0, 1) for no games."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def teacher_calibration(games: Sequence[TeacherGame]) -> dict[str, Any]:
    """Per teacher and Petra rung: games, win rate and its 95% Wilson interval, then the teacher to distil from.

    `best` follows `pick_teacher`: the best win rate against Petra-3, ties broken by Petra-4, then by name. A draw
    counts half, as everywhere on the board. With ~10 games in all, the intervals are wide on purpose: the report
    shows how little a difference means."""
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    models: dict[str, str | None] = {}
    for g in games:
        cells[g.teacher][g.opponent].append(g.result)
        models.setdefault(g.teacher, g.model)
    teachers: dict[str, Any] = {}
    for name in sorted(cells):
        rungs = {}
        for opponent, results in sorted(cells[name].items()):
            wins, n = sum(results), len(results)
            lo, hi = wilson_interval(wins, n)
            rungs[opponent] = {"games": n, "wins": wins, "win_rate": wins / n, "lo": lo, "hi": hi}
        teachers[name] = {"model": models[name], "rungs": rungs}
    summary: dict[str, Any] = {"games": len(games), "teachers": teachers}
    best = pick_teacher(summary)
    rule = f"best win rate vs {CALIBRATION_RUNG}, ties {CALIBRATION_TIEBREAK}"
    summary["best"] = {"name": best, "model": teachers[best]["model"], "rule": rule} if best else None
    return summary


def rank_teachers(summary: dict[str, Any]) -> list[str]:
    """The teachers that played Petra-3, best first: by win rate against it, ties by the win rate against Petra-4 (no
    games there counts as 0), then by name."""
    scored = []
    for name, teacher in summary["teachers"].items():
        rungs = teacher["rungs"]
        if CALIBRATION_RUNG in rungs:
            tie = rungs.get(CALIBRATION_TIEBREAK, {}).get("win_rate", 0.0)
            scored.append((-rungs[CALIBRATION_RUNG]["win_rate"], -tie, name))
    return [name for *_, name in sorted(scored)]


def pick_teacher(summary: dict[str, Any]) -> str | None:
    """The teacher with the best win rate against Petra-3 (it has to have played it), ties broken by the win rate
    against Petra-4, then by name. None when nobody played Petra-3."""
    ranked = rank_teachers(summary)
    return ranked[0] if ranked else None


# ---- choosing the SFT checkpoint --------------------------------------------------------------------------------
@dataclass(frozen=True)
class SftCandidate:
    name: str  # epoch1, epoch2 ...: candidates are listed in training order
    win_rate: float | None  # on the held-out panel (None: it was not played)
    games: int
    entropy_ratio: float | None  # mean token entropy over the base model's, from the epoch's manifest


MIN_ENTROPY_RATIO = 0.7


def select_sft_checkpoint(
    candidates: Sequence[SftCandidate], *, min_entropy_ratio: float = MIN_ENTROPY_RATIO
) -> dict[str, Any]:
    """Which SFT checkpoint RL should start from: by held-out win rate, never by the SFT loss.

    SFT loss falls with every epoch whether or not the policy plays better, and a checkpoint trained too long has
    collapsed its entropy, which RL then cannot explore from. So a candidate is rejected when its mean token entropy
    is under `min_entropy_ratio` (default 0.7) of the base model's, or when it has no panel games; of the rest the
    highest panel win rate wins, and a tie goes to the earlier (less trained) checkpoint. `chosen` is None when none
    passes: start RL from the base model."""
    rows, chosen = [], None
    for c in candidates:
        reason = None
        if c.win_rate is None:
            reason = "not played on the panel"
        elif c.entropy_ratio is None:
            reason = "no entropy measured"
        elif c.entropy_ratio < min_entropy_ratio:
            reason = f"entropy {c.entropy_ratio:.2f} of base is under {min_entropy_ratio:g}"
        rows.append(
            {
                "name": c.name,
                "win_rate": c.win_rate,
                "games": c.games,
                "entropy_ratio": c.entropy_ratio,
                "rejected": reason,
            }
        )
        if reason is None and (chosen is None or (c.win_rate or 0) > (chosen.win_rate or 0)):
            chosen = c
    return {
        "chosen": chosen.name if chosen else None,
        "min_entropy_ratio": min_entropy_ratio,
        "candidates": rows,
        "rule": "highest panel win rate among checkpoints whose token entropy is at least the floor times the base's; "
        "ties to the earlier epoch; never the SFT loss",
    }
