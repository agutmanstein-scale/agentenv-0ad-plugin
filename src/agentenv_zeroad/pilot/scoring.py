"""Reading a pilot game's results.json: the game's record, and each policy decision judged against its own position.

States diverge after the first few decisions of a game, so a decision is not compared with decisions in other
games. Instead two value estimates (ridge regressions over the observer's view of both players at the moment the
decision began: scores, economy, army, phase, game time, the opponent) predict how the policy's score margin will
move over the next `HORIZON_MS` and by the end of the game. A decision's advantage is how much better than predicted
the margin actually moved, short term plus `LONG_WEIGHT` times long term, each in standard deviations, less
`REJECT_WEIGHT` times the share of its orders the env rejected; then standardized over the batch and clipped. The
estimates use the observer's unfogged view, which the policy never sees: they only score, they don't play.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentenv_zeroad.pilot.games import GameSpec

HORIZON_MS = 120_000
LONG_WEIGHT = 0.5
REJECT_WEIGHT = 1.0
CLIP = 3.0
RIDGE = 1.0
PHASES = {"village": 0, "town": 1, "city": 2}
SIDE_KEYS = (  # (row key, scale)
    ("total", 1000),
    ("economy", 1000),
    ("military", 1000),
    ("exploration", 1000),
    ("pop", 100),
    ("army", 50),
    ("gatherers", 50),
    ("resources", 1000),
    ("buildings", 20),
    ("kills", 50),
    ("lost", 50),
)
RESULT = {"won": 1.0, "drawn": 0.5, "lost": 0.0}


@dataclass
class GameRecord:
    game_id: str
    round: int
    kind: str
    opponent: str
    seed: int
    policy_version: int
    status: str  # finished, failed, cancelled, missing, void (the opponent's model never played)
    outcome: str | None = None  # the policy's: won, lost, drawn
    margin: float | None = None  # the policy's final score minus the opponent's
    policy_score: float | None = None
    opponent_score: float | None = None
    ended_by: str | None = None
    game_time_s: float | None = None
    wall_s: float | None = None
    overruns: int | None = None
    calls: int = 0
    missed: int = 0
    orders: int = 0
    rejected: int = 0
    latency_p50_s: float | None = None
    spend_usd: float = 0.0
    detail: str | None = None

    @property
    def result(self) -> float | None:
        return RESULT.get(self.outcome or "")


@dataclass
class Decision:
    game_id: str
    round: int
    policy_version: int
    began_ms: int
    features: list[float]
    short: float  # the margin's move over the next HORIZON_MS
    long: float  # its move to the end of the game
    rejected_share: float
    prompt_token_ids: list[int]
    token_ids: list[int]
    advantage: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def record(spec: GameSpec, results: Mapping[str, Any] | None, detail: str | None = None) -> GameRecord:
    """The game's record; `results` None when the game left none (it failed before the match ended)."""
    rec = GameRecord(
        spec.game_id, spec.round, spec.kind, spec.opponent.id, spec.seed, spec.policy_version, "missing", detail=detail
    )
    if results is None:
        return rec
    me, opp = _sides(spec, results)
    seats = results["seats"]
    mine, theirs = seats[me], seats[opp]
    rec.status = results.get("status", "missing")
    rec.outcome = mine.get("outcome")
    if theirs.get("kind") == "agent" and theirs.get("forfeited"):
        # Its model calls failed (e.g. a refused key): its empire ran on standing orders alone, so the game says
        # nothing about either side. Kept, but out of the ratings and the ladder.
        rec.status, rec.outcome = "void", None
        detail = detail or f"the opponent's model calls failed ({_first_error(theirs)})"
    rec.policy_score = (mine.get("score") or {}).get("total")
    rec.opponent_score = (theirs.get("score") or {}).get("total")
    if rec.policy_score is not None and rec.opponent_score is not None:
        rec.margin = rec.policy_score - rec.opponent_score
    rec.ended_by = results.get("ended_by")
    rec.game_time_s = results.get("game_time_s")
    rec.wall_s = results.get("wall_s")
    rec.overruns = (results.get("clock") or {}).get("overruns")
    moves = mine.get("moves") or {}
    rec.calls, rec.missed = moves.get("made", 0) + moves.get("missed", 0), moves.get("missed", 0)
    turns = mine.get("turns") or []
    rec.orders = sum(len(t.get("orders") or []) for t in turns)
    rec.rejected = sum(1 for t in turns for o in t.get("orders") or [] if o.get("status") == "rejected")
    latency = ((mine.get("metrics") or {}).get("latency_s") or {}).get("p50")
    rec.latency_p50_s = latency
    rec.spend_usd = float(results.get("spend_usd_est") or 0.0)
    rec.detail = detail or results.get("detail")
    return rec


def decisions(spec: GameSpec, results: Mapping[str, Any]) -> list[Decision]:
    """The policy's logged turns that carry token ids, each with its features and how the margin moved after it."""
    me, opp = _sides(spec, results)
    timeline = results.get("timeline") or []
    if not timeline:
        return []
    times = [row["t_ms"] for row in timeline]
    margins = [_margin(row, me, opp) for row in timeline]
    seats = results["seats"]
    final = _final_margin(seats, me, opp, margins[-1])
    out = []
    for turn in seats[me].get("turns") or []:
        prompt, completion = turn.get("prompt_token_ids"), turn.get("token_ids")
        began = turn.get("began_ms")
        if not prompt or not completion or began is None:
            continue
        i = max(0, bisect.bisect_right(times, began) - 1)
        j = max(0, bisect.bisect_right(times, began + HORIZON_MS) - 1)
        orders = turn.get("orders") or []
        rejected = sum(1 for o in orders if o.get("status") == "rejected")
        out.append(
            Decision(
                spec.game_id,
                spec.round,
                spec.policy_version,
                int(began),
                features(timeline[i], me, opp, spec),
                short=margins[j] - margins[i],
                long=final - margins[i],
                rejected_share=rejected / len(orders) if orders else 0.0,
                prompt_token_ids=list(prompt),
                token_ids=list(completion),
            )
        )
    return out


def features(row: Mapping[str, Any], me: str, opp: str, spec: GameSpec) -> list[float]:
    players = row["players"]
    x = [1.0, row["t_ms"] / 60_000]
    for slot in (me, opp):
        p = players.get(slot) or {}
        x += [float(p.get(k) or 0) / scale for k, scale in SIDE_KEYS]
        x.append(float(PHASES.get(str(p.get("phase")), 0)))
    x.append(_margin(row, me, opp) / 1000)
    level = spec.opponent.level or 0
    x += [1.0 if spec.opponent.kind == "petra" and level == n else 0.0 for n in range(1, 6)]
    x += [1.0 if spec.opponent.kind == "snapshot" else 0.0, 1.0 if spec.opponent.kind == "frontier" else 0.0]
    return x


def advantages(batch: Sequence[Decision], history: Sequence[Decision] = ()) -> None:
    """Set each decision's advantage in `batch`; the value estimates are fit on `history` and `batch` together."""
    if not batch:
        return
    fit = list(history) + list(batch)
    short = _residuals(fit, [d.short for d in fit])[len(history) :]
    long = _residuals(fit, [d.long for d in fit])[len(history) :]
    raw = [
        s + LONG_WEIGHT * lg - REJECT_WEIGHT * d.rejected_share
        for s, lg, d in zip(_standard(short), _standard(long), batch, strict=True)
    ]
    for d, a in zip(batch, _standard(raw), strict=True):
        d.advantage = max(-CLIP, min(CLIP, a))


def _residuals(fit: Sequence[Decision], y: Sequence[float]) -> list[float]:
    """y minus a ridge regression's prediction from each decision's features (the mean, with too few decisions)."""
    dims = len(fit[0].features)
    if len(fit) < 2 * dims:
        mean = statistics.fmean(y)
        return [v - mean for v in y]
    w = ridge([d.features for d in fit], y, RIDGE)
    return [v - sum(a * b for a, b in zip(w, d.features, strict=True)) for d, v in zip(fit, y, strict=True)]


def ridge(xs: Sequence[Sequence[float]], y: Sequence[float], lam: float) -> list[float]:
    """Weights minimizing |Xw - y|^2 + lam |w|^2 (the bias, column 0, not penalized), by Gaussian elimination."""
    d = len(xs[0])
    a = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for x, v in zip(xs, y, strict=True):
        for i in range(d):
            xi = x[i]
            if xi == 0.0:
                continue
            b[i] += xi * v
            row = a[i]
            for j in range(d):
                row[j] += xi * x[j]
    for i in range(1, d):
        a[i][i] += lam
    return _solve(a, b)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            m[pivot][col] = 1e-12
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / m[col][col]
                m[r] = [u - f * v for u, v in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _standard(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return [(v - mean) / sd if sd > 1e-9 else 0.0 for v in values]


def _first_error(seat: Mapping[str, Any]) -> str:
    errors = [c.get("error") for c in seat.get("brain_calls") or [] if c.get("error")]
    return (errors[0] or "").splitlines()[0][:120] if errors else "no calls"


def _sides(spec: GameSpec, results: Mapping[str, Any]) -> tuple[str, str]:
    seats = results["seats"]
    me = spec.policy_slot
    [opp] = [s for s in seats if s != me]
    return me, opp


def _margin(row: Mapping[str, Any], me: str, opp: str) -> float:
    players = row["players"]
    return float((players.get(me) or {}).get("total") or 0) - float((players.get(opp) or {}).get("total") or 0)


def _final_margin(seats: Mapping[str, Any], me: str, opp: str, fallback: float) -> float:
    mine, theirs = (seats[me].get("score") or {}).get("total"), (seats[opp].get("score") or {}).get("total")
    if mine is None or theirs is None or not math.isfinite(mine - theirs):
        return fallback
    return float(mine - theirs)
