"""One rating per player from every pilot game: each policy version, each Petra level, each frontier model.

A Bradley-Terry fit (chess Elo's model: the chance a beats b is 1 / (1 + 10^((Rb - Ra) / 400))), with a draw as half
a win each, and a weak prior pulling every rating toward the mean so a player that never lost doesn't run off to
infinity. A snapshot opponent is a policy version, so self-play games link the versions directly. Ratings are put on
a scale where Petra at medium (petra-3) is 1000, or the mean is 1000 when Petra 3 never played.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

ELO = 400 / math.log(10)
ANCHOR = "petra-3"
PRIOR = 0.05  # L2 on the natural-unit ratings; about a 1,800-point standard deviation, so it only stops runaways


def fit(games: Sequence[tuple[str, str, float]], *, steps: int = 2000, rate: float = 0.5) -> dict[str, float]:
    """Ratings from `(a, b, score_of_a)` games (1 a won, 0.5 a draw, 0 b won)."""
    players = sorted({p for a, b, _ in games for p in (a, b)})
    if not players:
        return {}
    r = dict.fromkeys(players, 0.0)
    count = {p: 0 for p in players}
    for a, b, _ in games:
        count[a] += 1
        count[b] += 1
    for _ in range(steps):
        grad = {p: -PRIOR * r[p] for p in players}
        for a, b, s in games:
            p = 1 / (1 + math.exp(r[b] - r[a]))
            grad[a] += s - p
            grad[b] -= s - p
        for p in players:
            r[p] += rate * grad[p] / (count[p] + 1)
    zero = r[ANCHOR] if ANCHOR in r else sum(r.values()) / len(r)
    return {p: round(1000 + (v - zero) * ELO, 1) for p, v in r.items()}
