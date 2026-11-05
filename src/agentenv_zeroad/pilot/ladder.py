"""Who the policy plays: each round's training opponents, picked where it learns most, and the fixed probe panel.

Training games go mostly to opponents the policy beats about half the time: a group of games it always wins or
always loses gives every decision the same result, so none stands out. Each opponent's win rate is estimated from
the policy's recent games against it (wins, draws as halves) with a prior worth `PRIOR_GAMES` games: Petra's levels
start at rates that fall with difficulty, so round 1 plays mostly very easy and easy Petra; a snapshot starts at an
even match. Weights fall off with the distance from 50% (`weight`), and `MASTERED_SHARE` of the games always goes to
opponents it already beats (over `MASTERED_RATE`), so it doesn't forget them. Games are allotted by largest
remainder, so a round's mix is the same for the same history.

The probe panel: Petra 1 to 5, the previous policy version and the frontier models (all of them, or a rotating few
per round), each on its own fixed seed that no training game uses, so its scores compare from round to round.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agentenv_zeroad.pilot.games import Opponent, petra, snapshot

PETRA_PRIOR = {1: 0.6, 2: 0.45, 3: 0.3, 4: 0.15, 5: 0.05}
PRIOR_GAMES = 2.0
MASTERED_RATE = 0.8
MASTERED_SHARE = 0.1
WIDTH = 0.2
RECENT_ROUNDS = 3
PROBE_SEEDS = range(9001, 9100)  # training seeds are drawn below this
FRONTIER_SEEDS = {"sonnet": 9012, "opus": 9013, "astra": 9014, "sol": 9015}


@dataclass(frozen=True)
class Played:
    """One finished game, as the ladder reads it: who, when, and the policy's result (1 win, 0.5 draw, 0 loss)."""

    opponent: str
    round: int
    result: float


def prior(opponent: Opponent) -> float:
    return PETRA_PRIOR.get(opponent.level or 0, 0.5) if opponent.kind == "petra" else 0.5


def win_rates(opponents: Iterable[Opponent], played: Sequence[Played], round_: int) -> dict[str, float]:
    """Each opponent's estimated win rate for the policy, from its games in the last `RECENT_ROUNDS` rounds."""
    rates = {}
    for opp in opponents:
        recent = [g.result for g in played if g.opponent == opp.id and round_ - RECENT_ROUNDS <= g.round < round_]
        rates[opp.id] = (prior(opp) * PRIOR_GAMES + sum(recent)) / (PRIOR_GAMES + len(recent))
    return rates


def weight(rate: float) -> float:
    return math.exp(-(((rate - 0.5) / WIDTH) ** 2))


def allot(n: int, opponents: Sequence[Opponent], rates: Mapping[str, float]) -> list[Opponent]:
    """`n` training opponents: the learning share by `weight`, the mastered share spread over mastered opponents."""
    pool = [o for o in opponents if o.trainable]
    if n <= 0 or not pool:
        return []
    shares = {o.id: weight(rates[o.id]) for o in pool}
    total = sum(shares.values()) or 1.0
    mastered = [o for o in pool if rates[o.id] >= MASTERED_RATE]
    keep = MASTERED_SHARE if mastered and len(mastered) < len(pool) else 0.0
    quota = {o.id: n * (1 - keep) * shares[o.id] / total for o in pool}
    for o in mastered:
        quota[o.id] += n * keep / len(mastered)
    counts = {k: int(v) for k, v in quota.items()}
    left = n - sum(counts.values())
    for k in sorted(quota, key=lambda k: (counts[k] - quota[k], k))[:left]:
        counts[k] += 1
    return [o for o in pool for _ in range(counts[o.id])]


def ladder(versions_kept: Sequence[int]) -> list[Opponent]:
    """The training pool: Petra 1 to 5 and the kept snapshots of earlier policy versions."""
    return [petra(level) for level in range(1, 6)] + [snapshot(v) for v in versions_kept]


def probe_panel(previous_version: int | None, frontier: Sequence[Opponent] = ()) -> list[Opponent]:
    panel = [petra(level) for level in range(1, 6)]
    if previous_version is not None:
        panel.append(snapshot(previous_version))
    return panel + list(frontier)


def frontier_for(round_: int, frontier: Sequence[Opponent], per_round: int, offset: int) -> list[Opponent]:
    """The frontier probes of `round_`: all of them, or `per_round` taken in turn, starting `offset` along."""
    if per_round <= 0 or per_round >= len(frontier):
        return list(frontier)
    start = (round_ + offset) % len(frontier)
    return [frontier[(start + i) % len(frontier)] for i in range(per_round)]


def probe_seed(opponent: Opponent) -> int:
    """A fixed seed per probe opponent: Petra's levels, the snapshot, each frontier model (never a training seed)."""
    if opponent.kind == "petra":
        return PROBE_SEEDS[opponent.level or 0]
    if opponent.kind == "snapshot":
        return PROBE_SEEDS[10]
    return FRONTIER_SEEDS.get(opponent.id, PROBE_SEEDS[20 + sum(map(ord, opponent.id)) % 70])
