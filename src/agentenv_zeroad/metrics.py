"""Per-seat match metrics (spec §5): the numbers that say whether real-time play works.

All durations are in game seconds, taken from snapshot times, so they mean the same at speed 1.0 and 2.0:
- idle-villager-seconds and idle-production-seconds integrate the idle counts of each snapshot over the time until
  the next one (a sample-and-hold integral, so it is exact at the sampling rate the env chooses);
- `waiting_share` is the share of match time the seat had something it couldn't do until its brain answered. The
  brain layers (U5) call `begin_waiting` when they open such a question and `end_waiting` when the answer lands;
  overlapping waits count once. A seat whose waits nobody measures has `waiting_share` None in the summary (not
  measured), so it can't be read as a measured zero; it is measured once a wait is reported, its brain polls the
  seat's status (whose `open` the shipped brains gate their waits on), or its reflex sends a request;
- decision latency samples give p50/p95/max, and decisions per minute counts answered decisions per game minute;
- clock overruns come from `GameClock.metrics()`, which is per match, so every seat reports the same figure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentenv_zeroad.state import Snapshot


def percentile(samples: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for no samples."""
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class SeatMetrics:
    player: int
    idle_villager_s: float = 0.0
    idle_production_s: float = 0.0
    latencies_s: list[float] = field(default_factory=list)
    decisions: int = 0
    missed: int = 0
    waiting_s: float = 0.0
    start_s: float | None = None
    last_s: float | None = None
    _last_idle: tuple[int, int] = (0, 0)
    waits_reported: bool = False
    _waits: int = 0
    _wait_since: float | None = None

    def observe(self, snapshot: Snapshot) -> None:
        """Account the time since the previous snapshot at the previous snapshot's idle counts."""
        now = snapshot.time_ms / 1000
        me = snapshot.players.get(self.player)
        if self.last_s is not None and now > self.last_s:
            dt = now - self.last_s
            self.idle_villager_s += self._last_idle[0] * dt
            self.idle_production_s += self._last_idle[1] * dt
        if self.start_s is None:
            self.start_s = now
        self.last_s = max(now, self.last_s or now)
        self._last_idle = (len(me.idle_villagers), len(me.idle_production)) if me else (0, 0)

    def record_decision(self, latency_s: float, *, missed: bool = False) -> None:
        """One brain call finished (or missed: errored or ran past the forfeit limit, spec §4.7)."""
        self.latencies_s.append(latency_s)
        if missed:
            self.missed += 1
        else:
            self.decisions += 1

    def begin_waiting(self, now_s: float) -> None:
        self.waits_reported = True
        if self._waits == 0:
            self._wait_since = now_s
        self._waits += 1

    def end_waiting(self, now_s: float) -> None:
        if self._waits == 0:
            return
        self._waits -= 1
        if self._waits == 0 and self._wait_since is not None:
            self.waiting_s += max(0.0, now_s - self._wait_since)
            self._wait_since = None

    @property
    def match_s(self) -> float:
        if self.start_s is None or self.last_s is None:
            return 0.0
        return self.last_s - self.start_s

    def waiting_share(self, now_s: float | None = None) -> float:
        """Share of match time spent waiting on the brain; an open wait counts up to `now_s` (default: last seen)."""
        now = self.last_s if now_s is None else now_s
        waiting = self.waiting_s
        if self._wait_since is not None and now is not None:
            waiting += max(0.0, now - self._wait_since)
        span = (now - self.start_s) if (now is not None and self.start_s is not None) else 0.0
        return min(1.0, waiting / span) if span > 0 else 0.0

    def summary(self, clock: Mapping[str, Any] | None = None) -> dict[str, Any]:
        minutes = self.match_s / 60
        p50, p95 = percentile(self.latencies_s, 50), percentile(self.latencies_s, 95)
        return {
            "player": self.player,
            "match_s": round(self.match_s, 1),
            "waiting_share": round(self.waiting_share(), 4) if self.waits_reported else None,
            "idle_villager_s": round(self.idle_villager_s, 1),
            "idle_production_s": round(self.idle_production_s, 1),
            "decisions": self.decisions,
            "missed": self.missed,
            "decisions_per_min": round(self.decisions / minutes, 2) if minutes > 0 else 0.0,
            "latency_s": {
                "p50": None if p50 is None else round(p50, 3),
                "p95": None if p95 is None else round(p95, 3),
                "max": round(max(self.latencies_s), 3) if self.latencies_s else None,
                "n": len(self.latencies_s),
            },
            "clock_overruns": int((clock or {}).get("overruns", 0)),
            "worst_overrun_ms": float((clock or {}).get("worst_overrun_ms", 0.0)),
        }


class MatchMetrics:
    """All seats of one match; feed every snapshot, read `summary(clock.metrics())` for results."""

    def __init__(self, players: Sequence[int]):
        self.seats = {int(p): SeatMetrics(int(p)) for p in players}

    def observe(self, snapshot: Snapshot) -> None:
        for seat in self.seats.values():
            seat.observe(snapshot)

    def summary(self, clock: Mapping[str, Any] | None = None) -> dict[int, dict[str, Any]]:
        return {p: seat.summary(clock) for p, seat in self.seats.items()}
