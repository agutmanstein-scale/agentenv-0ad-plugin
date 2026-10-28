"""Per-seat metric counters (spec §5)."""

import json
from pathlib import Path

from agentenv_zeroad.metrics import MatchMetrics, SeatMetrics, percentile
from agentenv_zeroad.state import Snapshot

FIXTURES = Path(__file__).parent / "fixtures"


def snap_at(time_ms: int, idle_villagers: int, idle_production: int) -> Snapshot:
    data = json.loads((FIXTURES / "snapshot_2p_010s.json").read_text())
    data["time_ms"] = time_ms
    data["players"]["1"]["idle_villagers"] = list(range(idle_villagers))
    data["players"]["1"]["idle_production"] = list(range(idle_production))
    return Snapshot.from_json(data)


def test_idle_seconds_integrate_previous_counts_over_game_time():
    seat = SeatMetrics(1)
    seat.observe(snap_at(0, 4, 1))
    seat.observe(snap_at(5000, 2, 0))
    seat.observe(snap_at(15000, 0, 0))
    assert seat.idle_villager_s == 4 * 5 + 2 * 10
    assert seat.idle_production_s == 1 * 5
    assert seat.match_s == 15


def test_recorded_start_has_idle_villagers_and_production():
    seat = SeatMetrics(1)
    data = json.loads((FIXTURES / "snapshot_2p_010s.json").read_text())
    first = Snapshot.from_json({**data, "time_ms": 0})
    seat.observe(first)
    seat.observe(Snapshot.from_json({**data, "time_ms": 1000}))
    assert seat.idle_villager_s == len(first.players[1].idle_villagers)


def test_waiting_share_counts_overlapping_waits_once():
    seat = SeatMetrics(1)
    seat.observe(snap_at(0, 0, 0))
    seat.begin_waiting(10)
    seat.begin_waiting(12)
    seat.end_waiting(14)
    seat.end_waiting(20)
    seat.end_waiting(21)  # unmatched end is ignored
    seat.observe(snap_at(100_000, 0, 0))
    assert seat.waiting_s == 10
    assert seat.waiting_share() == 0.1
    seat.begin_waiting(90)
    assert seat.waiting_share() == 0.2  # an open wait counts up to the last snapshot


def test_waiting_share_is_unmeasured_until_a_wait_is_reported():
    seat = SeatMetrics(1)
    seat.observe(snap_at(0, 0, 0))
    seat.observe(snap_at(100_000, 0, 0))
    assert seat.summary()["waiting_share"] is None
    seat.begin_waiting(50)
    seat.end_waiting(60)
    assert seat.summary()["waiting_share"] == 0.1


def test_summary_has_latency_percentiles_rate_and_clock_overruns():
    seat = SeatMetrics(2)
    seat.observe(snap_at(0, 0, 0))
    seat.observe(snap_at(120_000, 0, 0))
    for latency in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        seat.record_decision(latency)
    seat.record_decision(130, missed=True)
    out = seat.summary({"overruns": 3, "worst_overrun_ms": 41.5})
    assert out["latency_s"] == {"p50": 6, "p95": 130, "max": 130, "n": 11}  # missed calls count as samples
    assert out["decisions"] == 10 and out["missed"] == 1
    assert out["decisions_per_min"] == 5.0
    assert (out["clock_overruns"], out["worst_overrun_ms"]) == (3, 41.5)


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([3.0], 95) == 3.0
    assert percentile(list(range(1, 101)), 95) == 95


def test_match_metrics_feeds_every_seat():
    match = MatchMetrics([1, 2])
    match.observe(snap_at(0, 3, 0))
    match.observe(snap_at(2000, 0, 0))
    summary = match.summary()
    assert summary[1]["idle_villager_s"] == 6
    assert set(summary) == {1, 2}
