"""Snapshot parsing and the event differ, on snapshots recorded from the real engine (scripts/record_snapshots.py)."""

import copy
import json
from pathlib import Path

import pytest

from agentenv_zeroad.engine import FakeEngine
from agentenv_zeroad.state import (
    RAID_QUIET_MS,
    ROLES,
    Event,
    EventTracker,
    Snapshot,
    SnapshotQuery,
    diff_events,
    snapshot_js,
)

FIXTURES = Path(__file__).parent / "fixtures"


def raw(seconds: int) -> dict:
    return json.loads((FIXTURES / f"snapshot_2p_{seconds:03d}s.json").read_text())


def snap(seconds: int) -> Snapshot:
    return Snapshot.from_json(raw(seconds))


def test_recorded_snapshot_parses_into_roles_resources_and_buildings():
    s = snap(60)
    assert s.time_ms == 60_000
    rome = s.players[1]
    assert (rome.name, rome.civ, rome.phase) == ("Rome", "rome", "village")
    assert set(rome.resources) == {"food", "wood", "stone", "metal"}
    assert set(rome.units) <= set(ROLES) | {"other"}
    assert rome.units["villager"] == 9
    assert rome.pop_cap >= rome.pop > 0
    assert "CivCentre" in rome.buildings_by_kind()
    assert rome.base is not None
    assert rome.stats["units_trained"] == 7


def test_differ_reports_training_between_recorded_snapshots():
    events = diff_events(snap(10).players[1], snap(60).players[1], 60_000)
    kinds = [e.kind for e in events]
    assert kinds == ["unit_trained"]
    assert events[0].data["count"] == 7
    assert events[0].data["villager"] == 5


def _with(seconds: int, player: int, **changes) -> Snapshot:
    data = copy.deepcopy(raw(seconds))
    data["players"][str(player)].update(changes)
    return Snapshot.from_json(data)


def test_differ_reports_attack_once_per_raid_and_again_on_building_damage():
    base = snap(120)
    enemy = {"2": {"units": {"cavalry": 5}, "buildings": [], "nearest": 30}}
    raided = _with(120, 1, enemies=enemy, enemies_near_base=5)
    first = diff_events(base.players[1], raided.players[1], 121_000)
    assert [e.kind for e in first] == ["under_attack"]
    assert first[0].urgent and first[0].data["by"] == [2]
    assert diff_events(raided.players[1], raided.players[1], 122_000) == []
    buildings = copy.deepcopy(raw(120)["players"]["1"]["buildings"])
    buildings[0]["hp"] = 40
    damaged = _with(120, 1, enemies=enemy, enemies_near_base=5, buildings=buildings)
    again = diff_events(raided.players[1], damaged.players[1], 123_000)
    assert [e.kind for e in again] == ["under_attack"]
    assert "damaged" in again[0].text


def test_differ_reports_building_and_unit_losses_from_engine_counters():
    prev = raw(120)["players"]["1"]
    keep = [b for b in prev["buildings"] if b["kind"] != "CivCentre"]
    stats = dict(prev["stats"], buildings_lost=1, units_lost=3)
    units = dict(prev["units"], villager=prev["units"]["villager"] - 3)
    cur = _with(120, 1, buildings=keep, stats=stats, units=units)
    events = diff_events(snap(120).players[1], cur.players[1], 130_000)
    by_kind = {e.kind: e for e in events}
    assert by_kind["building_lost"].data == {"count": 1, "kinds": ["CivCentre"]}
    assert by_kind["unit_lost"].data["count"] == 3 and by_kind["unit_lost"].data["villager"] == 3
    assert by_kind["building_lost"].urgent


def test_differ_reports_phase_tech_and_resource_exhaustion():
    prev = raw(120)["players"]["1"]
    nearby = copy.deepcopy(prev["nearby_resources"])
    nearby["food"] = {"amount": 0, "sources": 0}
    nearby["stone"]["sources"] = 0
    nearby["stone"]["amount"] = 0
    cur = _with(
        120, 1, phase="town", techs=[*prev["techs"], "phase_town", "gather_lumbering_ironaxes"], nearby_resources=nearby
    )
    events = diff_events(snap(120).players[1], cur.players[1], 200_000)
    kinds = [e.kind for e in events]
    assert kinds == ["phase_reached", "tech_researched", "resource_exhausted", "resource_exhausted"]
    assert events[1].data == {"tech": "gather_lumbering_ironaxes"}
    assert [e.data["resource"] for e in events[2:]] == ["food", "stone"]


def test_event_tracker_merges_training_until_drained():
    tracker = EventTracker(1)
    assert tracker.update(snap(10)) == []
    tracker.update(snap(60))
    tracker.update(snap(120))
    pending = tracker.peek()
    trained = snap(120).players[1].stats["units_trained"] - snap(10).players[1].stats["units_trained"]
    assert [(e.kind, e.data["count"], e.time_ms) for e in pending] == [("unit_trained", trained, 120_000)]
    assert not tracker.has_urgent()
    assert tracker.drain() == pending
    assert tracker.drain() == []


def _tick(seconds: int, t_ms: int, player: int = 1, **changes) -> Snapshot:
    data = copy.deepcopy(raw(seconds))
    data["players"][str(player)].update(changes)
    data["time_ms"] = t_ms
    return Snapshot.from_json(data)


RAID = {"2": {"units": {"cavalry": 5}, "buildings": [], "nearest": 30}}


def test_tracker_reports_a_damaging_raid_once_and_a_newly_hit_building_again():
    p1 = raw(120)["players"]["1"]
    tracker = EventTracker(1)
    tracker.update(_tick(120, 120_000))
    buildings = copy.deepcopy(p1["buildings"])
    for i in range(25):  # 5 s of a raid with the first building losing 1% hp per 200 ms tick
        buildings[0]["hp"] = 99 - i
        tracker.update(
            _tick(120, 120_200 + 200 * i, enemies=RAID, enemies_near_base=5, buildings=copy.deepcopy(buildings))
        )
    assert [e.kind for e in tracker.peek()].count("under_attack") == 1
    assert tracker.has_urgent()
    buildings[1]["hp"] = 50
    fresh = tracker.update(_tick(120, 125_400, enemies=RAID, enemies_near_base=5, buildings=copy.deepcopy(buildings)))
    assert [e.kind for e in fresh] == ["under_attack"] and fresh[0].data["damaged"] == [buildings[1]["id"]]
    tracker.drain()
    assert not tracker.has_urgent()
    quiet_end = 125_400 + RAID_QUIET_MS + 200
    tracker.update(_tick(120, quiet_end - 200, buildings=copy.deepcopy(buildings)))
    again = tracker.update(_tick(120, quiet_end, enemies=RAID, enemies_near_base=5, buildings=copy.deepcopy(buildings)))
    assert [e.kind for e in again] == ["under_attack"], "a new raid after a quiet spell is reported"


def test_tracker_keeps_the_attack_through_a_long_battle_of_losses():
    p1 = raw(120)["players"]["1"]
    tracker = EventTracker(1, keep=50)
    tracker.update(_tick(120, 120_000))
    tracker.update(_tick(120, 120_200, enemies=RAID, enemies_near_base=5))
    for i in range(1, 61):
        stats = dict(p1["stats"], units_lost=p1["stats"]["units_lost"] + i)
        tracker.update(_tick(120, 120_200 + 200 * i, enemies=RAID, enemies_near_base=5, stats=stats))
    pending = tracker.peek()
    assert [e.kind for e in pending] == ["under_attack", "unit_lost"]
    assert pending[1].data["count"] == 60
    assert tracker.has_urgent()
    tracker.drain()
    assert not tracker.has_urgent()


def test_tracker_cap_drops_routine_events_before_urgent_ones():
    tracker = EventTracker(1, keep=5)
    tracker._pending = [Event("building_lost", 0, "lost 1 building(s)", {"count": 1})]
    tracker._pending += [Event("tech_researched", i, f"researched t{i}", {"tech": f"t{i}"}) for i in range(1, 10)]
    tracker._cap()
    assert [e.kind for e in tracker.peek()][0] == "building_lost" and len(tracker.peek()) == 5
    assert tracker.peek()[-1].text == "researched t9"


def test_losing_the_civic_centre_does_not_report_resources_exhausted():
    prev = raw(120)["players"]["1"]
    keep = [b for b in prev["buildings"] if b["kind"] != "CivCentre"]
    nearby = {r: {"amount": 0, "sources": 0} for r in prev["nearby_resources"]}
    stats = dict(prev["stats"], buildings_lost=prev["stats"]["buildings_lost"] + 1)
    cur = _with(120, 1, buildings=keep, nearby_resources=nearby, stats=stats)
    events = diff_events(snap(120).players[1], cur.players[1], 130_000)
    assert [e.kind for e in events] == ["building_lost"]


def test_differ_reports_a_lost_mine_while_other_stone_remains():
    prev = raw(120)["players"]["1"]
    nearby = copy.deepcopy(prev["nearby_resources"])
    before = copy.deepcopy(nearby)
    before["stone"] = {"amount": 7000, "sources": 2}
    nearby["stone"] = {"amount": 4000, "sources": 1}
    events = diff_events(
        _with(120, 1, nearby_resources=before).players[1], _with(120, 1, nearby_resources=nearby).players[1], 1
    )
    assert [(e.kind, e.data["resource"], e.data["left"]) for e in events] == [("resource_exhausted", "stone", 4000)]
    assert "4000 left in 1 source(s)" in events[0].text


def test_differ_counts_a_training_and_a_death_in_the_same_interval_from_engine_totals():
    prev = raw(120)["players"]["1"]
    stats = dict(
        prev["stats"], units_trained=prev["stats"]["units_trained"] + 1, units_lost=prev["stats"]["units_lost"] + 1
    )
    events = diff_events(snap(120).players[1], _with(120, 1, stats=stats).players[1], 121_000)
    assert [(e.kind, e.data["count"]) for e in events] == [("unit_lost", 1), ("unit_trained", 1)]


def test_snapshot_js_carries_viewers_and_reads_every_player_from_the_engine():
    code = snapshot_js([1], base_radius=70)
    assert "__CONFIG__" not in code
    assert '{"players": [1], "baseRadius": 70}' in code
    assert "GetNumPlayers()" in code and "cfg.allPlayers" not in code
    assert SnapshotQuery(FakeEngine(), [1, 2]).code == snapshot_js([1, 2])


def test_snapshot_query_times_each_call_against_fake_engine():
    async def go():
        query = SnapshotQuery(FakeEngine(), [1, 2])
        engine = query.engine
        engine._evaluations[query.code] = raw(60)
        result = await query()
        return result, query

    import asyncio

    result, query = asyncio.run(go())
    assert result.players[2].civ == "pers"
    assert len(query.costs_s) == 1


def test_snapshot_query_rejects_non_object_reply():
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(SnapshotQuery(FakeEngine(), [1])())
