import asyncio
import json
from dataclasses import replace

import pytest

from agentenv_zeroad.placement import (
    DROPSITE_SERVES,
    EnginePlacer,
    Placement,
    Search,
    enemy_base_guess,
    placement_js,
    search_for,
)
from agentenv_zeroad.view import distance
from tests.executor_fixtures import load


def test_enemy_base_guess_mirrors_the_base_across_the_centre_for_two_players():
    seat, _ = load(300)
    x, z = enemy_base_guess(seat, 2, [1, 2])
    assert (round(x), round(z)) == (268, 140)  # Petra's real civic centre in this recorded game


def test_enemy_base_guess_spreads_four_players_on_the_base_circle():
    seat, _ = load(300)
    guesses = [enemy_base_guess(seat, q, [1, 2, 3, 4]) for q in (2, 3, 4)]
    radius = distance(seat.base, seat.center)
    assert all(abs(distance(g, seat.center) - radius) < 1 for g in guesses)
    assert len({(round(g[0]), round(g[1])) for g in guesses}) == 3


def test_base_search_rings_around_the_civic_centre():
    seat, _ = load(300)
    search = search_for(seat, "base", {})
    assert search.anchors == (seat.base,) and search.min_r > 0


def test_near_resource_anchors_on_the_nearest_site_no_drop_site_serves():
    seat, _ = load(300)
    search = search_for(seat, "near_wood", {})
    first = next(s for s in seat.sites["wood"] if s.distance > DROPSITE_SERVES)
    assert search.anchors == (first.pos, seat.base)
    served = replace(seat, sites={**seat.sites, "wood": tuple(replace(s, distance=5) for s in seat.sites["wood"])})
    assert search_for(served, "near_wood", {}).anchors[0] == seat.sites["wood"][0].pos


@pytest.mark.parametrize(("where", "metres"), [("toward_2", 45.0), ("toward_player_2", 45.0), ("front", 80.0)])
def test_toward_and_front_anchor_on_the_line_to_the_enemy(where, metres):
    seat, _ = load(300)
    enemy = (268.0, 140.0)
    search = search_for(seat, where, {2: enemy})
    assert distance(search.anchors[0], seat.base) == pytest.approx(metres, abs=0.01)
    assert distance(search.anchors[0], enemy) < distance(seat.base, enemy)
    assert search.anchors[-1] == seat.base  # falls back to the base when the forward rings have no valid spot


def test_no_base_means_no_search():
    seat, _ = load(300)
    assert search_for(replace(seat, base=None), "base", {}) is None


def test_placement_js_carries_the_search_and_engine_results_parse_into_plain_reasons():
    code = placement_js(1, "structures/rome/house", Search(((500.0, 628.0),), min_r=15, max_r=60))
    config = json.loads(code.rstrip()[code.rindex("})(") + 3 : -1])
    assert config["template"] == "structures/rome/house" and config["anchors"] == [[500.0, 628.0]]
    assert "CheckPlacement" in code and "AddLocalEntity" in code and "DestroyEntity" in code
    raw = {"ok": False, "tried": 9, "reasons": {"%(name)s cannot be built in unexplored area": 9}}
    assert Placement.from_json(raw).why() == "it cannot be built in unexplored area"


def test_engine_placer_runs_one_evaluate_per_search():
    class Engine:
        def __init__(self):
            self.codes = []

        async def evaluate(self, code):
            self.codes.append(code)
            return {"ok": True, "x": 520.5, "z": 640.0, "angle": 2.35, "tried": 4}

    engine = Engine()
    spot = asyncio.run(EnginePlacer(engine).find(1, "structures/rome/house", Search(((1.0, 2.0),))))
    assert spot.ok and (spot.x, spot.z, spot.tried) == (520.5, 640.0, 4) and len(engine.codes) == 1
