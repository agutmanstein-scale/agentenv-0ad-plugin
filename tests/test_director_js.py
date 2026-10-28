"""The director mod's camera logic (mod/zeroad-director/gui/session/zeroad_director.js), run under node with a stubbed
engine and clock: it cuts to the largest fight, else tours the bases, holds every cut at least 8 s, glides smoothly,
follows a moving fight without cutting, falls back to the stock entity calls, and hooks the session's onTick."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node to run the mod's JavaScript")
MOD = Path(__file__).resolve().parents[1] / "mod/zeroad-director"
DIRECTOR = (MOD / "gui/session/zeroad_director.js").read_text()

HARNESS = """
let sessionTicks = 0;
var onTick = function() { sessionTicks += 1; };
%s
const scenes = %s;          // [[from_ms, scene], ...]: the scene from that time on
let now = 0;
const moves = [];
let pivot = {x: 0, z: 0};
const engine = {
  GuiInterfaceCall(name, args) {
    if (name === "ZeroadDirectorScene") {
      if (%s) throw new Error("Invalid GuiInterface Call name");
      let scene = scenes[0][1];
      for (const [from, s] of scenes) if (now >= from) scene = s;
      return scene;
    }
    if (name === "GetNonGaiaEntities") return [11, 12, 13];
    if (name === "GetMultipleEntityStates") return [
      {entId: 11, state: {player: 1, position: {x: 100, z: 100}, identity: {classes: ["CivCentre"]}}},
      {entId: 12, state: {player: 2, position: {x: 500, z: 500}, identity: {classes: ["CivCentre"]}}},
      {entId: 13, state: null}];
    throw new Error(name);
  },
  GetCameraPivot() { return pivot; },
  CameraMoveTo(x, z) { pivot = {x, z}; moves.push([now, x, z]); },
  log() {}
};
const d = new ZeroadDirector(engine, () => now);
for (now = 0; now <= %d; now += 33) d.tick();
onTick(); onTick();  // no Engine under node: the director stops, the session's own tick goes on
console.log(JSON.stringify({cuts: d.cuts, moves, sessionTicks, failed: g_ZeroadDirectorFailed, fps: d.fps}));
"""

BASES = [{"x": 100, "z": 100, "owner": 1}, {"x": 500, "z": 500, "owner": 2}]


def fight_at(x: float, z: float, n: int = 6) -> list[dict]:
    return [{"x": x + i, "z": z - i, "owner": 1 + i % 2} for i in range(n)]


def direct(scenes: list[tuple[int, dict]], until_ms: int, *, no_mod_call: bool = False) -> dict:
    code = HARNESS % (DIRECTOR, json.dumps(scenes), "true" if no_mod_call else "false", until_ms)
    done = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(done.stdout)


def test_with_no_fight_it_tours_the_bases_holding_each_shot():
    out = direct([(0, {"fights": [], "bases": BASES})], 40_000)
    kinds = [(c["kind"], c["x"], c["z"]) for c in out["cuts"]]
    assert kinds[:3] == [("base", 100, 100), ("base", 500, 500), ("base", 100, 100)]
    gaps = [b["at"] - a["at"] for a, b in zip(out["cuts"], out["cuts"][1:], strict=False)]
    assert gaps and min(gaps) >= 14_000


def test_it_cuts_to_the_largest_fight_but_only_after_holding_the_shot_8_s():
    small = fight_at(480, 120, n=3)
    big = fight_at(300, 300, n=9)
    scenes = [(0, {"fights": [], "bases": BASES}), (3_000, {"fights": small + big, "bases": BASES})]
    out = direct(scenes, 20_000)
    first, second = out["cuts"][:2]
    assert first["kind"] == "base"
    assert second["kind"] == "fight" and abs(second["x"] - 304) < 10 and abs(second["z"] - 296) < 10
    assert second["at"] - first["at"] >= 8_000


def test_units_of_one_player_in_combat_are_a_hunt_not_a_fight_unless_many():
    hunt = [{"x": 300 + i, "z": 300, "owner": 1} for i in range(5)]
    out = direct([(0, {"fights": hunt, "bases": BASES})], 20_000)
    assert {c["kind"] for c in out["cuts"]} == {"base"}
    raid = [{"x": 300 + i, "z": 300, "owner": 1} for i in range(9)]
    out = direct([(0, {"fights": raid, "bases": BASES})], 2_000)
    assert out["cuts"][0]["kind"] == "fight"


def test_a_moving_fight_is_followed_without_new_cuts():
    scenes = [(0, {"fights": fight_at(300, 300), "bases": BASES})]
    scenes += [(t, {"fights": fight_at(300 + t / 1000, 300), "bases": BASES}) for t in range(1_000, 30_000, 1_000)]
    out = direct(scenes, 30_000)
    assert [c["kind"] for c in out["cuts"]] == ["fight"]
    assert out["moves"][-1][1] > 315  # the camera went along with the fight


def test_the_camera_glides_between_shots_with_no_jumps():
    out = direct([(0, {"fights": [], "bases": BASES})], 20_000)
    moves = out["moves"]
    steps = [((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 for a, b in zip(moves, moves[1:], strict=False)]
    # the cut from (100,100) to (500,500) is 566 m: a 3 s glide at 30 fps moves at most a few % of it per frame
    assert max(steps) < 0.05 * 566
    at_tour = next(m for m in moves if m[0] >= out["cuts"][1]["at"] + 3_000)
    assert abs(at_tour[1] - 500) < 1 and abs(at_tour[2] - 500) < 1


def test_without_the_mods_call_it_reads_the_stock_entity_states():
    out = direct([(0, {"fights": [], "bases": BASES})], 16_000, no_mod_call=True)
    assert [(c["kind"], c["x"], c["z"]) for c in out["cuts"][:2]] == [("base", 100, 100), ("base", 500, 500)]


def test_it_measures_the_frame_rate_from_its_ticks():
    out = direct([(0, {"fights": [], "bases": BASES})], 12_000)
    assert abs(out["fps"] - 1000 / 33) < 0.5


def test_it_hooks_the_sessions_tick_and_a_failing_director_never_stops_the_session():
    out = direct([(0, {"fights": [], "bases": BASES})], 0)
    assert out["sessionTicks"] == 2 and out["failed"] is True


def hurt_at(x: float, z: float, t_ms: int, *, owner: int = 2, n: int = 3) -> list[dict]:
    """n damaged entities at (x, z) losing a hitpoint every second from t=0."""
    return [{"id": 900 + i, "x": x + i, "z": z, "owner": owner, "hp": 500 - t_ms // 1000} for i in range(n)]


def test_units_stuck_in_combat_that_hit_nothing_hold_no_shot_once_the_scene_has_hitpoints():
    stuck = [{"x": 300 + i, "z": 300, "owner": 1 + i % 2} for i in range(20)]
    out = direct([(0, {"fights": stuck, "bases": BASES, "hurt": []})], 30_000)
    assert {c["kind"] for c in out["cuts"]} == {"base"}


def test_a_fight_that_deals_damage_is_shot():
    scenes = [
        (t, {"fights": fight_at(300, 300), "bases": BASES, "hurt": hurt_at(300, 300, t)})
        for t in range(0, 20_000, 1_000)
    ]
    out = direct(scenes, 20_000)
    shots = [c for c in out["cuts"] if c["kind"] == "fight"]
    assert shots and abs(shots[0]["x"] - 300) < 10 and abs(shots[0]["z"] - 300) < 10


def test_a_base_razed_by_one_player_is_a_fight_of_two():
    attackers = [{"x": 300 + i, "z": 300, "owner": 1} for i in range(4)]
    scenes = [
        (t, {"fights": attackers, "bases": BASES, "hurt": hurt_at(305, 300, t, owner=2, n=2)})
        for t in range(0, 20_000, 1_000)
    ]
    out = direct(scenes, 20_000)
    assert "fight" in [c["kind"] for c in out["cuts"]]


def test_a_bigger_clump_that_hits_nothing_loses_the_camera_to_a_real_fight():
    stuck = [{"x": 300 + i, "z": 300, "owner": 1 + i % 2} for i in range(20)]
    scenes = [
        (t, {"fights": stuck + fight_at(480, 120, n=4), "bases": BASES, "hurt": hurt_at(480, 120, t)})
        for t in range(0, 20_000, 1_000)
    ]
    out = direct(scenes, 20_000)
    shots = [c for c in out["cuts"] if c["kind"] == "fight"]
    assert shots and all(abs(c["x"] - 480) < 20 and abs(c["z"] - 120) < 20 for c in shots)
