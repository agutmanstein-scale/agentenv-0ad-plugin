import dataclasses
import json
from pathlib import Path

import pytest

from agentenv_zeroad.showcase.specs import ShowcaseConfig, next_starts, used_seeds
from agentenv_zeroad.train.config import PanelConfig, load_config
from agentenv_zeroad.train.mix import repin
from agentenv_zeroad.train.panels import frontier_specs, game_at, heldout_specs
from agentenv_zeroad.train.spec import TRAIN_SAMPLING

ROOT = Path(__file__).resolve().parents[1]
LIVE_S2 = ROOT / "live" / "s2"
SHOWCASE = ShowcaseConfig.load(ROOT / "live" / "showcase" / "showcase.toml")
CFG = SHOWCASE.run_config()
PANEL = next(p for p in CFG.panels if p.name == SHOWCASE.panel)


def _same_but_id(a, b):
    return dataclasses.replace(a, game_id="same") == dataclasses.replace(b, game_id="same")


def test_showcase_config_loads():
    assert SHOWCASE.rules_config == LIVE_S2 / "run1-frontier.toml"
    assert SHOWCASE.rotation == ("opus", "astra", "petra-3")
    assert (SHOWCASE.first_index, SHOWCASE.adapter_version) == (400, 1)


def test_game_at_matches_frontier_specs():
    frontier = frontier_specs(CFG, PANEL)
    for i, teacher in enumerate(("opus", "astra")):
        for j in range(16):
            assert _same_but_id(game_at(CFG, PANEL, j, teacher), frontier[i * 16 + j])


def test_petra_game_matches_heldout_specs():
    panel = PanelConfig(name="p3", at_hours=0, kind="heldout", games=16, opponents={"petra-3": 1})
    heldout = heldout_specs(CFG, panel)
    for j in range(16):
        assert _same_but_id(game_at(CFG, PANEL, j, "petra-3"), heldout[j])


def test_teacher_settings_are_the_13_3s():
    base = load_config(LIVE_S2 / "baselines.toml")
    theirs = {t.name: t for t in next(p for p in base.panels if p.name == "frontier").teachers}
    for name in ("opus", "astra"):
        seat = game_at(CFG, PANEL, 1, name).seats[0]
        assert (seat.model, seat.effort, seat.max_tokens) == (
            theirs[name].model,
            theirs[name].effort,
            theirs[name].max_tokens,
        )


def test_policy_seat_samples_like_training():
    spec = repin(game_at(CFG, PANEL, 0, "opus"), SHOWCASE.adapter_version, "http://vllm.invalid")
    seat = next(s for s in spec.seats if s.trainable)
    assert seat.sampling == TRAIN_SAMPLING
    assert seat.model == "local/policy-v1"
    assert seat.interface == "v2"


def test_rules_are_the_13_3s():
    for opponent in SHOWCASE.rotation:
        game = game_at(CFG, PANEL, 3, opponent)
        assert (game.latency_ms, game.time_cap_minutes, game.score_rule) == (5000, 30, "net_worth")
        for seat in game.seats:
            if seat.kind == "llm":
                assert (seat.interval_s, seat.min_gap_s) == (25, 8)


def test_showcase_seeds_avoid_every_earlier_panel():
    # sequence.toml holds the sequence's settings and sft-panels.toml is a template whose panels s2 renders in; neither
    # is a run config that loads alone
    not_run_configs = {"sequence.toml", "sft-panels.toml"}
    used = used_seeds(sorted(p for p in LIVE_S2.glob("*.toml") if p.name not in not_run_configs))
    starts = next_starts(CFG, SHOWCASE.first_index, used, 20)
    assert len(starts) == 20 and starts == sorted(starts) and starts[0] >= 400
    assert all(game_at(CFG, PANEL, j, "opus").seed not in used for j in starts)


def test_used_seeds_include_earlier_sessions(tmp_path):
    first = next_starts(CFG, SHOWCASE.first_index, set(), 1)[0]
    seed = game_at(CFG, PANEL, first, "opus").seed
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"rows": [{"seed": seed, "game_id": "x"}], "ended": None}))
    used = used_seeds([], [session])
    assert used == {seed}
    assert first not in next_starts(CFG, SHOWCASE.first_index, used, 5)


def test_starts_never_wrap():
    left = len(CFG.held_out.all_seeds()) - SHOWCASE.first_index
    with pytest.raises(ValueError):
        next_starts(CFG, SHOWCASE.first_index, set(), left + 1)
