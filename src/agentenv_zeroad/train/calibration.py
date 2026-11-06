"""The game specs of a run config's panels, and the teacher calibration panel's in particular.

Distillation (`train/distill.py`) needs a teacher, and the best one is the frontier model that plays this game best,
not the one that happened to beat the base Qwen. The calibration panel measures that where the policy is measured: each
configured teacher (Astra, Opus, any LiteLLM model name) plays Petra-3 and Petra-4 on interface v2 in lockstep at the
policy's own latency `L` (`games.latency_ms`, unless the panel sets one), turn-logged like every lockstep game, about 10
games in all by default. The Petra seat is the same v1 controller a training game's is. The games run under the
orchestrator's frontier spend caps (`game_cap_usd` per seat, then the run's budget) and land in ROOT/panels/NAME, kept
off the training board; `report.py` summarises them (win rate per teacher and rung with Wilson intervals) and
`distill.py --calibration` picks the teacher from that folder.

The games are dealt one cell per teacher and rung, rung-major so that a short count's extra games go to the lowest rung
that decides the pick (Petra-3), and the j-th game of every cell shares its map, seeds and civ pair, so the teachers
face the same positions. Held-out civ pairs, maps and seeds are used when the config has them, else the training
ones. Which side a teacher takes alternates.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping

from agentenv_zeroad.train.config import PanelConfig, RunConfig, TeacherConfig
from agentenv_zeroad.train.spec import GameSpec, SeatSpec, load_specs

LITELLM_URL_ENV = "LITELLM_BASE_URL"
UNSET_URL = "http://litellm.invalid/v1"  # in a spec made where no endpoint is configured (a config check)


def panel_specs(cfg: RunConfig, panel: PanelConfig, environ: Mapping[str, str] | None = None) -> list[GameSpec]:
    """The games of `panel`: its spec file's, or the calibration's."""
    if panel.kind == "teacher_calibration":
        return calibration_specs(cfg, panel, environ)
    assert panel.specs is not None
    return load_specs(panel.specs)


def calibration_specs(cfg: RunConfig, panel: PanelConfig, environ: Mapping[str, str] | None = None) -> list[GameSpec]:
    """`panel.games` games: every teacher against every rung, at the policy's latency, same positions each."""
    environ = os.environ if environ is None else environ
    cells = [(rung, teacher) for rung in panel.rungs for teacher in panel.teachers]
    per_cell = [panel.games // len(cells) + (i < panel.games % len(cells)) for i in range(len(cells))]
    base_url = panel.base_url or environ.get(LITELLM_URL_ENV) or UNSET_URL
    latency = panel.latency_ms if panel.latency_ms is not None else cfg.games.latency_ms
    positions = _positions(cfg, max(per_cell), panel.name)
    specs = []
    for (rung, teacher), games in zip(cells, per_cell, strict=True):
        for j in range(games):
            civ_teacher, civ_petra, map_name, seed, ai_seed = positions[j]
            seat = _teacher_seat(teacher, civ_teacher, base_url, panel.api_key_env, cfg)
            petra = SeatSpec(kind="petra", civ=civ_petra, difficulty=rung, behaviour="balanced")
            specs.append(
                GameSpec(
                    game_id=f"{_plain(teacher.name)}-petra{rung}-{j:02d}",
                    seats=(seat, petra) if j % 2 == 0 else (petra, seat),
                    map=map_name,
                    seed=seed,
                    ai_seed=ai_seed,
                    size=cfg.games.size,
                    biome=cfg.games.biome,
                    placement=cfg.games.placement,
                    time_cap_minutes=cfg.games.time_cap_minutes,
                    latency_ms=latency,
                    score_rule=cfg.games.score_rule,
                )
            )
    return specs


def _teacher_seat(teacher: TeacherConfig, civ: str, base_url: str, key_env: str, cfg: RunConfig) -> SeatSpec:
    return SeatSpec(
        kind="llm",
        civ=civ,
        name=teacher.name,
        model=teacher.model,
        base_url=base_url,
        api_key_env=key_env,
        effort=teacher.effort,
        max_tokens=teacher.max_tokens,
        interface=cfg.games.interface,
    )


def _positions(cfg: RunConfig, n: int, salt: str) -> list[tuple[str, str, str, int, int]]:
    """n (teacher civ, Petra civ, map, seed, AI seed) draws from the held-out configs when there are any."""
    rng = random.Random(f"{cfg.seed}:{salt}")
    held = cfg.held_out
    pairs = [tuple(p) for p in held.civ_pairs] or cfg.training_civ_pairs()
    maps = held.maps or cfg.games.maps
    seeds = list(held.seeds)
    out = []
    for j in range(n):
        civs = rng.choice(pairs)
        seed = seeds[j % len(seeds)] if seeds else rng.randrange(cfg.games.seed_low, cfg.games.seed_high)
        out.append((civs[0], civs[1], rng.choice(maps), seed, rng.randrange(cfg.games.seed_low, cfg.games.seed_high)))
    return out


def _plain(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "teacher"
