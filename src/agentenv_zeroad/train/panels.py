"""The game specs of the panels a config can ask for without a spec file: held-out panels and frontier baselines.

`heldout`: the policy seat (the version the panel is pinned to, v0 by default) plays `games` games against an opponent
mix (`opponents`, Petra levels, the scripted bot, the frozen base model), every game on a position drawn from the
held-out maps, civ pairs and seeds (`positions = "training"` draws from the training ones instead, which a short control
arm that must resemble a training run uses). The opponent of game `j` is dealt by running apportionment, so any prefix
of the panel is balanced across the mix, and the position of game `j` depends only on `j` and the run's seed, never on
the panel's size or its opponents: a 48-game v1 subset, a 192-game SFT-epoch panel and the 384-game panel see the same
maps, seeds and civ pairs in the same order (and the frozen policy seat takes the side `j` alternates it to).

`frontier`: each teacher in the panel plays the policy seat for its own `games` games at the policy's lockstep latency,
on the same positions, sides alternating. The policy seat is trainable and pinned to the panel's version, so the report
reads the game from its side and the teacher's name is the opponent on the board.

Games are lockstep games like every other, so both seats wait the same game-seconds for an answer whatever the model's
wall-clock latency is.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass

from agentenv_zeroad import grading
from agentenv_zeroad.interface import Interface
from agentenv_zeroad.train.config import PanelConfig, RunConfig, TeacherConfig
from agentenv_zeroad.train.mix import PETRA_BEHAVIORS, llm_seat
from agentenv_zeroad.train.spec import GameSpec, SeatSpec

UNSET_URL = "http://litellm.invalid/v1"


@dataclass(frozen=True)
class Position:
    """Where and how game j is played: the policy's and the opponent's civs, the map, the seeds, Petra's behaviour."""

    policy_civ: str
    opponent_civ: str
    map: str
    seed: int
    ai_seed: int
    behaviour: str


def position(cfg: RunConfig, kind: str, j: int) -> Position:
    """Game j's position, from the held-out configs (`kind` "held_out") or the training ones ("training")."""
    rng = random.Random(f"{cfg.seed}:{kind}:{j}")
    g, held = cfg.games, cfg.held_out
    if kind == "held_out":
        pairs = [tuple(p) for p in held.civ_pairs] or cfg.training_civ_pairs()
        maps = held.maps or g.maps
        seeds = held.all_seeds()
    else:
        pairs, maps, seeds = cfg.training_civ_pairs(), g.maps, []
    a, b = rng.choice(pairs)
    if rng.random() < 0.5:
        a, b = b, a
    return Position(
        a,
        b,
        rng.choice(maps),
        seeds[j % len(seeds)] if seeds else _seed(cfg, rng),
        _seed(cfg, rng, False),
        rng.choice(PETRA_BEHAVIORS),
    )


def _seed(cfg: RunConfig, rng: random.Random, avoid_held: bool = True) -> int:
    g = cfg.games
    for _ in range(10_000):
        seed = rng.randrange(g.seed_low, g.seed_high)
        if not (avoid_held and cfg.held_out.is_held_seed(seed)):
            return seed
    raise ValueError("no seed in games.seed_low..seed_high is outside the held-out seeds")


def schedule(shares: Mapping[str, float], n: int) -> list[str]:
    """n opponents by running apportionment: each game goes to the opponent furthest below its share, so every prefix
    holds the shares to within one game."""
    total = sum(shares.values())
    wanted = {k: w / total for k, w in shares.items() if w > 0}
    count = dict.fromkeys(wanted, 0)
    out = []
    for i in range(n):
        pick = max(wanted, key=lambda k: (wanted[k] * (i + 1) - count[k], -list(wanted).index(k)))
        count[pick] += 1
        out.append(pick)
    return out


def _interface(cfg: RunConfig, panel: PanelConfig) -> Interface:
    return panel.interface or cfg.games.interface


def _score_rule(cfg: RunConfig, interface: Interface) -> grading.ScoreRule | None:
    """The config's rule for the interface it names; another interface's panel takes that interface's own rule."""
    return cfg.games.score_rule if interface == cfg.games.interface else None


def _game(cfg: RunConfig, panel: PanelConfig, game_id: str, pos: Position, seats: tuple[SeatSpec, ...]) -> GameSpec:
    interface = _interface(cfg, panel)
    return GameSpec(
        game_id=game_id,
        seats=seats,
        map=pos.map,
        seed=pos.seed,
        ai_seed=pos.ai_seed,
        size=cfg.games.size,
        biome=cfg.games.biome,
        placement=cfg.games.placement,
        time_cap_minutes=panel.time_cap_minutes or cfg.games.time_cap_minutes,
        latency_ms=panel.latency_ms if panel.latency_ms is not None else cfg.games.latency_ms,
        score_rule=_score_rule(cfg, interface),
    )


def _policy(cfg: RunConfig, panel: PanelConfig, civ: str) -> SeatSpec:
    return llm_seat(cfg, civ, 0, cfg.servers.urls[0], trainable=True, interface=_interface(cfg, panel))


def heldout_specs(cfg: RunConfig, panel: PanelConfig) -> list[GameSpec]:
    """`panel.games` games of the policy seat against the panel's opponent mix, on held-out (or training) positions."""
    interface = _interface(cfg, panel)
    specs = []
    for j, opponent in enumerate(schedule(panel.opponents, panel.games)):
        pos = position(cfg, panel.positions, j)
        policy = _policy(cfg, panel, pos.policy_civ)
        if opponent == "bot":
            other = SeatSpec(kind="bot", civ=pos.opponent_civ)
        elif opponent == "base":
            other = llm_seat(cfg, pos.opponent_civ, 0, cfg.servers.urls[0], interface=interface)
        else:
            other = SeatSpec(
                kind="petra",
                civ=pos.opponent_civ,
                difficulty=int(opponent.removeprefix("petra-")),
                behaviour=pos.behaviour,
            )
        seats = (policy, other) if j % 2 == 0 else (other, policy)
        specs.append(_game(cfg, panel, f"{panel.name}-{j:04d}-{opponent}", pos, seats))
    return specs


def frontier_specs(cfg: RunConfig, panel: PanelConfig, base_url: str = UNSET_URL) -> list[GameSpec]:
    """Each teacher's own number of games against the policy seat, the same positions for every teacher."""
    specs = []
    for teacher in panel.teachers:
        for j in range(teacher.games):
            pos = position(cfg, panel.positions, j)
            seats = (
                _policy(cfg, panel, pos.policy_civ),
                _teacher_seat(cfg, panel, teacher, pos.opponent_civ, base_url),
            )
            specs.append(
                _game(
                    cfg,
                    panel,
                    f"{panel.name}-{_plain(teacher.name)}-{j:02d}",
                    pos,
                    seats if j % 2 == 0 else seats[::-1],
                )
            )
    return specs


def _teacher_seat(cfg: RunConfig, panel: PanelConfig, teacher: TeacherConfig, civ: str, base_url: str) -> SeatSpec:
    return SeatSpec(
        kind="llm",
        civ=civ,
        name=teacher.name,
        model=teacher.model,
        base_url=base_url,
        api_key_env=panel.api_key_env,
        effort=teacher.effort,
        max_tokens=teacher.max_tokens,
        interface=_interface(cfg, panel),
    )


def _plain(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "teacher"
