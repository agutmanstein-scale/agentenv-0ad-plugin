"""The stream of training game specs: Run 1's fixed opponent mix, random variety and the held-out exclusions.

`GameFactory.next(version, url)` makes the next game. The policy seat plays adapter `version` on the vLLM at `url`
(the orchestrator pins both at the moment the game starts); the opponent is drawn from the mix:

- the mix is exact per block, not just in expectation: opponents are dealt from a shuffled bag of `BLOCK` games with
  each share rounded by largest remainder, so every block of 100 games (games 0-99, 100-199, ...; a window that
  straddles two blocks can be off by a few) holds the configured shares and a rating or a milestone never waits on a
  lucky streak. Whether a game has no time cap (it ends by conquest or at the safety cap) is dealt from its own bag the
  same way, independently of the opponent;
- Petra's behaviour is drawn here from balanced, aggressive and defensive (never Petra's own `random`, so the spec
  records the one it played), the map seed and the AI's seed are separate draws, the policy's slot is random, and
  civs are drawn from every ordered pair the config allows;
- both LLM seats (the policy and the frozen base) play `games.interface`, and `games.score_rule` decides at the cap;
- nothing held out ever appears: no civ pair in `held_out.civ_pairs` (in either order), no map in `held_out.maps`
  (the config refuses overlap with the training maps) and no seed in `held_out.seeds`.

`repin` points an existing spec at another version or server, which is how a game voided by a vLLM outage is replayed
with the same map, civs, opponent and seeds. `GameFactory.state_dict()` / `load_state_dict()` carry the random state,
the bags and the game counter across an orchestrator restart, so a restarted run continues the same stream.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agentenv_zeroad.interface import Interface
from agentenv_zeroad.pilot.games import version_model
from agentenv_zeroad.scenario import AI_BEHAVIORS
from agentenv_zeroad.spend import LOCAL_PREFIX
from agentenv_zeroad.train.config import RunConfig
from agentenv_zeroad.train.spec import DEFAULT_MAX_TOKENS, TRAIN_SAMPLING, GameSpec, SeatSpec

BLOCK = 100
PETRA_BEHAVIORS = tuple(b for b in AI_BEHAVIORS if b != "random")
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def deal(shares: dict[str, float], n: int) -> list[str]:
    """n items with each key's count its share of n, rounded by largest remainder (the counts sum to n)."""
    exact = {k: share * n for k, share in shares.items()}
    counts = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    return [k for k, c in counts.items() for _ in range(c)]


class GameFactory:
    def __init__(self, cfg: RunConfig, *, start_index: int = 0):
        self.cfg = cfg
        self.index = start_index
        self.rng = random.Random(cfg.seed)
        self._opponents: list[str] = []
        self._uncapped: list[bool] = []
        self._pairs = cfg.training_civ_pairs()

    def next(self, version: int, url: str) -> GameSpec:
        """The next training game, its policy seat on adapter `version` served at `url`."""
        g, opponent = self.cfg.games, self._draw_opponent()
        policy_civ, opponent_civ = self.rng.choice(self._pairs)
        policy = self._llm_seat(policy_civ, version, url, trainable=True)
        other = self._opponent_seat(opponent, opponent_civ, url)
        seats = (policy, other) if self.rng.random() < 0.5 else (other, policy)
        game_id = f"{self.cfg.run_id}-g{self.index:06d}"
        self.index += 1
        return GameSpec(
            game_id=game_id,
            seats=seats,
            map=self.rng.choice(g.maps),
            seed=self._seed(),
            ai_seed=self._seed(),
            size=g.size,
            biome=g.biome,
            placement=g.placement,
            time_cap_minutes=None if self._draw_uncapped() else g.time_cap_minutes,
            latency_ms=g.latency_ms,
            score_rule=g.score_rule,
        )

    def _draw_opponent(self) -> str:
        if not self._opponents:
            self._opponents = deal(self.cfg.mix.shares(), BLOCK)
            self.rng.shuffle(self._opponents)
        return self._opponents.pop()

    def _draw_uncapped(self) -> bool:
        if not self._uncapped:
            n = round(self.cfg.games.no_cap_share * BLOCK)
            self._uncapped = [True] * n + [False] * (BLOCK - n)
            self.rng.shuffle(self._uncapped)
        return self._uncapped.pop()

    def _seed(self) -> int:
        g = self.cfg.games
        for _ in range(10_000):
            seed = self.rng.randrange(g.seed_low, g.seed_high)
            if not self.cfg.held_out.is_held_seed(seed):
                return seed
        raise ValueError("no seed in games.seed_low..seed_high is outside held_out.seeds")

    def state_dict(self) -> dict[str, Any]:
        version, internal, gauss = self.rng.getstate()
        return {
            "index": self.index,
            "rng": [version, list(internal), gauss],
            "opponents": list(self._opponents),
            "uncapped": list(self._uncapped),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.index = max(self.index, int(state["index"]))
        self._opponents, self._uncapped = list(state["opponents"]), list(state["uncapped"])

    def _llm_seat(self, civ: str, version: int, url: str, *, trainable: bool = False) -> SeatSpec:
        return llm_seat(self.cfg, civ, version, url, trainable=trainable)

    def _opponent_seat(self, opponent: str, civ: str, url: str) -> SeatSpec:
        if opponent == "bot":
            return SeatSpec(kind="bot", civ=civ)
        if opponent == "base":
            return self._llm_seat(civ, 0, url)
        level, behaviour = int(opponent.removeprefix("petra-")), self.rng.choice(PETRA_BEHAVIORS)
        return SeatSpec(kind="petra", civ=civ, difficulty=level, behaviour=behaviour)


def llm_seat(
    cfg: RunConfig, civ: str, version: int, url: str, *, trainable: bool = False, interface: Interface | None = None
) -> SeatSpec:
    """A seat on this box's vLLM playing adapter `version` (0 is the base model), sampled the way training samples."""
    return SeatSpec(
        kind="llm",
        civ=civ,
        model=version_model(version),
        base_url=url,
        api_key_env=cfg.servers.api_key_env,
        sampling=dict(TRAIN_SAMPLING),
        extra_body=dict(EXTRA_BODY),
        max_tokens=DEFAULT_MAX_TOKENS,
        trainable=trainable,
        interface=interface or cfg.games.interface,
    )


def repin(spec: GameSpec, version: int, url: str, *, game_id: str | None = None) -> GameSpec:
    """`spec` with its trainable seats on adapter `version` and every seat served by this box's vLLM (a `local/`
    model: the policy, a frozen version) pointed at `url`. A frontier seat is left as it is."""

    def pin(seat: SeatSpec) -> SeatSpec:
        if seat.kind != "llm":
            return seat
        if seat.trainable:
            return replace(seat, model=version_model(version), base_url=url)
        if (seat.model or "").startswith(LOCAL_PREFIX):
            return replace(seat, base_url=url)
        return seat

    return replace(spec, seats=tuple(pin(s) for s in spec.seats), game_id=game_id or spec.game_id)
