"""Credit for post-training v2: a finished training game in, the learner's inbox rows out.

The pilot scored every decision against a regression fitted on its own batch, and the scores carried no decision signal
(docs/post-training-v2.md). Here a decision gets only what the game's result says about it, less what that result was
expected to be:

  R   the learner seat's reward for the game. Conquest decides first: a win is +1, a loss -1. At a cap (time_cap or
      safety_cap) R = 0.5 sign(m) + 0.5 tanh(m / s), m the final net-worth margin and s the margin's within-opponent
      SD, so any cap win beats any cap loss and both sit inside a conquest win and loss. A game with no net_worth in its
      score falls back to the official score margin and is counted as such. A game that did not finish, or ended any
      other way, yields no rows.
  b   the baseline, a running mean of R per (opponent id, slot): an EMA over the last 64 games. With a critic
      (train/critic.py) that has earned its place, b is the critic's value at the decision's state.
  A   R - b, for each trained decision. Then centred within (opponent, 5-minute game-time bin) strata over a buffer of
      the last 4,096 decisions, divided by an EMA SD (of the mean square) over the last 4,096 decisions and clipped to
      +-5. A stratum is only centred once decisions from `min_stratum_games` games are in it: with one game's
      decisions all sharing one A, centring would zero them, and the first games of a run would teach nothing.

A seat-game trains 8 to 12 decisions (drawn per game), spread over the game-time bins. The advantages are mean-zero
within a stratum on purpose: a non-centred advantage brings back the self-imitation pull that shrank the pilot's
replies. The pre-centring mean and SD of every stratum are kept (`diagnostics()`); |mean / SD| above 0.1 for 3
consecutive updates' worth of decisions (256 each) is an alarm, because a good baseline leaves nothing to centre. The
batch is judged as a whole: its decisions pooled, the mean and SD of their pre-centring advantage, and the mean must
also be |z| > `drift_z` standard errors from zero, counting a game (not a decision) as one observation, since a game's
decisions share one A. Without that, a stratum's sampling noise alone puts the ratio over 0.1 most of the time.

Checked on every game; a failure is a CreditError, and the game gives no rows instead of rows from a game that makes
no sense (`on_error="raise"` lets the error out; by default it is counted and logged):
in a mirror game (two learner seats) R_seat1 = -R_seat2; at a cap the sign of m matches the recorded winner (when the
score rule is "net_worth", or the margin is the official one; under the official rule a net-worth margin may disagree
with the winner, which is counted instead).

Positive-control mode replaces R with the seat's standing army at game minute 10, z-scored per opponent: a signal the
policy can raise with the tools it has, to prove the whole path from game to update learns before any real run.

The learner seats are named by the caller or marked `"trainable": true` in the game spec's seat entries (the platform's
`SeatSpec.trainable`, which the mix sets on the policy seat).
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentenv_zeroad.train import results as rs

log = logging.getLogger(__name__)

SD_FLOOR = 1e-3
UNBOUNDED = 1e9  # the drift ratio of a batch with no spread at all, finite so the state stays valid JSON
STATE_VERSION = 2


class CreditError(ValueError):
    """A finished game whose results contradict themselves; no rows are emitted for it."""


class Critic(Protocol):
    """The part of train/critic.py that credit uses: a state value at a decision, and every game seen afterwards."""

    def baseline(self, results: Mapping[str, Any], slot: str, began_ms: float, fallback: float) -> float: ...

    def observe(self, results: Mapping[str, Any], slot: str, reward: float, baseline: float) -> None: ...


@dataclass(frozen=True)
class CreditConfig:
    margin_scale: float = 2000.0  # s: the net-worth margin's within-opponent SD
    baseline_games: int = 64
    decisions: tuple[int, int] = (8, 12)  # trained per seat-game, drawn uniformly per game
    bin_s: float = 300.0
    buffer_decisions: int = 4096
    sd_decisions: int = 4096
    clip: float = 5.0
    min_stratum_games: int = 4
    drift_limit: float = 0.1
    drift_z: float = 2.0
    on_error: str = "skip"  # "skip" a game that fails a check (counted, logged), or "raise" its CreditError
    drift_updates: int = 3
    decisions_per_update: int = 256
    positive_control: bool = False
    control_minute: float = 10.0
    control_games: int = 256  # the z-score's memory per opponent
    seed: int = 0


class Credit:
    def __init__(self, config: CreditConfig | None = None, *, critic: Critic | None = None):
        self.cfg = config or CreditConfig()
        self.critic = critic
        self.baselines: dict[str, list[float]] = {}  # "opponent|slot" -> [mean, games]
        self.buffer: deque[tuple[str, str, float]] = deque(maxlen=self.cfg.buffer_decisions)  # stratum, game, A
        self.sd_var: float | None = None
        self.sd_seen = 0
        self.control: dict[str, deque[float]] = {}
        self.counts: Counter[str] = Counter()
        self.strata: dict[str, dict[str, float]] = {}
        self.window: list[tuple[str, float]] = []  # (game:slot, raw A) of decisions not yet judged for drift
        self.batch: dict[str, float] = {}
        self.drift_streak = 0
        self.errors: list[str] = []
        self.last: dict[str, Any] = {}

    # ---- the game's rewards -----------------------------------------------------------------------------------
    def rows(self, results: Mapping[str, Any], learners: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """The inbox rows of one finished game, one per trained decision of each learner seat.

        A game that fails a check (`CreditError`) gives no rows and is counted and logged, so one bad game cannot
        take the orchestrator down; with `on_error="raise"` the error propagates instead."""
        try:
            return self._rows(results, learners)
        except CreditError as e:
            if self.cfg.on_error == "raise":
                raise
            game = _game_id(results)
            log.error("game %s not credited: %s", game, e)
            self.counts["credit_errors"] += 1
            self.errors = [*self.errors[-19:], f"{game}: {e}"]
            return []

    def _rows(self, results: Mapping[str, Any], learners: Sequence[str] | None) -> list[dict[str, Any]]:
        slots = rs.learner_slots(results, learners)
        if not slots:
            return self._skip("no_learner_seat")
        if results.get("status") != "finished":
            return self._skip(f"status_{results.get('status')}")
        if len(results["seats"]) != 2:
            return self._skip("not_1v1")
        if not self.cfg.positive_control and results.get("ended_by") not in ("conquest", *rs.CAP_ENDS):
            return self._skip(f"ended_by_{results.get('ended_by')}")
        rewards = {slot: self._reward(results, slot) for slot in slots}
        if len(slots) == 2 and not self.cfg.positive_control:
            first, second = (rewards[s][0] for s in slots)
            if not math.isclose(first, -second, abs_tol=1e-9):
                raise CreditError(f"mirror game: R of seat {slots[0]} is {first:.4f}, of seat {slots[1]} {second:.4f}")
        picked = {slot: self._decisions(results, slot, rewards[slot][0]) for slot in slots}
        for slot, (reward, _) in rewards.items():
            self._learn(results, slot, reward)
        out = self._normalised(results, picked)
        self.counts["games"] += 1
        self.counts["rows"] += len(out)
        self._tick(results, picked)
        self.last = {
            "game_id": (results.get("spec") or {}).get("game_id"),
            "ended_by": results.get("ended_by"),
            "rewards": {s: round(r, 4) for s, (r, _) in rewards.items()},
            "margin_sources": {s: m for s, (_, m) in rewards.items()},
        }
        return out

    def _skip(self, reason: str) -> list[dict[str, Any]]:
        self.counts[f"skipped:{reason}"] += 1
        return []

    def _reward(self, results: Mapping[str, Any], slot: str) -> tuple[float, str | None]:
        """(R, where the margin came from) for a learner seat."""
        opp = rs.other_slot(results, slot)
        assert opp is not None
        if self.cfg.positive_control:
            return self._control_reward(results, slot), None
        if results["ended_by"] == "conquest":
            outcome = results["seats"][slot].get("outcome")
            if outcome not in ("won", "lost", "drawn"):
                raise CreditError(f"seat {slot}: a conquest game with outcome {outcome!r}")
            return {"won": 1.0, "lost": -1.0, "drawn": 0.0}[outcome], None
        margin, source = rs.final_margin(results, slot, opp)
        if margin is None or source is None:
            raise CreditError(f"seat {slot}: a game that reached the cap with no score to take a margin from")
        if source != "net_worth":
            self.counts["margin_fallback_official_score"] += 1
        self._check_winner(results, slot, margin, source)
        return 0.5 * _sign(margin) + 0.5 * math.tanh(margin / self.cfg.margin_scale), source

    def _check_winner(self, results: Mapping[str, Any], slot: str, margin: float, source: str) -> None:
        outcome = results["seats"][slot].get("outcome")
        if margin == 0 or outcome not in ("won", "lost"):
            return
        agrees = _sign(margin) == (1.0 if outcome == "won" else -1.0)
        if agrees:
            return
        if results.get("score_rule", "official") == "net_worth" or source == "score":
            raise CreditError(f"seat {slot}: the margin {margin:.0f} at the cap disagrees with its outcome {outcome!r}")
        self.counts["net_worth_disagrees_with_official_winner"] += 1

    def _control_reward(self, results: Mapping[str, Any], slot: str) -> float:
        """Standing army at game minute 10, z-scored against the same opponent's earlier games."""
        row = rs.timeline_first_at_or_after(results, self.cfg.control_minute * 60_000)
        if row is None:
            raise CreditError("positive control needs the game's timeline")
        value = float(((row.get("players") or {}).get(slot) or {}).get("army") or 0)
        seen = self.control.setdefault(rs.player_id(results, rs.other_slot(results, slot) or ""), deque())
        z = 0.0
        if len(seen) >= 2 and statistics.pstdev(seen) > 1e-9:
            z = (value - statistics.fmean(seen)) / statistics.pstdev(seen)
        seen.append(value)
        while len(seen) > self.cfg.control_games:
            seen.popleft()
        return z

    # ---- the seat's decisions ---------------------------------------------------------------------------------
    def _key(self, results: Mapping[str, Any], slot: str) -> str:
        return f"{rs.player_id(results, rs.other_slot(results, slot) or '')}|{slot}"

    def _decisions(
        self, results: Mapping[str, Any], slot: str, reward: float
    ) -> list[tuple[Mapping[str, Any], float, str]]:
        """The trained turns of a seat with their raw advantage and stratum."""
        turns = []
        for turn in rs.decision_turns(results, slot):
            if len(turn["logprobs"]) != len(turn["token_ids"]):
                self.counts["skipped:logprobs_length"] += 1
                log.warning("game %s seat %s: a turn's log-probs and tokens differ in length", _game_id(results), slot)
                continue
            turns.append(turn)
        game = _game_id(results)
        rng = random.Random(f"{self.cfg.seed}:{game}:{slot}")
        count = rng.randint(*self.cfg.decisions)
        chosen = _stratified(turns, count, self.cfg.bin_s * 1000, rng)
        opp = rs.player_id(results, rs.other_slot(results, slot) or "")
        mean = self.baselines.get(self._key(results, slot), [0.0, 0])[0]
        out = []
        for turn in chosen:
            b = mean
            if self.critic is not None:
                b = self.critic.baseline(results, slot, turn["began_ms"], mean)
            stratum = f"{opp}|{int(turn['began_ms'] // (self.cfg.bin_s * 1000))}"
            out.append((turn, reward - b, stratum))
        return out

    def _learn(self, results: Mapping[str, Any], slot: str, reward: float) -> None:
        """Fold the game into the baselines (after its own decisions were scored, so a game never baselines itself)."""
        key = self._key(results, slot)
        mean, n = self.baselines.get(key, [0.0, 0])
        if self.critic is not None:
            self.critic.observe(results, slot, reward, mean)
        alpha = max(1 / (n + 1), 2 / (self.cfg.baseline_games + 1))
        self.baselines[key] = [mean + alpha * (reward - mean), min(n + 1, 10 * self.cfg.baseline_games)]

    # ---- centring and scale -----------------------------------------------------------------------------------
    def _normalised(
        self, results: Mapping[str, Any], picked: Mapping[str, list[tuple[Mapping[str, Any], float, str]]]
    ) -> list[dict[str, Any]]:
        game = _game_id(results)
        pending = [(slot, turn, a, stratum) for slot, items in picked.items() for turn, a, stratum in items]
        for slot, _, a, stratum in pending:
            self.buffer.append((stratum, f"{game}:{slot}", a))
        by_stratum: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for stratum, owner, a in self.buffer:
            by_stratum[stratum].append((owner, a))
        self.strata = {k: v for k, v in self.strata.items() if k in by_stratum}
        for stratum in {p[3] for p in pending}:
            self._record_stratum(stratum, by_stratum[stratum])
        centred = []
        for _, _, a, stratum in pending:
            games = {owner for owner, _ in by_stratum[stratum]}
            if len(games) >= self.cfg.min_stratum_games:
                a -= statistics.fmean(v for _, v in by_stratum[stratum])
            else:
                self.counts["uncentred_decisions"] += 1
            centred.append(a)
        scale = self._scale(centred)
        rows = []
        for (slot, turn, _, _), a in zip(pending, centred, strict=True):
            advantage = max(-self.cfg.clip, min(self.cfg.clip, a / scale))
            rows.append(self._row(results, slot, turn, advantage))
        return rows

    def _record_stratum(self, stratum: str, entries: list[tuple[str, float]]) -> None:
        values = [v for _, v in entries]
        self.strata[stratum] = {
            "decisions": len(values),
            "games": len({o for o, _ in entries}),
            "mean": statistics.fmean(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    def _scale(self, centred: Sequence[float]) -> float:
        """The EMA SD of centred advantages, updated with this game's decisions.

        An EMA of the mean square, not of the within-game variance: without a critic every decision of a game shares one
        A, so the spread that matters is between games. It averages all decisions seen until 4,096 have been, then
        forgets at that rate."""
        if centred:
            square = statistics.fmean(a * a for a in centred)
            window = min(self.sd_seen + len(centred), self.cfg.sd_decisions)
            alpha = len(centred) / window
            self.sd_var = square if self.sd_var is None else (1 - alpha) * self.sd_var + alpha * square
            self.sd_seen += len(centred)
        return max(math.sqrt(self.sd_var or 0.0), SD_FLOOR)

    def _row(self, results: Mapping[str, Any], slot: str, turn: Mapping[str, Any], advantage: float) -> dict[str, Any]:
        version = rs.version_of(turn.get("model"))
        if version is None:
            version = rs.policy_version(results, slot)
        if version is None:
            raise CreditError(f"seat {slot}: no policy version in its turns or model name")
        return {
            "game_id": f"{_game_id(results)}:{slot}",
            "version": version,
            "prompt_token_ids": list(turn["prompt_token_ids"]),
            "token_ids": list(turn["token_ids"]),
            "logprobs": [float(x) for x in turn["logprobs"]],
            "advantage": advantage,
            "is_hold": rs.is_hold(turn),
        }

    # ---- drift alarm, diagnostics, state ----------------------------------------------------------------------
    def _tick(
        self, results: Mapping[str, Any], picked: Mapping[str, list[tuple[Mapping[str, Any], float, str]]]
    ) -> None:
        """Each `decisions_per_update` trained decisions, judge the batch's drift; a streak of bad ones is the alarm."""
        game = _game_id(results)
        self.window += [(f"{game}:{slot}", a) for slot, items in picked.items() for _, a, _ in items]
        size = self.cfg.decisions_per_update
        while len(self.window) >= size:
            self.batch = _batch_drift(self.window[:size])
            del self.window[:size]
            bad = self.batch["ratio"] > self.cfg.drift_limit and abs(self.batch["z"]) > self.cfg.drift_z
            self.drift_streak = self.drift_streak + 1 if bad else 0

    def drift(self) -> float:
        """|mean / SD| of the pre-centring advantage over the last judged batch of decisions (0 before the first)."""
        return self.batch.get("ratio", 0.0)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "baselines": {k: {"mean": v[0], "games": int(v[1])} for k, v in sorted(self.baselines.items())},
            "scale": math.sqrt(self.sd_var) if self.sd_var is not None else None,
            "strata": {k: dict(v) for k, v in sorted(self.strata.items())},
            "drift": self.drift(),
            "drift_batch": dict(self.batch),
            "errors": list(self.errors),
            "drift_alarm": self.drift_streak >= self.cfg.drift_updates,
            "drift_streak": self.drift_streak,
            "last": dict(self.last),
        }

    def state_dict(self) -> dict[str, Any]:
        """Everything credit carries between games, as plain JSON (the critic keeps its own state)."""
        return {
            "version": STATE_VERSION,
            "baselines": {k: list(v) for k, v in self.baselines.items()},
            "buffer": [list(x) for x in self.buffer],
            "sd_var": self.sd_var,
            "sd_seen": self.sd_seen,
            "control": {k: list(v) for k, v in self.control.items()},
            "counts": dict(self.counts),
            "strata": {k: dict(v) for k, v in self.strata.items()},
            "window": [list(x) for x in self.window],
            "batch": dict(self.batch),
            "drift_streak": self.drift_streak,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"credit state version {state.get('version')!r}, expected {STATE_VERSION}")
        self.baselines = {k: list(v) for k, v in state["baselines"].items()}
        self.buffer = deque(((s, g, float(a)) for s, g, a in state["buffer"]), maxlen=self.cfg.buffer_decisions)
        self.sd_var, self.sd_seen = state["sd_var"], state["sd_seen"]
        self.control = {k: deque(v) for k, v in state["control"].items()}
        self.counts = Counter(state["counts"])
        self.strata = {k: dict(v) for k, v in state["strata"].items()}
        self.window = [(g, float(a)) for g, a in state["window"]]
        self.batch = dict(state["batch"])
        self.drift_streak = state["drift_streak"]


def _sign(x: float) -> float:
    return (x > 0) - (x < 0)


def _batch_drift(entries: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Mean, SD, |mean / SD| and the mean's z-score (a game one observation) of a batch of raw advantages."""
    values = [a for _, a in entries]
    games = len({owner for owner, _ in entries})
    mean, sd = statistics.fmean(values), statistics.pstdev(values)
    if sd > 1e-9:
        ratio, z = abs(mean) / sd, mean / (sd / math.sqrt(games))
    else:
        ratio, z = (UNBOUNDED, math.copysign(UNBOUNDED, mean)) if abs(mean) > 1e-9 else (0.0, 0.0)
    return {"mean": mean, "sd": sd, "games": games, "ratio": ratio, "z": z}


def _game_id(results: Mapping[str, Any]) -> str:
    return str((results.get("spec") or {}).get("game_id") or results.get("game_id") or "game")


def _stratified(
    turns: Sequence[Mapping[str, Any]], count: int, bin_ms: float, rng: random.Random
) -> list[Mapping[str, Any]]:
    """`count` turns (all of them if fewer), taken round-robin from the game-time bins so every phase is represented."""
    by_bin: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for turn in turns:
        by_bin[int(turn["began_ms"] // bin_ms)].append(turn)
    for items in by_bin.values():
        rng.shuffle(items)
    chosen: list[Mapping[str, Any]] = []
    while len(chosen) < min(count, len(turns)):
        for b in sorted(by_bin):
            if by_bin[b] and len(chosen) < count:
                chosen.append(by_bin[b].pop())
    return sorted(chosen, key=lambda t: t["began_ms"])
