"""A state critic for post-training v2's baseline: optional, and only used once it has earned its place.

Credit (train/credit.py) subtracts a baseline from each game's reward. The default baseline is a running mean per
opponent, which cannot tell a decision made while ahead from one made while behind. A critic can: a gradient-boosted
regressor (scikit-learn's HistGradientBoostingRegressor) that predicts the game's reward R from what an observer sees
at the moment a decision began. Its features are both sides' economy, military, population, army, gatherers, banked
resources, buildings, phase, kills, losses and net worth (missing when the game records none), their change over the
last 60 game-seconds, the game time, the opponent's kind, level and behaviour, the civ pair, the map and the lockstep
latency L. Exploration is left out: it adds score without winning anything, and a critic that leans on it would
re-teach exploring.

A critic must not guess its way into the loss. Each refit is a candidate, screened by cross-validation (GroupKFold, one
group per game, so a game's rows are never on both sides) against the baseline the games actually used; one that
passes is then judged prequentially, on the next `eval_games` games it has not seen, against both the critic in use and
the running mean. It replaces the baseline only if its squared error there beats both. On a switch the baseline moves
from the old value to the new over `blend_s` seconds, so the advantages do not jump. Rows are weighted toward recent
games (a half-life in games): the policy being scored keeps changing.

The data and the bookkeeping are JSON (`state_dict`), and so are the fitted models, as base64 pickles: the games are
trimmed to the newest `max_games`, so a fit made before a trim cannot be rebuilt from what is left, and a restart must
resume with the very models that were scoring decisions. The state file is the learner's own; do not load one you did
not write. A previous critic is dropped once the blend toward the active one is complete.
"""

from __future__ import annotations

import base64
import logging
import math
import pickle
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from agentenv_zeroad.train import results as rs

log = logging.getLogger(__name__)

STATS = ("economy", "military", "pop", "army", "gatherers", "resources", "buildings", "kills", "lost", "net_worth")
PHASES = {"village": 0, "town": 1, "city": 2}
KINDS = {"petra": 0, "bot": 1, "llm": 2}
BEHAVIOURS = {"balanced": 0, "aggressive": 1, "defensive": 2}
DELTA_MS = 60_000
CODES = 128
NAN = float("nan")
STATE_VERSION = 2
CATEGORICAL = ("kind", "behaviour", "civ", "opponent_civ", "map")
FEATURES = (
    ["minute"]
    + [f"{side}_{name}" for side in ("own", "opp") for name in (*STATS, "phase")]
    + [f"{side}_{name}_delta" for side in ("own", "opp") for name in STATS]
    + ["level", "L", *CATEGORICAL]
)


def _num(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else NAN


def _code(name: Any) -> float:
    return float(zlib.crc32(str(name).encode()) % CODES) if name else NAN


def features(results: Mapping[str, Any], slot: str, began_ms: float) -> list[float]:
    """The critic's feature vector for the decision `slot` began at `began_ms` (names in FEATURES)."""
    opp = rs.other_slot(results, slot)
    assert opp is not None
    now, before = rs.timeline_at(results, began_ms), rs.timeline_at(results, began_ms - DELTA_MS)
    assert now is not None and before is not None
    x = [began_ms / 60_000]
    for side in (slot, opp):
        p = (now.get("players") or {}).get(side) or {}
        x += [_num(p.get(k)) for k in STATS] + [float(PHASES.get(str(p.get("phase")), NAN))]
    for side in (slot, opp):
        p, q = ((r.get("players") or {}).get(side) or {} for r in (now, before))
        x += [_num(p.get(k)) - _num(q.get(k)) for k in STATS]
    spec = rs.spec_seats(results)
    opponent = spec.get(opp) or {}
    kind = rs.opponent_kind(results, opp)
    x += [
        float(opponent.get("difficulty", NAN)) if kind == "petra" else NAN,
        _num((results.get("lockstep") or {}).get("latency_ms")),
        float(KINDS.get(kind, NAN)),
        float(BEHAVIOURS.get(str(rs.behaviour(results, opp)), NAN)),
        _code(results["seats"][slot].get("civ")),
        _code(results["seats"][opp].get("civ")),
        _code((results.get("map") or {}).get("name")),
    ]
    return x


@dataclass(frozen=True)
class CriticConfig:
    min_games: int = 100  # games observed before the first candidate is fit
    eval_games: int = 200  # the prequential window a candidate is judged on
    cv_folds: int = 5
    half_life_games: float = 400.0  # a game this many games old counts half
    blend_s: float = 900.0
    max_games: int = 2000
    max_iter: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_samples_leaf: int = 40
    seed: int = 0


@dataclass
class GameRows:
    seq: int
    reward: float
    baseline: float  # the running mean credit used for this game
    x: np.ndarray  # one row per decision of the game


@dataclass
class Fit:
    model: HistGradientBoostingRegressor
    upto: int  # fit on the games observed up to this count


def _dump(fit: Fit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {"upto": fit.upto, "model": base64.b64encode(pickle.dumps(fit.model)).decode("ascii")}


def _load(blob: Mapping[str, Any] | None) -> Fit | None:
    if blob is None:
        return None
    return Fit(pickle.loads(base64.b64decode(blob["model"])), blob["upto"])


class GBCritic:
    def __init__(self, config: CriticConfig | None = None, *, clock: Callable[[], float] = time.time):
        self.cfg = config or CriticConfig()
        self.clock = clock
        self.games: list[GameRows] = []
        self.observed = 0
        self.cooldown_until = 0
        self.active: Fit | None = None
        self.previous: Fit | None = None
        self.candidate: Fit | None = None
        self.switched_at: float | None = None
        self.window = self._empty_window()
        self.history: list[dict[str, Any]] = []

    # ---- what credit calls ------------------------------------------------------------------------------------
    def baseline(self, results: Mapping[str, Any], slot: str, began_ms: float, fallback: float) -> float:
        """The baseline at a decision: the running mean until a critic is in use, then the critic, blended in."""
        if self.active is None or not results.get("timeline"):
            return fallback
        x = np.array([features(results, slot, began_ms)])
        value = float(self.active.model.predict(x)[0])
        w = self._weight()
        if w >= 1.0:
            self.previous = None
            return value
        old = float(self.previous.model.predict(x)[0]) if self.previous is not None else fallback
        return w * value + (1 - w) * old

    def observe(self, results: Mapping[str, Any], slot: str, reward: float, baseline: float) -> None:
        """One finished seat-game: judge the pending candidate on it, store it, refit when it is time."""
        turns = [t for t in results["seats"][slot].get("turns") or [] if t.get("began_ms") is not None]
        if not turns or not results.get("timeline"):
            return
        x = np.array([features(results, slot, t["began_ms"]) for t in turns])
        if self.candidate is not None:
            self._judge(x, reward, baseline)
        if self._weight() >= 1.0:
            self.previous = None
        self.observed += 1
        self.games.append(GameRows(self.observed, reward, baseline, x))
        del self.games[: -self.cfg.max_games]
        if self.candidate is not None and self.window["games"] >= self.cfg.eval_games:
            self._decide()
        if self.candidate is None and self.observed >= max(self.cooldown_until, self.cfg.min_games):
            self._fit_candidate()

    # ---- fitting ----------------------------------------------------------------------------------------------
    def _weight(self) -> float:
        if self.switched_at is None:
            return 1.0
        return max(0.0, min(1.0, (self.clock() - self.switched_at) / self.cfg.blend_s))

    @staticmethod
    def _empty_window() -> dict[str, float]:
        return {"games": 0, "rows": 0, "candidate": 0.0, "incumbent": 0.0, "mean": 0.0}

    def _model(self) -> HistGradientBoostingRegressor:
        cfg = self.cfg
        return HistGradientBoostingRegressor(
            max_iter=cfg.max_iter,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=1.0,
            categorical_features=[FEATURES.index(n) for n in CATEGORICAL],
            early_stopping=False,
            random_state=cfg.seed,
        )

    def _data(self, upto: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """X, y, recency weights, the baseline's value and the game of every row, for games observed up to `upto`."""
        games = [g for g in self.games if g.seq <= upto]
        x = np.concatenate([g.x for g in games])
        y = np.concatenate([np.full(len(g.x), g.reward) for g in games])
        base = np.concatenate([np.full(len(g.x), g.baseline) for g in games])
        group = np.concatenate([np.full(len(g.x), g.seq) for g in games])
        age = np.concatenate([np.full(len(g.x), upto - g.seq) for g in games])
        return x, y, 0.5 ** (age / self.cfg.half_life_games), base, group

    def _fit_candidate(self) -> None:
        upto = self.observed
        x, y, w, base, group = self._data(upto)
        folds = min(self.cfg.cv_folds, len(set(group.tolist())))
        if folds < 2:
            return
        oof = np.empty(len(y))
        for train, test in GroupKFold(n_splits=folds).split(x, y, group):
            oof[test] = self._model().fit(x[train], y[train], sample_weight=w[train]).predict(x[test])
        cv, mean = float(np.mean((y - oof) ** 2)), float(np.mean((y - base) ** 2))
        if cv >= mean:
            self.cooldown_until = upto + self.cfg.eval_games
            self._log({"event": "screened_out", "cv_loss": cv, "baseline_loss": mean})
            return
        self.candidate = Fit(self._model().fit(x, y, sample_weight=w), upto)
        self.window = self._empty_window()
        self._log({"event": "candidate", "cv_loss": cv, "baseline_loss": mean})

    def _judge(self, x: np.ndarray, reward: float, baseline: float) -> None:
        assert self.candidate is not None
        win = self.window
        win["candidate"] += float(np.sum((reward - self.candidate.model.predict(x)) ** 2))
        win["mean"] += len(x) * (reward - baseline) ** 2
        if self.active is not None:
            win["incumbent"] += float(np.sum((reward - self.active.model.predict(x)) ** 2))
        win["games"] += 1
        win["rows"] += len(x)

    def _decide(self) -> None:
        win, n = self.window, max(self.window["rows"], 1)
        losses = {"candidate": win["candidate"] / n, "mean": win["mean"] / n}
        beats = losses["candidate"] < losses["mean"]
        if self.active is not None:
            losses["incumbent"] = win["incumbent"] / n
            beats = beats and losses["candidate"] < losses["incumbent"]
        if beats:
            self.previous, self.active, self.switched_at = self.active, self.candidate, self.clock()
        self._log({"event": "switch" if beats else "kept", "games": win["games"], "loss": losses})
        self.candidate, self.cooldown_until = None, self.observed

    def _log(self, event: dict[str, Any]) -> None:
        event = {**event, "observed": self.observed, "time": self.clock()}
        self.history.append(event)
        log.info("critic: %s", event)

    # ---- state ------------------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "active": self.active.upto if self.active else None,
            "candidate": self.candidate.upto if self.candidate else None,
            "blend": self._weight() if self.active else None,
            "history": list(self.history[-20:]),
        }

    def state_dict(self) -> dict[str, Any]:
        def clean(row: list[float]) -> list[float | None]:
            return [None if math.isnan(v) else v for v in row]

        return {
            "version": STATE_VERSION,
            "observed": self.observed,
            "cooldown_until": self.cooldown_until,
            "games": [
                {"seq": g.seq, "reward": g.reward, "baseline": g.baseline, "x": [clean(r) for r in g.x.tolist()]}
                for g in self.games
            ],
            "fits": {k: _dump(getattr(self, k)) for k in ("active", "previous", "candidate")},
            "switched_at": self.switched_at,
            "window": dict(self.window),
            "history": list(self.history),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"critic state version {state.get('version')!r}, expected {STATE_VERSION}")
        self.games = [
            GameRows(
                g["seq"], g["reward"], g["baseline"], np.array([[NAN if v is None else v for v in r] for r in g["x"]])
            )
            for g in state["games"]
        ]
        self.observed, self.cooldown_until = state["observed"], state["cooldown_until"]
        self.switched_at, self.window = state["switched_at"], dict(state["window"])
        self.history = list(state["history"])
        for name in ("active", "previous", "candidate"):
            setattr(self, name, _load(state["fits"][name]))
