"""Teacher labels on hard states during RL: where reinforcement has no signal, a frontier teacher supplies one.

Outcome-only policy gradient learns nothing from a group of games that are all losses: every advantage is zero. Against
Petra-3 and above that is most of the time at first. This module watches the policy's rolling win rate against each
opponent (its last `window` games, draws half); once an opponent has `min_games` in the window and the rate is below
`threshold` (5% by default) the opponent is hard, and the decisions the policy made in its games against it are
candidates. Up to `per_hour` of them an hour are relabelled by the calibrated teacher and written to the learner's inbox
as rows flagged `sft`, which the learner trains on with a small decaying cross-entropy term (`learner.py`, the aux
term). Nothing is added where RL already has a signal.

The labelling is distillation's own (`distill.label`, `distill.sft_rows`): the policy's system prompt, reference text
and shown tools, the policy's exact prompt token ids, the teacher's tool calls as the completion in Qwen's format. Each
pass runs under a `SpendGuard` whose cap is what is left of the hour's `cap_usd_per_hour`; every pass is appended to the
run's ledger, so the labels count toward the run's budget. The teacher is `teacher` in the config, or by default the
calibration panel's best (`stats.pick_teacher`: best against Petra-3, ties Petra-4); with no calibration games yet
nothing is labelled.

A game is taken into the pool only if it is against an opponent that is hard once the game is counted, so the pool never
holds a state from a game the policy was winning against; the briefs are chosen by distillation's strata (losing
positions first). The windows and the hour's counters are saved by the orchestrator (`state_dict`); the waiting pool is
not, and a restart just refills it from the games that finish.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agentenv_zeroad import spend
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train import distill, stats
from agentenv_zeroad.train import report as rep
from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train.config import HardStateConfig

TeacherFactory = Callable[[str, SpendGuard], distill.Teacher]
PANEL = rep.CALIBRATION_PANEL


class HardStates:
    def __init__(
        self,
        cfg: HardStateConfig,
        *,
        root: Path,
        ledger: Path,
        policies: Mapping[str, distill.Policy],
        make_teacher: TeacherFactory,
        tokenizer: distill.Tokenizer,
        seed: int = 0,
    ):
        self.cfg, self.root, self.ledger = cfg, root, ledger
        self.policies, self.make_teacher, self.tokenizer = policies, make_teacher, tokenizer
        self.seed = seed
        self.windows: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=cfg.window))
        self.pool: list[distill.Brief] = []
        self.asked: dict[int, int] = defaultdict(int)  # decisions sent to the teacher, by run hour
        self.usd: dict[int, float] = defaultdict(float)  # what the labels cost, by run hour
        self.passes = 0

    # ---- what is hard -------------------------------------------------------------------------------------------
    def rate(self, opponent: str) -> float | None:
        """The policy's rolling win rate against `opponent`, None before `min_games` games."""
        window = self.windows.get(opponent)
        return statistics.fmean(window) if window and len(window) >= self.cfg.min_games else None

    def is_hard(self, opponent: str) -> bool:
        rate = self.rate(opponent)
        return rate is not None and rate < self.cfg.threshold

    def hard_opponents(self) -> list[str]:
        return sorted(o for o in self.windows if self.is_hard(o))

    def observe(self, results: Mapping[str, Any], name: str) -> bool:
        """Count a finished training game; when its opponent is hard now, keep its decisions for labelling. Returns
        whether it was kept."""
        slot = rs.policy_slot(results, None)
        opp = rs.other_slot(results, slot) if slot else None
        outcome = results["seats"][slot].get("outcome") if slot and opp else None
        finished = results.get("status") == "finished" and results.get("ended_by") in ("conquest", *rs.CAP_ENDS)
        if opp is None or outcome not in rep.RESULT or not finished:
            return False
        opponent = rs.player_id(results, opp)
        self.windows[opponent].append(rep.RESULT[outcome])
        if not self.is_hard(opponent):
            return False
        self.pool += distill.briefs_of(results, name, None, losing_opponents=None)
        del self.pool[: max(0, len(self.pool) - self.cfg.max_pool)]
        return True

    # ---- labelling ----------------------------------------------------------------------------------------------
    def quota(self, hour: int) -> int:
        return max(0, self.cfg.per_hour - self.asked[hour])

    def usd_left(self, hour: int) -> float:
        return max(0.0, self.cfg.cap_usd_per_hour - self.usd[hour])

    def teacher_model(self) -> str | None:
        """The configured teacher, or the calibration panel's best (None while it has no game against Petra-3)."""
        if self.cfg.teacher != "auto":
            return self.cfg.teacher
        folder = Path(self.cfg.calibration) if self.cfg.calibration else self.root / "panels" / PANEL
        summary = rep.teacher_calibration(folder)
        best = stats.pick_teacher(summary) if summary else None
        return (summary["teachers"][best]["model"] or best) if summary and best else None

    async def label(self, hour: int, alpha: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """One labelling pass for run hour `hour`: up to `batch` of the waiting decisions, within the hour's remaining
        count and dollars. Returns the learner rows (flagged sft) and what happened.

        `alpha` is the learner's latest auxiliary weight (its health record's `aux_alpha`): below `alpha_floor` the
        learner would barely train on a label, so none is bought. Only decisions from games against an opponent that
        is hard now are asked; those against one the policy has since learned to beat leave the pool.

        The pass's dollars, count and asked briefs are booked whatever happens in the labelling, an unexpected error
        included; the labels are tokenized with the tokenizer loaded at the start, which cannot fail for lack of an
        install."""
        info: dict[str, Any] = {"hour": hour, "asked": 0, "labelled": 0, "usd": 0.0, "skipped": None}
        self.pool = [b for b in self.pool if self.is_hard(b.opponent)]
        n, left = min(self.cfg.batch, self.quota(hour)), self.usd_left(hour)
        if alpha is not None and alpha < self.cfg.alpha_floor:
            info["skipped"] = f"the learner's aux alpha {alpha:.4f} is under the {self.cfg.alpha_floor:g} floor"
        elif not self.pool:
            info["skipped"] = "no hard states waiting"
        elif n <= 0:
            info["skipped"] = f"{self.cfg.per_hour} decisions already sampled this hour"
        elif left <= 0:
            info["skipped"] = f"the ${self.cfg.cap_usd_per_hour:g} hourly cap is spent"
        elif (model := self.teacher_model()) is None:
            info["skipped"] = "no calibrated teacher yet"
        if info["skipped"]:
            return [], info
        self.passes += 1
        chosen = distill.select(self.pool, n, per_game=self.cfg.per_game, seed=self.seed + self.passes)
        guard = SpendGuard(f"hard-state labels {model}", cap_usd=left)
        asked: set[str] = set()
        counts: dict[str, Any] = {"asked": 0}
        try:
            labels, counts = await distill.label(
                chosen, self.make_teacher(model, guard), self.policies, concurrency=self.cfg.concurrency,
                on_asked=lambda b: asked.add(b.key),
            )  # fmt: skip
            rows = distill.sft_rows(labels, self.tokenizer, model)
        finally:
            # a brief the teacher was sent is not asked again, even if it failed; one the cap left unasked waits
            self.pool = [b for b in self.pool if b.key not in asked]
            self.asked[hour] += len(asked)
            self.usd[hour] += guard.spent_usd
            if guard.spent_usd:
                spend.append_ledger(self.ledger, f"hard-state labels h{hour}", guard.spent_usd)
        info |= {
            "asked": counts["asked"], "labelled": len(rows), "usd": guard.spent_usd, "stopped": counts["stopped"],
            "teacher": model, "failed": counts["failed"], "no_calls": counts["no_calls"],
        }  # fmt: skip
        return rows, info

    # ---- state --------------------------------------------------------------------------------------------------
    def status(self, hour: int) -> dict[str, Any]:
        return {
            "hard": self.hard_opponents(),
            "rates": {o: round(r, 3) for o in sorted(self.windows) if (r := self.rate(o)) is not None},
            "waiting": len(self.pool),
            "hour": {"asked": self.asked[hour], "usd": round(self.usd[hour], 4)},
        }

    def state_dict(self) -> dict[str, Any]:
        recent = sorted(self.asked)[-2:]
        return {
            "windows": {o: list(w) for o, w in self.windows.items()},
            "asked": {str(h): self.asked[h] for h in recent},
            "usd": {str(h): self.usd[h] for h in sorted(self.usd)[-2:]},
            "passes": self.passes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for opponent, games in state["windows"].items():
            self.windows[opponent] = deque((float(g) for g in games), maxlen=self.cfg.window)
        self.asked = defaultdict(int, {int(h): int(n) for h, n in state["asked"].items()})
        self.usd = defaultdict(float, {int(h): float(v) for h, v in state["usd"].items()})
        self.passes = int(state.get("passes", 0))
