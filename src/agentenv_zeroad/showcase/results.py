"""What a showcase game came to, whether it kept the 13-3's rules, and the session's file.

`classify` turns one game's results.json into a `GameRow`: a win, loss or draw as grading wrote it, or a no result
when the game was not a fair test of the two sides (the engine or box failed, a side failed 5 calls in a row, or a
side's failed calls passed 10% of its decisions). A failed call is an error (the model's server or the proxy
answered badly, or not at all); a timeout cancels the call and misses the decision, which is not a failed call
(`StrategistModel` counts only calls that raised). `verify_lockstep` checks the game ran under training's rules.
`Session` keeps the rows in `session.json`, written atomically, which the viewer and the runner both read.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentenv_zeroad.spend import LOCAL_PREFIX
from agentenv_zeroad.train.spec import TRAIN_SAMPLING

LATENCY_MS = 5000
TIME_CAP_MINUTES = 30
SCORE_RULE = "net_worth"
INTERFACE = "v2"
MAX_FAILED_SHARE = 0.1
FAILURES_IN_A_ROW = 5
POLICY_NAME = "Qwen"
EXCEPTION_TYPE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*):")
RESULTS = {"won": "win", "lost": "loss", "drawn": "draw"}
HOW = {"conquest": "conquest", "time_cap": "net_worth", "safety_cap": "net_worth"}

Result = Literal["win", "loss", "draw", "no_result"]
How = Literal["conquest", "net_worth"]
Blame = Literal["policy", "opponent", "engine"]


@dataclass(frozen=True)
class GameRow:
    """One played game. `failed` and `decisions` are keyed `policy` and `opponent`; `blame` is who made a no result
    (`engine` for the engine or the box), which decides whether the opponent is dropped."""

    index: int
    game_id: str
    opponent: str
    seed: int
    policy_slot: str
    result: Result
    how: How | None
    reason: str | None
    failed: dict[str, int]
    decisions: dict[str, int]
    played_at: float
    problems: list[str] = field(default_factory=list)
    blame: Blame | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GameRow:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _counts(results: Mapping[str, Any], policy_slot: str) -> tuple[dict[str, int], dict[str, int], str | None]:
    """(failed calls, decisions) per role, and the opponent's slot (None for a seat with no model)."""
    decisions_by_slot: dict[str, int] = {}
    for d in (results.get("lockstep") or {}).get("decisions") or []:
        decisions_by_slot[str(d["slot"])] = decisions_by_slot.get(str(d["slot"]), 0) + 1
    sampler = results.get("sampler") or {}
    opponent_slots = [s for s in (results.get("seats") or {}) if s != policy_slot]
    opponent = opponent_slots[0] if opponent_slots else None

    def failed_of(slot: str | None) -> int:
        return int((sampler.get(slot) or {}).get("failed_calls", 0)) if slot is not None else 0

    failed = {"policy": failed_of(policy_slot), "opponent": failed_of(opponent)}
    decisions = {
        "policy": decisions_by_slot.get(policy_slot, 0),
        "opponent": decisions_by_slot.get(opponent, 0) if opponent is not None else 0,
    }
    return failed, decisions, opponent


def classify(results: Mapping[str, Any], *, index: int, opponent: str, policy_slot: str) -> GameRow:
    failed, decisions, _ = _counts(results, policy_slot)
    names = {"policy": POLICY_NAME, "opponent": opponent}
    result: Result
    how: How | None = None
    blame: Blame | None = None
    reason: str | None = None

    engine = results.get("failed") or (results.get("detail") if results.get("status") == "failed" else None)
    early = (results.get("showcase") or {}).get("ended_early")
    over = [
        role for role in ("policy", "opponent") if decisions[role] and failed[role] / decisions[role] > MAX_FAILED_SHARE
    ]
    if engine:
        result, reason, blame = "no_result", _engine_reason(str(engine)), "engine"
    elif early:
        result = "no_result"
        slot = str(early.get("slot")) if isinstance(early, Mapping) else None
        blame = "policy" if slot == policy_slot else "opponent" if slot is not None else None
        reason = (
            f"{names[blame]}, {FAILURES_IN_A_ROW} failed calls in a row"
            if blame
            else f"a seat failed {FAILURES_IN_A_ROW} calls in a row"
        )
    elif over:
        blame = max(over, key=lambda role: failed[role] / decisions[role])
        result, reason = "no_result", f"{names[blame]}, {failed[blame]} failed calls"
    else:
        outcome = ((results.get("seats") or {}).get(policy_slot) or {}).get("outcome")
        if outcome in RESULTS:
            result, how = RESULTS[outcome], HOW.get(str(results.get("ended_by")))  # type: ignore[assignment]
        else:
            result, reason, blame = "no_result", "the game ended without an outcome", "engine"
    spec = results.get("spec") or {}
    return GameRow(
        index=index,
        game_id=str(spec.get("game_id", "")),
        opponent=opponent,
        seed=int(spec.get("seed", -1)),
        policy_slot=policy_slot,
        result=result,
        how=how,
        reason=reason,
        failed=failed,
        decisions=decisions,
        played_at=time.time(),
        problems=verify_lockstep(results) if spec else [],
        blame=blame,
    )


def _engine_reason(detail: str) -> str:
    """A fixed phrase and the exception's type name only: the text of an exception can carry hosts and bucket names."""
    named = EXCEPTION_TYPE.match(detail)
    return f"engine failed ({named.group(1)})" if named else "engine failed"


def verify_lockstep(results: Mapping[str, Any]) -> list[str]:
    """What differs from the 13-3's rules: orders not landing at frozen + 5 s, or a spec that is not Run 1's."""
    problems: list[str] = []
    decisions = (results.get("lockstep") or {}).get("decisions") or []
    applied = [d for d in decisions if d.get("applied_ms") is not None]
    if not applied:
        problems.append("no decision was applied")
    for d in applied:
        gap = d["applied_ms"] - d["frozen_ms"]
        if gap != LATENCY_MS:
            problems.append(
                f"seat {d['slot']} decision frozen at {d['frozen_ms']} ms landed {gap} ms later, not {LATENCY_MS}"
            )
    spec = results.get("spec") or {}
    if spec.get("latency_ms") != LATENCY_MS:
        problems.append(f"spec latency is {spec.get('latency_ms')}, not {LATENCY_MS}")
    if spec.get("time_cap_minutes") != TIME_CAP_MINUTES:
        problems.append(f"spec time cap is {spec.get('time_cap_minutes')} minutes, not {TIME_CAP_MINUTES}")
    if spec.get("score_rule") != SCORE_RULE:
        problems.append(f"spec score rule is {spec.get('score_rule')}, not {SCORE_RULE}")
    for seat in spec.get("seats") or []:
        if seat.get("kind") != "llm":
            continue
        if seat.get("interface") != INTERFACE:
            problems.append(f"seat {seat.get('civ')} plays interface {seat.get('interface')}, not {INTERFACE}")
        if str(seat.get("model", "")).startswith(LOCAL_PREFIX) and dict(seat.get("sampling") or {}) != TRAIN_SAMPLING:
            problems.append(f"the policy's sampling is {seat.get('sampling')}, not {TRAIN_SAMPLING}")
    return problems


class Session:
    """The rows played so far, and how the session ended (`ended`, a short code, and `ended_why`)."""

    def __init__(self, path: Path, rotation: tuple[str, ...] = ()):
        self.path = path
        self.rotation = rotation
        self.rows: list[GameRow] = []
        self.ended: str | None = None
        self.ended_why: str | None = None

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.rows = [GameRow.from_json(r) for r in data.get("rows", [])]
        self.ended = data.get("ended")
        self.ended_why = data.get("ended_why")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        value = {"rows": [r.to_json() for r in self.rows], "ended": self.ended, "ended_why": self.ended_why}
        tmp.write_text(json.dumps(value, indent=1))
        os.replace(tmp, self.path)

    def add(self, row: GameRow) -> None:
        self.rows.append(row)

    def table(self) -> list[dict[str, Any]]:
        """One row per opponent, in rotation order (opponents outside the rotation follow, as first played)."""
        order = list(self.rotation)
        order += [r.opponent for r in self.rows if r.opponent not in order]
        order = list(dict.fromkeys(order))
        out = []
        for name in order:
            mine = [r.result for r in self.rows if r.opponent == name]
            out.append(
                {
                    "opponent": name,
                    "w": mine.count("win"),
                    "l": mine.count("loss"),
                    "d": mine.count("draw"),
                    "nr": mine.count("no_result"),
                }
            )
        return out

    def dropped(self) -> set[str]:
        """Opponents whose last two games were both no results made by their own failures."""
        dropped = set()
        for name in {r.opponent for r in self.rows}:
            last = [r for r in self.rows if r.opponent == name][-2:]
            if len(last) == 2 and all(r.result == "no_result" and r.blame == "opponent" for r in last):
                dropped.add(name)
        return dropped

    def next_index(self) -> int:
        return self.rows[-1].index + 1 if self.rows else 0

    def seeds(self) -> set[int]:
        return {r.seed for r in self.rows}
