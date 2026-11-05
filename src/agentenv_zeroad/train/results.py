"""Reading a training game's results.json: who played, how it ended, and what each logged decision looked like.

Credit, the critic, the report and the distillation probe all start from the same file (written by the runner,
`pilot/scoring.py` reads the same shape), so the reading lives here once. Nothing in this module judges a game.

- A seat's opponent id is what a rating board calls the opponent: `petra-3`, `bot`, `v0` (a frozen policy version) or
  a frontier model's name. It comes from the game spec in the results, or from the seat itself when a results file
  has no spec.
- A policy's version is read from its model name (`local/policy-v7`; `local/qwen-base` is version 0), because the
  results record the model each turn was sampled from and a version never changes inside a game.
- A game has learner seats: the seats whose decisions are trained on. The caller names them, or the spec's seat
  entries mark them `"trainable": true` (the platform's `SeatSpec.trainable`, set on the mix's policy seat).
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Mapping, Sequence
from typing import Any

CAP_ENDS = ("time_cap", "safety_cap")
BASE_MODEL = "local/qwen-base"
_POLICY_MODEL = re.compile(r"(?:^|/)policy-v(\d+)$")


def version_of(model: str | None) -> int | None:
    """The policy version a model name stands for, or None when it is not one of ours."""
    if model == BASE_MODEL:
        return 0
    match = _POLICY_MODEL.search(model or "")
    return int(match.group(1)) if match else None


def spec_seats(results: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The spec's seat entries by slot ("1", "2", ...), empty when the results carry no spec."""
    return {str(n): seat for n, seat in enumerate((results.get("spec") or {}).get("seats") or [], 1)}


def learner_slots(results: Mapping[str, Any], learners: Sequence[str] | None = None) -> list[str]:
    """The learner seats: `learners` if given, else the seats the spec marks `"trainable": true`."""
    if learners is not None:
        unknown = [s for s in learners if s not in results["seats"]]
        if unknown:
            raise ValueError(f"no such seat: {', '.join(unknown)}")
        return list(learners)
    return [slot for slot, seat in spec_seats(results).items() if seat.get("trainable")]


def other_slot(results: Mapping[str, Any], slot: str) -> str | None:
    """The opponent's slot in a 1v1 game, None in any other."""
    others = [s for s in results["seats"] if s != slot]
    return others[0] if len(results["seats"]) == 2 and len(others) == 1 else None


def opponent_kind(results: Mapping[str, Any], slot: str) -> str:
    """`petra`, `bot` or `llm`: what plays seat `slot`."""
    seat = spec_seats(results).get(slot)
    if seat is not None:
        return str(seat["kind"])
    kind = str(results["seats"][slot].get("kind"))
    return "llm" if kind == "agent" else kind


def seat_model(results: Mapping[str, Any], slot: str) -> str | None:
    seat = spec_seats(results).get(slot) or {}
    return seat.get("model") or results["seats"][slot].get("model")


def player_id(results: Mapping[str, Any], slot: str) -> str:
    """What a rating board calls whoever plays `slot`: petra-3, bot, v7, or a frontier model's name."""
    kind, seat = opponent_kind(results, slot), spec_seats(results).get(slot) or {}
    if kind == "petra":
        return f"petra-{seat.get('difficulty', 3)}"
    if kind == "bot":
        return "bot"
    version = version_of(seat_model(results, slot))
    if version is not None:
        return f"v{version}"
    return str(seat.get("name") or results["seats"][slot].get("name") or seat_model(results, slot))


def policy_version(results: Mapping[str, Any], slot: str) -> int | None:
    """The version that played `slot`: from its turns' model (the one actually sampled), else the seat's model."""
    for turn in results["seats"][slot].get("turns") or []:
        version = version_of(turn.get("model"))
        if version is not None:
            return version
    return version_of(seat_model(results, slot))


def behaviour(results: Mapping[str, Any], slot: str) -> str | None:
    return (spec_seats(results).get(slot) or {}).get("behaviour")


def final_margin(results: Mapping[str, Any], slot: str, opp: str) -> tuple[float | None, str | None]:
    """(slot's final margin over opp, where it came from): the net-worth margin, else the official score's."""
    mine, theirs = (results["seats"][s].get("score") or {} for s in (slot, opp))
    for source, key in (("net_worth", "net_worth"), ("score", "total")):
        a, b = mine.get(key), theirs.get(key)
        if a is not None and b is not None:
            return float(a) - float(b), source
    return None, None


def timeline_at(results: Mapping[str, Any], t_ms: float) -> Mapping[str, Any] | None:
    """The newest timeline row at or before `t_ms` (the first row when `t_ms` is earlier), None without a timeline."""
    timeline = results.get("timeline") or []
    if not timeline:
        return None
    times = [row["t_ms"] for row in timeline]
    return timeline[max(0, bisect.bisect_right(times, t_ms) - 1)]


def timeline_first_at_or_after(results: Mapping[str, Any], t_ms: float) -> Mapping[str, Any] | None:
    """The first timeline row at or after `t_ms`, or the last row when the game ended first."""
    timeline = results.get("timeline") or []
    if not timeline:
        return None
    times = [row["t_ms"] for row in timeline]
    return timeline[min(bisect.bisect_left(times, t_ms), len(timeline) - 1)]


def decision_turns(results: Mapping[str, Any], slot: str) -> list[Mapping[str, Any]]:
    """The seat's logged turns that a learner could use: prompt and completion ids with one log-prob per token."""
    out = []
    for turn in results["seats"][slot].get("turns") or []:
        prompt, completion, logprobs = turn.get("prompt_token_ids"), turn.get("token_ids"), turn.get("logprobs")
        if prompt and completion and logprobs is not None and turn.get("began_ms") is not None:
            out.append(turn)
    return out


def is_hold(turn: Mapping[str, Any]) -> bool:
    """Whether every order of the turn (say() aside, which changes nothing) is hold(); a turn without orders is not."""
    orders = [o for o in turn.get("orders") or [] if o.get("tool") != "say"]
    return bool(orders) and all(o.get("tool") == "hold" for o in orders)


def policy_slot(results: Mapping[str, Any], learners: Sequence[str] | None) -> str | None:
    """The seat whose side the game is reported from, or None when no seat is a policy."""
    flagged = learner_slots(results, learners)
    if flagged:
        return flagged[0]
    versions = {slot: policy_version(results, slot) for slot in results["seats"]}
    known = {slot: v for slot, v in versions.items() if v is not None}
    return max(sorted(known), key=lambda s: known[s]) if known else None
