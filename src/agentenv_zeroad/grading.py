"""How a 0 A.D. match is decided and graded (spec §4.7): conquest, a time cap with the engine score, draws, forfeits.

The engine decides conquest itself (its conquest victory condition marks the last player standing `won` and the others
`defeated`). At the time cap the highest score among the players still standing wins (a player already defeated stays
lost, whatever its score); players tied on top (within `draw_margin`) draw, and everyone else loses. The score is the
engine's official total (`official`, the default) or each player's net worth (`net_worth`: what its living units and
finished structures cost plus its banked resources), per the match's `score_rule`; `draw_margin` is in that measure's
own units. Grading follows agentenv-game-env's convention through its
`grade_match`: a win 1, a draw 0.5, a loss 0.

The forfeit rule is about brains, not orders: a brain call that errors or runs past `BRAIN_TIMEOUT_S` counts as
missed, and a seat that missed more than half of its calls scores 0, because its model was effectively unreachable.
Slow but answering models are never zeroed. The env reports each agent seat's calls as the match protocol's `moves`
(`made`, `missed`), and a task grades with `max_missed: 0.5`, so `grade_match` applies the rule; `forfeited` below
is the same rule for the results file. The brain layer (U5) reports each call it makes; a seat whose brain reports
none (a plain MCP agent) is counted by its tool calls instead, each one made, since the env can't see its model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

BRAIN_TIMEOUT_S = 120.0
FORFEIT_SHARE = 0.5
POINTS = {"won": 1.0, "drawn": 0.5, "lost": 0.0}

Outcome = Literal["won", "lost", "drawn"]
ScoreRule = Literal["official", "net_worth"]
SCORE_KEYS: dict[ScoreRule, str] = {"official": "total", "net_worth": "net_worth"}
PROMPT_RULES: dict[str, ScoreRule] = {"v1": "official", "v2": "net_worth"}


def resolve_score_rule(rule: ScoreRule | None, agent_interfaces: Iterable[str]) -> ScoreRule:
    """The score rule a match runs under, given what its agents are told.

    An interface's system prompt states how the time cap is decided (v1: the highest score; v2: the higher net
    worth), so the rule has to agree with it or the policy is trained, and the baseline measured, on a goal the engine
    does not score. `None` takes the rule the agents' prompt names (official when no agent plays, or when agents on
    both interfaces do, as in an interface comparison, where no one rule fits both); an explicit rule that contradicts
    the one prompt all the agents share is an error.
    """
    named = {PROMPT_RULES[i] for i in agent_interfaces}
    prompted = next(iter(named)) if len(named) == 1 else None
    if rule is None:
        return prompted or "official"
    if prompted is not None and rule != prompted:
        raise ValueError(f"score_rule {rule!r} contradicts the agents' prompt, which says the {prompted} rule decides")
    return rule


@dataclass(frozen=True)
class BrainCall:
    """One brain call of a seat: how long it took (wall seconds) and its error, if it failed."""

    seconds: float
    error: str | None = None
    game_ms: int | None = None

    @property
    def missed(self) -> bool:
        return self.error is not None or self.seconds > BRAIN_TIMEOUT_S


def moves(calls: Sequence[BrainCall]) -> tuple[int, int]:
    """(made, missed) for the match protocol's `moves`."""
    missed = sum(1 for c in calls if c.missed)
    return len(calls) - missed, missed


def forfeited(calls: Sequence[BrainCall]) -> bool:
    """More than half of the seat's brain calls missed. A seat with no calls is not forfeited by this rule."""
    made, missed = moves(calls)
    return bool(made + missed) and missed / (made + missed) > FORFEIT_SHARE


def time_cap_outcomes(scores: Mapping[str, float], *, draw_margin: float = 0.0) -> dict[str, Outcome]:
    """Outcomes at the time cap: the highest score wins; players within `draw_margin` of the top draw."""
    if not scores:
        return {}
    top = max(scores.values())
    leaders = [p for p, s in scores.items() if top - s <= draw_margin]
    if len(leaders) == 1:
        return {p: "won" if p == leaders[0] else "lost" for p in scores}
    return {p: "drawn" if p in leaders else "lost" for p in scores}


def conquest_outcomes(states: Mapping[str, str]) -> dict[str, Outcome] | None:
    """Outcomes from the engine's player states ("active", "won", "defeated"), or None while the game goes on.

    The game is over once someone has won or at most one player is still active. Everyone defeated at once (a mutual
    wipe-out) is a draw between them.
    """
    if not states:
        return None
    won = [p for p, s in states.items() if s == "won"]
    active = [p for p, s in states.items() if s == "active"]
    if won:
        return {p: "won" if p in won else "lost" for p in states}
    if len(active) == 1:
        return {p: "won" if p == active[0] else "lost" for p in states}
    if not active:
        return dict.fromkeys(states, "drawn")
    return None


def seat_score(outcome: Outcome | None, calls: Sequence[BrainCall], *, graded_by_calls: bool) -> float:
    """The seat's grade as `grade_match` computes it with `max_missed: 0.5`: outcome points, but 0 for an agent seat
    that forfeited or never answered once (grade_match's "played" gate)."""
    if graded_by_calls and (forfeited(calls) or moves(calls)[0] == 0):
        return 0.0
    return POINTS.get(outcome or "drawn", 0.5)
