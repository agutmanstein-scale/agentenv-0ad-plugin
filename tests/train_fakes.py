"""Synthetic results.json files in the shape the lockstep runner writes, for the learning stream's tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def turn(
    began_ms: int,
    *,
    tools: tuple[str, ...] = ("army",),
    status: str = "accepted",
    model: str = "local/policy-v3",
    tokens: int = 4,
    logprobs: int | None = None,
) -> dict[str, Any]:
    return {
        "began_ms": began_ms,
        "user": f"brief at {began_ms}",
        "text": "",
        "orders": [{"tool": t, "arguments": {}, "status": status} for t in tools],
        "prompt_token_ids": [1, 2, 3, began_ms // 1000 % 50],
        "token_ids": list(range(10, 10 + tokens)),
        "logprobs": [-0.5] * (tokens if logprobs is None else logprobs),
        "model": model,
    }


def timeline(
    minutes: float = 30, *, army: tuple[float, float] = (10, 10), net_worth: tuple[float, float] | None = (100, 100)
) -> list[dict[str, Any]]:
    """A row every 10 s; each side's army and net worth grow linearly at its own rate (net worth None: none)."""
    rows = []
    for t_ms in range(0, int(minutes * 60_000) + 1, 10_000):
        players = {}
        for slot, i in (("1", 0), ("2", 1)):
            row: dict[str, Any] = {
                "state": "active", "total": 10.0 * t_ms / 1000, "economy": 5.0 * t_ms / 1000, "military": 1.0 * i,
                "exploration": 3.0, "pop": 20 + t_ms / 60_000, "army": army[i] * t_ms / 600_000,
                "gatherers": 15 + t_ms / 60_000, "resources": 400.0, "phase": "village", "buildings": 5,
                "kills": 0, "lost": 0,
            }  # fmt: skip
            if net_worth is not None:
                row["net_worth"] = net_worth[i] * t_ms / 1000
            players[slot] = row
        rows.append({"t_ms": t_ms, "players": players})
    return rows


def results(
    game_id: str = "g1",
    *,
    ended_by: str = "time_cap",
    outcomes: tuple[str, str] = ("won", "lost"),
    net_worth: tuple[float | None, float | None] = (5000.0, 3000.0),
    totals: tuple[float, float] = (9000.0, 8000.0),
    turns: Mapping[str, list[dict[str, Any]]] | None = None,
    opponent: Mapping[str, Any] | None = None,
    trainable: tuple[bool, bool] = (True, False),
    score_rule: str | None = "net_worth",
    status: str = "finished",
    rows: list[dict[str, Any]] | None = None,
    model_1: str = "local/policy-v3",
    model_2: str = "local/policy-v0",
) -> dict[str, Any]:
    opponent = opponent or {"kind": "petra", "civ": "pers", "difficulty": 2, "behaviour": "balanced"}
    first = {"kind": "llm", "civ": "rome", "model": model_1, "base_url": "u", "api_key_env": "K"}
    second = {"kind": "llm", "civ": "pers", "model": model_2, "base_url": "u", "api_key_env": "K"}
    seats_spec = [{**first, "trainable": trainable[0]}]
    seats_spec.append({**(second if opponent.get("kind") == "llm" else opponent), "trainable": trainable[1]})
    seats = {}
    for i, slot in enumerate(("1", "2")):
        score: dict[str, Any] = {"total": totals[i]}
        if net_worth[i] is not None:
            score["net_worth"] = net_worth[i]
        kind = "agent" if seats_spec[i]["kind"] == "llm" else seats_spec[i]["kind"]
        seats[slot] = {
            "player": i + 1, "kind": kind, "civ": seats_spec[i]["civ"], "name": f"Seat {slot}",
            "outcome": outcomes[i], "score": score, "turns": (turns or {}).get(slot, []),
            "model": seats_spec[i].get("model"),
        }  # fmt: skip
    out: dict[str, Any] = {
        "status": status, "ended_by": ended_by, "game_time_s": 1800.0, "wall_s": 400.0, "map": {"name": "mainland"},
        "seats": seats, "timeline": rows if rows is not None else timeline(),
        "spec": {"game_id": game_id, "seats": seats_spec}, "lockstep": {"latency_ms": 5000},
        "interface": "v2",
    }  # fmt: skip
    if score_rule is not None:
        out["score_rule"] = score_rule
    return out


def turns_every(seconds: int, minutes: float = 30, **kwargs: Any) -> list[dict[str, Any]]:
    return [turn(t * 1000, **kwargs) for t in range(0, int(minutes * 60), seconds)]
