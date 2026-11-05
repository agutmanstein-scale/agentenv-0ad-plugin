"""One pilot game as an agent-env task: the policy (a zeroad-commander on the box's vLLM) against one opponent.

Opponents: Petra at a difficulty (1 very easy to 5 very hard), a snapshot of an earlier policy version (another
commander on the same vLLM, its own adapter), or a frontier model (a commander on the LiteLLM endpoint agent-env's
config names). The policy's commander counts its interval in game seconds and, in a training game, logs every turn
with its token ids. The match is unrendered, so it runs faster than real time; its files land in the game's folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

OpponentKind = Literal["petra", "snapshot", "frontier"]
CIVS = ("rome", "athen", "pers", "cart")  # the civs every earlier match played; the executor's roles cover them
PROMPT = (
    "You command an empire in a live, real-time 0 A.D. match against one opponent. The game never pauses for you. "
    "Win by conquest, or hold the highest score at the time cap. Read the brief, then give your orders with the tools."
)
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0}  # on-policy: the trainer scores the plain softmax
PLAY_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen's own non-thinking settings


@dataclass(frozen=True)
class Opponent:
    id: str  # petra-3, v2 (a policy version), sonnet
    kind: OpponentKind
    level: int | None = None
    model: str | None = None
    trainable: bool = True  # whether training games may be played against it (a frontier model: probes only)


def petra(level: int) -> Opponent:
    return Opponent(f"petra-{level}", "petra", level=level)


def version_model(version: int) -> str:
    """The vLLM model name of policy version `version`: 0 is the base model, n the adapter trained after round n-1."""
    return "local/qwen-base" if version == 0 else f"local/policy-v{version}"


def snapshot(version: int) -> Opponent:
    return Opponent(f"v{version}", "snapshot", model=version_model(version))


@dataclass(frozen=True)
class GameSpec:
    game_id: str  # r03-t07 (training) or r03-p-petra-2 (probe)
    round: int
    kind: Literal["train", "probe"]
    seed: int
    opponent: Opponent
    policy_version: int
    policy_slot: str  # "1" or "2": which start the policy gets
    policy_civ: str
    opponent_civ: str

    @property
    def policy_model(self) -> str:
        return version_model(self.policy_version)


@dataclass(frozen=True)
class PilotSettings:
    vllm_url: str  # as the agents' containers reach it, e.g. http://172.17.0.1:8000
    vllm_key: str
    time_cap_minutes: float = 30.0
    speed: float = 2.0
    size: int = 128
    frontier_cap_usd: float = 5.0  # one frontier opponent's spend cap for one game

    def local_agent_env(self, *, log_turns: bool, sampling: dict[str, Any]) -> dict[str, str]:
        body = {**sampling, "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True}
        return {
            "LITELLM_BASE_URL": self.vllm_url,
            "LITELLM_API_KEY": self.vllm_key,
            "ZEROAD_INTERVAL_CLOCK": "game",
            "ZEROAD_TURN_LOG": "1" if log_turns else "0",
            "ZEROAD_EXTRA_BODY": json.dumps(body, separators=(",", ":")),
            "ZEROAD_SHARE_REFLEX_KEY": "0",
        }


def task(spec: GameSpec, settings: PilotSettings, out_dir: str) -> list[dict[str, Any]]:
    """The agent-env task (its steps) that plays `spec` and keeps its files under `out_dir`."""
    env = "zeroad"
    ttl = int((settings.time_cap_minutes * 60 / settings.speed) + 3600)
    opp_slot = "2" if spec.policy_slot == "1" else "1"
    sampling = TRAIN_SAMPLING if spec.kind == "train" else PLAY_SAMPLING
    steps: list[dict[str, Any]] = [
        {"id": "deploy", "type": "deploy_env", "env_id": env, "ttl_seconds": ttl},
        _agent("policy", settings.local_agent_env(log_turns=spec.kind == "train", sampling=sampling), ttl),
    ]
    opp = spec.opponent
    if opp.kind == "snapshot":
        steps.append(_agent("opponent", settings.local_agent_env(log_turns=False, sampling=PLAY_SAMPLING), ttl))
    elif opp.kind == "frontier":
        vars_ = {
            "ZEROAD_MAX_COST_USD": f"{settings.frontier_cap_usd:g}",
            "ZEROAD_SHARE_REFLEX_KEY": "0",
            "ZEROAD_INTERVAL_CLOCK": "game",
        }  # the same call cadence as the policy at any speed
        steps.append(_agent("opponent", vars_, ttl))
    lobby = {
        "size": settings.size,
        "seed": spec.seed,
        "speed": settings.speed,
        "time_cap_minutes": settings.time_cap_minutes,
        "start_grace_seconds": 300,
    }
    steps.append({"id": "lobby", "type": "open_lobby", "env_id": env, "game_settings": lobby, "depends_on": ["deploy"]})
    steps.append(_slot(spec.policy_slot, "policy", spec.policy_civ, f"policy v{spec.policy_version}"))
    if opp.kind == "petra":
        steps.append(
            {
                "id": f"slot-{opp_slot}",
                "type": "add_player_slot",
                "env_id": env,
                "player_id": opp_slot,
                "player_kind": "ai",
                "game_settings": {"controller": "petra", "civ": spec.opponent_civ, "ai_difficulty": opp.level},
                "depends_on": ["lobby"],
            }
        )
    else:
        steps.append(_slot(opp_slot, "opponent", spec.opponent_civ, opp.id))
    steps.append({"id": "close", "type": "close_lobby", "env_id": env, "depends_on": ["slot-1", "slot-2"]})
    plays = [_prompt("policy", spec.policy_model, ttl)]
    if opp.kind != "petra":
        assert opp.model is not None
        plays.append(_prompt("opponent", opp.model, ttl))
    steps += plays
    steps += [
        {"id": "finish", "type": "finish_match", "env_id": env, "depends_on": [p["id"] for p in plays]},
        {"id": "grade", "type": "grade_match", "env_id": env, "max_missed": 0.5, "depends_on": ["finish"]},
        {"id": "results", "type": "save_zeroad_results", "env_id": env, "out_dir": out_dir, "depends_on": ["finish"]},
    ]
    return steps


def _agent(name: str, env_vars: dict[str, str], ttl: int) -> dict[str, Any]:
    return {
        "id": f"agent-{name}",
        "type": "deploy_agent",
        "agent_name": name,
        "a2a_agent_id": "zeroad-commander",
        "env_ids": [],
        "env_vars": env_vars,
        "ttl_seconds": ttl,
        "depends_on": ["deploy"],
    }


def _slot(slot: str, agent: str, civ: str, label: str) -> dict[str, Any]:
    return {
        "id": f"slot-{slot}",
        "type": "add_player_slot",
        "env_id": "zeroad",
        "player_id": slot,
        "player_kind": "agent",
        "player_name": agent,
        "game_settings": {"civ": civ, "label": label},
        "depends_on": ["lobby", f"agent-{agent}"],
    }


def _prompt(agent: str, model: str, ttl: int) -> dict[str, Any]:
    return {
        "id": f"play-{agent}",
        "type": "prompt_agent",
        "agent_name": agent,
        "prompt_id": agent,
        "prompt": PROMPT,
        "timeout_seconds": ttl,
        "fail_task_on_error": False,
        "depends_on": ["close"],
        "model": model,
    }
