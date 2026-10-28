"""zeroad-luna: GPT-6 Luna alone plays one 0 A.D. seat (Setup B, spec §4.6), graded like any agent seat.

Every second it reads the seat's compact JSON brief and sends one Decisions API request (`/openai/v1/decisions`,
model gpt-6-luna) with strategic and tactical choice and score questions, each answer mapped to an order tool
(`agentenv_zeroad.brains.LunaPlayer`). The model endpoint comes from agent-env's `[model]` config (LITELLM_BASE_URL /
LITELLM_API_KEY); the key is never logged or put in the result.

Environment: ZEROAD_INTERVAL_S (1), ZEROAD_MAX_COST_USD (this brain's spend cap; none if unset).
"""

import logging
import os

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    Usage,
    a2a_agent,
)

from agentenv_zeroad.brains import LunaPlayer, LunaSettings, seat_link, summary
from agentenv_zeroad.decisions import LUNA, DecisionsClient
from agentenv_zeroad.spend import SpendGuard

log = logging.getLogger("zeroad-luna")


class LunaConfig(AgentConfig):
    model: str = LUNA
    timeout_seconds: int = 7200


@a2a_agent(
    identity=AgentIdentity(
        name="zeroad-luna",
        description="GPT-6 Luna alone plays one 0 A.D. seat: one Decisions API request a second with strategic and "
        "tactical questions, each answer an order.",
        version="0.1.0",
    ),
    config=LunaConfig,
    extensions=(MCP_CONFIG_V1,),
)
class ZeroADLuna(AgentEnvAgent):
    async def run(self, request: TaskRequest[LunaConfig]) -> TaskResult:
        environ = dict(os.environ)
        base, key = environ.get("LITELLM_BASE_URL", ""), environ.get("LITELLM_API_KEY", "")
        if not base or not key or not request.mcp_servers:
            return TaskResult.failure(
                "no_endpoint", "needs LITELLM_BASE_URL, LITELLM_API_KEY and the seat's MCP server"
            )
        [server] = request.mcp_servers.values()
        cap = environ.get("ZEROAD_MAX_COST_USD")
        model = request.config.model.removeprefix("openai/")
        guard = SpendGuard("luna player", cap_usd=float(cap) if cap else None)
        player = LunaPlayer(
            DecisionsClient(base, key, guard, model=model),
            LunaSettings(interval_s=float(environ.get("ZEROAD_INTERVAL_S") or 1.0)),
        )
        try:
            async with seat_link(server) as link:
                await player.play(link)
        except Exception as e:  # noqa: BLE001 - the result says why the brain stopped
            player.record.end_reason = "error"
            player.record.last_error = f"{type(e).__name__}: {str(e).replace(key, '<key>')[:500]}"
        record = player.record
        text = f"{record.end_reason}: {record.calls} Decisions calls ({record.errors} failed), {record.orders} orders, "
        text += f"${guard.spent_usd:.4f}"
        return (
            TaskResult.builder()
            .succeeded()
            .add_text(text)
            .add_structured_output(summary(record, "zeroad-luna", model, guard.summary()))
            .usage(
                Usage(
                    tool_call_count=record.orders,
                    input_tokens=guard.input_tokens,
                    output_tokens=guard.output_tokens,
                    total_tokens=guard.input_tokens + guard.output_tokens,
                    cost_usd=round(guard.spent_usd, 6),
                )
            )
            .build()
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ZeroADLuna().serve()
