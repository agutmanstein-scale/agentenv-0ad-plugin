"""zeroad-commander: a big model (Claude, GPT-6, Gemini, Grok) plays one 0 A.D. seat as its strategist (spec §4.6).

agent-env deploys it with the seat's MCP server and the model endpoint from its own config (`[model]`, passed in as
LITELLM_BASE_URL / LITELLM_API_KEY); the task's prompt_agent step names the model. The loop is
`agentenv_zeroad.brains.Commander`: brief -> one model call with the order tools -> orders -> wait for the next trigger
(25 s, or an urgent event). With `ZEROAD_SHARE_REFLEX_KEY` on (default), it hands the endpoint to the seat's Luna
reflex (Setup A), which the env only uses when the slot sets `reflex: luna`.

Environment: ZEROAD_INTERVAL_S (25), ZEROAD_MIN_GAP_S (8), ZEROAD_MAX_COST_USD (this brain's spend cap; none if
unset), ZEROAD_SHARE_REFLEX_KEY (1). The key is never logged or put in the result.
"""

import logging
import os
from typing import Any

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

from agentenv_zeroad.brains import STRATEGIST_SYSTEM, Commander, CommanderSettings, seat_link, summary
from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.spend import SpendGuard

log = logging.getLogger("zeroad-commander")


class CommanderConfig(AgentConfig):
    model: str = "anthropic/claude-opus-5-5"
    system_prompt: str = ""
    effort: str | None = "low"
    model_params: dict[str, Any] = {"max_tokens": 2000}
    timeout_seconds: int = 7200


def _float(environ: dict[str, str], name: str, default: float | None) -> float | None:
    value = environ.get(name)
    return float(value) if value else default


@a2a_agent(
    identity=AgentIdentity(
        name="zeroad-commander",
        description="A big model plays one 0 A.D. seat as its strategist: one model call with the order tools every "
        "25 s or on an urgent event, through agent-env's model endpoint.",
        version="0.1.0",
    ),
    config=CommanderConfig,
    extensions=(MCP_CONFIG_V1,),
)
class ZeroADCommander(AgentEnvAgent):
    async def run(self, request: TaskRequest[CommanderConfig]) -> TaskResult:
        environ, config = dict(os.environ), request.config
        base, key = environ.get("LITELLM_BASE_URL", ""), environ.get("LITELLM_API_KEY", "")
        if not base or not key or not request.mcp_servers:
            return TaskResult.failure(
                "no_endpoint", "needs LITELLM_BASE_URL, LITELLM_API_KEY and the seat's MCP server"
            )
        [server] = request.mcp_servers.values()
        guard = SpendGuard(f"commander {config.model}", cap_usd=_float(environ, "ZEROAD_MAX_COST_USD", None))
        max_tokens = int((config.model_params or {}).get("max_tokens", 2000))
        model = StrategistModel(config.model, base, key, guard, effort=config.effort, max_tokens=max_tokens)
        settings = CommanderSettings(
            interval_s=_float(environ, "ZEROAD_INTERVAL_S", 25.0) or 25.0,
            min_gap_s=_float(environ, "ZEROAD_MIN_GAP_S", 8.0) or 8.0,
            system_prompt=config.system_prompt or STRATEGIST_SYSTEM,
            opening="\n".join(p.text for p in request.parts if getattr(p, "text", None)),
        )
        commander = Commander(model, settings)
        share = environ.get("ZEROAD_SHARE_REFLEX_KEY", "1") not in ("0", "false", "no")
        try:
            async with seat_link(server) as link:
                await commander.play(link, reflex_key=(base, key) if share else None)
        except Exception as e:  # noqa: BLE001 - the result says why the brain stopped
            commander.record.end_reason = "error"
            commander.record.last_error = f"{type(e).__name__}: {str(e).replace(key, '<key>')[:500]}"
        record = commander.record
        out = summary(record, "zeroad-commander", config.model, guard.summary())
        text = (
            f"{record.end_reason}: {record.calls} model calls ({record.errors} failed), {record.orders} orders, "
            f"${guard.spent_usd:.4f} ({config.model})"
        )
        return (
            TaskResult.builder()
            .succeeded()
            .add_text(text)
            .add_structured_output(out)
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
    ZeroADCommander().serve()
