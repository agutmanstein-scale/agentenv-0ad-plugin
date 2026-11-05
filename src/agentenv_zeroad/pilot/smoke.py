"""One strategist call through the box's vLLM, as a pilot game makes it: it must come back with a tool call and the
prompt's and the completion's token ids (box/pilot-bootstrap.sh runs it last). The key is read from $VLLM_API_KEY."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from agentenv_zeroad.brains import STRATEGIST_SYSTEM
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.pilot.games import TRAIN_SAMPLING
from agentenv_zeroad.spend import SpendGuard

RESOURCES = ("food", "wood", "stone", "metal")
ECONOMY = ToolSpec(
    "set_economy",
    "Set the share of gatherers on each resource (percentages that sum to 100).",
    {
        "type": "object",
        "properties": {r: {"type": "integer", "minimum": 0, "maximum": 100} for r in RESOURCES},
        "required": list(RESOURCES),
    },
)
BRIEF = (
    "Time 00:00. You are player 1 (Romans) on a small land map against one opponent. Village phase. Population 9/20: "
    "5 villagers, 4 soldiers. Resources: food 300, wood 300, stone 300, metal 300. No enemy in sight.\n\n"
    "Decide your orders now and call the tools."
)


async def main(url: str, model: str) -> int:
    body = {**TRAIN_SAMPLING, "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True}
    llm = StrategistModel(model, url, os.environ["VLLM_API_KEY"], SpendGuard("smoke"), extra_body=body)
    reply = await llm.call(STRATEGIST_SYSTEM, BRIEF, [ECONOMY])
    calls = [{"name": c.name, "arguments": c.arguments, "error": c.error} for c in reply.calls]
    tokens = {"prompt_tokens": len(reply.prompt_token_ids or []), "completion_tokens": len(reply.token_ids or [])}
    print(json.dumps({"seconds": round(reply.seconds, 2), "calls": calls, **tokens}))
    if not reply.calls or not reply.prompt_token_ids or not reply.token_ids:
        print("FAILED: the reply needs a tool call and both token id lists", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://172.17.0.1:8000")
    p.add_argument("--model", default="local/qwen-base")
    a = p.parse_args()
    sys.exit(asyncio.run(main(a.url, a.model)))
