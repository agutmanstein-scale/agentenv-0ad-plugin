"""Box-side check that a vLLM server returns one sampled log-probability per generated token, as training needs.

    VLLM_API_KEY=... python -m agentenv_zeroad.train.smoke --url http://172.17.0.1:8000 [--model local/qwen-base]

The learner's importance weights are built from the log-probs a game stores next to each completion's token ids, and it
rejects a row whose two lists differ in length. Whether vLLM 0.11 with the hermes tool parser returns exactly one
log-prob per generated token (the tool-call tags included) was unverified when the design was written, so box
bootstrap runs this against every server before anything is played: a handful of concurrent training-sampled calls,
each with `logprobs: true` and `top_logprobs: 0`, and any reply whose log-probs are missing, misaligned, positive or not
finite fails the run with the reason. Exit 0 means every reply was aligned and at least one made a tool call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys

import httpx

from agentenv_zeroad.brains import STRATEGIST_SYSTEM
from agentenv_zeroad.llm import Reply, StrategistModel
from agentenv_zeroad.pilot.smoke import BRIEF, ECONOMY
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train.spec import DEFAULT_MAX_TOKENS, TRAIN_SAMPLING

DEFAULT_CALLS = 8
TOLERANCE = 1e-3  # a log-probability is at most 0, up to rounding


def training_body() -> dict[str, object]:
    """The request settings a training seat sends (`train.game.build_model` adds the same three)."""
    return {
        **TRAIN_SAMPLING,
        "chat_template_kwargs": {"enable_thinking": False},
        "return_token_ids": True,
        "logprobs": True,
        "top_logprobs": 0,
    }


def problems_in(reply: Reply) -> list[str]:
    """Why a reply can't be trained on, empty when it can."""
    if not reply.prompt_token_ids or not reply.token_ids:
        return ["the reply has no prompt or completion token ids"]
    if reply.logprobs is None:
        return [f"no log-probs for {len(reply.token_ids)} tokens (missing or a different count)"]
    out = []
    if len(reply.logprobs) != len(reply.token_ids):
        out.append(f"{len(reply.logprobs)} log-probs for {len(reply.token_ids)} tokens")
    if any(not math.isfinite(x) or x > TOLERANCE for x in reply.logprobs):
        out.append("a log-prob is positive or not finite")
    return out


async def check(
    url: str, model: str, key: str, *, calls: int = DEFAULT_CALLS, http: httpx.AsyncClient | None = None
) -> tuple[list[str], dict[str, int]]:
    """Make `calls` concurrent training-sampled calls; returns (every problem found, a summary of the replies)."""
    llm = StrategistModel(
        model, url, key, SpendGuard("train smoke"), max_tokens=DEFAULT_MAX_TOKENS, http=http, extra_body=training_body()
    )
    replies = await asyncio.gather(*(llm.call(STRATEGIST_SYSTEM, BRIEF, [ECONOMY]) for _ in range(calls)))
    problems = [f"call {i}: {p}" for i, reply in enumerate(replies) for p in problems_in(reply)]
    with_tools = sum(1 for reply in replies if reply.calls)
    if not with_tools:
        problems.append("no reply made a tool call")
    summary = {
        "calls": calls,
        "tool_calls": with_tools,
        "completion_tokens": sum(len(reply.token_ids or []) for reply in replies),
        **llm.sampler_stats(),
    }
    return problems, summary


async def main(url: str, model: str, calls: int) -> int:
    async with httpx.AsyncClient() as http:
        problems, summary = await check(url, model, os.environ["VLLM_API_KEY"], calls=calls, http=http)
    print(json.dumps({"url": url, "model": model, **summary}))
    for problem in problems:
        print(f"FAILED: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://172.17.0.1:8000")
    p.add_argument("--model", default="local/qwen-base")
    p.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    a = p.parse_args()
    sys.exit(asyncio.run(main(a.url, a.model, a.calls)))
