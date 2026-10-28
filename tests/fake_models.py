"""A fake LiteLLM proxy for brain tests (no network, no keys): chat completions, responses and the Decisions route.

`FakeLiteLLM` is an httpx transport handler. It records every request (path, headers, JSON body) and answers by
route: `tool_calls` for the strategist routes, and for Decisions the first choice of every question (or a scripted
`choose` function), with usage. `fail` (a status code) or `delay_s` make it fail or answer slowly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx

KEY = "test-key-not-real"


class FakeLiteLLM:
    def __init__(
        self,
        *,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        choose: Callable[[dict[str, Any]], str] | None = None,
        usage: bool = True,
        fail: int | None = None,
        delay_s: float = 0.0,
    ):
        self.tool_calls = (
            tool_calls if tool_calls is not None else [("set_economy", _split()), ("say", {"text": "Ave"})]
        )
        self.choose = choose
        self.usage = usage
        self.fail = fail
        self.delay_s = delay_s
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [b for p, _, b in self.requests if p == path]

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append((request.url.path, dict(request.headers), body))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            return httpx.Response(self.fail, json={"error": "fake failure"})
        path = request.url.path
        if path == "/v1/chat/completions":
            calls = [
                {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(self.tool_calls)
            ]
            out: dict[str, Any] = {"choices": [{"message": {"content": "Plan.", "tool_calls": calls}}]}
            if self.usage:
                out["usage"] = {"prompt_tokens": 3000, "completion_tokens": 400}
            return httpx.Response(200, json=out)
        if path == "/v1/responses":
            items: list[dict[str, Any]] = [{"type": "reasoning", "summary": []}]
            items += [{"type": "function_call", "name": n, "arguments": json.dumps(a)} for n, a in self.tool_calls]
            items.append({"type": "message", "content": [{"type": "output_text", "text": "Plan."}]})
            out = {"output": items}
            if self.usage:
                out["usage"] = {"input_tokens": 3000, "output_tokens": 600}
            return httpx.Response(200, json=out)
        if path == "/openai/v1/decisions":
            answers = []
            for q in body["questions"]:
                if q["type"] == "choice":
                    value = self.choose(q) if self.choose else q["choices"][0]["value"]
                    answers.append({"type": "choice", "name": q["name"], "choice": value, "confidence": 0.9})
                else:
                    answers.append({"type": "score", "name": q["name"], "score": 1.0, "confidence": 0.8})
            out = {"answers": answers}
            if self.usage:
                out["usage"] = {"input_tokens": 1500, "output_tokens": 10}
            return httpx.Response(200, json=out)
        return httpx.Response(404, json={"error": f"no route {path}"})


def _split() -> dict[str, int]:
    return {"food": 50, "wood": 30, "stone": 10, "metal": 10}
