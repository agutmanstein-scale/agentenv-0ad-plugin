"""One strategist model call through a LiteLLM proxy, on the route each model takes (spec §4.6, spike results).

- OpenAI GPT-6 models (`openai/gpt-6-*`) go to `POST {base}/v1/responses`: flat function tools
  `{"type": "function", name, description, parameters}` and `reasoning: {"effort": ...}`, because chat completions
  refuses function tools together with `reasoning_effort`.
- Every other model (Claude, Gemini, Grok) goes to `POST {base}/v1/chat/completions` with `tools`.

A call is stateless (system prompt + one user message: the brief and a short memory), so its cost stays flat over a
match. Each call is checked against the brain's `SpendGuard` before it is sent and charged after; tool calls come
back as `(name, arguments)` with arguments parsed, or an error for arguments that aren't a JSON object.

`extra_body` goes into every chat request as is: a vLLM server takes sampling settings, `chat_template_kwargs` (e.g.
Qwen's `enable_thinking: false`) and `return_token_ids`, whose prompt and completion token ids the reply keeps, so a
trainer sees exactly what the model saw and sampled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentenv_zeroad.spend import SpendGuard, bound

DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT_S = 120.0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Reply:
    text: str
    calls: list[ToolCall]
    seconds: float
    usd: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None


def route(model: str) -> str:
    return "responses" if model.startswith("openai/gpt-6") else "chat"


def base_of(url: str) -> str:
    """The LiteLLM base without a trailing `/` or `/v1`, so routes can be appended."""
    return url.rstrip("/").removesuffix("/v1")


def _parse_args(raw: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw, dict):
        return raw, None
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError) as e:
        return {}, f"arguments are not valid JSON: {e}"
    return (value, None) if isinstance(value, dict) else ({}, "arguments must be a JSON object")


class StrategistModel:
    """Calls one model with the seat's order tools."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        guard: SpendGuard,
        *,
        effort: str | None = "low",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http: httpx.AsyncClient | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.base = base_of(base_url)
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.guard = guard
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._http = http
        self.extra_body = dict(extra_body or {})
        self.route = route(model)

    def request(self, system: str, user: str, tools: list[ToolSpec]) -> tuple[str, dict[str, Any]]:
        """(path, body) for this model's route."""
        if self.route == "responses":
            body: dict[str, Any] = {
                "model": self.model,
                "instructions": system,
                "input": user,
                "tools": [
                    {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ],
                "max_output_tokens": self.max_tokens,
            }
            if self.effort:
                body["reasoning"] = {"effort": self.effort}
            return "/v1/responses", body
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        return "/v1/chat/completions", body

    async def call(self, system: str, user: str, tools: list[ToolSpec]) -> Reply:
        """One call. Raises `SpendCapReached` before a call that could pass the cap; a failed call (charged at its
        bound) raises its httpx or parsing error."""
        path, body = self.request(system, user, tools)
        most = bound(self.model, body, self.max_tokens)
        self.guard.check(most)
        started = time.perf_counter()
        try:
            http = self._http or httpx.AsyncClient()
            try:
                response = await http.post(self.base + path, json=body, headers=self._headers, timeout=self.timeout_s)
            finally:
                if self._http is None:
                    await http.aclose()
            response.raise_for_status()
            data = response.json()
            text, calls, tokens_in, tokens_out = (
                self._parse_responses(data) if self.route == "responses" else self._parse_chat(data)
            )
        except Exception:
            self.guard.record(self.model, None, None, most_usd=most)
            raise
        usd = self.guard.record(self.model, tokens_in, tokens_out, most_usd=most)
        choice = (data.get("choices") or [{}])[0]
        return Reply(
            text,
            calls,
            time.perf_counter() - started,
            usd,
            tokens_in,
            tokens_out,
            data.get("usage") or {},
            prompt_token_ids=data.get("prompt_token_ids") or choice.get("prompt_token_ids"),
            token_ids=choice.get("token_ids"),
        )

    @staticmethod
    def _parse_chat(data: dict[str, Any]) -> tuple[str, list[ToolCall], int | None, int | None]:
        message = data["choices"][0]["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function") or {}
            args, error = _parse_args(fn.get("arguments"))
            calls.append(ToolCall(fn.get("name", ""), args, error))
        usage = data.get("usage") or {}
        return message.get("content") or "", calls, usage.get("prompt_tokens"), usage.get("completion_tokens")

    @staticmethod
    def _parse_responses(data: dict[str, Any]) -> tuple[str, list[ToolCall], int | None, int | None]:
        text, calls = [], []
        for item in data.get("output") or []:
            if item.get("type") == "function_call":
                args, error = _parse_args(item.get("arguments"))
                calls.append(ToolCall(item.get("name", ""), args, error))
            elif item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content") or [] if c.get("type") == "output_text"]
        usage = data.get("usage") or {}
        return "\n".join(text), calls, usage.get("input_tokens"), usage.get("output_tokens")
