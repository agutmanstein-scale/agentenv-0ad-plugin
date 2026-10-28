"""GPT-6 Luna on OpenAI's Decisions API (spec §3, §4.6): one request answers several fixed-choice questions at once.

The route through a LiteLLM proxy is `POST {base}/openai/v1/decisions` (`/v1/decisions` answers 403; spike results).
A request is `{"model": "gpt-6-luna", "input": <context>, "questions": [...]}` where a `choice` question lists
`choices` of `{value, description}` and a `score` question lists 2-10 `levels` of `{label, description}`, lowest first.
The response has `answers`: `{type, name, choice, probabilities, confidence}` for a choice, `{type, name, score, ...}`
for a score, where the score is a probability-weighted level index.

The client spends through a `SpendGuard` (usage when the response carries it, else the request's bound) and never
retries: a reflex that is late is useless, so a slow or failed call is the caller's to skip.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from agentenv_zeroad.spend import SpendGuard, approx_tokens, bound

LUNA = "gpt-6-luna"
ROUTE = "/openai/v1/decisions"
DEFAULT_TIMEOUT_S = 2.0
OUTPUT_ALLOWANCE = 64  # tokens per question the bound allows for; Luna's output is a few tokens per answer


@dataclass(frozen=True)
class Choice:
    value: str
    description: str


@dataclass(frozen=True)
class Question:
    """One question: `choice` (pick one of `choices`) or `score` (rate on `levels`, lowest first)."""

    name: str
    instructions: str
    kind: Literal["choice", "score"] = "choice"
    choices: tuple[Choice, ...] = ()
    levels: tuple[Choice, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind, "name": self.name, "instructions": self.instructions}
        if self.kind == "choice":
            out["choices"] = [{"value": c.value, "description": c.description} for c in self.choices]
        else:
            out["levels"] = [{"label": c.value, "description": c.description} for c in self.levels]
        return out


def choice(name: str, instructions: str, options: dict[str, str]) -> Question:
    return Question(name, instructions, "choice", tuple(Choice(v, d) for v, d in options.items()))


def score(name: str, instructions: str, levels: dict[str, str]) -> Question:
    return Question(name, instructions, "score", levels=tuple(Choice(v, d) for v, d in levels.items()))


@dataclass(frozen=True)
class Answer:
    name: str
    kind: str
    choice: str | None = None
    score: float | None = None
    confidence: float | None = None


@dataclass
class Decision:
    answers: dict[str, Answer]
    seconds: float
    usd: float
    raw_usage: dict[str, Any] = field(default_factory=dict)


def endpoint(base_url: str) -> str:
    """The Decisions URL for a LiteLLM base given with or without a trailing `/v1`."""
    return base_url.rstrip("/").removesuffix("/v1") + ROUTE


def parse_answers(body: dict[str, Any], questions: list[Question]) -> dict[str, Answer]:
    """Answers by question name; an answer naming an unknown question or an unlisted choice is dropped."""
    allowed = {q.name: {c.value for c in q.choices} for q in questions}
    out: dict[str, Answer] = {}
    for raw in body.get("answers") or []:
        name = raw.get("name")
        if name not in allowed:
            continue
        kind = raw.get("type", "choice")
        if kind == "choice":
            value = raw.get("choice")
            if value not in allowed[name]:
                continue
            out[name] = Answer(name, kind, choice=value, confidence=raw.get("confidence"))
        elif kind == "score" and isinstance(raw.get("score"), int | float):
            out[name] = Answer(name, kind, score=float(raw["score"]), confidence=raw.get("confidence"))
    return out


class DecisionsClient:
    """Asks Luna one batch of questions per call."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        guard: SpendGuard,
        *,
        model: str = LUNA,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http: httpx.AsyncClient | None = None,
    ):
        self.url = endpoint(base_url)
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.guard = guard
        self.model = model
        self.timeout_s = timeout_s
        self._http = http

    async def decide(self, context: str, questions: list[Question]) -> Decision:
        """One Decisions request. Raises `SpendCapReached` before sending one that could pass the cap, and httpx or
        ValueError errors for a failed call (charged at its bound, since it may have been billed)."""
        body = {"model": self.model, "input": context, "questions": [q.to_json() for q in questions]}
        most = bound(f"openai/{self.model}", body, OUTPUT_ALLOWANCE * len(questions))
        self.guard.check(most)
        started = time.perf_counter()
        try:
            http = self._http or httpx.AsyncClient()
            try:
                response = await http.post(self.url, json=body, headers=self._headers, timeout=self.timeout_s)
            finally:
                if self._http is None:
                    await http.aclose()
            response.raise_for_status()
            data = response.json()
        except Exception:
            self.guard.record(f"openai/{self.model}", None, None, most_usd=most)
            raise
        usage = data.get("usage") or {}
        tokens_in = usage.get("input_tokens", usage.get("prompt_tokens"))
        tokens_out = usage.get("output_tokens", usage.get("completion_tokens"))
        if tokens_in is None:  # no usage reported: count the input text, which is what Luna is priced on
            tokens_in, tokens_out = approx_tokens(str(body)), OUTPUT_ALLOWANCE * len(questions)
        usd = self.guard.record(f"openai/{self.model}", int(tokens_in), int(tokens_out or 0), most_usd=most)
        return Decision(parse_answers(data, questions), time.perf_counter() - started, usd, dict(usage))
