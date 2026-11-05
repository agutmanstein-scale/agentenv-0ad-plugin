"""Sampler log-probs: one per sampled token or none, counted either way, and the box-side smoke check that fails
loudly when vLLM does not return them aligned."""

import json

import httpx
import pytest

from agentenv_zeroad.brains import ORDER_TOOLS_V2, STRATEGIST_SYSTEM_V2
from agentenv_zeroad.llm import StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train import smoke
from agentenv_zeroad.train.spec import TRAIN_SAMPLING

pytestmark = pytest.mark.anyio

TOOLS = [ToolSpec("hold", "do nothing", {"type": "object", "properties": {}})]


def reply_json(*, token_ids=(7, 8), logprobs=(-0.5, -1.25), calls=True) -> dict:
    choice = {"message": {"content": "", "tool_calls": []}}
    if calls:
        call = {"id": "c", "type": "function", "function": {"name": "hold", "arguments": "{}"}}
        choice["message"]["tool_calls"] = [call]
    if token_ids is not None:
        choice["token_ids"] = list(token_ids)
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": "t", "logprob": x} for x in logprobs]}
    return {"choices": [choice], "prompt_token_ids": [1, 2, 3], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def served(body: dict | list[dict]) -> tuple[httpx.AsyncClient, list[dict]]:
    sent: list[dict] = []
    replies = iter(body if isinstance(body, list) else [])

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies) if isinstance(body, list) else body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), sent


def model(http: httpx.AsyncClient, *, logprobs: bool = True) -> StrategistModel:
    extra = {"return_token_ids": True, **({"logprobs": True, "top_logprobs": 0} if logprobs else {})}
    return StrategistModel("local/qwen-base", "http://vllm:8000", "k", SpendGuard("t"), http=http, extra_body=extra)


# ---- parsing -------------------------------------------------------------------------------------------------------
async def test_log_probs_one_per_token_come_back_in_token_order():
    http, sent = served(reply_json())
    llm = model(http)
    reply = await llm.call("system", "brief", TOOLS)
    assert reply.token_ids == [7, 8] and reply.logprobs == [-0.5, -1.25]
    assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 0 and sent[0]["return_token_ids"] is True
    assert llm.sampler_stats() == {
        "calls": 1, "failed_calls": 0, "logprobs_aligned": 1, "logprobs_mismatched": 0, "logprobs_missing": 0
    }  # fmt: skip


@pytest.mark.parametrize(
    "token_ids, logprobs",
    [((7, 8, 9), (-0.5, -1.25)), ((7,), (-0.5, -1.25)), (None, (-0.5, -1.25))],
    ids=["fewer log-probs", "more log-probs", "no token ids to align with"],
)
async def test_a_mismatched_turn_is_kept_with_null_log_probs_and_counted(token_ids, logprobs):
    http, _ = served(reply_json(token_ids=token_ids, logprobs=logprobs))
    llm = model(http)
    reply = await llm.call("system", "brief", TOOLS)
    assert reply.logprobs is None and [c.name for c in reply.calls] == ["hold"]
    assert reply.token_ids == (list(token_ids) if token_ids else None) and reply.prompt_token_ids == [1, 2, 3]
    assert llm.sampler_stats()["logprobs_mismatched"] == 1 and llm.sampler_stats()["logprobs_aligned"] == 0


async def test_a_reply_without_log_probs_is_counted_missing_when_they_were_asked_for():
    http, _ = served(reply_json(logprobs=None))
    llm = model(http)
    assert (await llm.call("system", "brief", TOOLS)).logprobs is None
    assert llm.sampler_stats()["logprobs_missing"] == 1


async def test_log_probs_nobody_asked_for_are_not_counted_at_all():
    http, _ = served(reply_json(logprobs=None))
    llm = model(http, logprobs=False)
    await llm.call("system", "brief", TOOLS)
    stats = llm.sampler_stats()
    assert stats["calls"] == 1 and stats["logprobs_missing"] == stats["logprobs_mismatched"] == 0


async def test_a_failed_request_is_counted_and_raises():
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})))
    llm = model(http)
    with pytest.raises(httpx.HTTPStatusError):
        await llm.call("system", "brief", TOOLS)
    assert llm.sampler_stats()["failed_calls"] == 1 and llm.sampler_stats()["calls"] == 0


def test_training_sampling_is_sent_explicitly_with_top_k_off():
    assert TRAIN_SAMPLING == {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    body = smoke.training_body()
    assert {k: body[k] for k in TRAIN_SAMPLING} == TRAIN_SAMPLING
    assert body["logprobs"] is True and body["top_logprobs"] == 0 and body["return_token_ids"] is True


# ---- the box-side smoke check --------------------------------------------------------------------------------------
async def run_smoke(replies: list[dict]) -> tuple[list[str], dict]:
    http, sent = served(replies)
    result = await smoke.check("http://vllm:8000", "local/qwen-base", "k", calls=len(replies), http=http)
    assert all(body["max_tokens"] == 512 and body["temperature"] == 1.0 and body["top_k"] == -1 for body in sent)
    return result


async def test_the_smoke_check_passes_when_every_reply_is_aligned_and_one_makes_a_tool_call():
    problems, summary = await run_smoke([reply_json(), reply_json(calls=False)])
    assert problems == [] and summary["tool_calls"] == 1 and summary["logprobs_aligned"] == 2
    assert summary["completion_tokens"] == 4


@pytest.mark.parametrize(
    "bad, expected",
    [
        (reply_json(logprobs=(-0.5,)), "no log-probs for 2 tokens"),
        (reply_json(logprobs=None), "no log-probs for 2 tokens"),
        (reply_json(logprobs=(-0.5, 0.7)), "positive or not finite"),
        (reply_json(token_ids=None, logprobs=None), "no prompt or completion token ids"),
    ],
    ids=["short", "missing", "positive", "no ids"],
)
async def test_the_smoke_check_fails_loudly_on_a_reply_training_could_not_use(bad, expected):
    problems, _ = await run_smoke([reply_json(), bad])
    assert any(expected in p and p.startswith("call 1:") for p in problems), problems


async def test_the_smoke_check_fails_when_no_reply_calls_a_tool():
    problems, _ = await run_smoke([reply_json(calls=False), reply_json(calls=False)])
    assert problems == ["no reply made a tool call"]


async def test_the_smoke_command_exits_nonzero_and_names_the_problem(monkeypatch, capsys):
    http, _ = served(reply_json(logprobs=(-0.5,)))
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setattr(smoke.httpx, "AsyncClient", lambda: http)
    assert await smoke.main("http://vllm:8000", "local/qwen-base", 1) == 1
    captured = capsys.readouterr()
    assert "FAILED: call 0: no log-probs for 2 tokens" in captured.err and '"calls": 1' in captured.out


async def test_the_smoke_check_sends_the_v2_system_prompt_and_the_v2_order_tools():
    http, sent = served([reply_json()])
    await smoke.check("http://vllm:8000", "local/qwen-base", "k", calls=1, http=http)
    messages, tools = sent[0]["messages"], sent[0]["tools"]
    assert messages[0]["content"] == STRATEGIST_SYSTEM_V2
    assert sorted(t["function"]["name"] for t in tools) == sorted(ORDER_TOOLS_V2)
