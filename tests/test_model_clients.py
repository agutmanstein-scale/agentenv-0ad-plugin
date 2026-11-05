"""The strategist model and Luna Decisions clients against a fake LiteLLM: each model's route and request shape,
parsed tool calls and answers, spend per call, and no request sent past the cap."""

import httpx
import pytest

from agentenv_zeroad.decisions import DecisionsClient, choice, endpoint, score
from agentenv_zeroad.llm import StrategistModel, ToolSpec, base_of, route
from agentenv_zeroad.spend import SpendCapReached, SpendGuard
from tests.fake_models import KEY, FakeLiteLLM

pytestmark = pytest.mark.anyio

TOOLS = [ToolSpec("say", "Say a line.", {"type": "object", "properties": {"text": {"type": "string"}}})]


def test_gpt6_models_take_the_responses_route_and_every_other_model_chat_completions():
    assert route("openai/gpt-6-astra") == route("openai/gpt-6-sol") == "responses"
    assert route("anthropic/claude-opus-5-5") == route("gemini/gemini-3.1-pro-preview") == "chat"
    assert route("xai/grok-4.7") == "chat"
    assert base_of("https://proxy.example/v1/") == base_of("https://proxy.example") == "https://proxy.example"
    assert endpoint("https://proxy.example/v1") == "https://proxy.example/openai/v1/decisions"


async def test_a_gpt6_call_sends_flat_tools_and_reasoning_effort_to_responses():
    fake = FakeLiteLLM(tool_calls=[("say", {"text": "Ave"}), ("army", {"action": "attack", "target": "player_2"})])
    guard = SpendGuard("t")
    model = StrategistModel("openai/gpt-6-astra", "https://proxy.example/v1", KEY, guard, http=fake.http())
    reply = await model.call("system", "brief", TOOLS)
    [(path, headers, body)] = fake.requests
    assert path == "/v1/responses" and headers["authorization"] == f"Bearer {KEY}"
    assert body["reasoning"] == {"effort": "low"} and body["instructions"] == "system" and body["input"] == "brief"
    assert body["tools"] == [
        {"type": "function", "name": "say", "description": "Say a line.", "parameters": TOOLS[0].parameters}
    ]
    assert "reasoning_effort" not in body and "messages" not in body
    assert [(c.name, c.arguments) for c in reply.calls] == [
        ("say", {"text": "Ave"}),
        ("army", {"action": "attack", "target": "player_2"}),
    ]
    assert reply.text == "Plan." and reply.input_tokens == 3000 and reply.output_tokens == 600
    assert reply.usd == pytest.approx((3000 * 15 + 600 * 75) / 1e6) == guard.spent_usd


async def test_other_models_send_nested_tools_to_chat_completions():
    fake = FakeLiteLLM()
    model = StrategistModel(
        "anthropic/claude-opus-5-5", "https://proxy.example", KEY, SpendGuard("t"), http=fake.http()
    )
    reply = await model.call("system", "brief", TOOLS)
    [(path, _, body)] = fake.requests
    assert path == "/v1/chat/completions"
    assert body["messages"][0] == {"role": "system", "content": "system"} and body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "say" and "reasoning" not in body
    assert [c.name for c in reply.calls] == ["set_economy", "say"] and reply.calls[0].arguments["food"] == 50


async def test_a_local_models_extra_body_goes_into_each_request_and_its_token_ids_come_back():
    fake = FakeLiteLLM(token_ids=True)
    extra = {"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True}
    guard = SpendGuard("t")
    url = "http://172.17.0.1:8000/v1"
    model = StrategistModel("local/qwen3-8b", url, KEY, guard, http=fake.http(), extra_body=extra)
    reply = await model.call("system", "brief", TOOLS)
    [(_, _, body)] = fake.requests
    assert body["temperature"] == 1.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["return_token_ids"] is True and body["max_tokens"] == 2000
    assert reply.prompt_token_ids == [1, 2, 3] and reply.token_ids == [7, 8] and guard.spent_usd == 0
    assert reply.logprobs == [-0.5, -1.25]


async def test_a_reply_without_logprobs_has_none():
    model = StrategistModel("local/qwen3-8b", "http://h:8000", KEY, SpendGuard("t"), http=FakeLiteLLM().http())
    assert (await model.call("system", "brief", TOOLS)).logprobs is None


async def test_no_model_request_is_sent_once_it_could_pass_the_cap_and_a_failed_call_is_charged():
    fake = FakeLiteLLM(fail=500)
    guard = SpendGuard("t", cap_usd=0.2)
    model = StrategistModel("anthropic/claude-opus-5-5", "https://p", KEY, guard, max_tokens=2000, http=fake.http())
    with pytest.raises(httpx.HTTPStatusError):
        await model.call("s", "b", TOOLS)
    assert guard.spent_usd > 0 and guard.unpriced_calls == 1
    tight = StrategistModel(
        "anthropic/claude-opus-5-5", "https://p", KEY, SpendGuard("t", cap_usd=0.01), http=fake.http()
    )
    with pytest.raises(SpendCapReached):
        await tight.call("s", "b", TOOLS)
    assert len(fake.requests) == 1


async def test_a_decisions_call_asks_luna_every_question_at_once_on_the_openai_route():
    fake = FakeLiteLLM(choose=lambda q: q["choices"][-1]["value"])
    guard = SpendGuard("reflex")
    client = DecisionsClient("https://proxy.example/v1", KEY, guard, http=fake.http())
    questions = [
        choice("army", "Army?", {"press": "go on", "hold": "defend"}),
        score("aggression", "How hard?", {"low": "economy", "mid": "steady", "high": "all in"}),
    ]
    decision = await client.decide("context", questions)
    [(path, _, body)] = fake.requests
    assert path == "/openai/v1/decisions" and body["model"] == "gpt-6-luna" and body["input"] == "context"
    assert body["questions"][0] == {
        "type": "choice",
        "name": "army",
        "instructions": "Army?",
        "choices": [{"value": "press", "description": "go on"}, {"value": "hold", "description": "defend"}],
    }
    assert body["questions"][1]["levels"][0] == {"label": "low", "description": "economy"}
    assert decision.answers["army"].choice == "hold" and decision.answers["aggression"].score == 1.0
    assert decision.usd == pytest.approx((1500 * 0.15 + 10 * 0.75) / 1e6) == guard.spent_usd


async def test_an_answer_outside_the_offered_choices_is_dropped():
    fake = FakeLiteLLM(choose=lambda q: "launch_nukes")
    client = DecisionsClient("https://p", KEY, SpendGuard("t"), http=fake.http())
    decision = await client.decide("c", [choice("army", "Army?", {"press": "go on", "hold": "defend"})])
    assert decision.answers == {}


async def test_a_decisions_reply_without_usage_is_priced_from_its_input():
    fake = FakeLiteLLM(usage=False)
    guard = SpendGuard("t")
    client = DecisionsClient("https://p", KEY, guard, http=fake.http())
    await client.decide("x" * 4000, [choice("a", "A?", {"x": "x", "y": "y"})])
    assert guard.input_tokens >= 1000 and guard.spent_usd > 0


async def test_no_decisions_request_is_sent_once_it_could_pass_the_cap():
    fake = FakeLiteLLM()
    client = DecisionsClient("https://p", KEY, SpendGuard("luna", cap_usd=1e-7), http=fake.http())
    with pytest.raises(SpendCapReached):
        await client.decide("context", [choice("army", "Army?", {"hold": "hold", "keep": "keep"})])
    assert fake.requests == [] and client.guard.stopped
