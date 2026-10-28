import asyncio
import json

import httpx
import pytest

from agentenv_zeroad.engine import SIM_TIME_JS, Engine, EngineError, format_commands


def make_engine(handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return Engine("http://engine:6000/", client=client), seen


def test_step_posts_one_player_prefixed_json_line_per_command():
    engine, seen = make_engine(lambda r: httpx.Response(200, text='{"timeElapsed": 200}'))
    walk = {"type": "walk", "entities": [7], "x": 1.5, "z": 2, "queued": False}

    state = asyncio.run(engine.step({1: [walk], 2: [{"type": "stop", "entities": [9]}]}))

    assert state == {"timeElapsed": 200}
    assert seen[0].url == "http://engine:6000/step"
    assert seen[0].content.decode().split("\n") == [
        '1;{"type":"walk","entities":[7],"x":1.5,"z":2,"queued":false}',
        '2;{"type":"stop","entities":[9]}',
    ]


def test_step_with_no_commands_posts_an_empty_body():
    engine, seen = make_engine(lambda r: httpx.Response(200, text="{}"))
    asyncio.run(engine.step())
    assert seen[0].content == b""


def test_format_commands_accepts_pairs_and_keeps_order():
    assert format_commands([(2, {"a": 1}), (1, {"b": "x\ny"})]) == '2;{"a":1}\n1;{"b":"x\\ny"}'


def test_reset_posts_scenario_json_with_player_id_and_optional_replay():
    engine, seen = make_engine(lambda r: httpx.Response(200, text='{"timeElapsed": 0}'))

    asyncio.run(engine.reset({"mapType": "random"}))
    asyncio.run(engine.reset('{"x": 1}', save_replay=True))

    assert str(seen[0].url) == "http://engine:6000/reset?playerID=-1"
    assert json.loads(seen[0].content) == {"mapType": "random"}
    assert str(seen[1].url) == "http://engine:6000/reset?playerID=-1&saveReplay=1"
    assert seen[1].content == b'{"x": 1}'


def test_evaluate_and_sim_time_parse_json_results():
    engine, seen = make_engine(lambda r: httpx.Response(200, text="60000"))
    assert asyncio.run(engine.sim_time_ms()) == 60000
    assert seen[0].content.decode() == SIM_TIME_JS


def known_templates(known: dict[str, str]):
    """Answer /templates like 0.28: one XML line per name it found, unknown names silently left out."""

    def handler(request: httpx.Request) -> httpx.Response:
        names = request.content.decode().split("\n")
        return httpx.Response(200, text="".join(f"{known[n]}\n" for n in names if n in known))

    return handler


def test_templates_maps_names_to_xml_lines_in_one_request():
    engine, seen = make_engine(known_templates({"a": "<A/>", "b": "<B/>"}))
    assert asyncio.run(engine.templates(["a", "b"])) == {"a": "<A/>", "b": "<B/>"}
    assert [r.content for r in seen] == [b"a\nb"]
    assert asyncio.run(engine.templates([])) == {}


def test_templates_leaves_out_unknown_names_without_shifting_the_others():
    engine, _ = make_engine(known_templates({"a": "<A/>", "c": "<C/>"}))
    assert asyncio.run(engine.templates(["a", "typo", "c"])) == {"a": "<A/>", "c": "<C/>"}


def test_engine_400_becomes_engine_error_with_its_message():
    engine, _ = make_engine(lambda r: httpx.Response(400, text="Game not running. Please create a scenario first."))
    with pytest.raises(EngineError, match="Game not running"):
        asyncio.run(engine.step())


def test_connection_failure_becomes_engine_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    engine, _ = make_engine(refuse)
    with pytest.raises(EngineError, match="failed"):
        asyncio.run(engine.evaluate("1"))


def test_wait_ready_treats_any_http_answer_as_open():
    engine, _ = make_engine(lambda r: httpx.Response(404, text="Unrecognised URI"))
    asyncio.run(engine.wait_ready(timeout=1))


def test_wait_ready_fails_fast_when_the_process_died():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    async def dead() -> bool:
        return False

    engine, _ = make_engine(refuse)
    with pytest.raises(EngineError, match="exited"):
        asyncio.run(engine.wait_ready(timeout=30, alive=dead))
