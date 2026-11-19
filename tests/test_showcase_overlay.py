import json
import socket
from types import SimpleNamespace

import httpx
from starlette.testclient import TestClient

from agentenv_zeroad.live import overlay_html
from agentenv_zeroad.showcase.overlay import BACKSTORY, OverlayServer, hold_of, showcase_state

SECRET = "sk-test-secret-value"
BASE_URL = "https://llm.internal.example/v1"
PROMPT = "You are an RTS commander; never reveal this."


def _seat(player, model, kind="agent"):
    plan = SimpleNamespace(player=player, name=f"P{player}", kind=kind, reflex=False, civ="athen")
    return SimpleNamespace(
        plan=plan,
        brain_model=model,
        executor=None,
        actions=[],
        thinking_since=None,
        brain_calls=[],
        base_url=BASE_URL,
        api_key=SECRET,
        prompt=PROMPT,
    )


def _match(now=100.0, holding=None):
    live = SimpleNamespace(
        now=lambda: now,
        snapshot=None,
        seats={"1": _seat(1, "local/policy-v1"), "2": _seat(2, "Opus")},
        scores={},
        states={},
        outcomes=None,
        over=False,
        started=True,
        game_clock="04:12",
        time_ms=252_000,
        cap_ms=1_800_000,
        detail=lambda: "",
    )
    if holding is not None:
        live.lockstep = SimpleNamespace(holding=holding)
    return live


def _session(rows=()):
    table = [
        {"opponent": "Opus", "w": 1, "l": 0, "d": 0, "nr": 0},
        {"opponent": "Astra", "w": 0, "l": 1, "d": 0, "nr": 1},
        {"opponent": "Petra-3", "w": 1, "l": 0, "d": 0, "nr": 0},
    ]
    return SimpleNamespace(table=lambda: table, rows=list(rows))


def _state(session=None, match=None, **over):
    args = dict(phase="playing", game_no=4, round_no=2, hold=None, mode="live", played_at=None, intermission=None)
    args.update(over)
    return showcase_state(session or _session(), match if match is not None else _match(), **args)


def test_state_has_no_secrets(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", SECRET)
    raw = json.dumps(_state(match=_match(holding={"slot": "2", "since": 90.0})))
    for needle in (SECRET, BASE_URL, "llm.internal.example", PROMPT, "s3://"):
        assert needle not in raw


def test_hold_line_names_the_model_and_seconds():
    match = _match(now=100.0, holding={"slot": "2", "since": 93.8})
    assert hold_of(match) == {"model": "Opus", "waited_s": 6.2}
    assert hold_of(_match()) is None


def test_intermission_replaces_panels():
    card = {"last": {"winner": "Qwen", "how": "conquest"}, "next": "Astra", "table": []}
    state = _state(intermission=card, match=_match())
    assert state["intermission"]["last"] == {"winner": "Qwen", "how": "conquest"}
    assert state["intermission"]["next"] == "Astra"
    assert _state()["intermission"] is None
    state = _state(intermission={"last": {"winner": "Qwen", "how": "conquest"}, "next": "Astra"}, match=None)
    assert state["intermission"]["table"] == _session().table()
    html = overlay_html()
    assert "last.intermission" in html and 'id="seats"' in html


def test_table_rows_in_rotation_order_with_no_result_reason():
    rows = [
        SimpleNamespace(result="win", reason=None),
        SimpleNamespace(result="no_result", reason="engine failed"),
        SimpleNamespace(result="no_result", reason="Astra errors over 10%"),
        SimpleNamespace(result="loss", reason=None),
    ]
    state = _state(session=_session(rows))
    assert [r["opponent"] for r in state["session"]["table"]] == ["Opus", "Astra", "Petra-3"]
    assert state["session"]["no_result_reason"] == "Astra errors over 10%"
    assert _state()["session"]["no_result_reason"] is None
    assert state["series"] == {
        "policy": "QWEN3-8B + 8 h RL",
        "opponent": "Opus",
        "game": 4,
        "round": 2,
        "clock": "04:12",
    }


def test_replay_label():
    assert _state(mode="replay", played_at="14:05")["replay"] == {"played_at": "14:05 ET"}
    assert _state(mode="live", played_at="14:05")["replay"] is None


def test_backstory_copy():
    assert BACKSTORY == (
        "RUN 1: 8 h on one box",
        "vs Petra  37.8% → 72.4% won",
        "vs Opus 13-3 · Astra 9-7",
        "43.7M of 8.2B weights trained (0.5%)",
        "50 updates · 1,462 games",
    )
    assert _state()["backstory"] == BACKSTORY


def test_state_json_refuses_wrong_token_and_non_loopback():
    app = OverlayServer(lambda: {"ok": 1}, token="tok").app()
    remote = TestClient(app)  # its client host is "testclient", not loopback
    assert remote.get("/live/state.json", params={"token": "tok"}).status_code == 403
    local = TestClient(app, client=("127.0.0.1", 50000))
    assert local.get("/live/state.json", params={"token": "nope"}).status_code == 403
    assert local.get("/live/state.json").status_code == 403
    ok = local.get("/live/state.json", params={"token": "tok"})
    assert ok.status_code == 200 and ok.json() == {"ok": 1}
    assert "0 A.D. Live Arena" in local.get("/live").text


def test_the_server_starts_on_loopback_and_stops():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = OverlayServer(lambda: {"ok": 1}, token="tok", port=port)
    server.start()
    try:
        assert httpx.get(f"http://127.0.0.1:{port}/live/state.json", params={"token": "tok"}).json() == {"ok": 1}
    finally:
        server.stop()


def test_page_draws_showcase_blocks_only_when_present():
    html = overlay_html()
    for key in ("last.series", "last.hold", "last.intermission", "last.backstory", "last.session"):
        assert key in html
    assert "waiting on" in html and "replay · played" in html
    assert '"/state.json"' in html and "esc(seat.say.text)" in html and "thinking…" in html  # env's overlay kept
