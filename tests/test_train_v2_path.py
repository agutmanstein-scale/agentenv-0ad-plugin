"""One v2 training game from the engine to the learner's inbox: a lockstep game on a fake engine with the policy on
interface v2 against Petra, its results.json, credit's rows, the orchestrator's inbox file and the learner reading it.

Run with: uv run --with torch --with transformers --with peft pytest tests/test_train_v2_path.py -q
"""

import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import learner as L  # noqa: E402

from agentenv_zeroad.server import tool_specs  # noqa: E402
from agentenv_zeroad.train.credit import Credit, CreditConfig  # noqa: E402
from agentenv_zeroad.train.game import build_model, play_game  # noqa: E402
from agentenv_zeroad.train.orchestrate import write_rows  # noqa: E402
from agentenv_zeroad.train.spec import GameSpec  # noqa: E402
from tests.fake_models import FakeLiteLLM  # noqa: E402
from tests.zeroad_fakes import Handle, ScriptedEngine, placer  # noqa: E402

pytestmark = pytest.mark.anyio

POLICY = {
    "kind": "llm",
    "civ": "rome",
    "model": "local/policy-v3",
    "base_url": "http://vllm:8000",
    "api_key_env": "VLLM_KEY",
    "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
    "trainable": True,
}


async def test_a_v2_lockstep_game_becomes_credit_rows_the_learner_takes_from_its_inbox(tmp_path):
    spec = GameSpec.from_json(
        {"game_id": "v2-g1", "seats": [POLICY, {"kind": "petra", "civ": "pers"}], "size": 128, "time_cap_minutes": 1}
    )
    engine = ScriptedEngine()
    engine.net_worth = {"1": 3000, "2": 2000}
    fake = FakeLiteLLM(token_ids=True, tool_calls=[("hold", {}), ("say", {"text": "Ave"})])
    await play_game(
        spec,
        Handle(engine),
        tmp_path / "games" / spec.game_id,
        tools=await tool_specs(),
        model_for=lambda seat, slot: build_model(seat, slot, environ={"VLLM_KEY": "k"}, http=fake.http()),
        match_options={"placer": placer()},
    )
    results = json.loads((tmp_path / "games" / spec.game_id / "results.json").read_text())
    policy = results["seats"]["1"]
    assert results["status"] == "finished" and results["ended_by"] == "time_cap" and policy["outcome"] == "won"
    assert policy["interface"] == "v2" and results["score_rule"] == "net_worth"
    assert policy["score"]["net_worth"] == 3000 and len(policy["turns"]) == 3

    credit = Credit(CreditConfig(on_error="raise"))
    rows = credit.rows(results)
    assert len(rows) == 3 and credit.last["margin_sources"] == {"1": "net_worth"}
    assert credit.last["rewards"]["1"] == pytest.approx(0.5 + 0.5 * math.tanh(1000 / 2000), abs=1e-4)
    assert not {k for k in credit.counts if k.startswith("skipped") or "fallback" in k or "error" in k}

    cfg = L.LearnerConfig(base="unused", root=tmp_path / "learner")
    inbox = L.Inbox(cfg)
    write_rows(cfg.inbox, spec.game_id, rows)
    assert inbox.poll() == 3 and inbox.drain_counts() == {"accepted": 3}
    for decision in inbox.pool:
        assert decision.version == 3 and decision.is_hold and decision.completion == [7, 8]
        assert decision.logprobs == [-0.5, -1.25] and math.isfinite(decision.advantage)
