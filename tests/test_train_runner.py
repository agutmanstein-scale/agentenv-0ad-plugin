"""The training runner: game specs, the strategist model an LLM seat gets, a batch of games on a pool of fake engines
whose results.json the pilot's scoring reads, a failing game, and the worker processes."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from agentenv_zeroad.llm import StrategistModel
from agentenv_zeroad.match import SAFETY_CAP_S
from agentenv_zeroad.pilot import scoring
from agentenv_zeroad.pilot.games import GameSpec as PilotSpec
from agentenv_zeroad.pilot.games import petra
from agentenv_zeroad.pool import EnginePool
from agentenv_zeroad.server import tool_specs
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train import runner
from agentenv_zeroad.train.game import build_model
from agentenv_zeroad.train.spec import GameSpec, SeatSpec, load_specs
from tests.fake_models import FakeLiteLLM
from tests.zeroad_fakes import FakeStarter, placer

pytestmark = pytest.mark.anyio

QWEN = {
    "kind": "llm",
    "civ": "rome",
    "model": "local/policy-v1",
    "base_url": "http://vllm:8000",
    "api_key_env": "VLLM_KEY",
    "sampling": {"temperature": 1.0, "top_p": 1.0},
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}


def spec_json(game_id="g1", **overrides) -> dict:
    return {
        "game_id": game_id,
        "seats": [QWEN, {"kind": "petra", "civ": "pers", "difficulty": 2, "behaviour": "aggressive"}],
        "seed": 11,
        "size": 128,
        "time_cap_minutes": 0.5,
        **overrides,
    }


# ---- specs -------------------------------------------------------------------------------------------------------
def test_a_spec_line_becomes_a_match_config_with_the_seats_as_agents_and_controllers():
    spec = GameSpec.from_json(spec_json(ai_seed=5, latency_ms=3000))
    config = spec.match_config()
    assert [(s.kind, s.civ, s.name, s.ai_difficulty, s.ai_behavior) for s in config.seats] == [
        ("agent", "rome", "Agent 1", 3, "balanced"),
        ("petra", "pers", "Petra 2", 2, "aggressive"),
    ]
    assert (config.seed, config.ai_seed, config.size, config.time_cap_s) == (11, 5, 128, 30.0)
    assert spec.latency_ms == 3000
    scenario = config.scenario()["settings"]
    assert scenario["AISeed"] == 5 and scenario["PlayerData"][1]["AIBehavior"] == "aggressive"


def test_a_game_without_a_time_cap_has_no_cap_in_its_config_and_the_safety_cap_behind_it():
    config = GameSpec.from_json(spec_json(time_cap_minutes=None)).match_config()
    assert config.time_cap_s == 0 and config.safety_cap_s == SAFETY_CAP_S == 45 * 60 and config.cap_s == 2700
    assert GameSpec.from_json(spec_json()).match_config().cap_s == 30.0


def test_the_interface_is_v1_until_the_interface_switch_is_merged():
    assert GameSpec.from_json(spec_json()).interface == "v1"
    with pytest.raises(ValueError, match="interface must be v1"):
        GameSpec.from_json(spec_json(interface="v2"))


def test_a_seat_can_be_marked_trainable_and_the_flag_survives_the_results_spec():
    spec = GameSpec.from_json(spec_json(seats=[{**QWEN, "trainable": True}, {"kind": "bot", "civ": "cart"}]))
    assert [s.trainable for s in spec.seats] == [True, False]
    assert [s["trainable"] for s in spec.to_json()["seats"]] == [True, False]


def test_a_spec_round_trips_through_json_without_a_key():
    spec = GameSpec.from_json(spec_json())
    assert GameSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec
    assert "VLLM_KEY" in json.dumps(spec.to_json()) and "key" not in {k for s in spec.to_json()["seats"] for k in s}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"size": 100}, "size must be one of"),
        ({"time_cap_minutes": 0}, "must be positive"),
        ({"game_id": "../x"}, "plain folder name"),
        ({"seats": [QWEN]}, "2 to 8 seats"),
        ({"surprise": 1}, "unknown game spec keys"),
        ({"seats": [QWEN, {"kind": "llm", "civ": "pers"}]}, "an llm seat needs model, base_url, api_key_env"),
        ({"seats": [QWEN, {"kind": "petra", "civ": "klingon"}]}, "civ must be one of"),
        ({"seats": [QWEN, {"kind": "petra", "civ": "pers", "difficulty": 9}]}, "difficulty is 0 to 5"),
        ({"seats": [QWEN, {"kind": "petra", "civ": "pers", "level": 2}]}, "unknown seat keys"),
        ({"seats": [QWEN, {"kind": "wizard", "civ": "pers"}]}, "seat kind must be"),
    ],
)
def test_a_bad_spec_is_refused_with_the_reason(change, message):
    with pytest.raises(ValueError, match=message):
        GameSpec.from_json(spec_json(**change))


def test_specs_load_from_json_lines_and_game_ids_must_be_unique(tmp_path):
    path = tmp_path / "specs.jsonl"
    path.write_text("\n".join(json.dumps(spec_json(f"g{i}")) for i in range(3)) + "\n\n")
    assert [s.game_id for s in load_specs(path)] == ["g0", "g1", "g2"]
    path.write_text("\n".join(json.dumps(spec_json("same")) for _ in range(2)))
    with pytest.raises(ValueError, match="unique"):
        load_specs(path)


# ---- the model an LLM seat gets ----------------------------------------------------------------------------------
async def test_an_llm_seats_model_asks_for_token_ids_with_its_sampling_and_its_key_from_the_environment():
    seat = SeatSpec(**{**QWEN, "max_tokens": 300})
    fake = FakeLiteLLM(token_ids=True)
    model = build_model(seat, "1", environ={"VLLM_KEY": "k1"}, http=fake.http())
    tools = await tool_specs()
    reply = await model.call("system", "user", tools)
    path, headers, body = fake.requests[0]
    assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer k1"
    assert body["temperature"] == 1.0 and body["top_p"] == 1.0 and body["max_tokens"] == 300
    assert body["return_token_ids"] is True and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["logprobs"] is True and body["top_logprobs"] == 0
    assert reply.prompt_token_ids == [1, 2, 3] and reply.token_ids == [7, 8] and reply.logprobs == [-0.5, -1.25]


def test_a_missing_key_variable_stops_the_game_before_it_starts():
    with pytest.raises(ValueError, match="VLLM_KEY is not set"):
        build_model(SeatSpec(**QWEN), "1", environ={})


# ---- a batch of games --------------------------------------------------------------------------------------------
def fake_model(fake: FakeLiteLLM):
    def model_for(seat: SeatSpec, slot: str) -> StrategistModel:
        return StrategistModel(
            seat.model, seat.base_url, "k", SpendGuard("t"), extra_body={"return_token_ids": True}, http=fake.http()
        )

    return model_for


def pilot_spec(game_id: str) -> PilotSpec:
    return PilotSpec(game_id, 1, "train", 11, petra(2), 1, policy_slot="1", policy_civ="rome", opponent_civ="pers")


async def test_a_batch_of_games_writes_results_the_pilots_scoring_reads_with_logprobs_and_applied_commands(tmp_path):
    fake = FakeLiteLLM(token_ids=True, delay_s=0.01)
    starter = FakeStarter()
    specs = [GameSpec.from_json(spec_json(f"g{i}", seed=i)) for i in range(3)]
    async with EnginePool(2, start_one=starter, mode="local", base_port=23_000) as pool:
        outcomes = await runner.run_games(
            specs,
            pool,
            tmp_path,
            concurrency=2,
            model_for=fake_model(fake),
            match_options={"placer": placer()},
        )
        assert pool._free.qsize() == 2
    assert [o.status for o in outcomes] == ["finished"] * 3
    assert [o.ended_by for o in outcomes] == ["time_cap"] * 3 and outcomes[0].outcomes == {"1": "won", "2": "lost"}
    ledger = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert sorted(row["game_id"] for row in ledger) == ["g0", "g1", "g2"]
    for spec in specs:
        results = json.loads((tmp_path / spec.game_id / "results.json").read_text())
        assert results["spec"]["game_id"] == spec.game_id and "commands" in results
        assert results["lockstep"]["latency_ms"] == 5000 and results["clock"]["lockstep"] is True
        assert results["interface"] == "v1" and results["score_rule"] == "official"
        assert results["sampler"]["1"]["calls"] == 2 and results["sampler"]["1"]["failed_calls"] == 0
        assert "2" not in results["sampler"]
        record = scoring.record(pilot_spec(spec.game_id), results)
        assert record.status == "finished" and record.outcome == "won" and record.calls == 2 and record.rejected == 0
        assert record.margin == 30.0 and record.game_time_s == 30.0
        [turn, *_] = results["seats"]["1"]["turns"]
        assert turn["logprobs"] == [-0.5, -1.25] and len(turn["logprobs"]) == len(turn["token_ids"])
        assert turn["began_ms"] == 0
        found = scoring.decisions(pilot_spec(spec.game_id), results)
        assert len(found) == 2 and found[0].prompt_token_ids == [1, 2, 3] and found[0].token_ids == [7, 8]
        assert found[0].began_ms == 0 and found[1].began_ms == 25_000
        assert not (tmp_path / spec.game_id / "commands.txt").exists()


async def test_games_run_at_most_concurrency_at_a_time_on_a_bigger_pool(tmp_path):
    starter = FakeStarter()
    async with EnginePool(4, start_one=starter, mode="local", base_port=23_100) as pool:
        running = peak = 0
        original = pool.lease

        @asynccontextmanager
        async def counting():
            nonlocal running, peak
            async with original() as handle:
                running += 1
                peak = max(peak, running)
                try:
                    yield handle
                finally:
                    running -= 1

        pool.lease = counting
        specs = [GameSpec.from_json(spec_json(f"g{i}")) for i in range(6)]
        fake = FakeLiteLLM(token_ids=True, delay_s=0.005)
        await runner.run_games(
            specs, pool, tmp_path, concurrency=3, model_for=fake_model(fake), match_options={"placer": placer()}
        )
    assert peak == 3


async def test_a_failing_game_is_an_outcome_its_engine_is_restarted_and_the_other_games_finish(tmp_path):
    starter = FakeStarter()
    fake = FakeLiteLLM(token_ids=True)
    good = fake_model(fake)

    def model_for(seat, slot):
        if model_for.calls == 0:
            model_for.calls += 1
            raise ValueError("the key variable is not set")
        return good(seat, slot)

    model_for.calls = 0
    specs = [GameSpec.from_json(spec_json(f"g{i}")) for i in range(3)]
    async with EnginePool(1, start_one=starter, mode="local", base_port=23_200) as pool:
        outcomes = await runner.run_games(
            specs, pool, tmp_path, concurrency=1, model_for=model_for, match_options={"placer": placer()}
        )
    assert [o.status for o in outcomes] == ["failed", "finished", "finished"]
    assert "the key variable is not set" in outcomes[0].detail
    assert len(starter.started) == 2 and starter.started[0].closed == 1
    assert (tmp_path / "g0" / "results.json").exists()  # the stopped match still wrote what it had


class DyingStarter(FakeStarter):
    """Starts engines like `FakeStarter`; the first one raises from its raw step once `after` steps were taken."""

    def __init__(self, after: int = 12) -> None:
        super().__init__()
        self.after = after

    async def __call__(self, options, home):
        running = await FakeStarter.__call__(self, options, home)
        if len(self.started) == 1:
            original, taken = running.engine.step_raw, 0

            async def dying(commands=()):
                nonlocal taken
                taken += 1
                if taken > self.after:
                    raise RuntimeError("engine gone")
                await original(commands)

            running.engine.step_raw = dying
        return running


async def test_an_engine_that_dies_mid_game_is_restarted_and_the_next_game_runs_on_the_new_one(tmp_path):
    starter = DyingStarter()
    fake = FakeLiteLLM(token_ids=True)
    specs = [GameSpec.from_json(spec_json(f"g{i}", time_cap_minutes=5)) for i in range(2)]
    async with EnginePool(1, start_one=starter, mode="local", base_port=23_400) as pool:
        outcomes = await runner.run_games(
            specs, pool, tmp_path, concurrency=1, model_for=fake_model(fake), match_options={"placer": placer()}
        )
    assert [o.status for o in outcomes] == ["failed", "finished"]
    assert "engine gone" in outcomes[0].detail
    assert len(starter.started) == 2 and starter.started[0].closed == 1
    assert starter.started[1].engine.raw_steps > 0


async def test_an_engine_that_will_not_restart_fails_its_game_and_the_worker_goes_on(tmp_path):
    starter = FakeStarter()
    fake = FakeLiteLLM(token_ids=True)
    good = fake_model(fake)
    restarts = 0

    def model_for(seat, slot):
        if model_for.calls == 0:
            model_for.calls += 1
            raise ValueError("the key variable is not set")
        return good(seat, slot)

    model_for.calls = 0
    specs = [GameSpec.from_json(spec_json(f"g{i}")) for i in range(3)]
    async with EnginePool(1, start_one=starter, mode="local", base_port=23_500) as pool:
        start_one = pool._start_one

        async def flaky(options, home):
            nonlocal restarts
            restarts += 1
            if restarts == 1:
                raise RuntimeError("docker run failed")
            return await start_one(options, home)

        pool._start_one = flaky
        outcomes = await runner.run_games(
            specs, pool, tmp_path, concurrency=1, model_for=model_for, match_options={"placer": placer()}
        )
        assert pool._free.qsize() == 1
    assert [o.status for o in outcomes] == ["failed", "failed", "finished"]
    assert outcomes[1].detail.startswith("no engine: RuntimeError: docker run failed")
    ledger = [json.loads(line)["game_id"] for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert ledger == ["g0", "g1", "g2"]


async def test_replays_are_copied_out_only_when_the_pool_keeps_them(tmp_path):
    starter = FakeStarter()
    fake = FakeLiteLLM(token_ids=True)
    async with EnginePool(1, start_one=starter, mode="local", keep_replays=True, base_port=23_300) as pool:
        await runner.run_games(
            [GameSpec.from_json(spec_json())],
            pool,
            tmp_path,
            concurrency=1,
            model_for=fake_model(fake),
            match_options={"placer": placer()},
        )
    assert (tmp_path / "g1" / "commands.txt").read_text().startswith('{"settings"')


# ---- worker processes --------------------------------------------------------------------------------------------
def test_specs_are_dealt_round_robin_and_engines_split_as_evenly_as_possible():
    assert runner.split(list(range(7)), 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert runner.engines_per_worker(10, 3) == [4, 3, 3] and runner.engines_per_worker(2, 2) == [1, 1]


async def test_the_supervisor_starts_each_worker_with_hash_seed_zero_its_shard_and_its_own_port_range(
    tmp_path, monkeypatch
):
    path = tmp_path / "specs.jsonl"
    path.write_text("\n".join(json.dumps(spec_json(f"g{i}")) for i in range(5)) + "\n")
    started = []

    class Child:
        async def wait(self):
            return 0

    async def fake_exec(*cmd, env):
        started.append((cmd, env))
        return Child()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    args = runner._parser().parse_args(
        ["--specs", str(path), "--out", str(tmp_path / "out"), "--engines", "5", "--concurrency", "2", "--workers", "2"]
    )
    assert await runner.supervise(args) == 0
    assert len(started) == 2
    for (cmd, env), expected_engines, ports in zip(started, ("3", "2"), (16_000, 17_000), strict=True):
        assert env["PYTHONHASHSEED"] == "0" and "--worker" in cmd
        assert cmd[cmd.index("--engines") + 1] == expected_engines
        assert cmd[cmd.index("--base-port") + 1] == str(ports) and cmd[cmd.index("--concurrency") + 1] == "2"
    shards = [json.loads(line)["game_id"] for line in (tmp_path / "out/.shards/shard-0.jsonl").read_text().splitlines()]
    assert shards == ["g0", "g2", "g4"]


async def test_a_worker_scans_ports_only_inside_its_own_range(tmp_path, monkeypatch):
    seen = {}

    class FakePool:
        def __init__(self, size, **options):
            seen.update(options, size=size)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    async def no_games(specs, pool, out_dir, **kwargs):
        return []

    monkeypatch.setattr(runner, "EnginePool", FakePool)
    monkeypatch.setattr(runner, "run_games", no_games)
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(spec_json()) + "\n")
    args = runner._parser().parse_args(
        ["--specs", str(path), "--out", str(tmp_path), "--engines", "2", "--worker", "--base-port", "17000"]
    )
    await runner.work(args)
    assert (seen["base_port"], seen["port_stop"], seen["size"]) == (17_000, 18_000, 2)


async def test_the_supervisor_refuses_a_bad_spec_before_starting_anything(tmp_path):
    path = tmp_path / "specs.jsonl"
    path.write_text(json.dumps(spec_json(size=7)) + "\n")
    args = runner._parser().parse_args(["--specs", str(path), "--out", str(tmp_path / "o"), "--engines", "1"])
    with pytest.raises(ValueError, match="size must be"):
        await runner.supervise(args)
