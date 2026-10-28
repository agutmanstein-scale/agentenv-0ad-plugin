"""The plugin as agent-env loads it: the bundle and its tasks, the results step against a served env, the CLI."""

import json
import os
from importlib.resources import files
from pathlib import Path

import pytest
from agent_env.config import reset_config
from agent_env.env.env import DeployedEnv
from agent_env.plugins._bundles import installed_bundles
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agentenv_game import SlotRequest
from agentenv_protocol import client
from agentenv_protocol.types import WELL_KNOWN_PATH
from click.testing import CliRunner

from agentenv_zeroad import cli, spend
from agentenv_zeroad.server import ZeroADEnv
from agentenv_zeroad.steps import SaveResultsTaskStep
from tests.served import served
from tests.zeroad_fakes import Handle, placer

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "smoke",
    "petra",
    "spike",
    "setup-a",
    "setup-b",
    "livestream",
    "livestream-4",
    "livestream-4-long",
    "livestream-4-conquest",
    "livestream-petra",
)
MODEL_TASKS = ("spike", "setup-a", "setup-b", "livestream-4")


@pytest.fixture
def local_stores(monkeypatch, tmp_path):
    """agent-env on its local default stores under tmp_path, whatever config the machine has."""
    config = tmp_path / "config.toml"
    config.write_text("")
    for var in [v for v in os.environ if v.startswith("AGENT_ENV_")]:
        monkeypatch.delenv(var)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    reset_config()
    yield tmp_path / "state"
    reset_config()


def task(name: str) -> list[dict]:
    return json.loads(files("agentenv_zeroad.bundles").joinpath(f"zeroad/tasks/{name}.json").read_text())


def test_the_bundle_is_installed_with_its_tasks():
    bundle = next(b for b in installed_bundles() if b.name == "zeroad")
    assert bundle.problem is None and bundle.package == "agentenv-zeroad"
    assert sorted(p.stem for p in (bundle.root / "tasks").glob("*.json")) == sorted(TASKS)


@pytest.mark.parametrize("name", TASKS)
def test_every_step_of_the_task_loads_as_agent_env_loads_it(name):
    registry = get_task_step_registry()
    for step in task(name):
        assert registry[step["type"]].from_dict(step).to_dict()["id"] == step["id"]


@pytest.mark.parametrize(("name", "controller"), [("smoke", "bot"), ("petra", "petra")])
def test_the_zero_dollar_tasks_seat_only_the_games_own_ai_and_grade_both_seats(name, controller):
    steps = task(name)
    types = [s["type"] for s in steps]
    assert "deploy_agent" not in types and "prompt_agent" not in types
    slots = [s for s in steps if s["type"] == "add_player_slot"]
    assert {(s["player_kind"], s["game_settings"]["controller"]) for s in slots} == {("ai", controller)}
    grade = next(s for s in steps if s["type"] == "grade_match")
    assert grade["player_ids"] == [s["player_id"] for s in slots]
    lobby = next(s for s in steps if s["type"] == "open_lobby")["game_settings"]
    assert lobby["speed"] == 2.0 and lobby["time_cap_minutes"] <= 10  # a few wall minutes
    assert {"save_match_files", "save_zeroad_results", "finish_match"} <= set(types)


def test_the_results_step_round_trips_through_the_registry():
    cls = get_task_step_registry()["save_zeroad_results"]
    assert cls is SaveResultsTaskStep
    step = cls.from_dict({"id": "r", "type": "save_zeroad_results", "env_id": "zeroad", "out_dir": "/tmp/x"})
    assert cls.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert step.fail_task_on_error is False


@pytest.mark.anyio
async def test_the_results_step_keeps_the_matchs_results_in_run_metadata_and_a_folder(local_stores, tmp_path):
    async def start():
        return Handle()

    env = ZeroADEnv(start, tmp_path / "env", tick_s=0.05, match_options={"placer": placer()})
    async with served(env) as base:
        env.new_lobby({"speed": 2.0, "time_cap_minutes": 0.02})
        env.fill_slot(SlotRequest(player_id="1", player_kind="ai", game_settings={"controller": "bot"}))
        env.fill_slot(SlotRequest(player_id="2", player_kind="ai"))
        await env.close_lobby()
        await env.finish_match()
        record = DeployedEnv(
            env_id="zeroad",
            env_version=1,
            environment_card_url=base + WELL_KNOWN_PATH,
            environment_card=await client.get_card(base),
        )
        context = TaskStepContext(
            deployed_envs=[record], metadata={"game_lobby": {"lobby_id": env.lobby.lobby_id}}, instance_id="i1"
        )
        step = SaveResultsTaskStep("results", None, env_id="zeroad", out_dir=str(tmp_path / "kept"))
        context = await step.execute(context)
    kept = context.metadata["zeroad_results"]
    assert kept["status"] == "finished" and kept["seats"]["1"]["kind"] == "bot"
    assert "calls" not in kept["seats"]["1"]["tool_calls"]
    names = sorted(p.name for p in (tmp_path / "kept" / "i1").iterdir())
    assert names == [f"zeroad-{env.lobby.lobby_id}-commands.txt", f"zeroad-{env.lobby.lobby_id}-results.json"]


def test_setup_builds_the_engine_image_when_missing_then_the_env_image_on_it(monkeypatch):
    monkeypatch.setattr(cli, "image_exists", lambda image: False)
    engine, env = cli.build_commands(ROOT, "mcp-server-zeroad", "zeroad-engine:0.28")
    assert engine == [str(ROOT / "scripts/build-engine.sh")]
    pinned = "https://github.com/earakely-scale/agentenv-game-env/archive/81ef9f2bf724aca2ba3a94212c8d539eb171c189"
    assert f"GAME_ENV={pinned}.tar.gz" in env and "ENGINE_IMAGE=zeroad-engine:0.28" in env
    assert env[-3:] == ["-t", "mcp-server-zeroad", str(ROOT)]
    monkeypatch.setattr(cli, "image_exists", lambda image: True)
    assert len(cli.build_commands(ROOT, "mcp-server-zeroad", "zeroad-engine:0.28")) == 1


def test_the_env_image_is_built_on_the_engine_image_and_runs_the_server_as_a_user():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ARG ENGINE_IMAGE=zeroad-engine:0.28\nFROM ${ENGINE_IMAGE}" in dockerfile
    assert "USER ubuntu" in dockerfile and "ZEROAD_ENGINE=local" in dockerfile
    assert 'CMD ["python", "-m", "agentenv_zeroad.server"]' in dockerfile


@pytest.mark.parametrize("name", MODEL_TASKS)
def test_the_model_tasks_play_real_time_with_capped_brains_graded_on_missed_calls(name):
    steps = task(name)
    by_type: dict[str, list[dict]] = {}
    for s in steps:
        by_type.setdefault(s["type"], []).append(s)
    lobby = by_type["open_lobby"][0]["game_settings"]
    assert lobby["speed"] == 1.0 and lobby["time_cap_minutes"] <= 10
    agents = {s["agent_name"]: s for s in by_type["deploy_agent"]}
    assert all(float(a["env_vars"]["ZEROAD_MAX_COST_USD"]) > 0 for a in agents.values())
    assert {a["a2a_agent_id"] for a in agents.values()} <= {"zeroad-commander", "zeroad-luna"}
    prompts = {s["agent_name"]: s for s in by_type["prompt_agent"]}
    assert set(prompts) == set(agents) and all(p["model"] for p in prompts.values())
    assert by_type["finish_match"][0]["depends_on"] == [p["id"] for p in by_type["prompt_agent"]]
    assert by_type["grade_match"][0]["max_missed"] == 0.5
    assert "sk-" not in json.dumps(steps) and "api_key" not in json.dumps(steps)


def test_the_livestream_renders_setup_as_lineup_for_40_game_minutes_on_the_live_env():
    live, setup_a = task("livestream"), task("setup-a")
    lobby = next(s for s in live if s["type"] == "open_lobby")["game_settings"]
    assert lobby["render"] is True and lobby["speed"] == 1.0 and lobby["time_cap_minutes"] == 40
    assert "rtmp" not in lobby  # record only
    assert {s.get("env_id") for s in live} - {None} == {"zeroad-live"}

    # Same lineup as setup-a. The caps are the livestream's own: setup-a's are sized for a 10-minute test match, and
    # a 40-minute stream must not have its brains stop a quarter of the way in.
    def lineup(steps):
        return [
            (s["player_name"], {k: v for k, v in s["game_settings"].items() if k != "reflex_max_usd"})
            for s in steps
            if s["type"] == "add_player_slot"
        ]

    def caps(steps):
        agents = {
            s["agent_name"]: float(s["env_vars"]["ZEROAD_MAX_COST_USD"]) for s in steps if s["type"] == "deploy_agent"
        }
        reflexes = {
            s["player_name"]: s["game_settings"]["reflex_max_usd"] for s in steps if s["type"] == "add_player_slot"
        }
        return agents | {f"{name}-reflex": usd for name, usd in reflexes.items()}

    assert lineup(live) == lineup(setup_a)
    assert caps(live).keys() == caps(setup_a).keys()
    assert all(caps(live)[k] >= caps(setup_a)[k] for k in caps(live))
    prompts = lambda steps: {s["agent_name"]: s["model"] for s in steps if s["type"] == "prompt_agent"}  # noqa: E731
    assert prompts(live) == prompts(setup_a)
    assert "sk-" not in json.dumps(live)


@pytest.mark.parametrize(
    ("name", "match_cap"), [("livestream-4", 9), ("livestream-4-long", 18), ("livestream-4-conquest", 410)]
)
def test_the_four_model_livestream_seats_four_priced_models_within_the_match_cap(name, match_cap):
    steps = task(name)
    assert {s.get("env_id") for s in steps} - {None} == {"zeroad-live"}
    lobby = next(s for s in steps if s["type"] == "open_lobby")["game_settings"]
    assert lobby["render"] is True and "rtmp" not in lobby
    slots = [s for s in steps if s["type"] == "add_player_slot"]
    assert [s["player_id"] for s in slots] == ["1", "2", "3", "4"]
    assert len({s["game_settings"]["civ"] for s in slots}) == 4
    models = [s["model"] for s in steps if s["type"] == "prompt_agent"]
    assert len(set(models)) == 4 and all(m in spend.PRICES for m in models)  # none priced at the unknown-model bound
    caps = sum(float(s["env_vars"]["ZEROAD_MAX_COST_USD"]) for s in steps if s["type"] == "deploy_agent")
    caps += sum(s["game_settings"]["reflex_max_usd"] for s in slots)
    assert caps <= match_cap  # LIVE_MATCH_CAP_USD


def test_the_zero_dollar_livestream_records_a_bot_against_petra_on_the_live_env():
    steps = task("livestream-petra")
    assert "deploy_agent" not in {s["type"] for s in steps}
    lobby = next(s for s in steps if s["type"] == "open_lobby")["game_settings"]
    assert lobby["render"] is True and lobby["time_cap_minutes"] <= 10 and "rtmp" not in lobby
    controllers = {s["game_settings"]["controller"] for s in steps if s["type"] == "add_player_slot"}
    assert controllers == {"bot", "petra"}
    assert {"save_match_files", "save_zeroad_results"} <= {s["type"] for s in steps}


def test_setup_live_builds_the_rt_engine_the_env_image_on_it_and_the_live_image(monkeypatch):
    monkeypatch.setattr(cli, "image_exists", lambda image: False)
    rt, env, live = cli.live_build_commands(ROOT, "mcp-server-zeroad-live")
    assert rt == [str(ROOT / "scripts/build-engine.sh"), "rt"]  # builds zeroad-engine:0.28 first
    assert "ENGINE_IMAGE=zeroad-engine:0.28-rt" in env and env[-3:-1] == ["-t", "mcp-server-zeroad-live-rt-base"]
    assert live[:4] == ["docker", "build", "-f", str(ROOT / "streamer/Dockerfile")]
    assert "ENV_IMAGE=mcp-server-zeroad-live-rt-base" in live and live[-3:] == [
        "-t",
        "mcp-server-zeroad-live",
        str(ROOT),
    ]
    streamer = (ROOT / "streamer/Dockerfile").read_text()
    assert "COPY mod/zeroad-director /usr/share/games/0ad/mods/zeroad-director" in streamer
    assert "google-chrome" in streamer and "!mod" in (ROOT / ".dockerignore").read_text()


def test_setup_a_gives_every_strategist_a_luna_reflex_and_setup_b_pits_luna_alone_against_one():
    a = {s["player_name"]: s["game_settings"] for s in task("setup-a") if s["type"] == "add_player_slot"}
    assert {g["reflex"] for g in a.values()} == {"luna"} and all(g["reflex_max_usd"] > 0 for g in a.values())
    models = {s["agent_name"]: s["model"] for s in task("setup-a") if s["type"] == "prompt_agent"}
    assert models == {"opus": "anthropic/claude-opus-5-5", "astra": "openai/gpt-6-astra"}
    b = task("setup-b")
    assert not any(s["game_settings"].get("reflex") for s in b if s["type"] == "add_player_slot")
    ids = {s["agent_name"]: s["a2a_agent_id"] for s in b if s["type"] == "deploy_agent"}
    assert ids == {"luna": "zeroad-luna", "opus": "zeroad-commander"}


def test_the_agents_are_built_from_the_repo_root_with_the_package_brains():
    cmd = cli.agent_build_command(ROOT, "zeroad-commander")
    assert cmd[:4] == ["docker", "build", "-t", "a2a-agent-zeroad-commander"] and cmd[-1] == str(ROOT)
    for agent in cli.AGENTS:
        dockerfile = (ROOT / "agents" / agent / "Dockerfile").read_text()
        assert "COPY src /tmp/pkg/src" in dockerfile and f"COPY agents/{agent}/agent.py" in dockerfile
        assert "USER agent" in dockerfile
    assert "!agents" in (ROOT / ".dockerignore").read_text()


def test_the_cli_group_lists_setup_and_reap():
    out = CliRunner().invoke(cli.zeroad, ["--help"]).output
    assert "setup" in out and "reap" in out


@pytest.mark.parametrize(
    ("args", "env_id", "hint"),
    [
        (["--live"], "zeroad-live", "livestream-petra"),
        (["--live", "--id", "my-live"], "my-live", "livestream-petra"),
        ([], "zeroad", "smoke"),
    ],
)
def test_setup_live_registers_the_live_image_as_zeroad_live_beside_the_plain_env(monkeypatch, args, env_id, hint):
    from types import SimpleNamespace

    import agent_env.artifact
    import agent_env.env

    put: dict[str, dict] = {}

    class Fake:
        def __init__(self, kind: str):
            self.kind = kind

        def put(self, **kwargs):
            put[self.kind] = kwargs
            return SimpleNamespace(id=kwargs["id"], version=1, image_name=kwargs.get("image_name"))

    monkeypatch.setattr(agent_env.artifact, "DockerImageArtifact", Fake("artifact"))
    monkeypatch.setattr(agent_env.env, "MCPServerEnv", Fake("env"))
    out = CliRunner().invoke(cli.zeroad, ["setup", "--image", "some-local-image", *args])
    assert out.exit_code == 0, out.output
    assert put["env"]["id"] == env_id and put["env"]["environment_name"] == "zeroad"
    assert put["artifact"]["image_name"] == "some-local-image"
    assert out.output.strip().endswith(f"agent-env run zeroad --task {hint}")
