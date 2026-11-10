"""The training orchestrator against fakes: a fake vLLM (httpx MockTransport), a fake game player, a fake credit, a fake
`aws` and a learner's folder written by hand. Nothing here starts an engine, a container or a network connection."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentenv_zeroad import spend
from agentenv_zeroad.brains import ORDER_TOOLS_V2
from agentenv_zeroad.train import budget, orchestrate
from agentenv_zeroad.train.config import RunConfig, load_config
from agentenv_zeroad.train.control import ControlWriter, TripWatch
from agentenv_zeroad.train.mix import BLOCK, GameFactory, deal, repin
from agentenv_zeroad.train.runner import GameOutcome
from agentenv_zeroad.train.serving import Adapters, Fleet
from agentenv_zeroad.train.spec import GameSpec, SeatSpec
from agentenv_zeroad.train.sync import Syncer, sync_command

pytestmark = pytest.mark.anyio

HOSTS = ("http://vllm0:8000", "http://vllm1:8000")


def policy_version(spec: GameSpec) -> int | None:
    """The adapter version of the spec's trainable seat (0 for the base model), None when it has none."""
    for seat in spec.seats:
        if seat.trainable:
            found = re.search(r"policy-v(\d+)$", seat.model or "")
            return int(found.group(1)) if found else 0
    return None


def raw_config(root: Path, **over) -> dict:
    raw = {
        "run_id": "r1",
        "root": str(root),
        "budget_usd": 100,
        "box_usd_per_hour": 10,
        "duration_hours": 10,
        "engines": 4,
        "concurrency": 4,
        "poll_s": 0.01,
        "learner_grace_s": 0,
        "sync_retry_s": 0,
        "servers": {"urls": list(HOSTS)},
        "s3": {"uri": "s3://bucket/prefix/r1"},
        **over,
    }
    if raw["servers"]:  # where the vLLM containers see this run's adapters: /train/<run folder>/adapters
        raw["servers"] = {"adapter_dir": f"/train/{root.name}/adapters", **raw["servers"]}
    return raw


def config(root: Path, **over) -> RunConfig:
    return RunConfig.from_dict(raw_config(root, **over))


# ---- config ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["box_usd_per_hour", "budget_usd", "duration_hours", "root", "run_id"])
def test_the_amounts_that_stop_a_run_have_no_defaults(tmp_path, missing):
    raw = raw_config(tmp_path)
    del raw[missing]
    with pytest.raises(ValueError, match=f"{missing} is required"):
        RunConfig.from_dict(raw)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"surprise": 1}, "unknown config keys: surprise"),
        ({"games": {"surprise": 1}}, "unknown games keys"),
        ({"mix": {"petra-9": 1.0}}, "unknown opponents"),
        ({"games": {"maps": ["mainland", "mountains"]}, "held_out": {"maps": ["mountains"]}}, "held-out maps"),
        ({"held_out": {"civ_pairs": [["rome", "kush"]]}}, "two of games.civs"),
        ({"concurrency": 9}, "concurrency"),
        ({"s3": {"uri": "https://x"}}, "s3://"),
        ({"servers": {}}, "at least one vLLM"),
        ({"box_usd_per_hour": 0}, "must be positive"),
        ({"games": {"interface": "v3"}}, "games.interface must be one of v1, v2"),
        ({"games": {"score_rule": "kills"}}, "games.score_rule must be one of"),
        ({"games": {"interface": "v1", "score_rule": "net_worth"}}, "contradicts the agents' prompt"),
    ],
)
def test_a_config_that_could_not_run_as_intended_is_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        config(tmp_path, **change)


def test_a_toml_file_loads_with_its_tables(tmp_path):
    panel = tmp_path / "panel.jsonl"
    path = tmp_path / "run.toml"
    path.write_text(
        f"""run_id = "run1"
root = "{tmp_path}/run1"
budget_usd = 140
box_usd_per_hour = 10.49
duration_hours = 10
shutdown = true
[servers]
urls = ["http://a:8000", "http://b:8001"]
adapter_dir = "/train/run1/adapters"
keep_versions = [4]
[mix]
bot = 0.5
petra-1 = 0.5
[held_out]
civ_pairs = [["rome", "pers"]]
seeds = [7, 8]
[s3]
uri = "s3://b/p"
[[panels]]
name = "mid"
at_hours = 5
specs = "{panel}"
"""
    )
    cfg = load_config(path)
    assert cfg.box_usd_per_hour == 10.49 and cfg.shutdown and cfg.s3_uri == "s3://b/p"
    assert cfg.servers.urls == ("http://a:8000", "http://b:8001") and cfg.servers.keep_versions == (4,)
    assert cfg.mix.shares() == {"bot": 0.5, "petra-1": 0.5} and cfg.held_out.seeds == (7, 8)
    assert cfg.panels[0].name == "mid" and cfg.panels[0].specs == panel
    assert cfg.ledger_path == tmp_path / "run1" / "spend.jsonl"


# ---- the game stream ------------------------------------------------------------------------------------------------
def opponent_of(spec: GameSpec) -> str:
    (other,) = [s for s in spec.seats if not s.trainable]
    if other.kind == "petra":
        return f"petra-{other.difficulty}"
    return "base" if other.kind == "llm" else other.kind


def test_largest_remainder_deals_exactly_n():
    dealt = deal({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100)
    assert len(dealt) == 100 and sorted(Counter(dealt).values()) == [33, 33, 34]


def test_every_block_of_a_hundred_games_holds_run_ones_mix_exactly(tmp_path):
    factory = GameFactory(config(tmp_path), start_index=0)
    specs = [factory.next(3, HOSTS[0]) for _ in range(5 * BLOCK)]
    for start in range(0, 5 * BLOCK, BLOCK):
        block = specs[start : start + BLOCK]
        assert Counter(opponent_of(s) for s in block) == {
            "bot": 10, "petra-1": 35, "petra-2": 35, "petra-3": 10, "base": 10
        }  # fmt: skip
        assert sum(s.time_cap_minutes is None for s in block) == 15
        assert all(s.time_cap_minutes in (None, 30.0) for s in block)


def test_games_vary_in_behaviour_civs_maps_seeds_and_the_policys_slot(tmp_path):
    factory = GameFactory(config(tmp_path, games={"maps": ["mainland", "islands"]}), start_index=0)
    specs = [factory.next(1, HOSTS[0]) for _ in range(400)]
    petra = [s for s in specs if opponent_of(s).startswith("petra")]
    assert {seat.behaviour for s in petra for seat in s.seats if seat.kind == "petra"} == {
        "balanced", "aggressive", "defensive"
    }  # fmt: skip
    assert {s.map for s in specs} == {"mainland", "islands"}
    assert {tuple(seat.trainable for seat in s.seats) for s in specs} == {(True, False), (False, True)}
    assert len({s.seed for s in specs}) > 300 and all(s.seed != s.ai_seed for s in specs[:50])
    assert len({s.game_id for s in specs}) == 400 and specs[0].game_id == "r1-g000000"


def test_held_out_civ_pairs_maps_and_seeds_never_appear_in_a_training_game(tmp_path):
    held = {"civ_pairs": [["rome", "pers"], ["athen", "cart"]], "maps": ["tundra"], "seeds": list(range(10, 40))}
    cfg = config(tmp_path, held_out=held, games={"seed_low": 1, "seed_high": 60, "maps": ["mainland", "islands"]})
    factory = GameFactory(cfg, start_index=0)
    seen_pairs = set()
    for _ in range(1500):
        spec = factory.next(2, HOSTS[1])
        pair = frozenset(seat.civ for seat in spec.seats)
        assert pair not in ({"rome", "pers"}, {"athen", "cart"}) and spec.map != "tundra"
        assert spec.seed not in held["seeds"] and spec.ai_seed not in held["seeds"]
        seen_pairs.add(pair)
    assert frozenset({"rome", "athen"}) in seen_pairs and frozenset({"rome"}) in seen_pairs  # mirrors are allowed


def test_the_policy_seat_is_pinned_and_trainable_and_the_frozen_base_is_not(tmp_path):
    factory = GameFactory(config(tmp_path, mix={"base": 1.0}), start_index=5)
    spec = factory.next(7, HOSTS[1])
    policy, base = sorted(spec.seats, key=lambda s: not s.trainable)
    assert (policy.model, policy.base_url, policy.trainable) == ("local/policy-v7", HOSTS[1], True)
    assert (base.model, base.base_url, base.trainable) == ("local/qwen-base", HOSTS[1], False)
    assert policy.sampling == {"temperature": 1.0, "top_p": 1.0, "top_k": -1} and policy.max_tokens == 512
    assert policy.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert spec.game_id == "r1-g000005" and policy_version(spec) == 7
    assert policy_version(factory.next(0, HOSTS[0])) == 0


def test_the_llm_seats_play_the_configured_interface_and_its_score_rule_decides_at_the_cap(tmp_path):
    spec = GameFactory(config(tmp_path, mix={"base": 1.0}), start_index=0).next(1, HOSTS[0])
    assert [s.interface for s in spec.seats] == ["v2", "v2"] and spec.match_config().score_rule == "net_worth"
    bot = GameFactory(config(tmp_path, mix={"bot": 1.0}), start_index=0).next(1, HOSTS[0])
    assert sorted(s.interface for s in bot.seats) == ["v1", "v2"]  # the scripted bot keeps its own interface
    cfg = config(tmp_path, games={"interface": "v1"}, mix={"base": 1.0})
    v1 = GameFactory(cfg, start_index=0).next(1, HOSTS[0])
    assert {s.interface for s in v1.seats} == {"v1"} and v1.match_config().score_rule == "official"


def test_repin_moves_the_policy_and_local_seats_and_leaves_a_frontier_seat_alone(tmp_path):
    frontier = SeatSpec(
        kind="llm", civ="cart", model="anthropic/claude-opus-5-5", base_url="http://litellm", api_key_env="K"
    )
    spec = GameSpec(
        "g",
        (GameFactory(config(tmp_path), start_index=0)._llm_seat("rome", 2, HOSTS[0], trainable=True), frontier),
    )
    moved = repin(spec, 9, HOSTS[1], game_id="g-r1")
    assert moved.game_id == "g-r1" and (moved.seats[0].model, moved.seats[0].base_url) == ("local/policy-v9", HOSTS[1])
    assert moved.seats[1] == frontier


# ---- fakes: vLLM, the game player, credit, aws ---------------------------------------------------------------------
class FakeVllm:
    def __init__(self):
        self.hosts = {
            h: {"up": True, "adapters": set(), "slow": False, "refuse_loads": False, "root": None} for h in HOSTS
        }
        self.calls: list[tuple[str, str, dict]] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def restart(self, host: str) -> None:
        self.hosts[host]["adapters"] = set()
        self.hosts[host]["up"] = True

    def loaded(self, host: str) -> set[str]:
        return set(self.hosts[host]["adapters"])

    def handle(self, request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        state = self.hosts[host]
        body = json.loads(request.content) if request.content else {}
        self.calls.append((host, request.url.path, body))
        if not state["up"]:
            raise httpx.ConnectError("connection refused", request=request)
        match request.url.path:
            case "/health":
                return httpx.Response(200)
            case "/v1/models":
                ids = ["local/qwen-base", *sorted(state["adapters"])]
                data = [{"id": i} for i in ids]
                if state["root"] is not None:
                    data = [{"id": i, "root": f"{state['root']}/v{i.rsplit('-v', 1)[-1]}"} for i in ids[1:]]
                    data.insert(0, {"id": "local/qwen-base"})
                return httpx.Response(200, json={"data": data})
            case "/v1/load_lora_adapter":
                if state["refuse_loads"]:
                    return httpx.Response(500, text="no such path")
                if state["root"] is not None:
                    state["root"] = None  # a load from the right path replaces the wrong one
                if body["lora_name"] in state["adapters"]:
                    return httpx.Response(400, json={"error": "the lora adapter has already been loaded"})
                state["adapters"].add(body["lora_name"])
                return httpx.Response(200, text="ok")
            case "/v1/unload_lora_adapter":
                if body["lora_name"] not in state["adapters"]:
                    return httpx.Response(404, text="no such adapter")
                state["adapters"].discard(body["lora_name"])
                return httpx.Response(200, text="ok")
            case "/v1/chat/completions":
                if state["slow"]:
                    raise httpx.ReadTimeout("the reply never came", request=request)
                if body["model"] != "local/qwen-base" and body["model"] not in state["adapters"]:
                    return httpx.Response(404, json={"error": "the model does not exist"})
                return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": []}}]})
        return httpx.Response(404)


class FakePlay:
    """Plays a spec instantly unless `gate` is closed; with `exercise` each llm seat's model makes one request first."""

    def __init__(self):
        self.specs: list[GameSpec] = []
        self.gate: asyncio.Event | None = None
        self.call_gate: asyncio.Event | None = None
        self.status = "finished"
        self.requested_by: dict[str, int] = {}

    def hold(self) -> None:
        self.gate = asyncio.Event()

    def release(self) -> None:
        assert self.gate is not None
        self.gate.set()

    async def __call__(self, spec: GameSpec, out_dir: Path, model_for) -> GameOutcome:
        self.specs.append(spec)
        if self.call_gate is not None:
            await self.call_gate.wait()
            for slot, seat in enumerate(spec.seats, 1):
                if seat.kind == "llm":
                    try:
                        await model_for(seat, str(slot)).call("s", "u", [])
                    except httpx.HTTPError:
                        pass
        if self.gate is not None:
            await self.gate.wait()
        folder = out_dir / spec.game_id
        folder.mkdir(parents=True)
        results = {
            "status": self.status,
            "ended_by": "conquest",
            "spec": spec.to_json(),
            "seats": {"1": {"outcome": "won"}, "2": {"outcome": "lost"}},
            "spend": {},
        }
        (folder / "results.json").write_text(json.dumps(results))
        return GameOutcome(spec.game_id, self.status, "conquest", {"1": "won", "2": "lost"}, 600.0, 7.0)


class FakeCredit:
    def __init__(self, rows_per_game: int = 2):
        self.games: list[str] = []
        self.rows_per_game = rows_per_game

    def rows(self, results: dict) -> list[dict]:
        game_id = results["spec"]["game_id"]
        self.games.append(game_id)
        return [{"game_id": game_id, "version": 1, "advantage": 0.5} for _ in range(self.rows_per_game)]

    def state_dict(self) -> dict:
        return {"games": list(self.games)}

    def load_state_dict(self, state: dict) -> None:
        self.games = list(state["games"])


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def publish(root: Path, version: int) -> None:
    folder = root / "adapters" / f"v{version}"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"version": version}))


def checkpoint(root: Path, version: int) -> None:
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "checkpoints" / f"ckpt-{version:06d}.pt").write_bytes(b"x")


def make(
    tmp_path: Path, monkeypatch, *, rows: int = 2, aws_code: int = 0, learner=True, free_gb=None, credit=None, **over
) -> SimpleNamespace:
    monkeypatch.setenv("VLLM_API_KEY", "k")
    cfg = config(tmp_path / "run", **over)
    fake, play, credit, clock = FakeVllm(), FakePlay(), credit or FakeCredit(rows), Clock()
    syncs: list[list[str]] = []
    powered: list[bool] = []

    async def aws(cmd: list[str]) -> tuple[int, str]:
        syncs.append(cmd)
        return aws_code, "boom" if aws_code else ""

    http = fake.client()
    orch = orchestrate.Orchestrator(
        cfg,
        fleet=Fleet(cfg.servers, "k", http),
        play=play,
        credit=credit,
        syncer=Syncer(cfg.root, cfg.s3_uri, run=aws),
        http=http,
        now=clock,
        power_off=lambda: powered.append(True),
        learner=learner,
        free_gb=free_gb,
    )
    return SimpleNamespace(
        orch=orch, fake=fake, play=play, credit=credit, clock=clock, syncs=syncs, powered=powered, cfg=cfg,
        root=cfg.root, http=http, aws=aws,
    )  # fmt: skip


async def until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.005)


async def settle(orch) -> None:
    await until(lambda: not orch.in_flight)


# ---- versions: pinning, hot-loading and unloading ----------------------------------------------------------------
async def test_a_new_adapter_is_loaded_into_both_instances_and_games_pin_the_version_current_at_their_start(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=2)
    publish(t.root, 1)
    await t.orch.refresh_serving()
    assert t.orch.current == 1 and t.fake.loaded(HOSTS[0]) == t.fake.loaded(HOSTS[1]) == {"local/policy-v1"}
    t.play.hold()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 2)
    first = list(t.play.specs)
    assert {policy_version(s) for s in first} == {1}
    assert Counter(s.seats[0].base_url if s.seats[0].trainable else s.seats[1].base_url for s in first) == {
        HOSTS[0]: 1, HOSTS[1]: 1
    }  # fmt: skip
    assert t.orch.adapters.pins == {1: 2}

    publish(t.root, 2)
    await t.orch.refresh_serving()
    assert t.orch.current == 2 and t.fake.loaded(HOSTS[0]) == {"local/policy-v1", "local/policy-v2"}
    t.play.release()
    await settle(t.orch)
    t.play.hold()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 4)
    assert {policy_version(s) for s in t.play.specs[2:]} == {2} and t.orch.adapters.pins == {2: 2}
    await t.orch.refresh_serving()
    assert t.fake.loaded(HOSTS[0]) == t.fake.loaded(HOSTS[1]) == {"local/policy-v2"}
    t.play.release()
    await settle(t.orch)


async def test_snapshots_stay_loaded_and_the_base_model_is_never_an_adapter(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, servers={"urls": list(HOSTS), "keep_versions": [1, 9]})
    for v in (1, 2, 3):
        publish(t.root, v)
    await t.orch.refresh_serving()
    assert t.orch.current == 3 and t.fake.loaded(HOSTS[0]) == {"local/policy-v1", "local/policy-v3"}
    assert not any(c[2].get("lora_name") == "local/qwen-base" for c in t.fake.calls)


async def test_the_current_version_does_not_move_until_every_instance_holds_it(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    publish(t.root, 1)
    await t.orch.refresh_serving()
    publish(t.root, 2)
    t.fake.hosts[HOSTS[1]]["up"] = False
    await t.orch.refresh_serving()
    assert t.orch.current == 2  # only the instance that is up counts
    t.fake.restart(HOSTS[1])
    await t.orch.refresh_serving()
    assert t.fake.loaded(HOSTS[1]) == {"local/policy-v2"} and t.orch.current == 2


async def test_games_wait_while_no_instance_serves_the_current_version(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    for host in HOSTS:
        t.fake.hosts[host]["up"] = False
    await t.orch.refresh_serving()
    t.orch.fill()
    assert not t.orch.in_flight and not t.play.specs


# ---- the watchdog --------------------------------------------------------------------------------------------------
async def test_a_restarted_vllm_gets_its_adapters_again_and_only_the_game_whose_requests_failed_is_replayed(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=2, mix={"bot": 1.0})
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.play.call_gate, t.play.gate = asyncio.Event(), asyncio.Event()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 2)
    by_host = {next(s.base_url for s in spec.seats if s.trainable): spec for spec in t.play.specs}
    victim, bystander = by_host[HOSTS[0]], by_host[HOSTS[1]]

    t.fake.hosts[HOSTS[0]]["up"] = False
    t.play.call_gate.set()
    await asyncio.sleep(0.05)
    t.play.release()
    await settle(t.orch)

    assert t.credit.games == [bystander.game_id]
    assert [p.name for p in (t.root / "voided").iterdir()] == [victim.game_id]
    assert not (t.root / "games" / victim.game_id).exists()
    assert [(p.base_id, p.attempt) for p in t.orch.requeue] == [(victim.game_id, 1)]
    assert t.orch.counts["voided"] == 1 and t.orch.counts["train_finished"] == 1

    await t.orch.refresh_serving()
    assert not t.orch.fleet.instances[0].up
    t.play.call_gate = t.play.gate = None
    t.orch.fill()
    await until(lambda: any(s.game_id == f"{victim.game_id}-r1" for s in t.play.specs))
    replay = next(s for s in t.play.specs if s.game_id == f"{victim.game_id}-r1")
    assert {s.base_url for s in replay.seats if s.kind == "llm"} == {HOSTS[1]}  # the dead instance gets nothing
    assert (replay.seed, replay.map, replay.seats[0].civ, replay.seats[1].civ) == (
        victim.seed, victim.map, victim.seats[0].civ, victim.seats[1].civ
    )  # fmt: skip

    before = len(t.fake.calls)
    t.fake.restart(HOSTS[0])
    await t.orch.refresh_serving()
    loads = [c for c in t.fake.calls[before:] if c[1] == "/v1/load_lora_adapter" and c[0] == HOSTS[0]]
    assert [c[2]["lora_name"] for c in loads] == ["local/policy-v1"]
    assert loads[0][2]["lora_path"] == "/train/run/adapters/v1" and t.fake.loaded(HOSTS[0]) == {"local/policy-v1"}
    await settle(t.orch)


async def test_a_restart_between_two_polls_is_seen_in_the_served_models_alone(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.fake.restart(HOSTS[0])  # /health never failed, but its adapters are gone
    await t.orch.refresh_serving()
    assert t.fake.loaded(HOSTS[0]) == {"local/policy-v1"}


async def test_a_game_on_an_instance_that_went_down_after_a_failed_request_is_cancelled_at_once(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0}, servers={"urls": [HOSTS[0]]})
    await t.orch.refresh_serving()
    t.play.call_gate, t.play.gate = asyncio.Event(), asyncio.Event()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 1)
    t.fake.hosts[HOSTS[0]]["up"] = False
    t.play.call_gate.set()
    await until(lambda: next(iter(t.orch.in_flight.values())).failed_calls == 1)
    await t.orch.refresh_serving()
    t.orch.cancel_games_on_down_instances()
    await settle(t.orch)
    assert t.orch.counts["voided"] == 1 and not t.credit.games and len(t.orch.requeue) == 1
    assert t.orch.requeue[0].attempt == 1


async def test_a_game_that_is_void_too_many_times_is_dropped(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0}, max_requeues=1, servers={"urls": [HOSTS[0]]})
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.fake.restart(HOSTS[0])  # the adapter is gone but nobody has polled yet: every request is a 404
    t.play.call_gate = asyncio.Event()
    t.play.call_gate.set()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 1)
    await settle(t.orch)
    assert t.orch.counts["voided"] == 1 and len(t.orch.requeue) == 1
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 2)
    await settle(t.orch)
    assert [s.game_id for s in t.play.specs] == ["r1-g000000", "r1-g000000-r1"]
    assert t.orch.counts["voided"] == 2 and t.orch.counts["dropped"] == 1 and not t.orch.requeue


# ---- rows into the inbox -------------------------------------------------------------------------------------------
async def test_a_finished_games_rows_reach_the_inbox_atomically_under_a_unique_name(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, rows=3, concurrency=2)
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: t.orch.counts["train_finished"] == 2)
    files = sorted(p.name for p in (t.root / "inbox").iterdir())
    assert files == ["r1-g000000.jsonl", "r1-g000001.jsonl"]
    rows = [json.loads(line) for line in (t.root / "inbox" / files[0]).read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["game_id"] == "r1-g000000" and t.orch.counts["rows"] == 6
    ledger = [json.loads(line) for line in (t.root / "games" / "games.jsonl").read_text().splitlines()]
    assert {row["game_id"] for row in ledger} >= {"r1-g000000", "r1-g000001"} and ledger[0]["rows"] == 3


def test_the_inbox_write_leaves_no_half_file_and_skips_empty_games(tmp_path):
    assert orchestrate.write_rows(tmp_path / "inbox", "g", []) is None and not (tmp_path / "inbox").exists()
    path = orchestrate.write_rows(tmp_path / "inbox", "g", [{"a": 1}])
    assert path == tmp_path / "inbox" / "g.jsonl" and [p.name for p in path.parent.iterdir()] == ["g.jsonl"]


async def test_a_failed_game_gets_no_credit_and_is_recorded(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, concurrency=1)
    t.play.status = "failed"
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: t.orch.counts["train_failed"] == 1)
    assert not t.credit.games and not t.orch.requeue
    assert json.loads((t.root / "games" / "games.jsonl").read_text().splitlines()[0])["status"] == "failed"


async def test_game_numbering_resumes_after_the_games_already_on_disk(tmp_path, monkeypatch):
    (tmp_path / "run" / "games" / "r1-g000041-r2").mkdir(parents=True)
    t = make(tmp_path, monkeypatch, concurrency=1)
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 1)
    assert t.play.specs[0].game_id == "r1-g000042"


# ---- tripwires -----------------------------------------------------------------------------------------------------
def health(root: Path, *records: dict) -> None:
    with (root / "health.jsonl").open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def update(n: int, *, version: int | None = None, trips=(), time: float = 1_000.0) -> dict:
    tripwires = {"trips": [{"name": t, "message": f"{t} tripped"} for t in trips], "alarms": []}
    return {"update": n, "version": version or n, "time": time, "tripwires": tripwires}


def commands(root: Path) -> list[dict]:
    folder = root / "control"
    names = sorted(p.name for p in folder.glob("*.json")) if folder.exists() else []
    return [json.loads((folder / n).read_text()) for n in names]


def trip_run(tmp_path, monkeypatch, **over):
    t = make(tmp_path, monkeypatch, **over)
    t.root.mkdir(parents=True, exist_ok=True)
    for v in range(1, 6):
        checkpoint(t.root, v)
    health(t.root, *(update(n) for n in range(1, 6)))
    return t


def test_a_trip_rolls_back_to_the_last_green_checkpoint_once_per_episode(tmp_path, monkeypatch):
    t = trip_run(tmp_path, monkeypatch)
    t.orch.poll_health(2_000.0)
    assert commands(t.root) == []
    checkpoint(t.root, 6)
    health(t.root, update(6, trips=["clipped_share"]))
    t.orch.poll_health(2_000.0)
    assert commands(t.root) == [{"op": "rollback", "version": 5}]  # one command: the rollback halves the lr itself
    health(t.root, update(7, trips=["clipped_share"]), update(8, trips=["grad_norm"]))
    t.orch.poll_health(2_000.0)
    assert len(commands(t.root)) == 1 and t.orch.stop_reason is None and t.orch.counts["rollbacks"] == 1


def test_a_green_version_older_than_the_trip_is_skipped_if_its_own_record_tripped(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.root.mkdir(parents=True)
    for v in range(1, 5):
        checkpoint(t.root, v)
    health(t.root, update(1), update(2), update(3, trips=["hold_share"]), update(4, trips=["hold_share"]))
    t.orch.poll_health(2_000.0)
    assert commands(t.root) == [{"op": "rollback", "version": 2}]


def test_a_trip_after_the_rollback_report_is_a_new_episode_and_a_second_within_three_hours_pauses(
    tmp_path, monkeypatch
):
    t = trip_run(tmp_path, monkeypatch)
    health(t.root, update(6, trips=["hold_share"], time=1_000.0))
    t.orch.poll_health(2_000.0)
    health(t.root, {"event": "rollback", "restored": 5, "version": 7, "lr": 5e-6, "time": 1_100.0})
    checkpoint(t.root, 7)
    health(t.root, update(7, version=8, time=1_200.0), update(8, version=9, trips=["invented_tools"], time=5_000.0))
    t.orch.poll_health(6_000.0)
    assert commands(t.root) == [{"op": "rollback", "version": 5}]
    assert t.orch.stop_reason.startswith("tripwire") and t.orch.counts["pauses"] == 1


def test_two_trips_more_than_three_hours_apart_are_two_rollbacks(tmp_path, monkeypatch):
    t = trip_run(tmp_path, monkeypatch)
    health(t.root, update(6, trips=["hold_share"], time=1_000.0))
    t.orch.poll_health(2_000.0)
    health(t.root, {"event": "rollback", "restored": 5, "version": 7, "lr": 5e-6, "time": 1_100.0})
    checkpoint(t.root, 7)
    health(
        t.root,
        update(7, version=8, time=1_200.0),
        update(8, version=9, trips=["hold_share"], time=1_000.0 + 3 * 3600 + 1),
    )
    t.orch.poll_health(20_000.0)
    assert commands(t.root) == [{"op": "rollback", "version": 5}, {"op": "rollback", "version": 7}]
    assert t.orch.stop_reason is None


def test_a_rollback_the_learner_never_reports_does_not_silence_later_trips(tmp_path, monkeypatch):
    t = trip_run(tmp_path, monkeypatch, trip_limit=5)
    health(t.root, update(6, trips=["hold_share"], time=1_000.0), update(7, trips=["hold_share"], time=1_100.0))
    t.orch.poll_health(2_000.0)
    assert len(commands(t.root)) == 1
    health(t.root, update(8, trips=["hold_share"], time=1_000.0 + 901))
    t.orch.poll_health(3_000.0)
    assert len(commands(t.root)) == 2


def test_a_trip_with_no_checkpoint_to_go_back_to_pauses(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.root.mkdir(parents=True)
    health(t.root, update(1, trips=["hold_share"]))
    t.orch.poll_health(2_000.0)
    assert commands(t.root) == [] and t.orch.stop_reason.startswith("tripwire")


def test_a_half_written_health_line_waits_for_its_newline(tmp_path):
    path = tmp_path / "health.jsonl"
    path.write_text(json.dumps(update(1)) + "\n" + json.dumps(update(2))[:20])
    watch = TripWatch(path, tmp_path / "checkpoints")
    watch.poll(0.0)
    assert [v for v, _ in watch.history] == [1]
    path.write_text(path.read_text() + json.dumps(update(2))[20:] + "\n")
    watch.poll(0.0)
    assert [v for v, _ in watch.history] == [1, 2]


def test_control_commands_are_numbered_after_those_already_read_and_never_half_written(tmp_path):
    writer = ControlWriter(tmp_path / "control")
    (tmp_path / "control" / "done").mkdir(parents=True)
    (tmp_path / "control" / "done" / "000007-rollback.json").write_text("{}")
    first = writer.send({"op": "rollback", "version": 3})
    second = writer.send({"op": "stop"})
    assert (first.name, second.name) == ("000008-rollback.json", "000009-stop.json")
    assert sorted(p.name for p in (tmp_path / "control").iterdir() if p.is_file()) == [first.name, second.name]


# ---- budget and stopping ---------------------------------------------------------------------------------------
def test_spend_is_box_hours_at_the_configured_rate_plus_the_ledger_across_restarts(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    spend.append_ledger(ledger, "sonnet", 6.5)
    first = budget.RunClock(tmp_path / "o" / "clock.json", now=1000.0)
    assert budget.spend(first, 1000.0 + 1800, 10.0, ledger) == budget.Spend(0.5, 5.0, 6.5)
    first.save(1000.0 + 1800)
    second = budget.RunClock(tmp_path / "o" / "clock.json", now=9_999_999.0)
    assert budget.spend(second, 9_999_999.0 + 3600, 10.0, ledger).box_hours == pytest.approx(1.5)


async def test_the_budget_check_stops_the_run_when_box_and_frontier_spend_reach_it(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, budget_usd=25)
    t.root.mkdir(parents=True)
    spend.append_ledger(t.cfg.ledger_path, "frontier", 8.0)
    t.clock.t += 3600
    spent = t.orch.check_budget(t.clock())
    assert (spent.box_usd, spent.model_usd, t.orch.stop_reason is None) == (10.0, 8.0, True)
    t.clock.t += 1800
    t.orch.check_budget(t.clock())
    assert t.orch.stop_reason is None  # $15 box + $8 model = $23 of $25
    spend.append_ledger(t.cfg.ledger_path, "frontier", 3.0)
    t.orch.check_budget(t.clock())
    assert t.orch.stop_reason.startswith("budget: box $15.00 (1.50 h) + model $11.00 = $26.00 of $25")
    status = json.loads((t.root / "orchestrator" / "status.json").read_text())
    assert status["spend"] == {"box": 15.0, "model": 11.0} and status["stop_reason"] is None


async def test_status_carries_the_orchestrators_cpu_and_loop_lag_every_minute(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.root.mkdir(parents=True)
    t.orch.monitor.start()
    t.orch.check_budget(t.clock())  # status.json exists, with no loop numbers yet
    status = json.loads((t.root / "orchestrator" / "status.json").read_text())
    assert status["loop"] == {}
    await asyncio.sleep(0.2)
    t.clock.t += 61
    await t.orch.tick()  # the loop step is due a minute after the budget check, and rewrites status.json
    status = json.loads((t.root / "orchestrator" / "status.json").read_text())
    loop = status["loop"]
    assert loop["cpu_s"] >= 0 and loop["cpu_cores"] is not None and loop["lag_samples"] > 0
    assert set(loop["lag_ms"]) == {"p50", "p95", "max"} and status["spend"]["box"] >= 0
    before = status["time"]
    t.clock.t += 5
    await t.orch.tick()  # not due again
    assert json.loads((t.root / "orchestrator" / "status.json").read_text())["time"] == before
    await t.orch.monitor.stop()


async def test_the_run_stops_after_its_duration(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, duration_hours=2)
    t.clock.t += 2 * 3600
    t.orch.check_budget(t.clock())
    assert t.orch.stop_reason == "duration"


async def test_a_budget_stop_cancels_the_games_tells_the_learner_uploads_and_powers_off_only_if_configured(
    tmp_path, monkeypatch
):
    for shutdown in (False, True):
        t = make(tmp_path / str(shutdown), monkeypatch, budget_usd=5, shutdown=shutdown, concurrency=2)
        publish(t.root, 1)
        checkpoint(t.root, 1)
        t.play.hold()
        t.clock.t += 3600
        reason = await asyncio.wait_for(t.orch.run(), 5)
        assert reason.startswith("budget")
        assert commands(t.root) == [{"op": "stop"}]
        assert len(t.syncs) == 1 and "--include" in t.syncs[-1] and t.syncs[-1][3] == str(t.root)  # the final one only
        assert "adapters/v1/*" in t.syncs[-1] and "checkpoints/ckpt-000001.pt" not in t.syncs[-1]
        assert json.loads((t.root / "orchestrator" / "final.json").read_text())["reason"] == reason
        assert t.powered == ([True] if shutdown else []) and not t.orch.in_flight


async def test_a_completed_run_lets_its_games_finish_before_the_final_upload(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, duration_hours=1, concurrency=2)
    t.play.hold()
    run = asyncio.create_task(t.orch.run())
    await until(lambda: len(t.play.specs) == 2)
    t.clock.t += 3600
    await asyncio.sleep(0.05)
    assert not run.done() and t.orch.stop_reason == "duration"
    t.play.release()
    assert await asyncio.wait_for(run, 5) == "duration"
    assert t.orch.counts["train_finished"] == 2 and len(t.credit.games) == 2 and len(t.syncs) >= 1


async def test_a_failed_final_upload_is_retried_and_leaves_the_box_up_instead_of_powering_it_off(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, budget_usd=1, aws_code=1, shutdown=True)
    t.clock.t += 3600
    await asyncio.wait_for(t.orch.run(), 5)
    assert len(t.syncs) == orchestrate.SYNC_ATTEMPTS  # the final upload's attempts; none before the first game
    assert t.orch.syncer.failures == len(t.syncs) and t.powered == []
    assert "final_upload" in t.orch.alarms
    assert json.loads((t.root / "orchestrator" / "final.json").read_text())["upload_failed"] is True


async def test_a_box_with_no_s3_prefix_still_powers_off_when_asked(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, budget_usd=1, shutdown=True, s3={})
    t.clock.t += 3600
    await asyncio.wait_for(t.orch.run(), 5)
    assert t.powered == [True]


# ---- S3 sync -------------------------------------------------------------------------------------------------------
def test_the_sync_command_names_the_prefix_and_lets_through_only_what_a_run_keeps(tmp_path):
    cmd = sync_command(tmp_path / "run", "s3://b/p/run1/", adapters=[12, 3, 12], checkpoint=14)
    assert cmd[:6] == ["aws", "s3", "sync", str(tmp_path / "run"), "s3://b/p/run1", "--only-show-errors"]
    assert cmd[6:8] == ["--exclude", "*"] and "--delete" not in cmd
    includes = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--include"]
    assert {"games/*", "inbox/*", "health.jsonl", "panels/*", "spend.jsonl", "adapters/*/manifest.json"} <= set(
        includes
    )
    assert "report/*" not in includes and "champion.json" not in includes  # nothing writes them
    assert includes[-3:] == ["adapters/v3/*", "adapters/v12/*", "checkpoints/ckpt-000014.pt"]
    assert "adapters/*" not in includes and "checkpoints/*" not in includes


def test_without_a_checkpoint_or_adapters_the_sync_names_neither(tmp_path):
    includes = [a for a in sync_command(tmp_path, "s3://b/p") if a.startswith(("adapters", "checkpoints"))]
    assert includes == ["adapters/*/manifest.json"]  # every manifest, no weights


async def test_syncs_run_in_the_background_every_interval_with_hourly_and_champion_adapters_and_the_healthy_ckpt(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=1)
    for v in (1, 2, 3):
        publish(t.root, v)
    for v in (1, 2, 3):
        checkpoint(t.root, v)
    health(t.root, update(1), update(2), update(3, trips=["hold_share"]))
    (t.root / "champion.json").write_text(json.dumps({"version": 1}))
    await t.orch.tick()
    assert t.syncs == []  # nothing is uploaded before the first game has run an interval
    t.clock.t += 3700
    await t.orch.tick()
    await t.orch.join_sync()
    assert t.orch.hourly == {1: 3} and len(t.syncs) == 1
    t.clock.t += 800
    await t.orch.tick()
    await t.orch.join_sync()
    assert len(t.syncs) == 1
    t.clock.t += 200
    await t.orch.tick()
    await t.orch.join_sync()
    last = t.syncs[-1]
    includes = [last[i + 1] for i, a in enumerate(last) if a == "--include"]
    assert len(t.syncs) == 2 and "adapters/v1/*" in includes and "adapters/v3/*" in includes
    assert "checkpoints/ckpt-000002.pt" in includes  # version 3's own record tripped


async def test_a_slow_sync_does_not_hold_up_the_tick(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, concurrency=1)
    release = asyncio.Event()

    async def slow(cmd):
        await release.wait()
        return 0, ""

    t.orch.syncer._run = slow
    await t.orch.tick()
    t.clock.t += 1000
    await asyncio.wait_for(t.orch.tick(), 1)  # returns with the upload still running
    assert t.orch._sync_task is not None and not t.orch._sync_task.done()
    release.set()
    await t.orch.join_sync()
    assert t.orch.syncer.last_ok is not None


async def test_an_unconfigured_prefix_uploads_nothing(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, s3={})
    assert await t.orch.sync_now() is False and t.syncs == []


# ---- panels --------------------------------------------------------------------------------------------------------
def write_panel(path: Path, n: int = 3) -> None:
    seats = [
        {
            "kind": "llm", "civ": "rome", "model": "local/policy-v0", "base_url": "http://x",
            "api_key_env": "VLLM_API_KEY", "trainable": True,
        },
        {"kind": "petra", "civ": "pers", "difficulty": 3},
    ]  # fmt: skip
    path.write_text(
        "\n".join(json.dumps({"game_id": f"p{i}", "seats": seats, "map": "tundra", "seed": 900 + i}) for i in range(n))
    )


async def drive(orch, predicate, timeout: float = 3.0) -> None:
    """tick() until `predicate`, the way run() would."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await orch.tick()
        await asyncio.sleep(0.005)


async def test_a_panel_plays_its_preregistered_games_on_a_pinned_version_apart_from_training(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel)
    panels = [{"name": "mid", "at_hours": 1, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=2, panels=panels)
    for v in (1, 2):
        publish(t.root, v)
    await t.orch.refresh_serving()
    assert t.orch.current == 2
    t.orch.start_due_panels()
    assert t.orch.panels == {}  # not due yet

    t.play.hold()
    t.clock.t += 3600
    await t.orch.tick()
    assert t.orch.adapters.pins == {1: 1 + 2}  # the panel's own pin and one per panel game in flight
    await until(lambda: len(t.play.specs) == 2)
    assert [s.game_id for s in t.play.specs] == ["mid-v1-p0", "mid-v1-p1"]  # panel games go first
    assert t.fake.loaded(HOSTS[0]) == {"local/policy-v1", "local/policy-v2"}
    assert all(policy_version(s) == 1 and s.map == "tundra" for s in t.play.specs)

    t.play.release()
    await drive(t.orch, lambda: t.orch.counts["panel_finished"] == 3)
    assert sorted(p.name for p in (t.root / "panels" / "mid").iterdir() if p.is_dir()) == [
        "mid-v1-p0", "mid-v1-p1", "mid-v1-p2"
    ]  # fmt: skip
    summary = json.loads((t.root / "panels" / "mid" / "summary.json").read_text())
    assert summary == {"name": "mid", "version": 1, "games": 3, "finished": 3, "voided": 0}
    assert not any(g.startswith("mid-") for g in t.credit.games)
    assert not any(p.name.startswith("mid-") for p in (t.root / "games").iterdir())
    assert 1 not in t.orch.adapters.pins  # the panel's pin is gone; training games since were pinned to v2
    await t.orch.refresh_serving()
    assert t.fake.loaded(HOSTS[0]) == {"local/policy-v2"}


async def test_a_panel_keeps_its_version_loaded_while_it_runs_and_a_missing_version_is_skipped(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 1)
    panels = [
        {"name": "old", "at_hours": 0, "specs": str(panel), "version": 1},
        {"name": "ghost", "at_hours": 0, "specs": str(panel), "version": 9},
    ]
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels)
    publish(t.root, 1)
    t.orch.start_due_panels()
    assert set(t.orch.panels) == {"old", "ghost"} and t.orch.panels["ghost"].remaining == 0
    assert [p.base_id for p in t.orch.panel_queue] == ["old-v1-p0"]
    publish(t.root, 2)
    await t.orch.refresh_serving()
    assert t.fake.loaded(HOSTS[0]) == {"local/policy-v1", "local/policy-v2"}


def test_a_panel_without_a_version_pins_whichever_is_current_when_it_starts(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 2)
    t = make(tmp_path, monkeypatch, panels=[{"name": "now", "at_hours": 0, "specs": str(panel)}])
    t.orch.current = 0
    t.orch.start_due_panels()
    assert t.orch.panels["now"].version == 0 and [p.base_id for p in t.orch.panel_queue] == ["now-v0-p0", "now-v0-p1"]


# ---- adapters on disk ------------------------------------------------------------------------------------------
def test_only_complete_version_directories_count(tmp_path):
    publish(tmp_path, 2)
    (tmp_path / "adapters" / ".v3.abc123").mkdir()
    (tmp_path / "adapters" / "v10").mkdir()
    (tmp_path / "adapters" / "notes").mkdir()
    adapters = Adapters(tmp_path, keep=[2])
    assert adapters.on_disk() == [2, 10]
    adapters.pin(5)
    adapters.pin(5)
    adapters.release(5)
    assert adapters.wanted(10, 0) == {2, 5, 10}
    adapters.release(5)
    assert adapters.wanted(10) == {2, 10}


def test_the_tools_command_prints_the_order_tools_the_policy_is_offered_on_v2_and_no_other_env_tool(capsys):
    assert orchestrate.main(["tools"]) == 0
    names = capsys.readouterr().out.strip().split(",")
    assert re.fullmatch(r"[a-z_,]+", ",".join(names))
    assert sorted(names) == sorted(ORDER_TOOLS_V2) and "hold" in names and "get_brief" not in names


# ---- a restarted orchestrator ----------------------------------------------------------------------------------
def reborn(t, credit=None, **kw) -> orchestrate.Orchestrator:
    """A new process of the same run: same config, same folder, nothing in memory."""
    return orchestrate.Orchestrator(
        t.cfg,
        fleet=Fleet(t.cfg.servers, "k", t.http),
        play=t.play,
        credit=credit or FakeCredit(),
        syncer=Syncer(t.cfg.root, t.cfg.s3_uri, run=t.aws),
        http=t.http,
        now=t.clock,
        power_off=lambda: t.powered.append(True),
        **kw,
    )


def test_a_fresh_tripwatch_does_not_replay_a_rollback_the_learner_already_made(tmp_path):
    health_path = tmp_path / "health.jsonl"
    ring = tmp_path / "checkpoints"
    ring.mkdir()
    for v in (1, 5, 10, 181):
        (ring / f"ckpt-{v:06d}.pt").write_bytes(b"x")
    records = [update(n, time=100.0 + n) for n in range(1, 11)]
    records += [update(11, trips=["hold_share"], time=200.0)]
    records += [{"event": "rollback", "restored": 10, "version": 12, "lr": 5e-6, "time": 210.0}]
    records += [update(n, version=n + 1, time=300.0 + n) for n in range(12, 200)]
    health_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert [a.kind for a in TripWatch(health_path, ring).poll(5_000.0)] == [
        "rollback"
    ]  # offset 0 and no state: replayed
    restarted = TripWatch(health_path, ring, state_path=tmp_path / "orchestrator" / "tripwatch.json")
    assert restarted.poll(5_000.0) == []  # no state file: the log is replayed silently
    health(tmp_path, update(200, version=201, trips=["grad_norm"], time=20_000.0))
    assert [a.kind for a in restarted.poll(20_000.0)] == ["rollback"]  # a new trip still acts
    again = TripWatch(health_path, ring, state_path=tmp_path / "orchestrator" / "tripwatch.json")
    assert again.poll(5_100.0) == [] and again.offset == restarted.offset  # and a second restart resumes, not replays


async def test_two_old_trips_do_not_pause_a_restarted_run_but_a_third_within_the_window_does(tmp_path, monkeypatch):
    t = trip_run(tmp_path, monkeypatch, trip_limit=3)
    health(t.root, update(6, trips=["hold_share"], time=1_000.0))
    t.orch.poll_health(2_000.0)
    health(t.root, {"event": "rollback", "restored": 5, "version": 7, "lr": 5e-6, "time": 1_100.0})
    health(t.root, update(7, version=8, trips=["hold_share"], time=1_200.0))
    t.orch.poll_health(2_000.0)
    assert t.orch.stop_reason is None and len(commands(t.root)) == 2
    for state in (t.root / "orchestrator" / "tripwatch.json", None):  # with its state, and with the state lost
        if state is not None:
            state.unlink()
        second = reborn(t)
        second.poll_health(3_000.0)
        assert second.stop_reason is None and second.counts["rollbacks"] == 0 and len(commands(t.root)) == 2
    health(t.root, {"event": "rollback", "restored": 5, "version": 9, "lr": 2e-6, "time": 1_300.0})
    health(t.root, update(9, version=10, trips=["hold_share"], time=1_400.0))
    second.poll_health(3_000.0)
    assert second.stop_reason.startswith("tripwire")  # the old trips still count towards the window


async def test_a_restart_keeps_credit_the_game_stream_the_panels_and_the_hourly_adapters(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "mid", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels)
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.orch.start_due_panels()
    await drive(t.orch, lambda: t.orch.counts["panel_finished"] == 1 and t.orch.counts["train_finished"] == 0)
    t.clock.t += 3700
    t.orch.check_budget(t.clock())
    await drive(t.orch, lambda: t.orch.counts["panel_finished"] == 3 and t.orch.counts["train_finished"] >= 2)
    t.orch.request_stop("test")  # no new games; let the one in flight end
    await settle(t.orch)
    first_specs = [s for s in t.play.specs if not s.game_id.startswith("mid")]
    assert t.credit.games == [s.game_id for s in first_specs] and t.orch.hourly == {1: 1}
    expected = GameFactory(t.cfg)
    for _ in first_specs:
        expected.next(1, HOSTS[0])
    next_spec = expected.next(1, HOSTS[0])

    second = reborn(t, FakeCredit())
    assert second.credit.games == t.credit.games and second.hourly == {1: 1}
    assert second.counts["restarts"] == 1 and set(second.panels) == {"mid"} and second.panels["mid"].remaining == 0
    assert not second.panel_queue
    await second.refresh_serving()
    second.fill()
    await until(lambda: len(t.play.specs) == len(first_specs) + 3 + 1)
    started = t.play.specs[-1]
    assert started.game_id == next_spec.game_id
    assert (started.seed, started.ai_seed, started.map) == (next_spec.seed, next_spec.ai_seed, next_spec.map)
    assert started.seats[0].civ == next_spec.seats[0].civ and started.game_id != first_specs[0].game_id
    await settle(second)


async def test_a_restart_resumes_a_panel_midway_and_replays_the_games_the_old_process_was_playing(
    tmp_path, monkeypatch
):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "mid", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=2, panels=panels, mix={"bot": 1.0})
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.play.hold()
    t.orch.start_due_panels()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 2)
    half = t.root / "panels" / "mid" / "mid-v1-p0"
    half.mkdir(parents=True)  # a folder the engine was still writing
    (half / "partial.txt").write_text("x")
    second = reborn(t)
    assert [p.game_id for p in second.panel_queue] == ["mid-v1-p0-r1", "mid-v1-p1-r1", "mid-v1-p2"]
    assert second.panels["mid"].remaining == 3 and second.adapters.pins == {1: 1}
    assert not half.exists() and (t.root / "voided" / "mid-v1-p0").is_dir()
    t.play.release()


async def test_a_restart_replays_a_train_game_that_was_in_flight_and_keeps_the_voided_replays_queued(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0})
    await t.orch.refresh_serving()
    t.play.hold()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 1)
    lost = t.play.specs[0]
    second = reborn(t)
    assert [p.game_id for p in second.requeue] == [f"{lost.game_id}-r1"]
    await second.refresh_serving()
    second.fill()
    await until(lambda: len(t.play.specs) == 2)
    replay = t.play.specs[1]
    assert (replay.seed, replay.map, replay.game_id) == (lost.seed, lost.map, f"{lost.game_id}-r1")
    t.play.release()
    await settle(second)


# ---- stopping, crashing, guarding ------------------------------------------------------------------------------
async def test_an_error_in_the_loop_still_stops_the_learner_uploads_and_powers_off(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, shutdown=True)

    async def broken() -> None:
        raise RuntimeError("tick blew up")

    t.orch.tick = broken
    reason = await asyncio.wait_for(t.orch.run(), 5)
    assert reason.startswith("crash: RuntimeError: tick blew up")
    assert commands(t.root) == [{"op": "stop"}] and len(t.syncs) == 1 and t.powered == [True]
    assert json.loads((t.root / "orchestrator" / "final.json").read_text())["reason"] == reason


async def test_a_step_that_raises_is_counted_and_the_other_steps_still_run(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, concurrency=1)

    def broken(now):
        raise RuntimeError("health blew up")

    t.orch.poll_health = broken
    await t.orch.tick()
    await until(lambda: len(t.play.specs) == 1)  # fill ran after the broken health step
    assert t.orch.counts["step_error_health"] == 1 and t.orch.stop_reason is None
    await settle(t.orch)


async def test_a_budget_check_that_keeps_failing_stops_the_run_rather_than_leave_it_unwatched(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, budget_interval_s=1)

    def broken(now):
        raise RuntimeError("no ledger")

    t.orch.check_budget = broken
    for _ in range(orchestrate.STEP_FAILURES_TO_STOP):
        t.clock.t += 2
        await t.orch.tick()
    assert t.orch.stop_reason.startswith("the budget check keeps failing")


def test_a_config_that_can_not_make_a_game_or_a_panel_is_refused_when_it_is_checked(tmp_path):
    from agentenv_zeroad.train.config import check_run

    with pytest.raises(ValueError, match="seed_low"):
        config(tmp_path, games={"seed_low": 5, "seed_high": 5})
    with pytest.raises(ValueError, match="every seed"):
        config(tmp_path, games={"seed_low": 1, "seed_high": 4}, held_out={"seeds": [1, 2, 3]})
    with pytest.raises(ValueError, match="unknown games.civs"):
        config(tmp_path, games={"civs": ["rome", "narnia"]})
    with pytest.raises(ValueError, match="interface"):
        check_run(config(tmp_path, games={"interface": "v9"}))
    with pytest.raises(ValueError, match="no spec file"):
        check_run(config(tmp_path, panels=[{"name": "x", "at_hours": 1, "specs": str(tmp_path / "gone.jsonl")}]))
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"game_id": "p", "seats": [{"kind": "bot", "civ": "narnia"}, {"kind": "bot", "civ": "rome"}]})
    )
    with pytest.raises(ValueError):
        check_run(config(tmp_path, panels=[{"name": "x", "at_hours": 1, "specs": str(bad)}]))
    assert check_run(config(tmp_path)) == []


def test_the_frontier_keys_a_panel_needs_are_reported(tmp_path):
    from agentenv_zeroad.train.config import check_run

    panel = tmp_path / "panel.jsonl"
    seats = [
        {
            "kind": "llm", "civ": "rome", "model": "local/policy-v0", "base_url": "http://x",
            "api_key_env": "VLLM_API_KEY",
        },
        {
            "kind": "llm", "civ": "pers", "model": "anthropic/claude-opus-5-5", "base_url": "http://l",
            "api_key_env": "LITELLM_API_KEY",
        },
    ]  # fmt: skip
    panel.write_text(json.dumps({"game_id": "p0", "seats": seats}))
    cfg = config(tmp_path, panels=[{"name": "frontier", "at_hours": 1, "specs": str(panel)}])
    assert check_run(cfg) == ["LITELLM_API_KEY"]


# ---- the vLLM watchdog, void or kept ---------------------------------------------------------------------------
async def test_a_request_that_timed_out_on_a_healthy_server_leaves_the_game_standing_and_credited(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0}, servers={"urls": [HOSTS[0]]})
    await t.orch.refresh_serving()
    t.fake.hosts[HOSTS[0]]["slow"] = True
    t.play.call_gate = asyncio.Event()
    t.play.call_gate.set()
    t.orch.fill()
    await until(lambda: t.orch.counts["train_finished"] == 1)
    assert t.credit.games == ["r1-g000000"] and t.orch.counts["voided"] == 0 and t.orch.counts["missed_calls"] == 1
    line = json.loads((t.root / "games" / "games.jsonl").read_text().splitlines()[0])
    assert line["failed_calls"] == 1 and "void" not in line


async def test_a_request_that_failed_during_an_observed_outage_voids_the_game_even_if_the_error_alone_does_not(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0}, servers={"urls": [HOSTS[0]]})
    await t.orch.refresh_serving()
    inst = t.orch.fleet.instances[0]
    t.fake.hosts[HOSTS[0]]["slow"] = True
    t.play.call_gate = asyncio.Event()
    t.play.call_gate.set()
    inst.outages.append([t.clock() - 5, None])  # the watchdog had seen it down
    t.orch.fill()
    await until(lambda: t.orch.counts["voided"] == 1)
    assert not t.credit.games and len(t.orch.requeue) == 1


async def test_a_5xx_or_a_connect_timeout_blames_the_server_and_a_read_timeout_does_not():
    request = httpx.Request("POST", "http://x")
    five = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(503, request=request))
    four = httpx.HTTPStatusError("bad", request=request, response=httpx.Response(400, request=request))
    assert orchestrate.is_server_error(five) and orchestrate.is_server_error(httpx.ConnectTimeout("t"))
    assert not orchestrate.is_server_error(four) and not orchestrate.is_server_error(httpx.ReadTimeout("t"))


async def test_a_game_on_an_instance_that_has_been_down_for_two_checks_is_cancelled_without_a_failed_request(
    tmp_path, monkeypatch
):
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0}, servers={"urls": [HOSTS[0]]})
    await t.orch.refresh_serving()
    t.play.hold()
    t.orch.fill()
    await until(lambda: len(t.play.specs) == 1)
    t.fake.hosts[HOSTS[0]]["up"] = False
    await t.orch.refresh_serving()
    t.orch.cancel_games_on_down_instances()
    assert len(t.orch.in_flight) == 1 and not next(iter(t.orch.in_flight.values())).cancelled  # one slow check
    await t.orch.refresh_serving()
    t.orch.cancel_games_on_down_instances()
    await settle(t.orch)
    assert t.orch.counts["voided"] == 1 and len(t.orch.requeue) == 1 and not t.credit.games


async def test_a_game_with_no_results_file_or_a_credit_that_raises_is_counted_and_recorded_not_fatal(
    tmp_path, monkeypatch
):
    class Raising(FakeCredit):
        def rows(self, results):
            raise KeyError("unexpected shape")

    t = make(tmp_path, monkeypatch, concurrency=1, credit=Raising())
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: t.orch.counts["credit_failed"] >= 1)
    assert t.orch.counts["train_finished"] >= 1 and t.orch.stop_reason is None
    line = json.loads((t.root / "games" / "games.jsonl").read_text().splitlines()[0])
    assert line["status"] == "finished" and line["error"].startswith("credit: KeyError") and line["rows"] == 0
    await settle(t.orch)

    async def no_results(spec, out_dir, model_for):
        return GameOutcome(spec.game_id, "finished", "conquest")

    t2 = make(tmp_path / "two", monkeypatch, concurrency=1)
    t2.orch.play = no_results
    await t2.orch.refresh_serving()
    t2.orch.fill()
    await until(lambda: t2.orch.counts["results_missing"] >= 1)
    first = json.loads((t2.root / "games" / "games.jsonl").read_text().splitlines()[0])
    assert first["error"].startswith("results.json:") and t2.orch.counts["train_failed"] >= 1
    await settle(t2.orch)


async def test_a_stop_records_the_games_in_flight_as_stopped_and_the_panels_and_queue_it_left(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "mid", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels, budget_usd=5)
    publish(t.root, 1)
    t.play.hold()
    run = asyncio.create_task(t.orch.run())
    await until(lambda: len(t.play.specs) == 1)
    t.clock.t += 3600
    reason = await asyncio.wait_for(run, 5)
    assert reason.startswith("budget")
    assert t.orch.counts["voided"] == 0 and t.orch.counts["requeued"] == 0 and t.orch.counts["panel_stopped"] == 1
    summary = json.loads((t.root / "panels" / "mid" / "summary.json").read_text())
    assert summary["stopped"] is True and summary["unfinished"] == 3 and summary["finished"] == 0
    lines = [json.loads(x) for x in (t.root / "panels" / "mid" / "games.jsonl").read_text().splitlines()]
    assert [(x["game_id"], x["status"]) for x in lines] == [
        ("mid-v1-p0", "stopped"), ("mid-v1-p1", "dropped"), ("mid-v1-p2", "dropped")
    ]  # fmt: skip
    assert not (t.root / "voided").exists()


async def test_a_restart_after_a_stop_replays_the_panel_games_the_stop_cancelled_once(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "mid", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels, budget_usd=5)
    publish(t.root, 1)
    t.play.hold()
    run = asyncio.create_task(t.orch.run())
    await until(lambda: len(t.play.specs) == 1)
    t.clock.t += 3600
    assert (await asyncio.wait_for(run, 5)).startswith("budget")
    second = reborn(t)
    assert [p.game_id for p in second.panel_queue] == ["mid-v1-p0-r1", "mid-v1-p1", "mid-v1-p2"]
    assert second.counts["panel_replayed_after_stop"] == 1 and second.panels["mid"].remaining == 3
    second.save_state()
    third = reborn(t)
    assert [p.game_id for p in third.panel_queue] == ["mid-v1-p0-r1", "mid-v1-p1", "mid-v1-p2"]
    assert third.counts["panel_replayed_after_stop"] == 1
    t.play.release()


def ledger(root: Path, panel: str, *lines: tuple[str, str]) -> None:
    folder = root / "panels" / panel
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "games.jsonl").open("a") as f:
        f.writelines(json.dumps({"game_id": g, "status": status, "kind": "panel"}) + "\n" for g, status in lines)


async def test_a_restart_plays_no_panel_game_again_that_its_record_finished_nor_a_panel_it_does_not_have(
    tmp_path, monkeypatch
):
    """10 Nov: the SFT epoch panels and then the rest share one root. Restarted for its epoch panels, the run took the
    rest's queue and a state that had lost the epoch panel it had finished, and dropped all of them at once."""
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "a", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels)
    publish(t.root, 1)
    t.orch.start_due_panels()  # queued, as the stale state had it
    spec = t.orch.panel_queue[0].spec
    t.orch.panel_queue.append(orchestrate.Pending(spec, "panel", "rest-v1-p9", "rest", 1))
    t.orch.save_state()
    ledger(t.root, "a", *[(f"a-v1-p{i}", "finished") for i in range(3)], *[(f"a-v1-p{i}", "dropped") for i in range(3)])
    second = reborn(t)
    assert list(second.panel_queue) == [] and second.counts["panel_already_finished"] == 3
    assert (second.panels["a"].remaining, second.panels["a"].finished, second.panels["a"].games) == (0, 3, 3)
    summary = json.loads((t.root / "panels" / "a" / "summary.json").read_text())
    assert (summary["finished"], summary["games"]) == (3, 3) and second.panels_finished()


async def test_a_panel_started_afresh_where_its_record_has_some_games_plays_only_the_ones_not_finished(
    tmp_path, monkeypatch
):
    """The SFT rest panel, run again after a stop: its record has 1 game finished, 1 a stop cut off, 1 never begun."""
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 3)
    panels = [{"name": "a", "at_hours": 0, "specs": str(panel), "version": 1}]
    t = make(tmp_path, monkeypatch, concurrency=2, panels=panels)
    publish(t.root, 1)
    ledger(t.root, "a", ("a-v1-p0", "finished"), ("a-v1-p1", "stopped"), ("a-v1-p2", "dropped"))
    t.orch.start_due_panels()
    assert [p.game_id for p in t.orch.panel_queue] == ["a-v1-p1-r1", "a-v1-p2-r1"]
    assert (t.orch.panels["a"].remaining, t.orch.panels["a"].finished, t.orch.panels["a"].games) == (2, 1, 3)
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: t.orch.panels["a"].remaining == 0)
    assert [s.game_id for s in t.play.specs] == ["a-v1-p1-r1", "a-v1-p2-r1"]
    summary = json.loads((t.root / "panels" / "a" / "summary.json").read_text())
    assert (summary["finished"], summary["games"]) == (3, 3)
    await settle(t.orch)


async def test_a_panel_whose_record_has_every_game_finished_is_done_without_its_adapter(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    write_panel(panel, 2)
    panels = [{"name": "a", "at_hours": 0, "specs": str(panel), "version": 7}]  # v7 pruned since
    t = make(tmp_path, monkeypatch, concurrency=1, panels=panels)
    ledger(t.root, "a", ("a-v7-p0", "finished"), ("a-v7-p1-r1", "finished"))
    t.orch.start_due_panels()
    assert not t.orch.panel_queue and t.orch.panels_finished() and 7 not in t.orch.adapters.pins
    assert json.loads((t.root / "panels" / "a" / "summary.json").read_text())["finished"] == 2


async def test_a_play_only_run_never_sends_stop_or_waits_for_a_learner(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, budget_usd=1, learner=False)
    t.clock.t += 3600
    await asyncio.wait_for(t.orch.run(), 5)
    assert commands(t.root) == [] and len(t.syncs) == 1


async def test_the_spend_of_a_void_or_failed_game_goes_in_the_box_ledger_and_seats_get_the_per_game_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZEROAD_MATCH_CAP_USD", "9")
    t = make(tmp_path, monkeypatch, concurrency=1, mix={"bot": 1.0})
    caps = []

    async def play(spec, out_dir, model_for):
        seat = next(s for s in spec.seats if s.kind == "llm")
        model = model_for(seat, "1")
        caps.append(model.guard.cap_usd)
        model.guard.spent_usd = 0.75  # what its calls cost before the game failed
        return GameOutcome(spec.game_id, "failed", detail="engine died")

    t.orch.play = play
    await t.orch.refresh_serving()
    t.orch.fill()
    await until(lambda: t.orch.counts["train_failed"] == 1)
    assert caps == [9.0] and spend.ledger_total(t.cfg.ledger_path) == pytest.approx(0.75)


# ---- adapters: where vLLM sees them, loads that fail ----------------------------------------------------------
def test_the_adapter_dir_is_derived_from_the_root_the_vllm_containers_mount(tmp_path):
    raw = raw_config(Path("/var/lib/zeroad-live/train/run7"))
    del raw["servers"]["adapter_dir"]
    cfg = RunConfig.from_dict(raw)
    assert cfg.servers.adapter_dir == "/train/run7/adapters"
    assert cfg.ledger_path == Path("/var/lib/zeroad-live/spend.jsonl")  # the box's one ledger
    assert cfg.boxtime_path == Path("/var/lib/zeroad-live/boxtime.json")
    raw["servers"]["adapter_dir"] = "/adapters"  # the old default, which the --train containers do not have
    with pytest.raises(ValueError, match="not where vLLM sees"):
        RunConfig.from_dict(raw)
    elsewhere = raw_config(tmp_path / "run")
    del elsewhere["servers"]["adapter_dir"]
    with pytest.raises(ValueError, match="adapter_dir is required"):
        RunConfig.from_dict(elsewhere)
    elsewhere["servers"]["adapter_dir"] = "/mnt/other/adapters"
    with pytest.raises(ValueError, match="not where vLLM sees"):
        RunConfig.from_dict(elsewhere)


async def test_an_adapter_served_from_another_path_is_unloaded_and_loaded_again_from_the_runs(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, servers={"urls": [HOSTS[0]]})
    publish(t.root, 1)
    await t.orch.refresh_serving()
    t.fake.hosts[HOSTS[0]]["root"] = "/adapters/other-run"  # as if loaded from the wrong mount
    before = len(t.fake.calls)
    await t.orch.refresh_serving()
    paths = [c[1] for c in t.fake.calls[before:] if c[1].startswith("/v1/") and "lora" in c[1]]
    assert paths == ["/v1/unload_lora_adapter", "/v1/load_lora_adapter"]
    assert t.fake.hosts[HOSTS[0]]["root"] is None and t.orch.fleet.instances[0].loaded == {1}


async def test_adapter_loads_that_keep_failing_alarm_and_stop_the_run(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, max_load_failures=3)
    publish(t.root, 1)
    t.fake.hosts[HOSTS[0]]["refuse_loads"] = True
    for _ in range(2):
        await t.orch.refresh_serving()
    assert t.orch.stop_reason is None
    await t.orch.refresh_serving()
    assert t.orch.stop_reason.startswith("adapter loads keep failing") and "adapter_loads" in t.orch.alarms
    t.fake.hosts[HOSTS[0]]["refuse_loads"] = False
    t2 = make(tmp_path / "ok", monkeypatch, max_load_failures=3)
    publish(t2.root, 1)
    t2.fake.hosts[HOSTS[0]]["refuse_loads"] = True
    await t2.orch.refresh_serving()
    t2.fake.hosts[HOSTS[0]]["refuse_loads"] = False
    await t2.orch.refresh_serving()
    assert t2.orch.fleet.worst_load_failures() == 0  # a load that succeeds clears the count


# ---- budget on a box ------------------------------------------------------------------------------------------
def test_box_hours_count_every_boot_the_way_budget_sh_does_and_never_less_than_the_run_clock(tmp_path):
    boxtime = tmp_path / "boxtime.json"
    boxtime.write_text(
        json.dumps({"1000": [1000, 1000 + 7200], "50000": [50000, 50300]})
    )  # 2 h before; this boot's own
    ledger = tmp_path / "spend.jsonl"
    box = budget.BoxTime(boxtime, boot_epoch=50000)
    assert box.hours(50000 + 1800) == pytest.approx(2.5)
    clock = budget.RunClock(tmp_path / "clock.json", now=50000 + 900)
    assert budget.spend(clock, 50000 + 1800, 10.0, ledger, box).box_hours == pytest.approx(2.5)
    assert budget.spend(clock, 50000 + 1800, 10.0, ledger, None).box_hours == pytest.approx(0.25)
    short = budget.RunClock(tmp_path / "c2.json", now=0.0)
    assert budget.spend(short, 4 * 3600.0, 10.0, ledger, box).box_hours == pytest.approx(4.0)  # the clock is larger


# ---- the disk, the learner's pulse, old adapters ---------------------------------------------------------------
async def test_a_box_short_of_disk_starts_no_games_and_alarms_until_there_is_room(tmp_path, monkeypatch):
    free = {"gb": 3.0}
    t = make(tmp_path, monkeypatch, concurrency=2, min_free_gb=8, free_gb=lambda: free["gb"])
    await t.orch.refresh_serving()
    await t.orch.housekeeping(t.clock())
    t.orch.fill()
    assert t.orch.disk_paused and "disk" in t.orch.alarms and not t.orch.in_flight
    free["gb"] = 50.0
    await t.orch.housekeeping(t.clock())
    t.orch.fill()
    assert not t.orch.disk_paused and "disk" not in t.orch.alarms and len(t.orch.in_flight) == 2
    await settle(t.orch)


def fill_inbox(root: Path, rows: int) -> None:
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "inbox" / "g.jsonl").write_text("{}\n" * rows)


async def test_a_learner_that_shows_no_progress_while_rows_wait_alarms_and_stops_the_run(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, learner_stall_minutes=30)
    t.root.mkdir(parents=True)
    fill_inbox(t.root, 300)
    t.orch.check_learner(t.clock())
    t.clock.t += 25 * 60
    t.orch.check_learner(t.clock())
    assert t.orch.stop_reason is None
    health(t.root, update(1))  # a health record is progress
    t.orch.check_learner(t.clock())
    t.clock.t += 25 * 60
    t.orch.check_learner(t.clock())
    assert t.orch.stop_reason is None
    t.clock.t += 6 * 60
    t.orch.check_learner(t.clock())
    assert t.orch.stop_reason.startswith("learner stalled: 300 rows wait") and "learner" in t.orch.alarms


async def test_few_rows_waiting_or_a_play_only_run_is_never_a_stalled_learner(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.root.mkdir(parents=True)
    fill_inbox(t.root, 255)
    t.clock.t += 10 * 3600
    t.orch.check_learner(t.clock())
    t.clock.t += 10 * 3600
    t.orch.check_learner(t.clock())
    assert t.orch.stop_reason is None
    fill_inbox(t.root, 400)
    p = make(tmp_path / "p", monkeypatch, learner=False)
    p.root.mkdir(parents=True)
    fill_inbox(p.root, 400)
    p.orch.check_learner(p.clock())
    p.clock.t += 10 * 3600
    p.orch.check_learner(p.clock())
    assert p.orch.stop_reason is None


async def test_adapters_nothing_wants_are_deleted_once_a_sync_has_their_manifest(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, servers={"urls": [HOSTS[0]], "keep_versions": [4]})
    for v in range(1, 8):
        publish(t.root, v)
    await t.orch.refresh_serving()
    assert t.orch.current == 7
    t.orch.hourly = {1: 2}
    (t.root / "champion.json").write_text(json.dumps({"version": 5}))
    t.orch.adapters.pin(3)
    t.orch.prune_adapters()
    assert t.orch.adapters.on_disk() == [1, 2, 3, 4, 5, 6, 7]  # the first sync has not yet taken their manifests
    t.orch.syncer.last_start_ok = 4_000_000_000.0
    t.orch.prune_adapters()
    assert t.orch.adapters.on_disk() == [2, 3, 4, 5, 7]  # hourly 2, pinned 3, configured 4, champion 5, current 7
    assert t.orch.counts["pruned"] == 2
    t.orch.adapters.release(3)


# ---- the credit and its config -------------------------------------------------------------------------------
def test_load_credit_builds_the_credit_and_the_critic_from_the_run_configs_tables(tmp_path, monkeypatch):
    import sys
    import types

    seen = {}

    class CreditConfig:
        def __init__(self, positive_control=False, margin_scale=2000.0, on_error="skip", decisions=(8, 12)):
            seen["config"] = dict(
                positive_control=positive_control, margin_scale=margin_scale, on_error=on_error, decisions=decisions
            )

    class RealCredit:
        def __init__(self, config=None, *, critic=None):
            seen["credit"] = (config, critic)

    class CriticConfig:
        def __init__(self, max_games=1000):
            self.max_games = max_games

    class GBCritic:
        def __init__(self, config):
            self.config = config

    monkeypatch.setitem(
        sys.modules, "agentenv_zeroad.train.credit", types.SimpleNamespace(Credit=RealCredit, CreditConfig=CreditConfig)
    )
    monkeypatch.setitem(
        sys.modules, "agentenv_zeroad.train.critic", types.SimpleNamespace(CriticConfig=CriticConfig, GBCritic=GBCritic)
    )
    cfg = RunConfig.from_dict(
        raw_config(
            tmp_path / "run",
            credit={
                "positive_control": True,
                "margin_scale": 1500.0,
                "on_error": "raise",
                "decisions": [4, 6],
                "critic": {"max_games": 50},
            },
        )
    )
    orchestrate.load_credit(cfg)
    config, critic = seen["credit"]
    assert seen["config"] == {
        "positive_control": True,
        "margin_scale": 1500.0,
        "on_error": "raise",
        "decisions": (4, 6),
    }
    assert critic.config.max_games == 50
    orchestrate.load_credit(RunConfig.from_dict(raw_config(tmp_path / "run")))
    assert seen["credit"] == (None, None)
    bad = RunConfig.from_dict(raw_config(tmp_path / "run", credit={"no_such_field": 1}))
    with pytest.raises(SystemExit, match="CreditConfig"):
        orchestrate.load_credit(bad)


def test_the_status_carries_the_sync_outcome_and_the_open_alarms(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.orch.alarm("disk", "full")
    t.orch.check_budget(t.clock())
    status = json.loads((t.root / "orchestrator" / "status.json").read_text())
    assert status["sync"] == {"last_ok": None, "failures": 0} and status["alarms"] == {"disk": "full"}


def test_the_game_stream_carries_its_state_across_a_restart(tmp_path):
    a = GameFactory(config(tmp_path))
    for _ in range(7):
        a.next(1, HOSTS[0])
    b = GameFactory(config(tmp_path))
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert [a.next(2, HOSTS[1]) for _ in range(5)] == [b.next(2, HOSTS[1]) for _ in range(5)]
