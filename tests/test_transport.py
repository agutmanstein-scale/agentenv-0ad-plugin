"""The multi-box transport: the stores, the bundle protocol and one learner orchestrator with two actor orchestrators
exchanging adapters and games through a LocalStore, against the fake vLLM, fake game player and fake credit of the
orchestrator's own tests. Nothing here starts an engine, a container or an AWS call."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentenv_zeroad.train import orchestrate
from agentenv_zeroad.train.config import TransportConfig
from agentenv_zeroad.train.mix import GameFactory
from agentenv_zeroad.train.serving import Fleet
from agentenv_zeroad.train.sync import Syncer
from agentenv_zeroad.train.transport import (
    ActorLink,
    BundleError,
    LocalStore,
    S3Store,
    StoreError,
    fetch_dir,
    flush_queue,
    get_json,
    publish_dir,
    put_json,
    unpack_game,
)
from tests import test_train_orchestrate as base

pytestmark = pytest.mark.anyio

URI_KEYS = ("current.json", "stop.json")


# ---- config -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change, message",
    [
        ({"role": "boss"}, "transport.role"),
        ({"enabled": True, "uri": "https://x"}, "transport.uri"),
        ({"enabled": True, "uri": "s3://b/p", "role": "actor"}, "transport.actor_id"),
        ({"enabled": True, "uri": "s3://b/p", "role": "actor", "actor_id": "a/b"}, "transport.actor_id"),
        ({"enabled": True, "uri": "s3://b/p", "max_version_lag": -1}, "max_version_lag"),
        ({"enabled": True, "uri": "s3://b/p", "poll_s": 0}, "must be positive"),
        ({"surprise": 1}, "unknown transport keys"),
    ],
)
def test_a_transport_section_that_could_not_work_is_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        base.config(tmp_path, transport=change)


def test_an_actor_plays_training_games_only(tmp_path):
    actor = {"enabled": True, "uri": "s3://b/p", "role": "actor", "actor_id": "a1"}
    with pytest.raises(ValueError, match="training games only"):
        base.config(tmp_path, transport=actor, hard_states={"enabled": True})
    cfg = base.config(tmp_path, transport=actor)
    assert cfg.transport.actor and not cfg.transport.learner and cfg.game_prefix == "r1-a1"
    assert base.config(tmp_path).game_prefix == "r1" and not base.config(tmp_path).transport.enabled


def test_actors_draw_different_games_from_each_other_and_from_the_learner(tmp_path):
    def stream(role, actor_id=""):
        t = {"enabled": True, "uri": "s3://b/p", "role": role, "actor_id": actor_id}
        cfg = base.config(tmp_path, transport=t, seed=7)
        factory = GameFactory(cfg)
        return [factory.next(0, base.HOSTS[0]) for _ in range(20)]

    one, two, learner = stream("actor", "a1"), stream("actor", "a2"), stream("learner")
    assert stream("actor", "a1") == one  # a restarted actor draws the same stream
    assert [s.seed for s in one] != [s.seed for s in two] and [s.seed for s in one] != [s.seed for s in learner]
    ids = {s.game_id for s in [*one, *two, *learner]}
    assert len(ids) == 60 and one[0].game_id == "r1-a1-g000000" and learner[0].game_id == "r1-g000000"


# ---- the stores ---------------------------------------------------------------------------------------------------
class FakeAws:
    """`aws s3 cp|rm` and `s3api list-objects-v2` over a directory, answering as the zeroad-train-box role does: it may
    list only under the run prefix, so a get of a key that is not there is a 403, as S3 answers without s3:ListBucket on
    the key (`missing_status` 404 is the open-bucket answer)."""

    def __init__(self, root: Path, bucket: str = "bkt", missing_status: int = 403):
        self.root, self.bucket, self.calls, self.missing_status = root, bucket, [], missing_status
        self.fail: str | None = None

    def _path(self, url: str) -> Path:
        assert url.startswith(f"s3://{self.bucket}/"), url
        return self.root / url.removeprefix(f"s3://{self.bucket}/")

    def __call__(self, cmd: list[str]) -> tuple[int, str]:
        self.calls.append(cmd)
        if self.fail and self.fail in " ".join(cmd):
            return 1, "Could not connect to the endpoint URL: port 8404"
        assert "head-object" not in cmd, "the role cannot HeadObject a key that is not there"
        match cmd[1:3]:
            case ["s3", "cp"]:
                src, dst = cmd[3], cmd[4]
                if src.startswith("s3://"):
                    path = self._path(src)
                    if not path.is_file():
                        return 1, f"An error occurred ({self.missing_status}) when calling the HeadObject operation"
                    shutil.copyfile(path, dst)
                else:
                    path = self._path(dst)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, path)
                return 0, ""
            case ["s3", "rm"]:
                self._path(cmd[3]).unlink(missing_ok=True)
                return 0, ""
            case ["s3api", "list-objects-v2"]:
                prefix = cmd[cmd.index("--prefix") + 1]
                found = [
                    [p.relative_to(self.root).as_posix(), f'"{p.stat().st_size}-{p.stat().st_mtime_ns}"']
                    for p in sorted(self.root.rglob("*"))
                    if p.is_file() and p.relative_to(self.root).as_posix().startswith(prefix)
                ]
                return 0, json.dumps(found or None)
        raise AssertionError(cmd)


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "bucket")
    return S3Store("s3://bkt/runs/r1", run=FakeAws(tmp_path / "bucket" / "runs" / "r1"))


def test_a_store_puts_gets_lists_checks_and_deletes(store, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    store.put("games/a1/g1.tgz", src)
    store.put("games/a1/g1.json", src)
    store.put("current.json", src)
    assert store.list("games") == ["games/a1/g1.json", "games/a1/g1.tgz"] and store.list("nothing") == []
    assert store.exists("current.json") and not store.exists("stop.json")
    out = tmp_path / "out.txt"
    store.get("current.json", out)
    assert out.read_text() == "hello"
    with pytest.raises(FileNotFoundError):
        store.get("stop.json", out)
    store.delete("current.json")
    store.delete("current.json")  # deleting what is not there is fine
    assert not store.exists("current.json")
    assert get_json(store, "stop.json") is None
    put_json(store, "stop.json", {"reason": "x"})
    assert get_json(store, "stop.json") == {"reason": "x"}
    with pytest.raises(ValueError):
        store.put("../escape", src)


def test_an_s3_failure_that_is_not_a_missing_object_raises_a_store_error(tmp_path):
    aws = FakeAws(tmp_path / "bucket")
    store = S3Store("s3://bkt", run=aws)
    put_json(store, "a.json", {})
    aws.fail = "s3 cp"
    with pytest.raises(StoreError):
        store.put("a.json", tmp_path)
    with pytest.raises(StoreError):
        store.get("a.json", tmp_path / "x")  # a failure on an object that a listing shows is not "missing"
    with pytest.raises(FileNotFoundError):
        store.get("b.json", tmp_path / "x")  # not listed: missing, whatever the text of the failure said
    aws.fail = "list-objects-v2"
    with pytest.raises(StoreError):
        store.exists("a.json")
    with pytest.raises(StoreError):
        store.get("b.json", tmp_path / "x")


@pytest.mark.parametrize("missing_status", [403, 404])
def test_a_missing_key_reads_as_missing_under_a_role_that_cannot_head_it(tmp_path, missing_status):
    aws = FakeAws(tmp_path / "bucket", missing_status=missing_status)
    store = S3Store("s3://bkt/runs/r1", run=aws)
    assert (
        not store.exists("stop.json")
        and get_json(store, "stop.json") is None
        and get_json(store, "current.json") is None
    )
    store.delete("stop.json")
    put_json(store, "current.json.bak", {})
    assert not store.exists("current.json")  # a longer key under the same prefix is not it
    put_json(store, "current.json", {"version": 1})
    assert store.exists("current.json") and get_json(store, "current.json") == {"version": 1}
    assert not any("head-object" in c for c in aws.calls)


def test_the_s3_store_names_objects_under_its_prefix_and_uses_the_cli(tmp_path):
    aws = FakeAws(tmp_path / "bucket")
    store = S3Store("s3://bkt/pre/fix/", run=aws)
    put_json(store, "actors/a1.json", {"actor_id": "a1"})
    assert (tmp_path / "bucket" / "pre" / "fix" / "actors" / "a1.json").is_file()
    assert aws.calls[0][:3] == ["aws", "s3", "cp"] and aws.calls[0][4] == "s3://bkt/pre/fix/actors/a1.json"
    assert store.list("actors") == ["actors/a1.json"]


# ---- marker-last items -------------------------------------------------------------------------------------------
def adapter_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "v3"
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(b"w" * 5000)
    (folder / "adapter_config.json").write_text("{}")
    (folder / "manifest.json").write_text(json.dumps({"version": 3}))
    return folder


def test_a_published_folder_round_trips_and_every_file_is_checked(store, tmp_path):
    publish_dir(store, "adapters/v3", adapter_folder(tmp_path))
    fetch_dir(store, "adapters/v3", tmp_path / "got")
    assert (tmp_path / "got" / "adapter_model.safetensors").read_bytes() == b"w" * 5000
    got = tmp_path / "got2"
    store.put("adapters/v3/adapter_model.safetensors", tmp_path / "v3" / "adapter_config.json")  # a corrupt object
    with pytest.raises(BundleError):
        fetch_dir(store, "adapters/v3", got)


def test_a_folder_without_its_ready_marker_is_not_there(store, tmp_path):
    folder = adapter_folder(tmp_path)
    for path in folder.iterdir():
        store.put(f"adapters/v3/{path.name}", path)  # every file, but the process died before READY
    with pytest.raises(FileNotFoundError):
        fetch_dir(store, "adapters/v3", tmp_path / "got")
    link = ActorLink(store, TransportConfig())
    assert link.pull_adapter(3, tmp_path / "adapters") is False
    assert not list((tmp_path / "adapters").glob("v*")) and not list((tmp_path / "adapters").glob(".dl-*"))


def test_a_pulled_adapter_appears_whole_or_not_at_all(store, tmp_path):
    publish_dir(store, "adapters/v3", adapter_folder(tmp_path))
    link = ActorLink(store, TransportConfig())
    adapters = tmp_path / "adapters"
    assert link.pull_adapter(3, adapters) is True
    assert sorted(p.name for p in (adapters / "v3").iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors", "manifest.json"
    ]  # fmt: skip
    assert not list(adapters.glob(".dl-*"))
    store.put("adapters/v3/manifest.json", tmp_path / "v3" / "adapter_config.json")
    shutil.rmtree(adapters / "v3")
    with pytest.raises(BundleError):
        link.pull_adapter(3, adapters)
    assert not (adapters / "v3").exists() and not list(adapters.glob(".dl-*"))


def test_an_archive_that_reaches_outside_its_game_folder_is_refused(tmp_path):
    import tarfile

    bad = tmp_path / "bad.tgz"
    (tmp_path / "x.txt").write_text("x")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(tmp_path / "x.txt", arcname="../evil.txt")
    with pytest.raises(BundleError):
        unpack_game(bad, "g1", tmp_path / "out")
    other = tmp_path / "other.tgz"
    with tarfile.open(other, "w:gz") as tar:
        tar.add(tmp_path / "x.txt", arcname="g2/x.txt")
    with pytest.raises(BundleError):
        unpack_game(other, "g1", tmp_path / "out")


# ---- one learner and two actors ----------------------------------------------------------------------------------
def node(tmp_path, monkeypatch, store, clock, name, role, *, aws_code=0, lag=4, **over):
    monkeypatch.setenv("VLLM_API_KEY", "k")
    transport = {"enabled": True, "uri": f"file://{store.root}", "role": role, "poll_s": 0.01, "max_version_lag": lag}
    if role == "actor":
        transport["actor_id"] = name
    extra = {"s3": {}} if role == "actor" else {}
    cfg = base.config(tmp_path / name, transport=transport, engines=2, concurrency=2, **{**extra, **over})
    fake, play = base.FakeVllm(), base.FakePlay()
    credit = base.FakeCredit(2) if role == "learner" else orchestrate.PlayOnly()
    powered: list[bool] = []
    syncs: list[list[str]] = []

    async def aws(cmd: list[str]) -> tuple[int, str]:
        syncs.append(cmd)
        return aws_code, ""

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
        learner=role == "learner",
        store=store,
    )
    return SimpleNamespace(
        orch=orch, fake=fake, play=play, credit=credit, root=cfg.root, cfg=cfg, powered=powered, syncs=syncs
    )


def games_in(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "inbox").glob("*.jsonl")) if (root / "inbox").is_dir() else []


async def play_games(actor, n: int) -> None:
    """Let the actor's orchestrator finish `n` games (it starts `concurrency` at a time)."""
    while actor.orch.counts["train_finished"] < n:
        actor.orch.fill()
        await base.until(lambda: not actor.orch.in_flight)


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "bucket")
    clock = base.Clock()
    learner = node(tmp_path, monkeypatch, store, clock, "learner", "learner", lag=3, drain_timeout_s=0.2)
    actors = {n: node(tmp_path, monkeypatch, store, clock, n, "actor") for n in ("a1", "a2")}
    return SimpleNamespace(store=store, clock=clock, learner=learner, actors=actors, tmp=tmp_path, mp=monkeypatch)


async def bring_up(w, version: int = 1) -> None:
    """The learner serves `version` and publishes it; both actors read it, pull it and serve it."""
    base.publish(w.learner.root, version)
    await w.learner.orch.refresh_serving()
    w.learner.orch.xport.start()
    await w.learner.orch.xport.run_all()
    for a in w.actors.values():
        a.orch.xport.start()
        await a.orch.xport.run_all()
        await a.orch.refresh_serving()


async def test_actor_games_are_credited_once_on_the_learner(world):
    await bring_up(world)
    for a in world.actors.values():
        await play_games(a, 3)
        await a.orch.xport.run_all()
    learner = world.learner
    await learner.orch.xport.run_all()
    credited = sorted(learner.credit.games)
    assert len(credited) == len(set(credited)) >= 6
    assert {g.split("-g")[0] for g in credited} == {"r1-a1", "r1-a2"}
    assert games_in(learner.root) == sorted(f"{g}.jsonl" for g in credited)
    assert all(
        (learner.root / "remote" / g.split("-g")[0].removeprefix("r1-") / g / "results.json").is_file()
        for g in credited
    )
    await learner.orch.xport.run_all()
    assert sorted(learner.credit.games) == credited  # a second poll credits nothing again
    assert learner.orch.counts["remote_credited"] == len(credited) and learner.orch.counts["rows"] == 2 * len(credited)
    first = {a: world.actors[a].play.specs[0] for a in world.actors}
    assert first["a1"].seed != first["a2"].seed


async def test_a_partial_bundle_without_its_manifest_is_ignored(world, tmp_path):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    folder_id = next(iter(a.orch.xport.queue))
    link = ActorLink(world.store, a.cfg.transport)
    link.ship("a1", a.root / "games" / folder_id, 1, "bot")
    (world.store.root / "games" / "a1" / f"{folder_id}.json").unlink()  # the upload died before the marker
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == [] and world.learner.orch.counts["remote_credited"] == 0
    (world.store.root / "games" / "a1" / "junk.json").write_text("{not json")
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == []


async def test_a_bundle_whose_sha_does_not_match_is_rejected_and_not_retried_forever(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    bad, good = sorted((world.store.root / "games" / "a1").glob("*.tgz"))
    manifest = json.loads(bad.with_suffix(".json").read_text())
    bad.write_bytes(b"x" * manifest["size"])  # the right length, the wrong bytes
    for _ in range(5):
        await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == [good.stem] and world.learner.orch.counts["remote_rejected"] == 1
    assert world.learner.orch.xport.status(world.clock())["dropped"] == {"rejected": 1}


async def test_a_game_pinned_to_a_version_too_far_behind_is_dropped_and_counted(world, tmp_path):
    await bring_up(world, version=10)
    folder = tmp_path / "made" / "r1-a1-g000900"
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps({"spec": {"game_id": folder.name}}))
    link = ActorLink(world.store, world.actors["a1"].cfg.transport, world.clock)
    link.ship("a1", folder, 6, "bot")  # lag 4 against max_version_lag 3
    folder2 = tmp_path / "made" / "r1-a1-g000901"
    shutil.copytree(folder, folder2)
    (folder2 / "results.json").write_text(json.dumps({"spec": {"game_id": folder2.name}}))
    link.ship("a1", folder2, 7, "bot")  # lag 3: still trained on
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == ["r1-a1-g000901"]
    assert world.learner.orch.counts["remote_stale"] == 1
    assert world.learner.orch.xport.status(world.clock())["dropped"] == {"stale": 1}
    assert not (world.learner.root / "remote" / "a1" / "r1-a1-g000900").exists()  # not even downloaded


async def test_a_new_adapter_version_reaches_the_actors_and_their_new_games_pin_it(world):
    await bring_up(world, version=1)
    a = world.actors["a1"]
    await play_games(a, 2)
    assert {base.policy_version(s) for s in a.play.specs} == {1}
    base.publish(world.learner.root, 2)
    await world.learner.orch.refresh_serving()
    await world.learner.orch.xport.run_all()
    assert get_json(world.store, "current.json")["version"] == 2 and world.store.exists("adapters/v2/READY")
    await a.orch.xport.run_all()
    await a.orch.refresh_serving()
    assert a.orch.current == 2 and (a.root / "adapters" / "v2" / "manifest.json").is_file()
    before = len(a.play.specs)
    await play_games(a, a.orch.counts["train_finished"] + 2)
    assert {base.policy_version(s) for s in a.play.specs[before:]} == {2}
    await a.orch.xport.beat_pass()
    assert get_json(world.store, "actors/a1.json")["version"] == 2


async def test_the_learner_publishes_each_current_version_once(world):
    await bring_up(world, version=1)
    puts = []
    original = world.store.put
    world.store.put = lambda key, src: (puts.append(key), original(key, src))[1]
    await world.learner.orch.xport.run_all()
    assert not [k for k in puts if k.startswith("adapters/")]  # v1 is already there
    assert world.learner.orch.counts["published"] == 1


async def test_actors_do_not_start_games_before_the_learner_has_published_a_pulse(world):
    a = world.actors["a1"]
    a.orch.xport.start()
    await a.orch.xport.run_all()
    await a.orch.refresh_serving()
    a.orch.fill()
    assert a.play.specs == [] and not a.orch.xport.ready
    world.learner.orch.xport.start()
    await a.orch.xport.run_all()
    a.orch.fill()
    await base.until(lambda: len(a.play.specs) == 2)  # concurrency 2, at the base model


async def test_the_heartbeats_become_the_actors_block_of_status_json(world):
    await bring_up(world)
    for a in world.actors.values():
        await play_games(a, 2)
        await a.orch.xport.run_all()
    learner = world.learner
    await learner.orch.xport.run_all()
    learner.orch.check_budget(world.clock())
    status = json.loads((learner.root / "orchestrator" / "status.json").read_text())
    members = status["actors"]["members"]
    assert set(members) == {"a1", "a2"} and members["a1"]["state"] == "alive" and members["a1"]["version"] == 1
    assert members["a1"]["games_done"] >= 2 and members["a1"]["lag"] == 0
    world.clock.t += 11 * 60  # actor_stale_minutes is 10
    await learner.orch.xport.run_all()
    status = learner.orch.xport.status(world.clock())
    assert {m["state"] for m in status["members"].values()} == {"stale"}
    assert set(learner.orch.alarms) == {"actor:a1", "actor:a2"}


async def test_the_budget_counts_every_actors_box_hours_and_frontier_spend(world):
    learner = world.learner
    learner.orch.xport.start()
    link = ActorLink(world.store, world.actors["a1"].cfg.transport, world.clock)
    now = world.clock()
    beat = {"time": now, "version": 0, "box_hours": 6.0, "box_usd_per_hour": 10.0, "model_usd": 5.0, "in_flight": 0}
    link.heartbeat("a1", beat)
    link.heartbeat("a2", {**beat, "box_hours": 2.0, "box_usd_per_hour": 15.0, "model_usd": 0.0})
    await learner.orch.xport.run_all()
    spent = learner.orch.check_budget(now)
    assert spent.box_usd == pytest.approx(60 + 30) and spent.model_usd == pytest.approx(5)
    assert learner.orch.stop_reason is None  # 95 of 100
    link.heartbeat("a1", {**beat, "box_hours": 6.6})
    await learner.orch.xport.run_all()
    learner.orch.check_budget(now)
    assert learner.orch.stop_reason and learner.orch.stop_reason.startswith("budget:")


async def test_an_actor_that_went_quiet_keeps_costing_for_the_stale_window_only(world):
    learner = world.learner
    learner.orch.xport.start()
    now = world.clock()
    ActorLink(world.store, learner.cfg.transport, world.clock).heartbeat(
        "a1", {"time": now, "box_hours": 1.0, "box_usd_per_hour": 10.0, "model_usd": 0.0}
    )
    await learner.orch.xport.run_all()
    world.clock.t += 3600
    assert learner.orch.xport.spend_extra(world.clock())[0] == pytest.approx(10 + 10 * 10 / 60)
    ActorLink(world.store, learner.cfg.transport, world.clock).heartbeat(
        "a1", {"time": now, "box_hours": 1.0, "box_usd_per_hour": 10.0, "final": True}
    )
    await learner.orch.xport.run_all()
    assert learner.orch.xport.spend_extra(world.clock())[0] == pytest.approx(10)


async def run_actor_until_stopped(a) -> None:
    """What an actor's loop does: poll until the stop arrives, then finish (the held games are let go meanwhile)."""
    while a.orch.stop_reason is None:
        await a.orch.xport.run_all()
        await asyncio.sleep(0.005)
    asyncio.get_running_loop().call_later(0.05, a.play.release)
    await a.orch.finish(a.orch.stop_reason)


async def test_a_complete_or_duration_stop_drains_the_actors_and_the_learner_credits_what_they_ship(
    tmp_path, monkeypatch
):
    store, clock = LocalStore(tmp_path / "bucket"), base.Clock()
    w = SimpleNamespace(store=store, clock=clock)
    w.learner = node(tmp_path, monkeypatch, store, clock, "learner", "learner")
    w.actors = {"a1": node(tmp_path, monkeypatch, store, clock, "a1", "actor", shutdown=True)}
    await bring_up(w)
    a = w.actors["a1"]
    a.play.hold()
    a.orch.fill()
    await base.until(lambda: len(a.play.specs) == 2)
    await asyncio.gather(w.learner.orch.finish("duration"), run_actor_until_stopped(a))
    stop = get_json(store, "stop.json")
    assert stop["reason"] == "duration" and stop["by"] == "learner" and stop["drain"] is True
    assert stop["epoch"] == get_json(store, "current.json")["epoch"]
    assert a.orch.stop_reason == "transport-drain: stop requested by learner: duration"
    assert a.orch.counts["train_finished"] == 2 and a.orch.counts["train_stopped"] == 0  # drained, not cancelled
    assert a.orch.xport.unshipped() == 0 and len(list((store.root / "games" / "a1").glob("*.json"))) == 2
    final = get_json(store, "actors/a1.json")
    assert final["final"] is True and final["unshipped"] == 0
    assert a.powered == [True]
    assert len(w.learner.credit.games) == 2  # the learner kept ingesting while the actor drained


async def test_a_budget_or_tripwire_stop_cancels_the_actors_games_at_once(tmp_path, monkeypatch):
    store, clock = LocalStore(tmp_path / "bucket"), base.Clock()
    w = SimpleNamespace(store=store, clock=clock)
    w.learner = node(tmp_path, monkeypatch, store, clock, "learner", "learner")
    w.actors = {"a1": node(tmp_path, monkeypatch, store, clock, "a1", "actor", shutdown=True)}
    await bring_up(w)
    a = w.actors["a1"]
    a.play.hold()
    a.orch.fill()
    await base.until(lambda: len(a.play.specs) == 2)
    await w.learner.orch.finish("budget: box $90 + model $10 = $100 of $100")
    assert get_json(store, "stop.json")["drain"] is False
    await a.orch.xport.run_all()
    assert a.orch.stop_reason == "transport: stop requested by learner: budget: box $90 + model $10 = $100 of $100"
    await a.orch.finish(a.orch.stop_reason)
    assert a.orch.counts["train_stopped"] == 2 and a.orch.counts["train_finished"] == 0
    assert w.learner.credit.games == []


@pytest.mark.parametrize("reason", ["signal SIGTERM", "crash: ValueError: x", "cancelled", "learner stalled: 300 rows"])
async def test_only_a_final_stop_writes_stop_json_so_actors_survive_a_learner_restart(world, reason):
    await bring_up(world)
    await world.learner.orch.finish(reason)
    assert not world.store.exists("stop.json")
    await world.actors["a1"].orch.xport.run_all()
    assert world.actors["a1"].orch.stop_reason is None


@pytest.mark.parametrize("reason", ["complete", "duration", "budget: x", "tripwire: [rollback]"])
async def test_the_final_stops_write_stop_json(world, reason):
    await bring_up(world)
    await world.learner.orch.finish(reason)
    stop = get_json(world.store, "stop.json")
    assert stop["reason"] == reason and stop["drain"] is (reason in ("complete", "duration"))


async def test_an_actor_obeys_only_a_stop_of_the_learner_epoch_it_follows(world):
    await bring_up(world)
    a = world.actors["a1"]
    epoch = get_json(world.store, "current.json")["epoch"]
    put_json(world.store, "stop.json", {"reason": "old run", "by": "learner", "time": world.clock(), "epoch": "other"})
    await a.orch.xport.run_all()
    assert a.orch.stop_reason is None
    put_json(world.store, "stop.json", {"reason": "by hand", "by": "human", "time": world.clock(), "epoch": epoch})
    await a.orch.xport.run_all()
    assert a.orch.stop_reason == "transport: stop requested by human: by hand"


async def test_an_actor_started_after_the_learner_stopped_obeys_its_stop(world):
    await bring_up(world)
    await world.learner.orch.finish("budget: x")
    late = node(world.tmp, world.mp, world.store, world.clock, "a9", "actor")
    late.orch.xport.start()
    await late.orch.xport.run_all()
    assert late.orch.stop_reason == "transport: stop requested by learner: budget: x"


async def test_a_restarted_learner_is_a_new_epoch_that_actors_follow_and_the_old_stop_is_cleared(world):
    await bring_up(world)
    first = get_json(world.store, "current.json")["epoch"]
    again = node(world.tmp, world.mp, world.store, world.clock, "learner", "learner")
    again.orch.xport.start()
    second = get_json(world.store, "current.json")["epoch"]
    assert first != second
    await world.actors["a1"].orch.xport.run_all()
    assert world.actors["a1"].orch.xport.epoch == second and world.actors["a1"].orch.stop_reason is None


async def test_a_learner_that_goes_stale_stops_the_actors(world):
    await bring_up(world)
    a = world.actors["a1"]
    world.clock.t += 14 * 60
    await a.orch.xport.run_all()
    assert a.orch.stop_reason is None  # learner_stale_minutes is 15
    world.clock.t += 2 * 60
    await a.orch.xport.run_all()
    assert a.orch.stop_reason.startswith("transport: the learner's pulse is 16 min old")
    await world.learner.orch.xport.run_all()  # the learner is back: a pulse, but the actor has stopped
    assert a.orch.stop_reason.startswith("transport: the learner's")


async def test_an_actor_started_with_no_learner_gives_up_after_the_stale_window(world):
    a = world.actors["a1"]
    await a.orch.xport.run_all()
    assert a.orch.stop_reason is None
    world.clock.t += 16 * 60
    await a.orch.xport.run_all()
    assert a.orch.stop_reason == "transport: no learner has published a pulse"


async def test_a_restarted_learner_does_not_ingest_again(world, tmp_path, monkeypatch):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 2)
    await a.orch.xport.run_all()
    await world.learner.orch.xport.run_all()
    credited = sorted(world.learner.credit.games)
    assert credited
    put_json(world.store, "stop.json", {"reason": "x", "by": "learner", "time": world.clock()})
    again = node(tmp_path, monkeypatch, world.store, world.clock, "learner", "learner")
    again.orch.current = 1
    again.orch.xport.start()
    assert not world.store.exists("stop.json")  # a new learner process is a new go-ahead
    await again.orch.xport.run_all()
    assert (
        sorted(again.credit.games) == credited and again.orch.counts["remote_credited"] == 0
    )  # restored, not added to
    assert again.orch.xport.status(world.clock())["ingested"] == len(credited)


async def test_a_game_credited_before_a_crash_could_record_it_is_not_credited_twice(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    game_id = next(iter(a.orch.xport.shipped))
    inbox = world.learner.root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"{game_id}.jsonl").write_text("{}\n")  # rows written, ingested.json never saved
    await world.learner.orch.xport.run_all()
    assert game_id not in world.learner.credit.games


async def test_a_failed_upload_is_retried_with_backoff_and_the_game_is_not_lost(world):
    await bring_up(world)
    a = world.actors["a1"]
    failing = {"on": True}
    original = world.store.put

    def put(key, src):
        if failing["on"] and key.startswith("games/"):
            raise StoreError("the endpoint is unreachable")
        original(key, src)

    world.store.put = put
    await play_games(a, 1)
    await a.orch.xport.run_all()
    queued = a.orch.xport.unshipped()
    assert queued >= 1 and "unreachable" in a.orch.xport.last_error and a.orch.counts["ship_failed"] >= 1
    failed = a.orch.counts["ship_failed"]
    await a.orch.xport.ship_pass()
    assert a.orch.counts["ship_failed"] == failed  # still inside the backoff
    failing["on"] = False
    world.clock.t += 6
    await a.orch.xport.ship_pass()
    assert a.orch.xport.unshipped() == 0 and a.orch.counts["shipped"] == queued
    await a.orch.xport.beat_pass()
    assert get_json(world.store, "actors/a1.json")["unshipped"] == 0


async def test_an_actor_that_cannot_ship_stays_up_when_it_finishes(world, tmp_path, monkeypatch):
    store, clock = world.store, world.clock
    a = node(tmp_path, monkeypatch, store, clock, "a3", "actor", shutdown=True, sync_retry_s=0)
    await bring_up(world)
    a.orch.xport.start()
    await a.orch.xport.run_all()
    await a.orch.refresh_serving()
    store.put = lambda key, src: (_ for _ in ()).throw(StoreError("down")) if key.startswith("games/") else None
    await play_games(a, 1)
    await a.orch.finish("duration")
    assert a.powered == [] and "unshipped" in a.orch.alarms
    assert json.loads((a.root / "orchestrator" / "final.json").read_text())["unshipped"] >= 1


async def test_a_shipped_game_leaves_the_actors_disk_but_a_failed_one_stays(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 2)
    await a.orch.xport.run_all()
    assert not list((a.root / "games").glob("*/results.json"))
    assert a.orch.xport.unshipped() == 0


async def test_an_actor_learns_nothing_and_has_no_inbox_or_panels(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 2)
    assert not (a.root / "inbox").exists() and not (a.root / "panels").exists() and a.orch.counts["rows"] == 0
    assert not a.orch.learner and a.orch.hard is None


async def test_the_cli_reads_and_stops(world, capsys):
    from agentenv_zeroad.train import transport

    await bring_up(world)
    uri = f"file://{world.store.root}"
    await world.actors["a1"].orch.xport.beat_pass()
    assert transport.main(["status", "--uri", uri]) == 0
    out = capsys.readouterr().out
    assert '"version": 1' in out and "a1: v1" in out
    assert transport.main(["stop", "--uri", uri, "--reason", "enough"]) == 0
    assert get_json(world.store, "stop.json")["reason"] == "enough"


# ---- review fixes: a fresh actor, replacements, drift, retries, disk -------------------------------------------
async def test_a_fresh_actors_first_wave_pins_the_learners_version_even_when_the_passes_run_as_tasks(world):
    base.publish(world.learner.root, 5)
    await world.learner.orch.refresh_serving()
    world.learner.orch.xport.start()
    await world.learner.orch.xport.run_all()
    a = world.actors["a1"]
    a.orch.xport.start()
    await a.orch.refresh_serving()
    a.orch.xport.kick()  # what tick() does: every pass as its own task
    a.orch.fill()
    assert a.play.specs == []  # nothing before the learner has been read and v5 is held
    for task in list(a.orch.xport._tasks.values()):
        await task
    await a.orch.refresh_serving()
    a.orch.fill()
    await base.until(lambda: len(a.play.specs) == 2)
    assert {base.policy_version(s) for s in a.play.specs} == {5}


async def test_an_adapter_that_keeps_failing_to_download_lets_a_game_start_only_within_the_version_lag(world):
    await bring_up(world, version=1)
    base.publish(world.learner.root, 3)
    await world.learner.orch.refresh_serving()
    await world.learner.orch.xport.run_all()
    a = world.actors["a1"]
    (world.store.root / "adapters" / "v3" / "READY").unlink()  # published, then broken
    for _ in range(3):
        await a.orch.xport.run_all()
    assert a.orch.xport.pull_failures == 3 and a.orch.xport.ready  # v1 against v3: within the lag of 4
    base.publish(world.learner.root, 9)
    await world.learner.orch.refresh_serving()
    await world.learner.orch.xport.run_all()
    (world.store.root / "adapters" / "v9" / "READY").unlink()
    a.orch.xport.pull_failures = 0
    await a.orch.xport.run_all()
    assert not a.orch.xport.ready


async def test_a_crash_after_crediting_when_the_trainer_has_already_consumed_the_rows_is_not_credited_again(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    await world.learner.orch.xport.run_all()
    credited = list(world.learner.credit.games)
    inbox = world.learner.root / "inbox"
    (inbox / "consumed").mkdir()
    for f in inbox.glob("*.jsonl"):
        shutil.move(str(f), str(inbox / "consumed" / f.name))
    (world.learner.root / "orchestrator" / "ingested.json").write_text(json.dumps({"ingested": {}, "dropped": {}}))
    again = node(world.tmp, world.mp, world.store, world.clock, "learner", "learner", lag=3)
    again.orch.current = 1
    await again.orch.xport.run_all()
    assert again.credit.games == credited and again.orch.counts["remote_credited"] == 0


async def test_an_intent_left_by_a_crash_is_resolved_at_the_next_start(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    [manifest] = sorted((world.store.root / "games" / "a1").glob("*.json"))[:1]
    game = manifest.stem
    odir = world.learner.root / "orchestrator"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "intent.json").write_text(json.dumps({"key": f"a1/{game}", "game_id": game, "sha256": "x"}))
    (world.learner.root / "inbox").mkdir(exist_ok=True)
    (world.learner.root / "inbox" / f"{game}.jsonl").write_text("{}\n")  # the rows were written
    again = node(world.tmp, world.mp, world.store, world.clock, "learner", "learner", lag=3)
    assert f"a1/{game}" in again.orch.xport.ingested and not (odir / "intent.json").exists()


async def test_a_bundle_without_results_json_or_an_empty_one_is_rejected_not_retried_forever(world, tmp_path):
    await bring_up(world)
    link = ActorLink(world.store, world.actors["a1"].cfg.transport, world.clock)
    nothing = tmp_path / "made" / "r1-a1-g000900"
    nothing.mkdir(parents=True)
    (nothing / "other.txt").write_text("x")
    link.ship("a1", nothing, 1, "bot")
    empty = tmp_path / "made" / "r1-a1-g000901"
    empty.mkdir()
    link.ship("a1", empty, 1, "bot")
    for _ in range(3):
        await world.learner.orch.xport.run_all()
    c = world.learner.orch.counts
    assert c["remote_rejected"] == 2 and c["transport_error_fetch"] == 0
    assert world.learner.orch.xport.status(world.clock())["dropped"] == {"rejected": 2}


async def test_a_game_whose_archive_is_gone_is_given_up_on_after_five_tries(world, tmp_path):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    victim = sorted((world.store.root / "games" / "a1").glob("*.tgz"))[0]
    victim.unlink()  # the manifest says it is there
    for _ in range(8):
        await world.learner.orch.xport.run_all()
    assert (
        world.learner.orch.counts["remote_unfetchable"] == 1 and world.learner.orch.counts["transport_error_fetch"] == 5
    )


async def test_one_actors_stuck_games_cannot_block_another_actors_games(world, tmp_path):
    await bring_up(world)
    link = ActorLink(world.store, world.actors["a1"].cfg.transport, world.clock)
    for n in range(20):
        folder = tmp_path / "made" / f"r1-a1-g9{n:05d}"
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps({"spec": {"game_id": folder.name}}))
        link.ship("a1", folder, 1, "bot")
        (world.store.root / "games" / "a1" / f"{folder.name}.tgz").unlink()  # every one stuck
    good = tmp_path / "made" / "r1-a2-g000001"
    good.mkdir()
    (good / "results.json").write_text(json.dumps({"spec": {"game_id": good.name}}))
    link.ship("a2", good, 1, "bot")
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == ["r1-a2-g000001"]


async def test_a_replacement_actor_with_the_same_id_plays_new_games_that_the_learner_credits(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 2)
    await a.orch.xport.run_all()
    await world.learner.orch.xport.run_all()
    first = sorted(world.learner.credit.games)
    shipped = sorted(p.stem for p in (world.store.root / "games" / "a1").glob("*.json"))
    b = node(world.tmp / "replacement", world.mp, world.store, world.clock, "a1", "actor")
    b.orch.xport.start()
    await b.orch.xport.run_all()
    await b.orch.refresh_serving()
    await play_games(b, 2)
    await b.orch.xport.run_all()
    await world.learner.orch.xport.run_all()
    ids = [s.game_id for s in b.play.specs]
    assert not set(ids) & set(shipped) and min(ids) > max(shipped)
    assert sorted(world.learner.credit.games) == sorted({*first, *ids})
    assert [s.seed for s in b.play.specs[:2]] != [s.seed for s in a.play.specs[:2]]  # not the same specs replayed


async def test_a_game_shipped_again_with_different_contents_is_refused_and_alarmed(world, tmp_path):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    await world.learner.orch.xport.run_all()
    credited = list(world.learner.credit.games)
    game = credited[0]
    again = tmp_path / "made" / game
    again.mkdir(parents=True)
    (again / "results.json").write_text(json.dumps({"spec": {"game_id": game}, "other": 1}))
    ActorLink(world.store, a.cfg.transport, world.clock).ship("a1", again, 1, "bot")
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == credited and world.learner.orch.counts["remote_conflict"] == 1
    assert f"conflict:a1/{game}" in world.learner.orch.alarms


async def test_two_boxes_heartbeating_as_one_actor_id_raise_an_alarm(world):
    await bring_up(world)
    learner = world.learner
    link = ActorLink(world.store, world.actors["a1"].cfg.transport, world.clock)
    beat = {"version": 1, "box_hours": 1, "box_usd_per_hour": 1, "process_start": 1.0}
    for n in range(4):
        link.heartbeat("a1", {**beat, "time": world.clock(), "instance_id": "i-aaa" if n % 2 else "i-bbb"})
        await learner.orch.xport.liveness_pass()
        world.clock.t += 15
    assert "duplicate:a1" in learner.orch.alarms
    link.heartbeat("a1", {**beat, "time": world.clock(), "instance_id": "i-bbb"})
    world.clock.t += 20 * 60  # a replacement long after the first died is not a duplicate
    link.heartbeat("a1", {**beat, "time": world.clock(), "instance_id": "i-ccc"})
    learner.orch.alarms.clear()
    learner.orch.xport._dup.clear()
    await learner.orch.xport.liveness_pass()
    assert "duplicate:a1" not in learner.orch.alarms


async def test_an_actors_heartbeat_names_its_instance_and_process(world):
    world.mp.setenv("ZEROAD_INSTANCE_ID", "i-0abc")
    a = node(world.tmp, world.mp, world.store, world.clock, "a7", "actor")
    await a.orch.xport.beat_pass()
    beat = get_json(world.store, "actors/a7.json")
    assert beat["instance_id"] == "i-0abc" and beat["process_start"] == world.clock() and beat["fingerprint"]


async def test_an_actor_whose_games_config_differs_from_the_learners_refuses_to_start(world, tmp_path):
    await bring_up(world)
    other = node(world.tmp, world.mp, world.store, world.clock, "a5", "actor", mix={"bot": 1.0})
    other.orch.xport.start()
    await other.orch.xport.run_all()
    await other.orch.refresh_serving()
    other.orch.fill()
    assert other.play.specs == [] and not other.orch.xport.ready
    assert other.orch.stop_reason.startswith("transport: config mismatch")
    await other.orch.xport.beat_pass()
    assert "differ from the learner's" in get_json(world.store, "actors/a5.json")["last_error"]
    assert get_json(world.store, "current.json")["fingerprint"] == world.actors["a1"].orch.xport.fingerprint


async def test_a_game_played_on_held_out_positions_is_dropped_with_reason_config(tmp_path, monkeypatch):
    store, clock = LocalStore(tmp_path / "bucket"), base.Clock()
    learner = node(
        tmp_path, monkeypatch, store, clock, "learner", "learner",
        held_out={"maps": ["tundra"], "seeds": [77], "civ_pairs": [["rome", "pers"]]},
    )  # fmt: skip
    learner.orch.xport.start()
    link = ActorLink(store, learner.cfg.transport, clock)
    specs = {
        "r1-a1-g000001": {"map": "tundra", "seed": 1, "seats": []},
        "r1-a1-g000002": {"map": "mainland", "seed": 77, "seats": []},
        "r1-a1-g000003": {"map": "mainland", "seed": 5, "ai_seed": 6, "seats": [{"civ": "pers"}, {"civ": "rome"}]},
        "r1-a1-g000004": {"map": "mainland", "seed": 5, "seats": [{"civ": "rome"}, {"civ": "athen"}]},
    }
    for game_id, spec in specs.items():
        folder = tmp_path / "made" / game_id
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps({"spec": {"game_id": game_id, **spec}}))
        link.ship("a1", folder, 0, "bot")
    await learner.orch.xport.run_all()
    assert learner.credit.games == ["r1-a1-g000004"] and learner.orch.counts["remote_config"] == 3
    assert learner.orch.xport.status(clock())["dropped"] == {"config": 3}


async def test_an_ingested_game_keeps_only_a_slim_record_when_the_run_does_not_sync_to_s3(tmp_path, monkeypatch):
    store, clock = LocalStore(tmp_path / "bucket"), base.Clock()
    learner = node(tmp_path, monkeypatch, store, clock, "learner", "learner", s3={})
    learner.orch.xport.start()
    folder = tmp_path / "made" / "r1-a1-g000001"
    folder.mkdir(parents=True)
    turn = {
        "began_ms": 0,
        "orders": [],
        "token_ids": [1] * 1000,
        "prompt_token_ids": [2] * 1000,
        "logprobs": [0.1] * 1000,
    }
    (folder / "results.json").write_text(
        json.dumps({"spec": {"game_id": folder.name}, "seats": {"1": {"turns": [turn]}}})
    )
    (folder / "decisions.log").write_text("x" * 5000)
    ActorLink(store, learner.cfg.transport, clock).ship("a1", folder, 0, "bot")
    await learner.orch.xport.run_all()
    kept = learner.root / "remote" / "a1" / folder.name
    assert [p.name for p in kept.iterdir()] == ["results.json"]
    slim = json.loads((kept / "results.json").read_text())
    assert slim["seats"]["1"]["turns"] == [{"began_ms": 0, "orders": []}]


async def test_with_s3_configured_the_remote_folder_syncs_first_and_is_slimmed_after_a_successful_sync(world, tmp_path):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    learner = world.learner
    await learner.orch.xport.run_all()
    folders = sorted((learner.root / "remote" / "a1").iterdir())
    assert folders and all(len(list(f.iterdir())) >= 1 for f in folders)
    os_utime = __import__("os").utime
    for f in folders:
        os_utime(f, (0, 1))  # older than the sync that is about to start
    learner.orch.syncer.last_start_ok = None
    assert await learner.orch.sync_now() is True
    [cmd] = [c for c in learner.syncs if "sync" in c][-1:]
    assert "remote/*" in cmd
    assert all([p.name for p in f.iterdir()] == ["results.json"] for f in folders)


async def test_the_single_box_sync_does_not_include_remote(tmp_path, monkeypatch):
    t = base.make(tmp_path, monkeypatch)
    await t.orch.sync_now()
    assert "remote/*" not in t.syncs[-1]


async def test_ingest_waits_while_the_disk_is_short(world):
    await bring_up(world)
    a = world.actors["a1"]
    await play_games(a, 1)
    await a.orch.xport.run_all()
    world.learner.orch.disk_paused = True
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games == []
    world.learner.orch.disk_paused = False
    await world.learner.orch.xport.run_all()
    assert world.learner.credit.games


async def test_the_learner_prunes_old_published_versions_but_not_hourly_or_champion_ones(world):
    learner = world.learner
    learner.orch.xport.start()
    learner.orch.hourly = {1: 2}
    for v in range(1, 12):
        base.publish(learner.root, v)
        await learner.orch.refresh_serving()
        await learner.orch.xport.publish_pass()
    versions = sorted(int(k.split("/")[1][1:]) for k in world.store.list("adapters") if k.endswith("READY"))
    assert versions == [2, 6, 7, 8, 9, 10, 11]  # lag 3 + 2 behind v11 is v6; v2 is the hourly one
    assert not [k for k in world.store.list("adapters") if "/v1/" in k]


def test_a_flushed_queue_ships_with_backoff_and_ends_with_a_final_heartbeat(tmp_path):
    store = LocalStore(tmp_path / "bucket")
    cfg = TransportConfig(enabled=True, uri=f"file://{store.root}", role="actor", actor_id="a1")
    root, clock = tmp_path / "root", base.Clock()
    queue = {}
    for n in range(2):
        folder = root / "games" / f"r1-a1-g00000{n}"
        folder.mkdir(parents=True)
        (folder / "results.json").write_text("{}")
        queue[folder.name] = {"version": 1, "opponent": "bot"}
    (root / "orchestrator").mkdir()
    (root / "orchestrator" / "ship.json").write_text(json.dumps({"queue": queue, "shipped": []}))
    link = ActorLink(store, cfg, clock)
    link.heartbeat("a1", {"time": clock(), "version": 3, "games_done": 9, "unshipped": 2, "final": False})
    failing = {"n": 3}
    original = store.put

    def put(key, src):
        if key.startswith("games/") and failing["n"] > 0:
            failing["n"] -= 1
            raise StoreError("network down")
        original(key, src)

    store.put = put
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock.t += seconds

    left = flush_queue(link, root, "a1", timeout_s=1800, keep_local=False, now=clock, sleep=sleep)
    assert left == 0 and slept == [10.0, 20.0] and len(list((store.root / "games" / "a1").glob("*.json"))) == 2
    beat = get_json(store, "actors/a1.json")
    assert beat["final"] is True and beat["unshipped"] == 0 and beat["games_done"] == 9
    assert json.loads((root / "orchestrator" / "ship.json").read_text())["queue"] == {}


def test_a_flush_that_gives_up_reports_what_is_left_and_exits_nonzero(tmp_path, capsys):
    from agentenv_zeroad.train import transport

    uri = f"file://{tmp_path / 'bucket'}"
    toml = tmp_path / "actor.toml"
    root = tmp_path / "run1-a1"
    toml.write_text(
        f'run_id = "r1"\nroot = "{root}"\nbudget_usd = 10\nbox_usd_per_hour = 1\nduration_hours = 1\n'
        f'servers = {{ urls = ["http://h:1"], adapter_dir = "/train/run1-a1/adapters" }}\n'
        f'[transport]\nenabled = true\nuri = "{uri}"\nrole = "actor"\nactor_id = "a1"\n'
    )
    folder = root / "games" / "r1-a1-g000000"
    folder.mkdir(parents=True)
    (root / "orchestrator").mkdir()
    (root / "orchestrator" / "ship.json").write_text(
        json.dumps({"queue": {folder.name: {"version": 0, "opponent": "bot"}}, "shipped": []})
    )  # no results.json: it can never ship
    assert transport.main(["flush", "--config", str(toml), "--timeout-minutes", "0.01"]) == 1
    assert "1 games left unshipped" in capsys.readouterr().out
    beat = get_json(LocalStore(tmp_path / "bucket"), "actors/a1.json")
    assert beat["final"] is True and beat["unshipped"] == 1 and "r1-a1-g000000" in beat["last_error"]


async def test_one_failed_pass_while_draining_does_not_end_the_shipping_and_the_heartbeats(world):
    await bring_up(world)
    a = world.actors["a1"]
    a.play.hold()
    a.orch.fill()
    await base.until(lambda: len(a.play.specs) == 2)
    calls = {"n": 0}
    original = a.orch.xport.beat_pass

    async def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise StoreError("one bad upload")
        await original(**kw)

    a.orch.xport.beat_pass = flaky
    drain = asyncio.create_task(a.orch._drain(5))
    await base.until(lambda: calls["n"] >= 3)
    a.play.release()
    await drain
    assert a.orch.counts["transport_error_keep"] == 1


def test_the_human_stop_names_the_running_epoch_and_can_ask_for_a_drain(world, capsys):
    from agentenv_zeroad.train import transport

    uri = f"file://{world.store.root}"
    put_json(world.store, "current.json", {"version": 1, "published_at": 0, "learner_alive_at": 0, "epoch": "E1"})
    assert transport.main(["stop", "--uri", uri, "--reason", "enough", "--drain"]) == 0
    stop = get_json(world.store, "stop.json")
    assert (stop["epoch"], stop["by"], stop["drain"]) == ("E1", "human", True)


def test_the_report_counts_actor_games_from_the_learners_remote_folder_and_panels_stay_as_they_are(tmp_path):
    from agentenv_zeroad.train import report
    from tests.train_fakes import results

    for folder, game in ((tmp_path / "games" / "g1", "g1"), (tmp_path / "remote" / "a1" / "g2", "g2")):
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(results(game)))
    local, _ = report.read_games(tmp_path / "games", None)
    both, _ = report.read_games(tmp_path / "games", None, tmp_path / "remote")
    assert [g.game_id for g in local] == ["g1"] and sorted(g.game_id for g in both) == ["g1", "g2"]
