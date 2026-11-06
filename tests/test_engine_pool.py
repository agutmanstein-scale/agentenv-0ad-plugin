"""The engine pool and what it relies on: ports it picks itself, one reap at start, an own HOME and replay folder per
engine, restart of an engine whose game failed, and the raw step."""

import asyncio
import socket

import httpx
import pytest

from agentenv_zeroad import engine as engine_module
from agentenv_zeroad import pool as pool_module
from agentenv_zeroad.engine import Engine, EngineError, EngineOptions, LocalLauncher, free_ports
from agentenv_zeroad.pool import EnginePool
from tests.zeroad_fakes import FakeStarter

pytestmark = pytest.mark.anyio


def test_free_ports_are_distinct_free_and_inside_the_range():
    ports = free_ports(5, start=20_000, stop=20_100)
    assert len(set(ports)) == 5 and all(20_000 <= p < 20_100 for p in ports)
    for port in ports:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def test_free_ports_skip_ports_in_use_and_fail_when_the_range_runs_out():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        taken = busy.getsockname()[1]
        assert taken not in free_ports(3, start=taken, stop=taken + 10)
        with pytest.raises(EngineError, match="found only"):
            free_ports(3, start=taken, stop=taken + 2)


async def test_the_pool_picks_a_distinct_port_for_every_engine_before_starting_any():
    starter = FakeStarter()
    async with EnginePool(4, start_one=starter, base_port=21_000, mode="local") as pool:
        ports = [r.options.port for r in starter.started]
        assert len(set(ports)) == 4 and all(p >= 21_000 for p in ports)
        assert pool.size == 4


async def test_the_pool_reaps_dead_owners_once_at_its_start_not_for_each_engine_or_game(monkeypatch):
    reaped = []

    async def reap():
        reaped.append(1)
        return []

    monkeypatch.setattr(pool_module, "reap_orphans", reap)
    starter = FakeStarter()
    async with EnginePool(3, start_one=starter, mode="docker", base_port=21_100) as pool:
        for _ in range(4):
            async with pool.lease():
                pass
    assert reaped == [1]


async def test_a_pool_of_docker_engines_starts_its_launchers_without_a_reap_of_their_own(monkeypatch):
    seen = {}

    async def fake_start(options, **kwargs):
        seen.update(kwargs, port=options.port)
        return "running"

    monkeypatch.setattr(pool_module, "start_engine", fake_start)
    pool = EnginePool(1, mode="docker", image="img:1", base_port=21_200)
    assert await pool._start_engine(EngineOptions(port=21_200), None) == "running"
    assert seen["docker"] == {"reap": False} and seen["image"] == "img:1" and seen["mode"] == "docker"


async def test_a_docker_launcher_started_with_reap_off_does_not_look_for_orphans(monkeypatch):
    calls = []

    async def reap():
        calls.append("reap")
        return []

    async def run(cmd):
        calls.append(cmd[1])
        return 0, ""

    monkeypatch.setattr(engine_module, "reap_orphans", reap)
    monkeypatch.setattr(engine_module, "_run", run)
    await engine_module.DockerLauncher(EngineOptions(port=6111), reap=False).start()
    await engine_module.DockerLauncher(EngineOptions(port=6112)).start()
    assert calls == ["run", "reap", "run"]


async def test_a_local_engine_gets_its_own_home_under_the_root_and_a_docker_engine_none(tmp_path):
    starter = FakeStarter()
    async with EnginePool(3, start_one=starter, mode="local", home_root=tmp_path, base_port=21_300):
        homes = [r.home for r in starter.started]
    assert homes == [tmp_path / f"engine-{i}" for i in range(3)]
    docker = FakeStarter()
    async with EnginePool(2, start_one=docker, mode="docker", home_root=tmp_path, base_port=21_400):
        pass
    assert [r.home for r in docker.started] == [None, None]


async def test_a_local_pool_without_a_home_root_gives_each_engine_a_temporary_home_and_removes_it_at_close():
    starter = FakeStarter()
    async with EnginePool(2, start_one=starter, mode="local", base_port=21_350) as pool:
        homes = [r.home for r in starter.started]
        assert all(home is not None and home.parent == pool.home_root for home in homes) and len(set(homes)) == 2
        root = pool.home_root
        assert root is not None and root.is_dir()
    assert not root.exists()


async def test_a_local_launcher_with_the_shared_home_refuses_to_delete_its_replays(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    replays = tmp_path / ".local/share/0ad/replays"
    replays.mkdir(parents=True)
    with pytest.raises(EngineError, match="shared HOME"):
        await LocalLauncher(EngineOptions()).clear_replays()
    assert replays.is_dir()


async def test_a_pool_scans_for_ports_only_below_its_stop():
    starter = FakeStarter()
    async with EnginePool(2, start_one=starter, base_port=24_000, port_stop=24_010, mode="local"):
        assert all(24_000 <= r.options.port < 24_010 for r in starter.started)
    with pytest.raises(EngineError, match="found only"):
        await EnginePool(3, start_one=starter, base_port=24_100, port_stop=24_102, mode="local").start()


async def test_a_local_launcher_copies_the_replay_of_its_own_home_and_never_another_engines(tmp_path):
    def write(home, text):
        folder = home / ".local/share/0ad/replays/2026-11-05/run"
        folder.mkdir(parents=True)
        (folder / "commands.txt").write_text(text)

    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "game a")
    write(b, "game b")
    launcher = LocalLauncher(EngineOptions(), home=a)
    assert (await launcher.copy_replay(tmp_path / "out.txt")).read_text() == "game a"
    await launcher.clear_replays()
    assert await launcher.copy_replay(tmp_path / "out2.txt") is None
    assert (b / ".local/share/0ad/replays").exists()


async def test_a_local_launcher_starts_its_engine_with_the_home_in_its_environment(tmp_path, monkeypatch):
    seen = {}

    async def fake_exec(*cmd, env, **kwargs):
        seen["env"] = env
        raise RuntimeError("stop here")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    home = tmp_path / "h"
    with pytest.raises(RuntimeError):
        await LocalLauncher(EngineOptions(), binary="/bin/true", home=home).start()
    assert seen["env"]["HOME"] == str(home) and home.is_dir()


async def test_a_lease_waits_for_a_free_engine_and_hands_out_each_engine_to_one_game_at_a_time():
    starter = FakeStarter()
    async with EnginePool(2, start_one=starter, mode="local", base_port=21_500) as pool:
        held = []
        release = asyncio.Event()

        async def game(i):
            async with pool.lease() as handle:
                held.append((i, handle.port))
                await release.wait()

        tasks = [asyncio.create_task(game(i)) for i in range(3)]
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(held) == 2 and len({port for _, port in held}) == 2
        release.set()
        await asyncio.gather(*tasks)
        assert len(held) == 3


async def test_a_leased_engine_has_its_old_replays_cleared_and_only_gives_a_replay_when_asked_for(tmp_path):
    starter = FakeStarter()
    async with EnginePool(1, start_one=starter, mode="local", base_port=21_600) as pool:
        async with pool.lease() as handle:
            assert starter.started[0].cleared == 1
            assert await handle.copy_replay(tmp_path / "r.txt") is None
    starter = FakeStarter()
    async with EnginePool(1, start_one=starter, mode="local", keep_replays=True, base_port=21_700) as pool:
        async with pool.lease() as handle:
            assert (await handle.copy_replay(tmp_path / "r.txt")).read_text().startswith("{")
            await handle.close()  # a match closes its handle when it ends; the engine stays leased
            assert pool._free.qsize() == 0
        assert pool._free.qsize() == 1


async def test_an_engine_marked_broken_is_restarted_on_its_port_before_its_next_game():
    starter = FakeStarter()
    async with EnginePool(1, start_one=starter, mode="local", base_port=21_800) as pool:
        async with pool.lease() as handle:
            handle.mark_broken()
        first = starter.started[0]
        async with pool.lease() as again:
            assert again.engine is starter.started[1].engine and again.engine is not first.engine
        assert first.closed == 1 and starter.started[1].options.port == first.options.port
        async with pool.lease() as third:
            assert third.engine is starter.started[1].engine and len(starter.started) == 2


async def test_a_lease_whose_restart_fails_leaves_the_engine_broken_so_the_next_lease_restarts_it_again():
    starter = FakeStarter()
    async with EnginePool(1, start_one=starter, mode="local", base_port=21_950) as pool:
        async with pool.lease() as handle:
            handle.mark_broken()
        good = pool._start_one

        async def failing(options, home):
            raise EngineError("no port")

        pool._start_one = failing
        for _ in range(2):
            with pytest.raises(EngineError):
                async with pool.lease():
                    pass
        pool._start_one = good
        async with pool.lease() as handle:
            assert handle.engine is starter.started[-1].engine and len(starter.started) == 2
        async with pool.lease() as again:
            assert again.engine is starter.started[-1].engine and len(starter.started) == 2


async def test_a_failed_restart_surfaces_in_that_lease_and_the_engine_stays_in_the_pool():
    starter = FakeStarter()
    async with EnginePool(1, start_one=starter, mode="local", base_port=21_900) as pool:
        async with pool.lease() as handle:
            handle.mark_broken()
        good = pool._start_one

        async def failing(options, home):
            raise EngineError("no port")

        pool._start_one = failing
        with pytest.raises(EngineError):
            async with pool.lease():
                pass
        pool._start_one = good
        async with pool.lease() as handle:
            assert handle.engine is starter.started[-1].engine
        assert pool._free.qsize() == 1


async def test_a_pool_whose_engine_fails_to_start_closes_the_others_and_raises():
    starter = FakeStarter()

    async def flaky(options, home):
        if len(starter.started) == 2:
            raise EngineError("did not open")
        return await starter(options, home)

    pool = EnginePool(3, start_one=flaky, mode="local", base_port=22_000)
    with pytest.raises(EngineError, match="did not open"):
        await pool.start()
    assert [r.closed for r in starter.started] == [1, 1]


async def test_closing_the_pool_closes_every_engine():
    starter = FakeStarter()
    pool = EnginePool(3, start_one=starter, mode="local", base_port=22_100)
    await pool.start()
    await pool.close()
    assert [r.closed for r in starter.started] == [1, 1, 1]


def test_a_pool_needs_at_least_one_engine():
    with pytest.raises(ValueError):
        EnginePool(0)


async def test_a_raw_step_posts_the_commands_and_never_decodes_the_reply():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.content.decode()))
        return httpx.Response(200, text="this is not json {")

    engine = Engine("http://engine:6000", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await engine.step_raw({1: [{"type": "stop", "entities": [9]}]}) is None
    assert seen == [("/step", '1;{"type":"stop","entities":[9]}')]
    with pytest.raises(ValueError):
        await engine.step()


# ---- the image's patches: what the pool refuses and what it turns on ------------------------------------------------
def fake_image_patches(monkeypatch, result):
    async def inspect(image):
        return result

    monkeypatch.setattr(pool_module, "image_patches", inspect)


async def started_docker_pool(monkeypatch, **options):
    async def no_reap():
        return []

    monkeypatch.setattr(pool_module, "reap_orphans", no_reap)
    monkeypatch.setattr(pool_module, "free_ports", lambda count, start, stop: [21_700 + i for i in range(count)])
    seen = []

    async def fake_start(opts, **kwargs):
        seen.append(kwargs)
        raise EngineError("stop here")

    monkeypatch.setattr(pool_module, "start_engine", fake_start)
    pool = EnginePool(1, mode="docker", **options)
    with pytest.raises(EngineError, match="stop here"):
        await pool.start()
    return pool, seen


async def test_a_pool_refuses_an_image_docker_does_not_have(monkeypatch):
    fake_image_patches(monkeypatch, None)
    with pytest.raises(EngineError, match="not built here"):
        await EnginePool(1, mode="docker", image="img:1").start()


async def test_a_pool_refuses_an_image_without_the_patches_it_requires(monkeypatch):
    fake_image_patches(monkeypatch, ("rl-realtime",))  # an rt image from before the idle patch
    with pytest.raises(EngineError, match="lacks the patches rl-idle-wait") as e:
        await EnginePool(
            1, mode="docker", image=pool_module.TRAIN_IMAGE, require_patches=pool_module.TRAIN_PATCHES
        ).start()
    assert "REBUILD=1" in str(e.value)
    fake_image_patches(monkeypatch, ())  # the PPA image under any tag
    with pytest.raises(EngineError, match="lacks the patches rl-realtime, rl-idle-wait"):
        await EnginePool(1, mode="docker", require_patches=pool_module.TRAIN_PATCHES).start()


async def test_an_image_with_the_idle_patch_gets_step_without_state_and_the_ppa_image_does_not(monkeypatch):
    fake_image_patches(monkeypatch, pool_module.TRAIN_PATCHES)
    pool, seen = await started_docker_pool(monkeypatch, require_patches=pool_module.TRAIN_PATCHES)
    assert pool.no_state_step is True and seen[0]["no_state_step"] is True
    fake_image_patches(monkeypatch, ())
    pool, seen = await started_docker_pool(monkeypatch)  # a comparison run on the PPA image requires nothing
    assert pool.no_state_step is False and seen[0]["no_state_step"] is False


async def test_a_pool_with_its_own_starter_does_not_ask_docker_about_the_image(monkeypatch):
    def forbidden(image):
        raise AssertionError("docker was asked")

    monkeypatch.setattr(pool_module, "image_patches", forbidden)
    async with EnginePool(1, mode="local", start_one=FakeStarter(), require_patches=pool_module.TRAIN_PATCHES):
        pass


async def test_step_raw_asks_for_no_state_only_on_an_engine_built_for_it():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text="{}")

    for no_state in (True, False):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        engine = Engine("http://e:6000", client=client, no_state_step=no_state)
        await engine.step_raw({1: [{"type": "stop", "entities": [1]}]})
        await engine.step({})
    assert seen == ["http://e:6000/step?state=0", "http://e:6000/step", "http://e:6000/step", "http://e:6000/step"]


async def test_image_patches_reads_the_label_and_tells_missing_from_unlabelled(monkeypatch):
    answers = iter([(0, "rl-realtime rl-idle-wait\n"), (0, "\n"), (0, "<no value>\n"), (1, "no such image")])

    async def run(cmd):
        assert cmd[:3] == ["docker", "image", "inspect"] and "zeroad.patches" in cmd[4]
        return next(answers)

    monkeypatch.setattr(engine_module, "_run", run)
    assert await engine_module.image_patches("i") == ("rl-realtime", "rl-idle-wait")
    assert await engine_module.image_patches("i") == ()
    assert await engine_module.image_patches("i") == ()
    assert await engine_module.image_patches("i") is None
