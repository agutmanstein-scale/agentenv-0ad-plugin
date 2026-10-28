"""Engine containers carry their owner process, and only containers whose owner is gone are reaped."""

import asyncio
import os
import socket

from agentenv_zeroad import engine
from agentenv_zeroad.engine import LABEL_HOST, LABEL_PID, LABEL_START, DockerLauncher, EngineOptions, owner_labels


def test_each_engine_container_is_labelled_with_its_owner_process():
    labels = owner_labels()
    assert labels[LABEL_PID] == str(os.getpid()) and labels[LABEL_START].isdigit()
    assert labels[LABEL_HOST] == socket.gethostname()
    cmd = DockerLauncher(EngineOptions(port=6111)).command()
    for key, value in labels.items():
        assert f"{key}={value}" in cmd


def test_only_this_hosts_engine_containers_with_a_dead_owner_are_removed(monkeypatch):
    me, host = owner_labels(), socket.gethostname()
    listed = "\n".join(
        [
            f"zeroad-engine-dead1111\t999999999\t1\t{host}",  # owner pid gone
            f"zeroad-engine-reused22\t{os.getpid()}\t1\t{host}",  # pid alive but another process (start differs)
            f"zeroad-engine-live3333\t{os.getpid()}\t{me[LABEL_START]}\t{host}",  # owner alive
            "zeroad-engine-other444\t999999999\t1\tsomewhere-else",  # another host's
            f"agentenv-registry\t999999999\t1\t{host}",  # not an engine container
        ]
    )
    calls = []

    async def fake_run(cmd):
        calls.append(cmd)
        return (0, listed) if cmd[:2] == ["docker", "ps"] else (0, "")

    monkeypatch.setattr(engine, "_run", fake_run)
    removed = asyncio.run(engine.reap_orphans())
    assert removed == ["zeroad-engine-dead1111", "zeroad-engine-reused22"]
    assert [c for c in calls if c[:2] == ["docker", "rm"]] == [
        ["docker", "rm", "-f", "zeroad-engine-dead1111"],
        ["docker", "rm", "-f", "zeroad-engine-reused22"],
    ]
    assert calls[0][:5] == ["docker", "ps", "-a", "--filter", f"label={LABEL_PID}"]


def test_the_launcher_reaps_orphans_before_it_starts_an_engine(monkeypatch):
    order = []

    async def fake_reap():
        order.append("reap")
        return []

    async def fake_run(cmd):
        order.append(cmd[1])
        return 0, ""

    monkeypatch.setattr(engine, "reap_orphans", fake_reap)
    monkeypatch.setattr(engine, "_run", fake_run)
    asyncio.run(DockerLauncher(EngineOptions(port=6111)).start())
    assert order == ["reap", "run"]


def test_nothing_is_reaped_where_the_owner_check_cannot_be_made(monkeypatch):
    """Without /proc (a macOS dev machine) every owner would look dead, live matches' engines included."""
    calls = []

    async def fake_run(cmd):
        calls.append(cmd)
        return 0, f"zeroad-engine-live3333\t{os.getpid()}\t\t{socket.gethostname()}"

    monkeypatch.setattr(engine, "_process_start", lambda pid: None)
    monkeypatch.setattr(engine, "_run", fake_run)
    assert asyncio.run(engine.reap_orphans()) == []
    assert not [c for c in calls if c[:2] == ["docker", "rm"]]
