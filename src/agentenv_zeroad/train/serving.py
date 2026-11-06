"""The vLLM instances a training run plays on, and the adapter versions they hold.

Two vLLM servers (TP=1, one GPU each) serve the base model and, through runtime LoRA loading, one adapter per policy
version: model `local/qwen-base` is version 0, `local/policy-v7` is adapter `ROOT/adapters/v7`, which the learner
publishes atomically (a `v7` directory is always complete). `Fleet` talks to every instance:

- `check()` reads each one's /health and, when it is up, the models it serves. An instance that comes back with fewer
  adapters than it had (a restarted container forgets every runtime-loaded adapter) shows up in the same read;
- `reconcile(wanted)` makes each up instance serve exactly the wanted adapter versions: it loads the missing ones and
  unloads the policy versions nobody wants. Hot-loading a new version, unloading a version no game pins any more and
  re-registering the resident set after a restart are all this one call;
- `serving(version)` lists the up instances that hold a version, which is where a game pinned to it may play.

Every instance also keeps the outages `check()` saw (`Instance.outages`: from the last time it was seen up to the check
that found it back, open while it is down), which is what tells a model request that failed because of the server from
one that failed on its own. A served adapter whose `root` is not where this run's adapter lives (a vLLM that was given
another run's folder, or the wrong mount) is not counted as loaded: it is unloaded and loaded again from the right path.
`Fleet.load_failures` counts consecutive failed loads of one adapter into one instance, for the orchestrator's alarm.

`Adapters` is the run's bookkeeping: the versions on disk, the games pinned to each (a game pins the version current
at its start for its whole length) and so which versions must stay loaded: the newest, the current one, every pinned
one and the configured snapshots.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agentenv_zeroad.pilot.games import version_model
from agentenv_zeroad.train.config import ServerConfig

log = logging.getLogger(__name__)

VERSION_DIR = re.compile(r"^v(\d+)$")
POLICY_MODEL = re.compile(r"^local/policy-v(\d+)$")
LOAD_TIMEOUT_S = 300.0


@dataclass
class Instance:
    url: str
    up: bool = False
    loaded: set[int] = field(default_factory=set)  # adapter versions it served at the last check
    misrooted: set[int] = field(default_factory=set)  # served, but from a path other than this run's
    last_up: float | None = None  # when a check last found it up
    down_polls: int = 0  # consecutive checks that found it down
    outages: list[list[float | None]] = field(default_factory=list)  # [start, end or None while still down]

    def overlaps_outage(self, when: float, margin_s: float = 0.0) -> bool:
        """Whether `when` falls in an outage seen so far (an open one runs to now)."""
        return any(start - margin_s <= when and (end is None or when <= end + margin_s) for start, end in self.outages)


class Fleet:
    def __init__(self, cfg: ServerConfig, key: str, http: httpx.AsyncClient):
        self.cfg = cfg
        self.instances = [Instance(url.rstrip("/")) for url in cfg.urls]
        self._headers = {"Authorization": f"Bearer {key}"}
        self._http = http
        self.load_failures: Counter[tuple[str, int]] = Counter()  # (url, version) -> consecutive failed loads

    def adapter_path(self, version: int) -> str:
        return f"{self.cfg.adapter_dir}/v{version}"

    async def check(self, now: float | None = None) -> list[Instance]:
        """Refresh every instance's health and served adapters; returns those whose state changed (up or down, or a
        different adapter set). `now` stamps the outages."""
        now = time.time() if now is None else now
        changed = []
        for inst in self.instances:
            was = (inst.up, set(inst.loaded))
            inst.up, inst.loaded, inst.misrooted = await self._read(inst)
            self._track(inst, was[0], now)
            if (inst.up, inst.loaded) != was:
                changed.append(inst)
        return changed

    @staticmethod
    def _track(inst: Instance, was_up: bool, now: float) -> None:
        if inst.up:
            inst.down_polls = 0
            if inst.outages and inst.outages[-1][1] is None:
                inst.outages[-1][1] = now
            inst.last_up = now
            return
        inst.down_polls += 1
        if not inst.outages or inst.outages[-1][1] is not None:
            inst.outages.append([inst.last_up if inst.last_up is not None else now, None])
            del inst.outages[:-50]

    async def _read(self, inst: Instance) -> tuple[bool, set[int], set[int]]:
        try:
            health = await self._http.get(f"{inst.url}/health", timeout=self.cfg.health_timeout_s)
            if health.status_code != 200:
                return False, set(), set()
            models = await self._http.get(
                f"{inst.url}/v1/models", headers=self._headers, timeout=self.cfg.health_timeout_s
            )
            models.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("vLLM %s is not answering: %s", inst.url, type(e).__name__)
            return False, set(), set()
        loaded, misrooted = set(), set()
        for entry in models.json().get("data", []):
            found = POLICY_MODEL.match(entry["id"])
            if not found:
                continue
            version = int(found.group(1))
            root = entry.get("root")
            if root is not None and str(root).rstrip("/") != self.adapter_path(version):
                log.error("vLLM %s serves v%d from %s, not %s", inst.url, version, root, self.adapter_path(version))
                misrooted.add(version)
            else:
                loaded.add(version)
        return True, loaded, misrooted

    async def reconcile(self, wanted: set[int]) -> None:
        """Load the wanted adapter versions every up instance lacks and unload the policy versions it has but nobody
        wants. A failure on one instance is logged and tried again at the next call."""
        for key in [k for k in self.load_failures if k[1] not in wanted]:
            del self.load_failures[key]
        for inst in self.instances:
            if not inst.up:
                continue
            for version in sorted(inst.misrooted):
                if await self._unload(inst, version):
                    inst.misrooted.discard(version)
            for version in sorted(wanted - inst.loaded - {0}):
                if await self._load(inst, version):
                    inst.loaded.add(version)
                    self.load_failures.pop((inst.url, version), None)
                else:
                    self.load_failures[(inst.url, version)] += 1
            for version in sorted(inst.loaded - wanted):
                if await self._unload(inst, version):
                    inst.loaded.discard(version)

    async def _load(self, inst: Instance, version: int) -> bool:
        body = {"lora_name": version_model(version), "lora_path": self.adapter_path(version)}
        try:
            r = await self._http.post(
                f"{inst.url}/v1/load_lora_adapter", json=body, headers=self._headers, timeout=LOAD_TIMEOUT_S
            )
        except httpx.HTTPError as e:
            log.warning("loading v%d into %s failed: %s", version, inst.url, type(e).__name__)
            return False
        if r.status_code == 200 or "already" in r.text:
            return True
        log.warning("%s refused adapter v%d: %d %s", inst.url, version, r.status_code, r.text[:200])
        return False

    async def _unload(self, inst: Instance, version: int) -> bool:
        body = {"lora_name": version_model(version)}
        try:
            r = await self._http.post(
                f"{inst.url}/v1/unload_lora_adapter", json=body, headers=self._headers, timeout=LOAD_TIMEOUT_S
            )
        except httpx.HTTPError as e:
            log.warning("unloading v%d from %s failed: %s", version, inst.url, type(e).__name__)
            return False
        return r.status_code == 200 or r.status_code == 404

    def worst_load_failures(self) -> int:
        """The most consecutive failed loads of any one adapter into any one instance."""
        return max(self.load_failures.values(), default=0)

    def serving(self, version: int) -> list[Instance]:
        """The up instances a game pinned to `version` may use."""
        return [i for i in self.instances if i.up and (version == 0 or version in i.loaded)]

    def newest_everywhere(self, on_disk: Iterable[int]) -> int:
        """The newest on-disk version every up instance holds, 0 when there is none (or no instance is up)."""
        up = [i for i in self.instances if i.up]
        shared = set.intersection(*(i.loaded for i in up)) if up else set()
        return max((v for v in on_disk if v in shared), default=0)


class Adapters:
    """The adapter versions on disk and which games pin which."""

    def __init__(self, root: Path, keep: Iterable[int] = ()):
        self.dir = root / "adapters"
        self.keep = set(keep)
        self.pins: Counter[int] = Counter()

    def on_disk(self) -> list[int]:
        """Published versions, oldest first. The learner renames a finished directory into place, so none is partial."""
        if not self.dir.is_dir():
            return []
        found = (VERSION_DIR.match(p.name) for p in self.dir.iterdir() if p.is_dir())
        return sorted(int(m.group(1)) for m in found if m)

    def pin(self, version: int) -> None:
        self.pins[version] += 1

    def release(self, version: int) -> None:
        self.pins[version] -= 1
        if self.pins[version] <= 0:
            del self.pins[version]

    def wanted(self, *extra: int) -> set[int]:
        """The adapter versions every instance must serve: `extra` (the newest on disk, the current one), the pinned
        ones and the configured snapshots that exist. Version 0 is the base model, never an adapter."""
        return ({*extra} | set(self.pins) | (self.keep & set(self.on_disk()))) - {0}
