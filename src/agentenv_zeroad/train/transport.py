"""Several boxes, one learner: rollout actors feed the learner through S3 (`[transport]` in the run config, off by
default).

The boxes are SSM-only with no inbound ports and may sit in different subnets or regions, so they do not talk to each
other. They meet in one S3 prefix (the transport URI), which each reaches with its instance role. A box that dies
halfway through anything leaves objects the other side ignores, and nothing needs VPC networking. Layout under the URI:

    adapters/v{N}/...  adapters/v{N}/READY   the learner box publishes each new current version; READY lists every
                                             file with its size and sha256 and is written last
    current.json                             {version, published_at, learner_alive_at}: which version actors should
                                             serve, and the learner's pulse (rewritten every poll)
    games/{actor}/{game}.tgz  .json          a finished game's folder and, written after it, its manifest (version
                                             pinned, opponent, size, sha256)
    actors/{actor}.json                      an actor's heartbeat, every poll: the version it serves, games in flight
                                             and done, box hours and rate, frontier spend, its last error
    stop.json                                the learner's (or a human's) signal for actors to drain, upload, power off

Every multi-object item has its marker object written last (READY, the game's .json) and readers ignore an item without
one, so a half-uploaded adapter or game is never loaded or credited; a reader also checks every sha256, so a truncated
or corrupt object is refused rather than trained on.

`Store` is the one interface to the prefix: `LocalStore` (a directory, for tests) and `S3Store` (`aws s3` subprocess
calls with the instance role, like `sync.py`, with an injectable runner).

The learner side (`LearnerSide`, inside the orchestrator when `role = "learner"`) publishes each new current version off
the main loop, ingests the actors' games through the same credit path as local ones (so the per-opponent baselines,
strata centring and critic stay single and global), drops games pinned to a version more than `max_version_lag`
behind, remembers what it has ingested in ROOT/orchestrator/ingested.json, reads the heartbeats into status.json's
`actors` block and the budget (every actor's box hours at its rate plus its frontier spend count toward `budget_usd`;
an actor that went quiet without a final heartbeat is counted for at most `actor_stale_minutes` more), and writes
stop.json when it stops. The actor side (`ActorSide`, `role = "actor"`) pulls each new version into ROOT/adapters
atomically, plays the usual game loop with no learner, credit, inbox or panels, ships every finished game with retries
and backoff, beats, and stops on a stop.json written after it started, on a stale learner pulse, or on its own budget
or duration.

`python -m agentenv_zeroad.train.transport status|stop --uri URI` reads the prefix or writes stop.json by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agentenv_zeroad.train import budget
from agentenv_zeroad.train.config import TransportConfig

if TYPE_CHECKING:
    from agentenv_zeroad.train.orchestrate import Orchestrator
    from agentenv_zeroad.train.spec import GameSpec

log = logging.getLogger(__name__)

CURRENT, STOP, READY = "current.json", "stop.json", "READY"
S3_TIMEOUT_S = 900.0
MAX_BUNDLES_PER_PASS = 16  # games ingested per pass: a long backlog must not starve the other passes
BAD_BUNDLE_TRIES = 3  # downloads that fail their sha256 before a game is rejected for good
SHIP_BACKOFF_S = (5.0, 300.0)  # first retry delay and its cap, doubling
SHIP_ATTEMPTS = 3  # the final flush's tries per game
TASK_JOIN_S = 120.0


class StoreError(Exception):
    """The store failed (a network or permission error), as opposed to an object that is not there."""


class BundleError(Exception):
    """An object that arrived but is not what its marker says (a sha256 or size mismatch, an unsafe archive)."""


class Store(Protocol):
    def put(self, key: str, src: Path) -> None: ...
    def get(self, key: str, dst: Path) -> None:
        """Copy the object to `dst`; FileNotFoundError when it does not exist."""
        ...

    def list(self, prefix: str) -> list[str]:
        """Every key under `prefix` (a folder name, no trailing slash needed), sorted."""
        ...

    def tags(self, prefix: str) -> dict[str, str]:
        """Every key under `prefix` with a tag that changes when the object is rewritten (S3's ETag)."""
        ...

    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


def _check_key(key: str) -> str:
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"bad store key {key!r}")
    return key


class LocalStore:
    """A directory standing in for the bucket. Puts are atomic, as an S3 put is."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    def put(self, key: str, src: Path) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, path)

    def get(self, key: str, dst: Path) -> None:
        try:
            shutil.copyfile(self._path(key), dst)
        except FileNotFoundError:
            raise FileNotFoundError(key) from None

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix.strip("/")
        if base.is_file():
            return [prefix.strip("/")]
        found = (p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")) if base.is_dir() else ()
        return sorted(p.relative_to(self.root).as_posix() for p in found)

    def tags(self, prefix: str) -> dict[str, str]:
        out = {}
        for key in self.list(prefix):
            stat = self._path(key).stat()
            out[key] = f"{stat.st_size}-{stat.st_mtime_ns}"
        return out

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


Runner = Callable[[list[str]], tuple[int, str]]


def run_aws(cmd: list[str]) -> tuple[int, str]:
    """(exit code, stdout when it succeeded, else stderr and stdout)."""
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=S3_TIMEOUT_S, env={**os.environ, "AWS_PAGER": ""}, check=False
        )
    except subprocess.TimeoutExpired:
        return 124, "aws timed out"
    return done.returncode, done.stdout if done.returncode == 0 else done.stderr + done.stdout


class S3Store:
    """`aws s3` calls with the box's instance role. A failure other than a missing object raises StoreError."""

    def __init__(self, uri: str, run: Runner = run_aws):
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        self.bucket, self.prefix, self._run = bucket, prefix.strip("/"), run

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{_check_key(key)}" if self.prefix else _check_key(key)

    def _url(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full(key)}"

    def _ok(self, cmd: list[str]) -> str:
        code, out = self._run(cmd)
        if code != 0:
            raise StoreError(f"{' '.join(cmd[:4])} failed ({code}): {out.strip()[-300:]}")
        return out

    def put(self, key: str, src: Path) -> None:
        self._ok(["aws", "s3", "cp", str(src), self._url(key), "--only-show-errors"])

    def get(self, key: str, dst: Path) -> None:
        code, out = self._run(["aws", "s3", "cp", self._url(key), str(dst), "--only-show-errors"])
        if code == 0:
            return
        # The role may list only its own prefix, so S3 answers 403 rather than 404 for a key that is not there: only a
        # listing says whether the object exists, and a failure on one that does is an error.
        if not self.exists(key):
            raise FileNotFoundError(key)
        raise StoreError(f"get {key} failed ({code}): {out.strip()[-300:]}")

    def _pairs(self, prefix: str) -> list[tuple[str, str]]:
        out = self._ok(
            ["aws", "s3api", "list-objects-v2", "--bucket", self.bucket, "--prefix", prefix, "--query",
             "Contents[].[Key,ETag]", "--output", "json"]
        )  # fmt: skip
        strip = len(self.prefix) + 1 if self.prefix else 0
        return sorted((k[strip:], str(tag)) for k, tag in json.loads(out or "null") or [])

    def list(self, prefix: str) -> list[str]:
        return [k for k, _ in self._pairs(self._full(prefix.strip("/")) + "/")]

    def tags(self, prefix: str) -> dict[str, str]:
        return dict(self._pairs(self._full(prefix.strip("/")) + "/"))

    def exists(self, key: str) -> bool:
        return any(k == key for k, _ in self._pairs(self._full(key)))

    def delete(self, key: str) -> None:
        self._ok(["aws", "s3", "rm", self._url(key), "--only-show-errors"])


def open_store(uri: str) -> Store:
    return LocalStore(Path(uri.removeprefix("file://"))) if uri.startswith("file://") else S3Store(uri)


def put_json(store: Store, key: str, value: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "item.json"
        path.write_text(json.dumps(value))
        store.put(key, path)


def get_json(store: Store, key: str) -> Any:
    """The object's JSON, or None when it is missing or not valid JSON (a reader ignores what it cannot read)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "item.json"
        try:
            store.get(key, path)
            return json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_dir(store: Store, prefix: str, folder: Path) -> None:
    """Upload every file of `folder` under `prefix`, then READY (their sizes and sha256) last."""
    listing: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        listing[rel] = {"size": path.stat().st_size, "sha256": sha256_of(path)}
        store.put(f"{prefix}/{rel}", path)
    put_json(store, f"{prefix}/{READY}", {"files": listing})


def fetch_dir(store: Store, prefix: str, dest: Path) -> None:
    """Download a published folder into `dest` and check every file against READY. FileNotFoundError when READY is
    missing (not published yet); BundleError when a file is not what READY says."""
    ready = get_json(store, f"{prefix}/{READY}")
    if not isinstance(ready, dict) or not ready.get("files"):
        raise FileNotFoundError(f"{prefix}/{READY}")
    for rel, meta in ready["files"].items():
        target = dest / _check_key(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        store.get(f"{prefix}/{rel}", target)
        if target.stat().st_size != meta["size"] or sha256_of(target) != meta["sha256"]:
            raise BundleError(f"{prefix}/{rel} does not match its READY entry")


@dataclass(frozen=True)
class Current:
    version: int
    published_at: float
    learner_alive_at: float

    @classmethod
    def parse(cls, raw: Any) -> Current | None:
        try:
            return cls(int(raw["version"]), float(raw["published_at"]), float(raw["learner_alive_at"]))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Manifest:
    """A shipped game: what the learner needs to decide on it before downloading it."""

    actor_id: str
    game_id: str
    version: int
    opponent: str
    size: int
    sha256: str
    shipped_at: float

    @property
    def key(self) -> str:
        return f"{self.actor_id}/{self.game_id}"

    @classmethod
    def parse(cls, raw: Any) -> Manifest | None:
        try:
            return cls(
                str(raw["actor_id"]), str(raw["game_id"]), int(raw["version"]), str(raw.get("opponent", "")),
                int(raw["size"]), str(raw["sha256"]), float(raw.get("shipped_at", 0)),
            )  # fmt: skip
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


def opponent_of(spec: GameSpec) -> str:
    for seat in spec.seats:
        if not seat.trainable:
            return f"petra-{seat.difficulty}" if seat.kind == "petra" else "base" if seat.kind == "llm" else seat.kind
    return ""


def pinned_version(spec: GameSpec) -> int:
    for seat in spec.seats:
        if seat.trainable:
            tail = (seat.model or "").rsplit("-v", 1)[-1]
            return int(tail) if tail.isdigit() else 0
    return 0


def pack_game(folder: Path, out: Path) -> None:
    with tarfile.open(out, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)


def unpack_game(archive: Path, game_id: str, dest: Path) -> None:
    """Extract a game archive (one folder named `game_id`) into `dest`, refusing anything else in it."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if not parts or parts[0] != game_id or ".." in parts or Path(m.name).is_absolute():
                raise BundleError(f"unsafe path {m.name!r} in the archive of {game_id}")
            if not (m.isfile() or m.isdir()):
                raise BundleError(f"{m.name!r} in the archive of {game_id} is not a plain file")
        tar.extractall(dest, members=members, filter="data")


class LearnerLink:
    """The learner's blocking calls on the store (the orchestrator runs them in threads)."""

    def __init__(self, store: Store, cfg: TransportConfig, now: Callable[[], float] = time.time):
        self.store, self.cfg, self.now = store, cfg, now
        self._lock = threading.Lock()
        self.version = 0  # the newest version published (0: the base model, nothing to publish)
        self.published_at = now()

    def start(self) -> None:
        """A new learner process: drop the previous stop signal and continue from the published version."""
        self.store.delete(STOP)
        current = Current.parse(get_json(self.store, CURRENT))
        if current is not None:
            self.version, self.published_at = current.version, current.published_at
        self.touch()

    def touch(self) -> None:
        with self._lock:
            put_json(
                self.store, CURRENT,
                {"version": self.version, "published_at": self.published_at, "learner_alive_at": self.now()},
            )  # fmt: skip

    def publish(self, version: int, folder: Path) -> None:
        """Upload adapter `version` (a no-op when its READY is already there), then point current.json at it."""
        prefix = f"adapters/v{version}"
        if not self.store.exists(f"{prefix}/{READY}"):
            publish_dir(self.store, prefix, folder)
        with self._lock:
            self.version, self.published_at = max(self.version, version), self.now()
        self.touch()

    def heartbeats(self) -> dict[str, dict[str, Any]]:
        found = {}
        for key in self.store.list("actors"):
            beat = get_json(self.store, key)
            if isinstance(beat, dict) and beat.get("actor_id"):
                found[str(beat["actor_id"])] = beat
        return found

    def new_manifests(self, seen: set[str], limit: int = MAX_BUNDLES_PER_PASS) -> list[Manifest]:
        """Manifests of games not in `seen`, oldest first; a game with a .tgz but no manifest is not listed."""
        manifests = []
        for key in self.store.list("games"):
            if not key.endswith(".json") or key.count("/") != 2:
                continue
            _, actor, name = key.split("/")
            if f"{actor}/{name.removesuffix('.json')}" in seen:
                continue
            manifest = Manifest.parse(get_json(self.store, key))
            if manifest is not None and manifest.key == f"{actor}/{name.removesuffix('.json')}":
                manifests.append(manifest)
            if len(manifests) >= limit:
                break
        return sorted(manifests, key=lambda m: (m.shipped_at, m.key))

    def fetch_game(self, manifest: Manifest, remote_root: Path) -> Path:
        """Download, check and unpack a game into REMOTE_ROOT/ACTOR/GAME. BundleError on a sha256 or size mismatch."""
        final = remote_root / manifest.actor_id / manifest.game_id
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "game.tgz"
            self.store.get(f"games/{manifest.key}.tgz", archive)
            if archive.stat().st_size != manifest.size or sha256_of(archive) != manifest.sha256:
                raise BundleError(f"{manifest.key}: the archive does not match its manifest")
            staging = Path(tmp) / "unpacked"
            unpack_game(archive, manifest.game_id, staging)
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(final, ignore_errors=True)
            shutil.move(str(staging / manifest.game_id), str(final))
        return final

    def write_stop(self, reason: str, by: str = "learner") -> None:
        put_json(self.store, STOP, {"reason": reason, "by": by, "time": self.now()})


class ActorLink:
    """An actor's blocking calls on the store."""

    def __init__(self, store: Store, cfg: TransportConfig, now: Callable[[], float] = time.time):
        self.store, self.cfg, self.now = store, cfg, now

    def current(self) -> Current | None:
        return Current.parse(get_json(self.store, CURRENT))

    def stop(self) -> dict[str, Any] | None:
        raw = get_json(self.store, STOP)
        return raw if isinstance(raw, dict) else None

    def pull_adapter(self, version: int, adapters: Path) -> bool:
        """Download v{version} into a hidden folder of ROOT/adapters and rename it into place: a `v{N}` there is always
        complete, as the learner's own are. False when it is not published yet."""
        adapters.mkdir(parents=True, exist_ok=True)
        for old in adapters.glob(".dl-*"):
            shutil.rmtree(old, ignore_errors=True)
        tmp = adapters / f".dl-v{version}-{os.getpid()}"
        try:
            fetch_dir(self.store, f"adapters/v{version}", tmp)
            final = adapters / f"v{version}"
            if final.exists():
                return True
            os.rename(tmp, final)
            return True
        except FileNotFoundError:
            return False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def heartbeat(self, actor_id: str, payload: dict[str, Any]) -> None:
        put_json(self.store, f"actors/{actor_id}.json", {"actor_id": actor_id, **payload})

    def ship(self, actor_id: str, folder: Path, version: int, opponent: str) -> None:
        """Upload the game folder as a .tgz, then its manifest (the marker) last."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "game.tgz"
            pack_game(folder, archive)
            manifest = Manifest(
                actor_id, folder.name, version, opponent, archive.stat().st_size, sha256_of(archive), self.now()
            )
            self.store.put(f"games/{manifest.key}.tgz", archive)
        put_json(self.store, f"games/{manifest.key}.json", asdict(manifest))


Task = Callable[[], Awaitable[None]]


class _Side:
    """Background passes of one side: each runs in its own task so that a slow upload or download never delays a
    pulse, and a pass that raises is counted and logged, never fatal."""

    def __init__(self, host: Orchestrator):
        self.host = host
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self.closing = False

    def _passes(self) -> dict[str, Task]:
        raise NotImplementedError

    def kick(self) -> None:
        if self.closing:
            return
        for name, fn in self._passes().items():
            task = self._tasks.get(name)
            if task is None or task.done():
                self._tasks[name] = asyncio.create_task(self._guard(name, fn))

    async def _guard(self, name: str, fn: Task) -> None:
        try:
            await fn()
        except Exception:  # noqa: BLE001 - a transport pass that fails is tried again at the next poll
            self.host.counts[f"transport_error_{name}"] += 1
            log.exception("transport pass %s failed", name)

    async def run_all(self) -> None:
        """Every pass once, in order (what the tests drive; the loop kicks them as tasks)."""
        for name, fn in self._passes().items():
            await self._guard(name, fn)

    async def join(self) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=TASK_JOIN_S)


class LearnerSide(_Side):
    def __init__(self, host: Orchestrator, link: LearnerLink):
        super().__init__(host)
        self.link, self.cfg = link, host.cfg.transport
        self.remote = host.root / "remote"
        saved = host._read("ingested.json") or {}
        self.ingested: set[str] = set(saved.get("ingested", ()))
        self.dropped: dict[str, str] = dict(saved.get("dropped", {}))
        self.beats: dict[str, dict[str, Any]] = {}
        self._bad: Counter[str] = Counter()

    def _passes(self) -> dict[str, Task]:
        return {"publish": self.publish_pass, "live": self.liveness_pass, "ingest": self.ingest_pass}

    def start(self) -> None:
        self.link.start()

    async def publish_pass(self) -> None:
        version = self.host.current
        folder = self.host.adapters.dir / f"v{version}"
        if version <= self.link.version or not folder.is_dir():
            return
        await asyncio.to_thread(self.link.publish, version, folder)
        self.host.counts["published"] += 1
        log.info("published adapter v%d to the transport", version)

    async def liveness_pass(self) -> None:
        await asyncio.to_thread(self.link.touch)
        self.beats = await asyncio.to_thread(self.link.heartbeats)
        now = self.host.now()
        for actor, beat in self.beats.items():
            stale = self._stale(beat, now)
            self.host.alarm(
                f"actor:{actor}",
                f"actor {actor} has not reported for {self._age(beat, now) / 60:.0f} min" if stale else None,
            )

    def _age(self, beat: dict[str, Any], now: float) -> float:
        return max(0.0, now - float(beat.get("time", 0)))

    def _stale(self, beat: dict[str, Any], now: float) -> bool:
        return not beat.get("final") and self._age(beat, now) > self.cfg.actor_stale_minutes * 60

    async def ingest_pass(self) -> None:
        manifests = await asyncio.to_thread(self.link.new_manifests, self.ingested | set(self.dropped))
        for manifest in manifests:
            if self.closing or self.host.stop_reason is not None:
                return
            await self._ingest(manifest)

    def _drop(self, manifest: Manifest, reason: str) -> None:
        self.dropped[manifest.key] = reason
        self.host.counts[f"remote_{reason}"] += 1
        self._save()

    async def _ingest(self, manifest: Manifest) -> None:
        host = self.host
        if manifest.version < host.current - self.cfg.max_version_lag:
            self._drop(manifest, "stale")
            log.info("dropped %s: pinned v%d, current is v%d", manifest.key, manifest.version, host.current)
            return
        if (host.root / "inbox" / f"{manifest.game_id}.jsonl").exists():
            self.ingested.add(manifest.key)  # a crash between crediting and recording: it was credited
            self._save()
            return
        try:
            folder = await asyncio.to_thread(self.link.fetch_game, manifest, self.remote)
            results = json.loads(await asyncio.to_thread((folder / "results.json").read_text))
        except BundleError as e:
            self._bad[manifest.key] += 1
            log.error("%s (try %d of %d)", e, self._bad[manifest.key], BAD_BUNDLE_TRIES)
            if self._bad[manifest.key] >= BAD_BUNDLE_TRIES:
                self._drop(manifest, "rejected")
            return
        except (StoreError, FileNotFoundError):
            host.counts["transport_error_fetch"] += 1
            log.exception("fetching %s failed", manifest.key)
            return
        except (OSError, ValueError) as e:
            log.error("%s has no usable results.json: %s", manifest.key, e)
            self._drop(manifest, "rejected")
            return
        host._observe_supply(manifest.game_id, results)
        rows, error = await host._credit_id(manifest.game_id, results)
        host.counts["remote_failed" if error else "remote_credited"] += 1
        self.ingested.add(manifest.key)
        self._save()
        log.info(
            "credited %s from %s: %d rows%s", manifest.game_id, manifest.actor_id, rows, f" ({error})" if error else ""
        )

    def _save(self) -> None:
        self.host._write("ingested.json", {"ingested": sorted(self.ingested), "dropped": self.dropped})

    def spend_extra(self, now: float) -> tuple[float, float]:
        """(box dollars, frontier dollars) of every actor from its last heartbeat. One that went quiet without a final
        beat is counted for the time since too, but for at most the stale window: a box that vanished may still be
        billing, and a guard that stopped counting it would be wrong; one counted forever would stop the run."""
        box = model = 0.0
        for beat in self.beats.values():
            hours = float(beat.get("box_hours", 0))
            if not beat.get("final"):
                hours += min(self._age(beat, now), self.cfg.actor_stale_minutes * 60) / 3600
            box += hours * float(beat.get("box_usd_per_hour", 0))
            model += float(beat.get("model_usd", 0))
        return box, model

    def status(self, now: float) -> dict[str, Any]:
        members: dict[str, Any] = {}
        for actor, beat in sorted(self.beats.items()):
            hours = float(beat.get("run_hours", 0))
            version = int(beat.get("version", 0))
            members[actor] = {
                "state": "stopped" if beat.get("final") else "stale" if self._stale(beat, now) else "alive",
                "age_s": round(self._age(beat, now)),
                "version": version,
                "lag": self.host.current - version,
                "in_flight": beat.get("in_flight", 0),
                "games_done": beat.get("games_done", 0),
                "games_per_h": round(beat.get("games_done", 0) / hours, 1) if hours > 0.01 else None,
                "box_hours": beat.get("box_hours", 0),
                "box_usd": round(float(beat.get("box_hours", 0)) * float(beat.get("box_usd_per_hour", 0)), 2),
                "frontier_usd": beat.get("model_usd", 0),
                "last_error": beat.get("last_error"),
            }
        return {
            "published": self.link.version,
            "ingested": len(self.ingested),
            "dropped": dict(Counter(self.dropped.values())),
            "members": members,
        }

    async def close(self, reason: str) -> None:
        """Write stop.json (the actors drain, upload and power off), then let the passes in flight end."""
        self.closing = True
        try:
            await asyncio.to_thread(self.link.write_stop, reason)
        except Exception:  # noqa: BLE001 - the stop still goes on; the actors stop on the pulse that ends
            log.exception("writing stop.json failed")
        await self.join()


class ActorSide(_Side):
    def __init__(self, host: Orchestrator, link: ActorLink):
        super().__init__(host)
        self.link, self.cfg, self.actor_id = link, host.cfg.transport, host.cfg.transport.actor_id
        self.started = host.now()
        self.ready = False  # the learner's current.json has been read: games start only then
        self.learner_version = 0
        self.learner_alive_at: float | None = None
        self.last_error: str | None = None
        saved = host._read("ship.json") or {}
        self.queue: dict[str, dict[str, Any]] = dict(saved.get("queue", {}))
        self.shipped: set[str] = set(saved.get("shipped", ()))
        self._tries: Counter[str] = Counter()
        self._next: dict[str, float] = {}
        self._ship_lock = asyncio.Lock()

    def _passes(self) -> dict[str, Task]:
        return {"poll": self.poll_pass, "pull": self.pull_pass, "ship": self.ship_pass, "beat": self.beat_pass}

    def start(self) -> None:
        pass

    async def poll_pass(self) -> None:
        host, now = self.host, self.host.now()
        try:
            current = await asyncio.to_thread(self.link.current)
            stop = await asyncio.to_thread(self.link.stop)
        except StoreError as e:
            self.last_error = f"transport: {e}"
            host.counts["transport_error_poll"] += 1
            log.error("reading the transport failed: %s", e)
            current, stop = None, None
        else:
            self.last_error = None
        if current is not None:
            self.ready, self.learner_version, self.learner_alive_at = True, current.version, current.learner_alive_at
        if stop is not None and float(stop.get("time", 0)) >= self.started:
            host.request_stop(f"transport: stop requested by {stop.get('by', '?')}: {stop.get('reason', '')}")
        elif self.learner_alive_at is not None and now - self.learner_alive_at > self.cfg.learner_stale_minutes * 60:
            host.request_stop(f"transport: the learner's pulse is {(now - self.learner_alive_at) / 60:.0f} min old")
        elif self.learner_alive_at is None and now - self.started > self.cfg.learner_stale_minutes * 60:
            host.request_stop("transport: no learner has published a pulse")

    async def pull_pass(self) -> None:
        host = self.host
        have = host.adapters.on_disk()
        if not self.ready or self.learner_version <= (have[-1] if have else 0):
            return
        try:
            got = await asyncio.to_thread(self.link.pull_adapter, self.learner_version, host.adapters.dir)
        except (StoreError, BundleError, OSError) as e:
            self.last_error = f"adapter v{self.learner_version}: {e}"
            host.counts["transport_error_pull"] += 1
            log.error("pulling adapter v%d failed: %s", self.learner_version, e)
            return
        if got:
            host.counts["pulled"] += 1
            host._wake.set()
            log.info("pulled adapter v%d", self.learner_version)

    def enqueue(self, game_id: str, spec: GameSpec) -> None:
        self.queue[game_id] = {"version": pinned_version(spec), "opponent": opponent_of(spec)}
        self._save()

    def unshipped(self) -> int:
        return len(self.queue)

    async def ship_pass(self, *, force: bool = False) -> None:
        async with self._ship_lock:
            now = self.host.now()
            for game_id in list(self.queue):
                if not force and self._next.get(game_id, 0) > now:
                    continue
                if await self._ship(game_id):
                    continue
                self._tries[game_id] += 1
                delay = min(SHIP_BACKOFF_S[1], SHIP_BACKOFF_S[0] * 2 ** (self._tries[game_id] - 1))
                self._next[game_id] = now + delay

    async def _ship(self, game_id: str) -> bool:
        info = self.queue[game_id]
        folder = self.host.root / "games" / game_id
        try:
            if not (folder / "results.json").is_file():
                raise FileNotFoundError(f"{folder}/results.json")
            await asyncio.to_thread(self.link.ship, self.actor_id, folder, info["version"], info["opponent"])
        except (StoreError, OSError) as e:
            self.last_error = f"shipping {game_id}: {e}"
            self.host.counts["ship_failed"] += 1
            log.error("shipping %s failed: %s", game_id, e)
            return False
        del self.queue[game_id]
        self.shipped.add(game_id)
        self._save()
        self.host.counts["shipped"] += 1
        if self.host.cfg.s3_uri is None:
            shutil.rmtree(
                folder, ignore_errors=True
            )  # the bundle is the durable copy; a day of games would fill the disk
        return True

    async def flush(self) -> int:
        """The final upload: every queued game, retried, ignoring the backoff. Returns how many are still unshipped."""
        for attempt in range(SHIP_ATTEMPTS):
            await self.ship_pass(force=True)
            if not self.queue or attempt == SHIP_ATTEMPTS - 1:
                break
            await asyncio.sleep(self.host.cfg.sync_retry_s)
        return len(self.queue)

    def _save(self) -> None:
        self.host._write("ship.json", {"queue": self.queue, "shipped": sorted(self.shipped)})

    async def beat_pass(self, *, final: bool = False) -> None:
        host, now = self.host, self.host.now()
        spent = budget.spend(host.clock, now, host.cfg.box_usd_per_hour, host.cfg.ledger_path, host.boxtime)
        payload = {
            "time": now, "version": host.current, "in_flight": len(host.in_flight),
            "games_done": host.counts["train_finished"], "shipped": len(self.shipped), "unshipped": len(self.queue),
            "run_hours": round(host.clock.hours(now), 4), "box_hours": round(spent.box_hours, 4),
            "box_usd_per_hour": host.cfg.box_usd_per_hour, "model_usd": round(spent.model_usd, 4),
            "last_error": self.last_error or next(iter(host.alarms.values()), None),
            "stop_reason": host.stop_reason, "final": final,
        }  # fmt: skip
        await asyncio.to_thread(self.link.heartbeat, self.actor_id, payload)

    async def close(self, reason: str) -> None:
        """After the games have drained or been cancelled: ship what is left, then the final heartbeat."""
        self.closing = True
        await self.join()
        left = await self.flush()
        if left:
            log.error("%d games could not be shipped and stay on this box's disk", left)
            self.host.alarm("unshipped", f"{left} finished games could not be uploaded; the box stays up")
        with contextlib.suppress(Exception):
            await self.beat_pass(final=True)


def build_side(host: Orchestrator, store: Store) -> LearnerSide | ActorSide:
    cfg = host.cfg.transport
    if cfg.actor:
        return ActorSide(host, ActorLink(store, cfg, host.now))
    return LearnerSide(host, LearnerLink(store, cfg, host.now))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentenv_zeroad.train.transport", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, text in (
        ("status", "print current.json, stop.json and every actor's heartbeat"),
        ("stop", "write stop.json"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--uri", required=True, help="the run's transport URI ([transport] uri)")
        if name == "stop":
            p.add_argument("--reason", default="stopped by hand")
    args = parser.parse_args(argv)
    store = open_store(args.uri)
    if args.cmd == "stop":
        put_json(store, STOP, {"reason": args.reason, "by": "human", "time": time.time()})
        print("stop.json written: the actors drain, upload and power off")
        return 0
    now = time.time()
    print("current:", json.dumps(get_json(store, CURRENT)))
    print("stop:", json.dumps(get_json(store, STOP)))
    for key in store.list("actors"):
        beat = get_json(store, key) or {}
        age = now - float(beat.get("time", 0))
        error = f", error: {beat['last_error']}" if beat.get("last_error") else ""
        print(
            f"{beat.get('actor_id', key)}: v{beat.get('version')}, {beat.get('in_flight')} in flight, "
            f"{beat.get('games_done')} done, {beat.get('unshipped')} unshipped, {age:.0f} s ago"
            f"{', final' if beat.get('final') else ''}{error}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
