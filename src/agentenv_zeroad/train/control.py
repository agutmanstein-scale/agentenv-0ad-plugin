"""The orchestrator's side of the learner's files: its health log (tripwires) and its control commands.

The learner (trainer/learner.py) appends one record per update to ROOT/health.jsonl, with `"tripwires": {"trips": [...],
"alarms": [...]}`, and acts on commands dropped in ROOT/control/. It only reports; this module decides.

`TripWatch.poll()` reads the records written since the last call and returns what to do:

- a record with trips is a trip. The first one rolls back to the last green checkpoint: the newest version whose own
  record had no trips and that is still in the learner's checkpoint ring. The learner's rollback also halves its
  learning rate and publishes the restored weights as a new version, so one `rollback` command is both halves of the
  response (a second `set_lr` would quarter the rate);
- until the learner reports `{"event": "rollback"}`, further trip records are the same episode (it keeps updating
  until it reads the command) and are not counted again; if the report never comes (the learner ignored the command)
  the next trip record after `rollback_wait_s` counts as a new trip;
- a trip within `window_s` of `limit - 1` earlier ones is not rolled back but paused: the run uploads and stops.

The watch survives an orchestrator restart. It keeps its read offset, the trip times, the episode it is waiting on and
its history in `state_path` (saved after every poll that read something); with no state file it replays the whole log
once, silently (`replay()`): the state it rebuilds is the state it would have, but it emits no action, so a restart
never repeats a rollback the learner already made and never pauses a run over trips from hours ago.

`ControlWriter` numbers commands so the learner (which reads them oldest name first) applies them in order, and renames
each into place so it never reads a half-written one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHECKPOINT = re.compile(r"^ckpt-(\d+)\.pt$")


class ControlWriter:
    def __init__(self, control: Path):
        self.dir = control

    def send(self, command: dict[str, Any]) -> Path:
        """Write `command` as the next numbered file (atomically) and return its path."""
        self.dir.mkdir(parents=True, exist_ok=True)
        name = f"{self._next():06d}-{command['op']}.json"
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(command))
        os.replace(tmp, self.dir / name)
        return self.dir / name

    def _next(self) -> int:
        numbers = [
            int(p.name.split("-", 1)[0])
            for folder in (self.dir, self.dir / "done")
            if folder.is_dir()
            for p in folder.glob("[0-9]*-*.json")
        ]
        return max(numbers, default=0) + 1

    def pending(self) -> list[str]:
        """Commands the learner hasn't read yet."""
        return sorted(p.name for p in self.dir.glob("*.json")) if self.dir.is_dir() else []


@dataclass(frozen=True)
class Action:
    kind: str  # "rollback" or "pause"
    version: int | None  # the checkpoint to restore (rollback only)
    trips: list[dict[str, Any]]
    time: float


@dataclass
class TripWatch:
    health: Path
    checkpoints: Path
    window_s: float = 3 * 3600.0
    limit: int = 2
    offset: int = 0
    rollback_wait_s: float = 900.0
    awaiting_since: float | None = None
    trip_times: list[float] = field(default_factory=list)
    history: list[tuple[int, bool]] = field(default_factory=list)  # (version, had trips), since the last rollback
    state_path: Path | None = None

    def __post_init__(self) -> None:
        if self.state_path is not None and self.state_path.exists():
            self._load()
        elif self.state_path is not None and self.health.exists():
            self.replay()

    def poll(self, now: float) -> list[Action]:
        actions = []
        before = self.offset
        for record in self._new_records():
            action = self._handle(record, now)
            if action is not None:
                actions.append(action)
                if action.kind == "pause":
                    break
        if self.offset != before or actions:
            self.save()
        return actions

    def replay(self) -> None:
        """Read the whole log to rebuild the state, emitting nothing."""
        self.offset = 0
        self.awaiting_since, self.trip_times, self.history = None, [], []
        for record in self._new_records():
            self._handle(record, float(record.get("time") or 0.0))
        self.awaiting_since = None  # a rollback asked for before the restart is not waited on after it
        self.save()

    def save(self) -> None:
        if self.state_path is None:
            return
        state = {
            "offset": self.offset,
            "awaiting_since": self.awaiting_since,
            "trip_times": self.trip_times,
            "history": [[v, t] for v, t in self.history],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.state_path)

    def _load(self) -> None:
        assert self.state_path is not None
        try:
            state = json.loads(self.state_path.read_text())
            self.offset = int(state["offset"])
            self.awaiting_since = state["awaiting_since"]
            self.trip_times = [float(t) for t in state["trip_times"]]
            self.history = [(int(v), bool(t)) for v, t in state["history"]]
        except (ValueError, KeyError, TypeError) as e:
            log.error("the tripwire state %s is unreadable (%s): replaying the health log", self.state_path, e)
            self.replay()
        if self.health.exists() and self.health.stat().st_size < self.offset:
            log.warning("the health log is shorter than the saved offset: replaying it")
            self.replay()

    def _new_records(self) -> list[dict[str, Any]]:
        if not self.health.exists():
            return []
        with self.health.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]
        self.offset += len(complete)
        records = []
        for line in complete.decode().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("ignoring an unreadable health line")
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _handle(self, record: dict[str, Any], now: float) -> Action | None:
        if record.get("event") == "rollback":
            self.awaiting_since = None
            self.history = [(int(record["restored"]), False), (int(record["version"]), False)]
            return None
        if "update" not in record:
            return None
        tripwires = record.get("tripwires") or {}
        trips = tripwires.get("trips") or []
        for alarm in tripwires.get("alarms") or []:
            log.warning("alarm at version %s: %s", record.get("version"), alarm.get("message"))
        self.history.append((int(record["version"]), bool(trips)))
        when = float(record.get("time") or now)
        if self.awaiting_since is not None and when - self.awaiting_since > self.rollback_wait_s:
            log.warning("the learner never reported the rollback it was told to make")
            self.awaiting_since = None
        if not trips or self.awaiting_since is not None:
            return None
        self.trip_times = [t for t in self.trip_times if when - t <= self.window_s] + [when]
        log.warning("tripwire at version %s: %s", record.get("version"), [t.get("message") for t in trips])
        target = self.last_green()
        if len(self.trip_times) >= self.limit or target is None:
            return Action("pause", None, trips, when)
        self.awaiting_since = when
        return Action("rollback", target, trips, when)

    def ring(self) -> set[int]:
        if not self.checkpoints.is_dir():
            return set()
        return {int(m.group(1)) for p in self.checkpoints.iterdir() if (m := CHECKPOINT.match(p.name))}

    def latest_healthy(self) -> int | None:
        """The newest version without trips whose checkpoint is in the ring."""
        ring = self.ring()
        return next((v for v, tripped in reversed(self.history) if not tripped and v in ring), None)

    def last_green(self) -> int | None:
        """The version to roll back to: the latest healthy one, failing that the oldest checkpoint in the ring."""
        latest = self.latest_healthy()
        ring = self.ring()
        return latest if latest is not None else min(ring, default=None)
