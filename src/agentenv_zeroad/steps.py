"""The plugin's task step: `save_zeroad_results` keeps a finished match's results where a run's reader looks.

agentenv-game-env's `save_match_files` stores the match's files (results.json and the replay) as file artifacts, and
`grade_match` grades each player from the match protocol. This step, like the OpenCiv3 plugin's `save_env_recording`,
adds what those leave out: the results themselves (each seat's outcome, engine score, grade with the forfeit rule, §5
metrics, tool and brain call timings) in the run's `metadata["zeroad_results"]`, and, with `out_dir`, a copy of each
file in a local folder named after the run, so a match can be read without the artifact store.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import ClassVar

from agent_env.entity_refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agentenv_game import MATCH
from agentenv_game.steps import _deployed, _download, _invoke

log = logging.getLogger(__name__)


def summary(results: dict) -> dict:
    """The results without the per-call logs, small enough for run metadata."""
    seats = {}
    for slot, seat in (results.get("seats") or {}).items():
        calls = seat.get("tool_calls") or {}
        seats[slot] = {
            **{k: v for k, v in seat.items() if k not in ("tool_calls", "brain_calls")},
            "tool_calls": {k: v for k, v in calls.items() if k != "calls"},
            "brain_calls": len(seat.get("brain_calls") or []),
        }
    return {**results, "seats": seats}


class SaveResultsTaskStep(TaskStep):
    """Read the finished match's results.json from the env into `metadata["zeroad_results"]` (and copy every match
    file into `out_dir/<instance id>/` when given). Put it after finish_match."""

    type: ClassVar[str] = "save_zeroad_results"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: int | None,
        env_id: str,
        out_dir: str | None = None,
        timeout_seconds: int = 600,
        depends_on: list | None = None,
        fail_task_on_error: bool = False,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id, self.out_dir, self.timeout_seconds = env_id, out_dir, timeout_seconds

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "env_id": self.env_id,
            "out_dir": self.out_dir,
            "timeout_seconds": self.timeout_seconds,
            "fail_task_on_error": self.fail_task_on_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SaveResultsTaskStep:
        return cls(
            **{**cls._base_from_dict(data), "fail_task_on_error": data.get("fail_task_on_error", False)},
            env_id=data["env_id"],
            **{k: data[k] for k in ("out_dir", "timeout_seconds") if k in data},
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = _deployed(context, self.env_id)
        lobby_id = (context.metadata.get("game_lobby") or {}).get("lobby_id")
        request = {"lobby_id": lobby_id} if lobby_id else {}
        listed = await _invoke(deployed, MATCH, "files", request, self.timeout_seconds)
        for note in listed["notes"]:
            log.warning("save_zeroad_results: %s", note)
        base = deployed.environment_url.rstrip("/")
        keep = Path(self.out_dir).expanduser() / (context.instance_id or lobby_id or "match") if self.out_dir else None
        results = None
        with tempfile.TemporaryDirectory() as tmp:
            for file in listed["files"]:
                path = (keep or Path(tmp)) / file["name"]
                path.parent.mkdir(parents=True, exist_ok=True)
                await _download(base + file["path"], path, self.timeout_seconds)
                if file["kind"] == "results":
                    results = json.loads(path.read_text())
                if keep is not None:
                    log.info("save_zeroad_results: %s kept at %s", file["name"], path)
        if results is None:
            raise RuntimeError(f"env {self.env_id!r} kept no results for its match: {listed['notes']}")
        context.metadata["zeroad_results"] = summary(results)
        for slot, seat in results["seats"].items():
            log.info(
                "save_zeroad_results: seat %s (%s, %s): %s, score %s, grade %s",
                slot,
                seat["kind"],
                seat["civ"],
                seat["outcome"],
                (seat.get("score") or {}).get("total"),
                seat.get("grade"),
            )
        return context
