"""Getting a run off the box: `aws s3 sync` of the run folder to a prefix, with the box's instance role.

Results used to leave the box only through laptop-signed URLs that died with the SSO session after about an hour, and a
box that stops or is replaced takes its disk with it. Every 15 minutes (and at the end) the orchestrator syncs what a
run is worth keeping, not the whole folder:

- games (every results.json, with its token ids and commands), the decision rows the learner read or has waiting
  (`inbox/`), `health.jsonl`, the panels, the control commands and the orchestrator's own state (which includes the
  run config and a copy of the box's ledger and box-time files);
- every published adapter's `manifest.json` (a few KB: the update, health and time of each version), and the weights of
  the hourly and champion adapters (a version is about 175 MB, so the hundreds the learner publishes are not all
  kept), named by the caller;
- the latest healthy checkpoint, the one a restart or a rollback would resume from.

`sync_command` builds the command without running it, so tests can check it without AWS; `Syncer` runs it and keeps the
outcome. It never uses `--delete`: a file removed from the box stays in the bucket.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

log = logging.getLogger(__name__)

ALWAYS = (
    "games/*", "inbox/*", "health.jsonl", "panels/*", "control/*", "orchestrator/*", "spend.jsonl",
    "adapters/*/manifest.json",
)  # fmt: skip
SYNC_TIMEOUT_S = 600.0  # train.sh's TimeoutStopSec covers SYNC_ATTEMPTS of these (orchestrate.SYNC_ATTEMPTS)

Run = Callable[[list[str]], Awaitable[tuple[int, str]]]


def sync_command(
    root: Path, uri: str, *, adapters: Iterable[int] = (), checkpoint: int | None = None, extra: Iterable[str] = ()
) -> list[str]:
    """`aws s3 sync ROOT URI` limited to the files worth keeping: ALWAYS, `adapters/v{N}` for each version in
    `adapters`, and `checkpoints/ckpt-{checkpoint}.pt`. An include after `--exclude "*"` is what lets a file through."""
    cmd = ["aws", "s3", "sync", str(root), uri.rstrip("/"), "--only-show-errors", "--exclude", "*"]
    for pattern in (*ALWAYS, *extra):
        cmd += ["--include", pattern]
    for version in sorted(set(adapters)):
        cmd += ["--include", f"adapters/v{version}/*"]
    if checkpoint is not None:
        cmd += ["--include", f"checkpoints/ckpt-{checkpoint:06d}.pt"]
    return cmd


async def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, env={**os.environ, "AWS_PAGER": ""}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), SYNC_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "aws s3 sync timed out"
    return proc.returncode or 0, out.decode(errors="replace")


class Syncer:
    """Runs syncs one at a time; a failed sync is logged and tried again at the next interval, never fatal."""

    def __init__(self, root: Path, uri: str | None, run: Run = run_command):
        self.root = root
        self.uri = uri
        self._run = run
        self._lock = asyncio.Lock()
        self.last_ok: float | None = None  # when the last successful sync ended
        self.last_start_ok: float | None = None  # when it started: what it saw is on S3
        self.failures = 0

    async def sync(
        self, *, adapters: Iterable[int] = (), checkpoint: int | None = None, extra: Iterable[str] = ()
    ) -> bool:
        """Whether the sync succeeded. With no URI configured nothing is uploaded and this is False."""
        if self.uri is None:
            return False
        async with self._lock:
            started = time.time()
            code, out = await self._run(
                sync_command(self.root, self.uri, adapters=adapters, checkpoint=checkpoint, extra=extra)
            )
        if code != 0:
            self.failures += 1
            log.error("s3 sync failed (%d): %s", code, out.strip()[-500:])
            return False
        self.last_ok, self.last_start_ok = time.time(), started
        return True
