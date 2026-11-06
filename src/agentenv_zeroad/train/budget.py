"""What a run has cost so far, the way live/box/budget.sh counts it: box hours at the configured rate plus model spend.

Box time on a box is `$STATE/boxtime.json`, which the zeroad-boxtime timer keeps (budget.sh tick: one `[boot epoch,
last seen epoch]` per boot of the box), plus the current boot up to now: every hour the box has been up, not just this
run's, so the box's stage budget is one number whichever script counts it. Off a box (tests, a laptop) it is the hours
this run's orchestrator has been up, kept in `ROOT/orchestrator/clock.json`. The run clock is also what the run's
`duration_hours` and its panels' `at_hours` are measured on, and a box whose timer file is missing or behind it is
counted by the clock, whichever is larger. Model spend is the ledger's (`$STATE/spend.jsonl`, the box's one ledger,
one `{"what", "usd_est"}` per line, `agentenv_zeroad.spend`'s format): the orchestrator appends every game's frontier
spend to it, and other box tools append their own.

There is no default rate. The guard once priced a g6e.12xlarge at the 1.32 $/h of a stream box, which is why the run
config requires it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agentenv_zeroad.spend import ledger_total


@dataclass(frozen=True)
class Spend:
    box_hours: float
    box_usd: float
    model_usd: float

    @property
    def total(self) -> float:
        return self.box_usd + self.model_usd

    def line(self, budget_usd: float) -> str:
        return (
            f"box ${self.box_usd:.2f} ({self.box_hours:.2f} h) + model ${self.model_usd:.2f} = ${self.total:.2f}"
            f" of ${budget_usd:g}"
        )


class RunClock:
    """Run hours across restarts: the hours before this process started plus the time since."""

    def __init__(self, path: Path, now: float):
        self.path = path
        self.started = now
        self.before_h = 0.0
        if path.exists():
            self.before_h = float(json.loads(path.read_text()).get("hours", 0.0))

    def hours(self, now: float) -> float:
        return self.before_h + max(0.0, now - self.started) / 3600

    def save(self, now: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"hours": self.hours(now), "updated": now}))
        os.replace(tmp, self.path)


class BoxTime:
    """The box's running hours over every boot, from budget.sh's boxtime.json."""

    def __init__(self, path: Path, boot_epoch: float | None = None):
        self.path = path
        self.boot = boot_epoch if boot_epoch is not None else _boot_epoch()

    def hours(self, now: float) -> float:
        try:
            boots = json.loads(self.path.read_text())
        except (OSError, ValueError):
            boots = {}
        current = str(int(self.boot))
        earlier = sum(
            max(0, last - start) for key, (start, last) in boots.items() if key != current
        )  # the timer's own record of the earlier boots
        return (earlier + max(0.0, now - self.boot)) / 3600


def _boot_epoch() -> float:
    """This boot's start (/proc/stat btime), or now where there is none."""
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime"):
                return float(line.split()[1])
    except OSError:
        pass
    return time.time()


def spend(clock: RunClock, now: float, rate_usd_per_hour: float, ledger: Path, boxtime: BoxTime | None = None) -> Spend:
    hours = clock.hours(now)
    if boxtime is not None:
        hours = max(hours, boxtime.hours(now))
    return Spend(hours, hours * rate_usd_per_hour, ledger_total(ledger))
