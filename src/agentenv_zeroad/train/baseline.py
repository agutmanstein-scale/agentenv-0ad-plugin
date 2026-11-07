"""v0's decision rows for the learner's tripwire baseline (docs/s2-box-day.md).

The banded tripwires (completion length, tool calls per decision, exact-completion share, distinct fraction, entropy)
judge a window of updates against the base model's own distribution. That distribution is the same health record a
training update writes, taken over v0's decisions in batches of 256. This module makes the rows: v0's finished
ordinary games (the baselines' `games/`, not the panels) go through the same `Credit` Run 1 uses, so the decisions are
chosen and flagged (hold, version) as a training batch's are, and are written as one inbox-format file. The learner's
`--stats-only` mode (trainer/learner.py) then measures them without changing the weights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentenv_zeroad.train.config import RunConfig
from agentenv_zeroad.train.credit import Credit, CreditConfig


def credit_for(cfg: RunConfig) -> Credit:
    """Run 1's credit settings (its `[credit]` table), without the critic: only the rows are wanted."""
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in dict(cfg.credit).items()}
    return Credit(CreditConfig(**fields))


def v0_rows(games: Path, out: Path, credit: Credit) -> dict[str, Any]:
    """Credit every finished game of `games` (oldest name first) and write v0's rows to `out`; returns the counts."""
    rows = skipped = played = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    with tmp.open("w") as f:
        for path in sorted(games.glob("*/results.json")):
            try:
                results: Mapping[str, Any] = json.loads(path.read_text())
            except (OSError, ValueError):
                skipped += 1
                continue
            mine = [r for r in credit.rows(results) if r["version"] == 0]
            if not mine:
                skipped += 1
                continue
            played += 1
            rows += len(mine)
            f.write("".join(json.dumps(r) + "\n" for r in mine))
    tmp.replace(out)
    return {"rows": rows, "games": played, "skipped": skipped}
