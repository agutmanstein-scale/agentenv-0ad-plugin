"""Which games the showcase plays: the 13-3's rules from its config, on starts no earlier panel or session used."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agentenv_zeroad.train.calibration import panel_specs
from agentenv_zeroad.train.config import RunConfig, load_config
from agentenv_zeroad.train.panels import position


@dataclass(frozen=True)
class ShowcaseConfig:
    """`showcase.toml`: the config whose rules the games follow, the panel whose teachers play, the opponents in turn,
    the first held-out index to draw starts from and the adapter version served as the policy."""

    rules_config: Path
    panel: str
    rotation: tuple[str, ...]
    first_index: int
    adapter_version: int

    @classmethod
    def load(cls, path: Path) -> ShowcaseConfig:
        raw = tomllib.loads(path.read_text())
        rules = Path(raw["rules_config"])
        if not rules.is_absolute():
            path = path.resolve()
            roots = [p for p in path.parents if (p / rules).is_file()]
            rules = (roots[0] if roots else path.parent) / rules
        return cls(
            rules_config=rules,
            panel=raw["panel"],
            rotation=tuple(raw["rotation"]),
            first_index=int(raw["first_index"]),
            adapter_version=int(raw["adapter_version"]),
        )

    def run_config(self) -> RunConfig:
        return load_config(self.rules_config)


def used_seeds(configs: Iterable[Path], sessions: Iterable[Path] = ()) -> set[int]:
    """Every seed any panel of any config plays, and every seed of the rows of the given `session.json` files."""
    seeds: set[int] = set()
    for path in configs:
        cfg = load_config(path)
        for panel in cfg.panels:
            seeds |= {spec.seed for spec in panel_specs(cfg, panel)}
    for path in sessions:
        seeds |= {row["seed"] for row in json.loads(path.read_text())["rows"]}
    return seeds


def next_starts(cfg: RunConfig, first_index: int, used: set[int], count: int) -> list[int]:
    """The first `count` indices j >= first_index whose position seed is not in `used`. Positions wrap at the number of
    held-out seeds, so running past it would replay a start: that raises instead."""
    limit = len(cfg.held_out.all_seeds())
    starts: list[int] = []
    for j in range(first_index, limit):
        if len(starts) == count:
            break
        if position(cfg, "held_out", j).seed not in used:
            starts.append(j)
    if len(starts) < count:
        raise ValueError(f"only {len(starts)} unused starts from index {first_index} before {limit}, not {count}")
    return starts
