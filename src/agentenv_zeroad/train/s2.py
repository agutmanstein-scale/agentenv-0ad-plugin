"""Box day 1 (S2) and Run 1 (S3) as one unattended sequence on one training box (docs/s2-box-day.md).

    python -m agentenv_zeroad.train.s2 check CONFIGDIR [--budget USD]
    python -m agentenv_zeroad.train.s2 run --config-dir CONFIGDIR --budget USD [--resume] [--no-shutdown]
    python -m agentenv_zeroad.train.s2 status S2ROOT

`run` is the process of the box's zeroad-s2 unit (live/box/s2-run.sh). It runs the steps in order, each under its own
wall-clock cap, records each step's start, end and gate verdicts in ROOT/s2/gates.json, and starts the next step only
if the gates of the one before passed:

  e1          GPU gates: log-prob alignment on every vLLM server (`smoke`), decisions per second per GPU at 16 to 64
              calls in flight on real v2 prompts (`probe`, gate >= 3), and the learner's seconds per trained decision
              (<= 2) and log-prob gap to vLLM (< 0.05 nats per token) from `learner.py --probe`
  baselines   play-only v0 on interface v2: the held-out panel, a base-v1 subset, the frontier panel, the teacher
              calibration panel and enough ordinary games to hold the distillation briefs
  calibration the teacher is picked (`stats.pick_teacher`)
  tripwire    the learner's banded-tripwire baseline: v0's health records over the ordinary baseline games' decisions
              (`train.baseline` rows through the same credit, `learner.py --stats-only`, no training) into
              ROOT/s2/baseline.json, which E2 and Run 1 are given as `--baseline`
  e2          the positive control: an RL run on the army-at-minute-10 reward against a frozen-v0 arm played beside it
              (`gates.positive_control_gates`); its adapters stay in its own root and are never used again
  e3_check    E3's check, play-only v0 (a petra-3 cap panel and two panels of v0 on identical starts, whose paired test
              must not reject: `gates.null_pair_gate`), then the learner's bench twice on the recorded rows, with E2's
              flags and with E3's, for seconds per row
  e3          the RL run on the game outcome with same-start leave-one-out credit, trained for `train_hours`, then two
              end panels (v0 and the final adapter, identical starts) judged by `gates.paired_margin_gate`
  e4_check    E4's check, play-only: 100 games each of v0 and of E3's final adapter on identical 30-minute starts
              against Petra-2 and -3 (E3's adapter is copied into this step's own root as version 1); the finished
              shares, and the margin scale E4's reward uses (the pooled within-opponent SD of v0's cap margins)
  e4          the RL run from E3's final adapter on the reward with a cap weight under 1 (conquest outweighs a cap
              lead), then two end panels on the check's starts (the start adapter and the final one) judged by
              `gates.paired_reward_gate`
  labels      `distill.py`: the teacher relabels the briefs under a spend cap
  sft         the learner in --sft mode, a checkpoint per epoch
  sft_panels  each epoch's adapter plays the first panel_games of the held-out panel; `stats.select_sft_checkpoint`
              picks one, which then plays the remaining configs, so it has all 384 that v0 has
  init        Run 1's starting point: the chosen adapter if it beats v0 on all 384 paired configs (one-sided p < 0.05),
              else the base model
  run1        the RL run, panels at its end; then the report, the final S3 upload and (unless --no-shutdown) the box's
              own `shutdown -h now`, which stops it (it is never terminated from here)
  run1_frontier  play-only: Run 1's final adapter (copied into this step's own root as version 1) plays 16 games each
              against Opus and Astra at equal latency, on the starts of the baselines' frontier panel, so its game j
              pairs with v0's game j; counts per teacher from the policy seat's side

A failed gate, a step cap, a budget refusal or a tripwire pause stops the sequence with the reason in gates.json and
status.json; the report and the final upload still run. Every step is a subprocess (a stuck one gets SIGTERM at its cap,
so an orchestrator uploads, then SIGKILL after the grace), and every step that ends is on disk, so `--resume` after a
crash or an SSM hiccup skips the steps that passed.

sequence.toml's [sequence] table holds a person's calls on a run, read again on every --resume: `accept_failed` lists
steps whose recorded failure was looked at and accepted, so the sequence goes on past them (their gates stay on record,
the status becomes `accepted`; a step that has not failed is never skipped), and `stop_after` names a step after which
the sequence stops, uploads and powers off as at any other stop (only the steps up to it count in the expected hours).
`rerun` lists steps to run again although gates.json records them (whatever their status; the old entry is kept under
`earlier`); once rerun, a step follows the normal rules again, and accept_failed never skips a step listed in `rerun`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tomllib
import traceback
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentenv_zeroad.train import gates as G
from agentenv_zeroad.train import poweroff, stats
from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train.config import RunConfig, check_run, load_config

log = logging.getLogger("s2")

STOPPED_BY_HAND = "stopped by hand (SIGTERM)"
STEPS = (
    "e1", "baselines", "calibration", "tripwire", "e2", "e3_check", "e3", "e4_check", "e4", "labels", "sft",
    "sft_panels", "init", "run1", "run1_frontier"
)  # fmt: skip
GRACE_S = 2400.0  # an orchestrator told to stop uploads for up to 3 x 10 minutes (sync.SYNC_TIMEOUT_S)
SFT_PREFIX = "sft-epoch"
SFT_REST = "sft-rest"  # the chosen epoch's second panel: the held-out configs the epoch panels did not play
BASELINE_FILE = "baseline.json"  # ROOT/s2/baseline.json, the tripwire baseline; its records are in the baselines root
E2_BASELINE_FILE = "baseline-e2.json"  # ROOT/s2/baseline-e2.json, E2's own when [e2] baseline_config is set
DECISIONS_PER_UPDATE = 256  # the learner's batch: one health record per this many decisions
E3_END_PANELS = ("end-v0", "end-final")  # e3.toml's end-of-run panels: v0, then the adapter current when they start
E3_CHECK_PANELS = (
    "petra3-cap",
    "pair-a",
    "pair-b",
)  # e3-check.toml's panels: the cap rate, and v0 twice on identical starts
E4_CHECK_PANELS = ("long-base", "long-e3")  # e4-check.toml's panels: v0 and E3's final adapter on identical starts
E4_END_PANELS = (
    "end-start",
    "end-final",
)  # e4.toml's end panels: the start adapter, then the adapter current at the end
RUN1_END_PANELS = ("end-base", "end-final")  # run1.toml's end panels: v0, then the adapter current at the end
RUN1_FRONTIER_PANEL = "frontier-final"  # run1-frontier.toml's one panel: Run 1's final adapter against the teachers
RUN1_FRONTIER_VERSION = 1  # Run 1's final adapter is published as this version in the root of the frontier check
RUN1_INIT_VERSION = 1  # the SFT adapter, when Run 1 starts from it, is published as this version in Run 1's root
E4_INIT_VERSION = 1  # E3's adapter is published as this version in the root of E4's check and of E4
E4_MIN_CAP_GAMES = 30  # the margin scale is measured on at least this many of v0's cap-ended games
E4_SCALE_RANGE = (2000.0, 20000.0)  # and clamped to this
PANEL_FINISHED_SHARE = 0.9  # of each E3 panel's games that must have finished

# Every table of sequence.toml with its keys and their defaults; a key not listed is an error.
SCHEMA: dict[str, dict[str, Any]] = {
    "e1": {
        "config": "e1.toml",
        "cap_hours": 0.75,
        "expected_hours": 0.5,
        "levels": [16, 32, 48, 64],
        "window_s": 60.0,
        "rows": 48,
        "min_rows": 32,
        "agree_rows": 5,
        "min_decisions_per_s_per_gpu": 3.0,
        "max_learner_s_per_decision": 2.0,
        "max_logprob_gap": 0.05,
    },
    "e2": {
        "config": "e2.toml",
        "cap_hours": 2.6,
        "expected_hours": 2.3,
        "control_panel": "control-v0",
        "window_games": 40,
        "min_games": 30,
        "min_median_army": 6.0,
        "max_p": 0.01,
        "min_updates": 5,
        "minute": 10.0,
        "baseline_config": "",  # a play-only v0 run on E2's kind of games whose health records band E2's tripwires
        "lr": 0.0,  # the learner's learning rate for E2; 0 keeps the learner's default
        "drift": "trip",  # the learner's --drift for E2: trip, or alarm to report the banded tripwires, no rollback
        "grad_norm": "trip",  # the learner's --grad-norm for E2: trip, or alarm to report a gradient-norm spike
    },
    "e3_check": {
        "config": "e3-check.toml",
        "cap_hours": 1.5,
        "expected_hours": 1.0,
        "bench_rows": "/var/lib/zeroad-live/train/s2/e2d/inbox/consumed",  # recorded rows the learner bench trains on
        "bench_updates": 3,
    },
    "e3": {
        "config": "e3.toml",
        "cap_hours": 5.0,
        "expected_hours": 4.4,
        "lr": 3e-5,
        "drift": "alarm",
        "grad_norm": "alarm",
        "baseline_config": "e2-baseline.toml",  # as E2's: a play-only v0 run whose health records band the tripwires
        "min_updates": 5,
        "max_p": 0.05,  # the paired margin test's one-sided Wilcoxon p
        "anchor_beta0": 0.0,
        "warmup_updates": 2,
        "decisions_per_update": 512,  # the learner's batch; e3.toml's [credit] decisions_per_update must match
        "kl_check_fraction": 0.25,
        "probe_rows": 64,
        "whole_groups": True,
    },
    "e4_check": {"config": "e4-check.toml", "cap_hours": 3.0, "expected_hours": 2.0},
    "run1_frontier": {"config": "run1-frontier.toml", "cap_hours": 2.0, "expected_hours": 1.2, "planned_usd": 100.0},
    "e4": {
        "config": "e4.toml",
        "cap_hours": 7.5,
        "expected_hours": 6.0,
        "lr": 3e-5,
        "drift": "alarm",
        "grad_norm": "alarm",
        "baseline_config": "e2-baseline.toml",
        "min_updates": 5,
        "max_p": 0.05,  # the paired R test's one-sided Wilcoxon p
        "anchor_beta0": 0.0,
        "warmup_updates": 2,
        "decisions_per_update": 512,  # the learner's batch; e4.toml's [credit] decisions_per_update must match
        "kl_check_fraction": 0.25,
        "probe_rows": 64,
        "whole_groups": True,
    },
    "baselines": {
        "config": "baselines.toml",
        "cap_hours": 3.5,
        "expected_hours": 2.6,
        "min_finished_share": 0.9,
        "heldout_panel": "heldout",
        "v1_panel": "heldout-v1",
        "frontier_panel": "frontier",
        "calibration_panel": "teacher-calibration",
        "planned_usd": 45.0,
    },
    "tripwire": {"cap_hours": 0.5, "expected_hours": 0.3, "min_records": 10},
    "distill": {
        "cap_hours": 1.0,
        "expected_hours": 0.4,
        "n": 3000,
        "min_labels": 2400,
        "cap_usd": 60.0,
        "second_cap_usd": 8.0,
        "check": 200,
        "per_game": 8,
        "tokenizer": "Qwen/Qwen3-8B",
        "weights": "losing=0.6,late=0.2,rejected=0.2",
    },
    "sft": {"config": "sft.toml", "cap_hours": 1.25, "expected_hours": 0.8, "epochs": 2, "lr": 1e-4},
    "sft_panels": {
        "config": "sft-panels.toml",
        "cap_hours": 3.0,
        "expected_hours": 2.1,
        "panel_games": 192,  # per epoch, to choose between epochs; the chosen one then plays the other 192
        "heldout_games": 384,  # the held-out panel's size: the chosen epoch has this many configs, like v0
        "min_entropy_ratio": 0.7,
        "min_games": 20,
        "max_p": 0.05,
    },
    "run1": {
        "config": "run1.toml",
        "cap_hours": 13.5,
        "expected_hours": 12.5,
        "lr": 3e-5,
        "drift": "alarm",
        "grad_norm": "alarm",
        "baseline_config": "",  # "": the tripwire step's baseline.json, made from the 30-minute baseline games
        "min_updates": 5,
        "max_p": 0.05,  # the paired R test's one-sided Wilcoxon p
        "anchor_beta0": 0.0,
        "warmup_updates": 2,
        "decisions_per_update": 512,  # the learner's batch; run1.toml's [credit] decisions_per_update must match
        "kl_check_fraction": 0.25,
        "probe_rows": 64,
        "whole_groups": True,
    },
}
DEFAULT_ROOT = "/var/lib/zeroad-live/train/s2"


class Runner(Protocol):
    def run(self, argv: Sequence[str], *, timeout_s: float, label: str, grace_s: float = GRACE_S) -> int:
        """Run argv to its end (its exit code), or stop it at `timeout_s` (SIGTERM to its process group, SIGKILL after
        `grace_s`) and return 124."""


class SubprocessRunner:
    def __init__(self, logs: Path, env: Mapping[str, str] | None = None):
        self.logs, self.env = logs, dict(env if env is not None else os.environ)

    def run(self, argv: Sequence[str], *, timeout_s: float, label: str, grace_s: float = GRACE_S) -> int:
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / f"{label}.log").open("ab") as out:
            out.write(f"\n$ {' '.join(argv)}\n".encode())
            out.flush()
            proc = subprocess.Popen(
                list(argv), stdout=out, stderr=subprocess.STDOUT, env=self.env, start_new_session=True
            )
            try:
                return proc.wait(timeout=max(timeout_s, 1.0))
            except (
                KeyboardInterrupt
            ):  # SIGTERM to the sequence (s2.sh stop): the unit's cgroup stop reached the child too
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=grace_s)
                raise
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                return 124


# ---- settings ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    root: Path
    s3_uri: str | None
    tables: Mapping[str, Mapping[str, Any]]
    accept_failed: tuple[str, ...] = ()
    stop_after: str = ""
    rerun: tuple[str, ...] = ()

    def __getitem__(self, step: str) -> Mapping[str, Any]:
        return self.tables[step]

    def cap_h(self, step: str) -> float:
        return float(self.tables[step]["cap_hours"])

    def counted_tables(self, skipped: Collection[str] = ()) -> list[str]:
        """The tables of the steps this sequence can still reach (all of them, or those up to stop_after), less the
        `skipped` steps a resume passes over."""
        steps = STEPS[: STEPS.index(self.stop_after) + 1] if self.stop_after else STEPS
        return [CAP_TABLE.get(s, s) for s in steps if s not in skipped and CAP_TABLE.get(s, s) in self.tables]

    @property
    def total_cap_hours(self) -> float:
        return sum(self.cap_h(t) for t in self.counted_tables())

    def expected_hours(self, skipped: Collection[str] = ()) -> float:
        return sum(float(self.tables[t]["expected_hours"]) for t in self.counted_tables(skipped))

    @property
    def total_expected_hours(self) -> float:
        return self.expected_hours()


def load_settings(path: Path) -> Settings:
    raw = tomllib.loads(path.read_text())
    unknown = set(raw) - {"root", "s3", "sequence", *SCHEMA}
    if unknown:
        raise ValueError(f"unknown sequence.toml tables: {', '.join(sorted(unknown))}")
    s3 = dict(raw.get("s3", {}))
    if set(s3) - {"uri"}:
        raise ValueError(f"unknown [s3] keys: {', '.join(sorted(set(s3) - {'uri'}))}")
    calls = dict(raw.get("sequence", {}))
    allowed_calls = {"accept_failed", "stop_after", "rerun"}
    if set(calls) - allowed_calls:
        raise ValueError(f"unknown [sequence] keys: {', '.join(sorted(set(calls) - allowed_calls))}")
    accept, stop_after = tuple(calls.get("accept_failed", ())), str(calls.get("stop_after", ""))
    rerun = tuple(calls.get("rerun", ()))
    for name in (*accept, *rerun, *([stop_after] if stop_after else [])):
        if name not in STEPS:
            raise ValueError(f"[sequence] names no step {name!r}; the steps are {', '.join(STEPS)}")
    tables = {}
    for step, defaults in SCHEMA.items():
        given = dict(raw.get(step, {}))
        extra = set(given) - set(defaults)
        if extra:
            raise ValueError(f"unknown [{step}] keys: {', '.join(sorted(extra))}")
        tables[step] = {**defaults, **given}
        if float(tables[step]["cap_hours"]) <= 0:
            raise ValueError(f"[{step}] cap_hours must be positive")
    for table in ("e2", "e3", "e4", "run1"):
        for switch in ("drift", "grad_norm"):
            if tables[table][switch] not in ("trip", "alarm"):
                raise ValueError(f"[{table}] {switch} must be 'trip' or 'alarm', not {tables[table][switch]!r}")
    return Settings(Path(raw.get("root", DEFAULT_ROOT)), s3.get("uri"), tables, accept, stop_after, rerun)


def render_sft_panels(template: str, versions: Mapping[int, int], games: int) -> str:
    """The SFT-panels config: its template plus one held-out panel per epoch, pinned to that epoch's adapter version."""
    blocks = "".join(
        f'\n[[panels]]\nname = "{SFT_PREFIX}{epoch}"\nkind = "heldout"\nat_hours = 0\n'
        f"version = {version}\ngames = {games}\n"
        for epoch, version in sorted(versions.items())
    )
    return template.rstrip() + "\n" + blocks


def render_rest_panel(template: str, version: int, start: int, games: int, hours_before: float = 0.0) -> str:
    """The chosen epoch's second config: the template plus one held-out panel of the configs `start` .. `start + games`
    of the held-out sequence (the ones the epoch panels did not play), pinned to the chosen adapter version. The root's
    run clock counts on across runs, so its duration_hours is the template's on top of the `hours_before` the clock
    already shows: the epoch panels' hours."""
    own = float(tomllib.loads(template)["duration_hours"])
    text, n = re.subn(
        r"^duration_hours\s*=.*$", f"duration_hours = {hours_before + own:.4f}", template, count=1, flags=re.M
    )
    if n != 1:
        raise ValueError("the sft_panels config has no duration_hours line")
    return (
        text.rstrip()
        + f'\n\n[[panels]]\nname = "{SFT_REST}"\nkind = "heldout"\nat_hours = 0\nversion = {version}\n'
        + f"start = {start}\ngames = {games}\n"
    )


class ConfigDir:
    """The step configs of a config directory, loaded and checked."""

    def __init__(self, path: Path, settings: Settings):
        self.path, self.settings = path, settings

    def file(self, step: str) -> Path:
        return self.path / str(self.settings[step]["config"])

    def load(self, step: str, *, sft_versions: Mapping[int, int] | None = None) -> RunConfig:
        if step == "sft_panels":
            games = int(self.settings["sft_panels"]["panel_games"])
            text = render_sft_panels(self.file(step).read_text(), sft_versions or {1: 1, 2: 2}, games)
            return RunConfig.from_dict(tomllib.loads(text))
        return load_config(self.file(step))


def check_dir(path: Path, budget: float | None = None, resume_from: Path | None = None) -> dict[str, Any]:
    """Validate a config directory (every step's config through `check_run`, the guards the sequence relies on) and
    return its numbers. With `resume_from` (a sequence's folder), the expected hours are those of the steps a --resume
    of it still runs. Raises ValueError with the first problem."""
    settings = load_settings(path / "sequence.toml")
    configs = ConfigDir(path, settings)
    rates, frontier = set(), False
    for step in (
        "e1",
        "e2",
        "e3_check",
        "e3",
        "e4_check",
        "e4",
        "baselines",
        "sft",
        "sft_panels",
        "run1",
        "run1_frontier",
    ):
        cfg = configs.load(step)
        keys = check_run(cfg)
        frontier = frontier or bool(keys)
        rates.add(cfg.box_usd_per_hour)
        try:
            cfg.root.relative_to(settings.root)
        except ValueError as e:
            raise ValueError(f"{step}: root {cfg.root} is not under the sequence root {settings.root}") from e
        if cfg.shutdown:
            raise ValueError(
                f"{step}: shutdown must be false: the sequence powers the box off itself, once, at the end"
            )
        if budget is not None and cfg.budget_usd > budget:
            raise ValueError(f"{step}: budget_usd {cfg.budget_usd:g} is above the stage budget {budget:g}")
        if cfg.duration_hours > settings.cap_h(step):
            raise ValueError(
                f"{step}: duration_hours {cfg.duration_hours:g} is past its cap_hours {settings.cap_h(step):g}"
            )
        if cfg.s3_uri is None:
            raise ValueError(f"{step}: s3.uri is required: a step nobody uploads is lost with the box")
    for table in ("e2", "e3", "e4"):
        if not settings[table]["baseline_config"]:
            continue
        cfg = load_config(path / str(settings[table]["baseline_config"]))
        check_run(cfg)
        rates.add(cfg.box_usd_per_hour)
        if cfg.shutdown or cfg.s3_uri is None or cfg.panels:
            raise ValueError(
                f"{table} baseline_config: shutdown must be false, s3.uri set and no panels (its games are v0's)"
            )
        try:
            cfg.root.relative_to(settings.root)
        except ValueError as e:
            raise ValueError(f"{table} baseline_config: root {cfg.root} is not under the sequence root") from e
        if cfg.root == configs.load(table).root:
            raise ValueError(f"{table} baseline_config: its root must not be {table.upper()}'s own")
    if len(rates) != 1:
        raise ValueError(f"the step configs name different box rates: {sorted(rates)}")
    _check_panel_names(configs, settings)
    if configs.load("e2").credit.get("positive_control") is not True:
        raise ValueError("e2: [credit] positive_control must be true")
    _check_e3(configs, settings)
    _check_e4(configs, settings)
    _check_run1(configs, settings)
    _check_run1_frontier(configs, settings)
    sp = settings["sft_panels"]
    if not 0 < int(sp["panel_games"]) < int(sp["heldout_games"]):
        raise ValueError(
            "sft_panels: panel_games must be under heldout_games, so the chosen epoch has configs left to play"
        )
    held = next(p for p in configs.load("baselines").panels if p.name == settings["baselines"]["heldout_panel"])
    if held.games != int(sp["heldout_games"]):
        raise ValueError(
            f"sft_panels heldout_games {sp['heldout_games']} is not the baselines' held-out panel's {held.games} games"
        )
    if configs.load("sft").root != configs.load("sft_panels").root:
        raise ValueError("sft and sft_panels must share one root: the epochs and the panels are read together")
    return {
        "root": str(settings.root),
        "cap_hours": settings.total_cap_hours,
        "expected_hours": settings.expected_hours(resumed_steps(settings, resume_from) if resume_from else ()),
        "run1_hours": configs.load("run1").duration_hours,
        "rate": rates.pop(),
        "frontier": frontier,
    }


def _same_starts(cfg: RunConfig, first: str, second: str, other: RunConfig | None = None) -> bool:
    """Whether two panels play identical starts game by game: their specs differ at most by the game id. They are
    panels of `cfg`, or `second` is a panel of `other`."""
    from agentenv_zeroad.train.calibration import panel_specs

    other = other or cfg
    a, b = (panel_specs(c, next(p for p in c.panels if p.name == n)) for c, n in ((cfg, first), (other, second)))
    return len(a) == len(b) and all(
        dataclasses.replace(x, game_id="x") == dataclasses.replace(y, game_id="x") for x, y in zip(a, b, strict=True)
    )


def _check_e3(configs: ConfigDir, settings: Settings) -> None:
    """What E3 and its check rely on: their panels exist and pair, E3 trains on the outcome with same-start credit in
    batches the learner's size, and the check plays no training game."""
    e3, check = configs.load("e3"), configs.load("e3_check")
    for cfg, names in ((e3, E3_END_PANELS), (check, E3_CHECK_PANELS)):
        missing = set(names) - {p.name for p in cfg.panels}
        if missing:
            raise ValueError(f"{cfg.run_id}: no panel named {', '.join(sorted(missing))}")
    for cfg, pair in ((e3, E3_END_PANELS), (check, ("pair-a", "pair-b"))):
        if not _same_starts(cfg, *pair):
            raise ValueError(f"{cfg.run_id}: panels {pair[0]} and {pair[1]} do not play identical starts")
    if e3.train_hours is None or any(p.at_hours < e3.train_hours for p in e3.panels if p.name in E3_END_PANELS):
        raise ValueError("e3: train_hours must be set and no end panel may start before it")
    if not check.panels_only or not check.stop_when_done:
        raise ValueError("e3_check: the check plays panels only and ends when they are done")
    credit = e3.credit
    if credit.get("positive_control") or credit.get("baseline") != "start_loo":
        raise ValueError("e3: [credit] must train on the outcome (no positive_control) with baseline = start_loo")
    if int(credit.get("group_size", 1)) != e3.games.seed_group:
        raise ValueError("e3: [credit] group_size must be games.seed_group")
    if int(credit.get("decisions_per_update", 256)) != int(settings["e3"]["decisions_per_update"]):
        raise ValueError("e3: [credit] decisions_per_update must match [e3] decisions_per_update, the learner's batch")
    if len({e3.root, check.root}) != 2:
        raise ValueError("e3 and e3_check need roots of their own")


def _check_e4(configs: ConfigDir, settings: Settings) -> None:
    """What E4 and its check rely on: the four panels exist and all pair (the check's two with E4's two, on identical
    starts), E4 starts from the base model or E3's adapter and end-start plays that model, it trains on a cap weight
    under 1 with same-start credit and a margin scale the step renders, and the check plays no training game."""
    e4, check = configs.load("e4"), configs.load("e4_check")
    for cfg, names in ((e4, E4_END_PANELS), (check, E4_CHECK_PANELS)):
        missing = set(names) - {p.name for p in cfg.panels}
        if missing:
            raise ValueError(f"{cfg.run_id}: no panel named {', '.join(sorted(missing))}")
    for other, name in ((check, E4_CHECK_PANELS[1]), (e4, E4_END_PANELS[0]), (e4, E4_END_PANELS[1])):
        if not _same_starts(check, E4_CHECK_PANELS[0], name, other):
            raise ValueError(
                f"e4_check's {E4_CHECK_PANELS[0]} and {other.run_id}'s {name} do not play identical starts"
            )
    panel = {p.name: p for p in check.panels}
    if panel[E4_CHECK_PANELS[0]].version != 0 or panel[E4_CHECK_PANELS[1]].version != E4_INIT_VERSION:
        raise ValueError(f"e4_check: long-base plays v0 and long-e3 plays v{E4_INIT_VERSION}, E3's adapter")
    end = {p.name: p for p in e4.panels}
    if e4.init_version not in (0, E4_INIT_VERSION):
        raise ValueError(f"e4: init_version is 0 (the base model) or {E4_INIT_VERSION} (E3's adapter)")
    if end[E4_END_PANELS[0]].version != e4.init_version or end[E4_END_PANELS[1]].version is not None:
        raise ValueError(
            f"e4: end-start plays v{e4.init_version}, the model E4 starts from, and end-final the final adapter"
        )
    if e4.train_hours is None or any(p.at_hours < e4.train_hours for p in e4.panels if p.name in E4_END_PANELS):
        raise ValueError("e4: train_hours must be set and no end panel may start before it")
    if not check.panels_only or not check.stop_when_done:
        raise ValueError("e4_check: the check plays panels only and ends when they are done")
    credit = e4.credit
    if credit.get("positive_control") or credit.get("baseline") != "start_loo":
        raise ValueError("e4: [credit] must train on the outcome (no positive_control) with baseline = start_loo")
    if int(credit.get("group_size", 1)) != e4.games.seed_group:
        raise ValueError("e4: [credit] group_size must be games.seed_group")
    if int(credit.get("decisions_per_update", 256)) != int(settings["e4"]["decisions_per_update"]):
        raise ValueError("e4: [credit] decisions_per_update must match [e4] decisions_per_update, the learner's batch")
    if not 0 < float(credit.get("cap_weight", 1.0)) < 1:
        raise ValueError(
            "e4: [credit] cap_weight must be above 0 and under 1, so a conquest outweighs a lead at the cap"
        )
    if "margin_scale" not in credit:
        raise ValueError("e4: [credit] needs a margin_scale line for the step to replace with the check's measured one")
    if len({e4.root, check.root, configs.load("e3").root}) != 3:
        raise ValueError("e3, e4 and e4_check need roots of their own")


def _check_run1(configs: ConfigDir, settings: Settings) -> None:
    """What Run 1 relies on: its two end panels exist, pair with each other and with the baselines' held-out panel on
    identical starts, no end panel starts before train_hours, and it trains on the outcome with same-start credit in
    batches the learner's size and a decisions range that is a pair."""
    run1, base = configs.load("run1"), configs.load("baselines")
    missing = set(RUN1_END_PANELS) - {p.name for p in run1.panels}
    if missing:
        raise ValueError(f"{run1.run_id}: no panel named {', '.join(sorted(missing))}")
    heldout = str(settings["baselines"]["heldout_panel"])
    for cfg, name in ((run1, RUN1_END_PANELS[1]), (base, heldout)):
        if not _same_starts(run1, RUN1_END_PANELS[0], name, cfg):
            raise ValueError(f"run1's {RUN1_END_PANELS[0]} and {cfg.run_id}'s {name} do not play identical starts")
    panel = {p.name: p for p in run1.panels}
    if run1.init_version not in (0, RUN1_INIT_VERSION):
        raise ValueError(f"run1: init_version is 0 (the base model) or {RUN1_INIT_VERSION} (the SFT adapter)")
    if panel[RUN1_END_PANELS[0]].version != 0 or panel[RUN1_END_PANELS[1]].version is not None:
        raise ValueError("run1: end-base plays v0, the base model, and end-final the final adapter")
    if run1.train_hours is None or any(panel[n].at_hours < run1.train_hours for n in RUN1_END_PANELS):
        raise ValueError("run1: train_hours must be set and no end panel may start before it")
    credit = run1.credit
    if credit.get("positive_control") or credit.get("baseline") != "start_loo":
        raise ValueError("run1: [credit] must train on the outcome (no positive_control) with baseline = start_loo")
    if int(credit.get("group_size", 1)) != run1.games.seed_group:
        raise ValueError("run1: [credit] group_size must be games.seed_group")
    if int(credit.get("decisions_per_update", 256)) != int(settings["run1"]["decisions_per_update"]):
        raise ValueError(
            "run1: [credit] decisions_per_update must match [run1] decisions_per_update, the learner's batch"
        )
    decisions = credit.get("decisions", (8, 12))
    if len(decisions) != 2 or not 1 <= int(decisions[0]) <= int(decisions[1]):
        raise ValueError("run1: [credit] decisions is a pair lo, hi with 1 <= lo <= hi")
    if "margin_scale" not in credit:
        raise ValueError(
            "run1: [credit] needs a margin_scale line for the step to replace with the check's measured one"
        )
    if not run1.stop_when_done:
        raise ValueError("run1: stop_when_done must be set, so the run ends when the end panels are played")


def teacher_specs(cfg: RunConfig, panel: Any) -> dict[str, list[Any]]:
    """A frontier panel's game specs by teacher, in game order (game j of a teacher is its j-th)."""
    from agentenv_zeroad.train.calibration import panel_specs

    specs, start, out = panel_specs(cfg, panel), 0, {}
    for teacher in panel.teachers:
        out[teacher.name] = specs[start : start + teacher.games]
        start += teacher.games
    return out


def _check_run1_frontier(configs: ConfigDir, settings: Settings) -> None:
    """What the frontier check relies on: its one panel plays Run 1's final adapter as version 1, the run plays panels
    only, and each teacher, set as in the baselines' frontier panel, plays game j on that panel's game j start (specs
    equal but for the game id)."""
    check, base = configs.load("run1_frontier"), configs.load("baselines")
    if [p.name for p in check.panels] != [RUN1_FRONTIER_PANEL] or check.panels[0].kind != "frontier":
        raise ValueError(f"{check.run_id}: its one panel is a frontier panel named {RUN1_FRONTIER_PANEL}")
    mine = check.panels[0]
    if mine.version != RUN1_FRONTIER_VERSION:
        raise ValueError(f"run1_frontier: {mine.name} plays v{RUN1_FRONTIER_VERSION}, Run 1's final adapter")
    if not check.panels_only or not check.stop_when_done:
        raise ValueError("run1_frontier: the check plays panels only and ends when they are done")
    theirs = next(p for p in base.panels if p.name == settings["baselines"]["frontier_panel"])
    old = {t.name: t for t in theirs.teachers}
    before, after = teacher_specs(base, theirs), teacher_specs(check, mine)
    for teacher in mine.teachers:
        other = old.get(teacher.name)
        same = other is not None and all(
            getattr(teacher, k) == getattr(other, k) for k in ("model", "effort", "max_tokens")
        )
        if not same:
            raise ValueError(f"run1_frontier: teacher {teacher.name} is not set as in the baselines' frontier panel")
        if any(
            dataclasses.replace(x, game_id="x") != dataclasses.replace(y, game_id="x")
            for x, y in zip(before[teacher.name], after[teacher.name], strict=False)
        ):
            raise ValueError(f"run1_frontier: {teacher.name}'s games are not on the baselines' frontier starts")


def _check_panel_names(configs: ConfigDir, settings: Settings) -> None:
    wanted = {
        "baselines": ("heldout_panel", "v1_panel", "frontier_panel", "calibration_panel"),
        "e2": ("control_panel",),
    }
    for step, keys in wanted.items():
        names = {p.name for p in configs.load(step).panels}
        for key in keys:
            if settings[step][key] not in names:
                raise ValueError(
                    f"sequence.toml [{step}] {key} = {settings[step][key]!r}: {step}'s config has no such panel"
                )


# ---- the record ----------------------------------------------------------------------------------
@dataclass
class Outcome:
    gates: list[G.Gate] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None  # a failure that is not a gate (a step cap, a crash, a missing file)

    @property
    def passed(self) -> bool:
        return self.reason is None and all(g.passed for g in self.gates)

    def why(self) -> str | None:
        if self.reason:
            return self.reason
        failed = [g.line() for g in self.gates if not g.passed]
        return "; ".join(failed) or None


class StepFailed(Exception):
    pass


class Record:
    """ROOT/s2/gates.json (every step's verdicts) and status.json (where the sequence is), written atomically."""

    def __init__(self, s2: Path, now: Callable[[], float]):
        self.s2, self.now = s2, now
        self.gates_path, self.status_path = s2 / "gates.json", s2 / "status.json"
        self.data: dict[str, Any] = {"verdict": "running", "reason": None, "started": now(), "steps": {}}
        if self.gates_path.exists():
            self.data = json.loads(self.gates_path.read_text())
            self.data.update(verdict="running", reason=None)

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        self.s2.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=1))
        os.replace(tmp, path)

    def passed(self, step: str) -> bool:
        return self.status(step) == "passed"

    def status(self, step: str) -> str | None:
        return self.data["steps"].get(step, {}).get("status")

    def accept(self, step: str) -> None:
        """Mark a failed step accepted by hand, keeping its gates and reason."""
        entry = self.data["steps"][step]
        if entry["status"] != "accepted":
            entry.update(status="accepted", accepted=self.now())
        self.save(step)

    def begin(self, step: str, rerun: bool = False) -> None:
        """Start a step's entry. Its earlier runs stay on record: a rerun adds the entry it replaces, a retry keeps the
        list the entry already had."""
        old = self.data["steps"].get(step, {})
        earlier = list(old.get("earlier", []))
        if rerun and old:
            earlier.append({k: v for k, v in old.items() if k != "earlier"})
        self.data["steps"][step] = {
            "status": "running",
            "started": self.now(),
            "ended": None,
            "gates": [],
            "detail": {},
            **({"earlier": earlier} if earlier else {}),
        }
        self.save(step)

    def end(self, step: str, outcome: Outcome) -> None:
        entry = self.data["steps"][step]
        entry.update(
            status="passed" if outcome.passed else "failed",
            ended=self.now(),
            seconds=round(self.now() - entry["started"]),
            gates=[g.to_json() for g in outcome.gates],
            detail=outcome.detail,
            reason=outcome.why(),
        )
        self.save(step)

    def finish(self, verdict: str, reason: str | None) -> None:
        self.data.update(verdict=verdict, reason=reason, ended=self.now())
        self.save(None)

    def save(self, current: str | None) -> None:
        self._write(self.gates_path, self.data)
        self._write(
            self.status_path,
            {
                "time": self.now(),
                "step": current,
                "verdict": self.data["verdict"],
                "reason": self.data["reason"],
                "steps": {k: v["status"] for k, v in self.data["steps"].items()},
            },
        )


# ---- the context a step runs in ------------------------------------------------------------------
@dataclass
class Ctx:
    settings: Settings
    configs: ConfigDir
    box_home: Path
    runner: Runner
    budget_usd: float
    record: Record
    now: Callable[[], float] = time.time
    python: str = sys.executable
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    step: str = ""
    step_started: float = 0.0
    step_cap_h: float = 0.0

    @property
    def s2(self) -> Path:
        return self.settings.root

    def say(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime(self.now()))} {message}"
        log.info(message)
        self.s2.mkdir(parents=True, exist_ok=True)
        with (self.s2 / "progress.log").open("a") as f:
            f.write(line + "\n")

    def remaining_s(self) -> float:
        return self.step_cap_h * 3600 - (self.now() - self.step_started)

    def run(self, argv: Sequence[str], label: str) -> int:
        """Run a command inside the step's cap; 124 (and a step failure) when the cap ran out under it."""
        left = self.remaining_s()
        if left <= 0:
            raise StepFailed(f"the step's {self.step_cap_h:g} h cap ran out before {label}")
        self.say(f"{self.step}: {label}")
        rc = self.runner.run(argv, timeout_s=left, label=f"{self.step}-{label}".replace(" ", "-"))
        if rc == 124:
            raise StepFailed(f"{label} was still running at the step's {self.step_cap_h:g} h cap and was stopped")
        return rc

    def train_run(self, config: Path, flags: Sequence[str] = (), label: str = "run") -> int:
        return self.run(["bash", str(self.box_home / "box" / "train-run.sh"), str(config), *flags], label)

    def budget_ok(self, hours: float, usd: float = 0.0) -> str | None:
        """None when budget.sh says the planned box hours and frontier dollars fit LIVE_BUDGET_USD, else its refusal."""
        argv = [
            "bash",
            str(self.box_home / "box" / "budget.sh"),
            "check",
            "--planned-usd",
            f"{usd:g}",
            "--planned-hours",
            f"{hours:g}",
            "--budget",
            f"{self.budget_usd:g}",
        ]
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        text = (done.stdout + done.stderr).strip().splitlines()
        return None if done.returncode == 0 else (text[-1] if text else "budget.sh refused")

    def cfg(self, step: str, **kw: Any) -> RunConfig:
        return self.configs.load(step, **kw)


def rerun_due(settings: Settings, step: str, entry: Mapping[str, Any] | None) -> bool:
    """Whether [sequence] rerun makes `step` run again: it is listed, has an entry, and that is not yet a rerun."""
    return step in settings.rerun and bool(entry) and not (entry or {}).get("earlier")


def resumed_steps(settings: Settings, s2: Path) -> set[str]:
    """The steps a --resume of the sequence in `s2` passes over: those that passed, those accepted by hand, and the
    failed ones its [sequence] accept_failed now accepts. A step [sequence] rerun lists is not passed over until it has
    been rerun, and accept_failed never passes it over."""
    steps = (read_json(s2 / "gates.json") or {}).get("steps", {})
    return {
        name
        for name, entry in steps.items()
        if not rerun_due(settings, name, entry)
        and (
            entry.get("status") in ("passed", "accepted")
            or (entry.get("status") == "failed" and name in settings.accept_failed and name not in settings.rerun)
        )
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def final_reason(root: Path) -> str:
    final = read_json(root / "orchestrator" / "final.json") or {}
    return str(final.get("reason") or "no final.json")


def trips_in(health: Path) -> int:
    """Learner updates whose record carried a tripwire trip."""
    count = 0
    with contextlib.suppress(OSError):
        for line in health.read_text().splitlines():
            with contextlib.suppress(ValueError):
                if (json.loads(line).get("tripwires") or {}).get("trips"):
                    count += 1
    return count


def updates_in(health: Path) -> int:
    n = 0
    with contextlib.suppress(OSError):
        for line in health.read_text().splitlines():
            with contextlib.suppress(ValueError):
                n += "version" in json.loads(line) and "event" not in json.loads(line)
    return n


def ended_normally(root: Path, ok: Sequence[str]) -> G.Gate:
    reason = final_reason(root)
    return G.Gate(f"the run ended normally ({reason[:60]})", 1.0 if reason.split(":")[0] in ok else 0.0, ">=", 1.0)


# ---- E1 ------------------------------------------------------------------------------------------
def step_e1(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["e1"], ctx.cfg("e1")
    out = Outcome()
    smoke = [
        ctx.run([ctx.python, "-m", "agentenv_zeroad.train.smoke", "--url", url], f"smoke {i}")
        for i, url in enumerate(cfg.servers.urls)
    ]
    out.gates.append(
        G.Gate(
            "log-prob alignment on every vLLM server",
            float(sum(rc == 0 for rc in smoke)),
            ">=",
            float(len(smoke)),
            "smoke.py: one aligned log-prob per generated token",
        )
    )
    if not out.gates[-1].passed:
        return out
    urls = [a for url in cfg.servers.urls for a in ("--url", url)]
    probe = [
        ctx.python,
        "-m",
        "agentenv_zeroad.train.probe",
        *urls,
        "--out",
        str(cfg.root),
        "--levels",
        ",".join(str(x) for x in s["levels"]),
        "--window-s",
        str(s["window_s"]),
        "--rows",
        str(s["rows"]),
    ]
    if ctx.run(probe, "throughput probe") != 0:
        out.reason = "the throughput probe failed (see logs/e1-throughput-probe.log)"
        return out
    summary = read_json(cfg.root / "throughput.json") or {}
    rates = {int(k): float(v) for k, v in (summary.get("per_gpu") or {}).items()}
    out.detail["throughput"] = summary
    out.gates.append(G.throughput_gate(rates, float(s["min_decisions_per_s_per_gpu"])))
    kept = int(summary.get("rows_kept") or 0)
    if kept < int(s["min_rows"]):
        out.reason = f"the probe kept {kept} sampled decisions, under the {s['min_rows']} the learner's probe needs"
        return out
    rc = ctx.train_run(ctx.configs.file("e1"), ["--probe", str(cfg.root / "probe-rows.jsonl")], "learner probe")
    result = read_json(cfg.root / "probe.json")
    if rc != 0 or not result:
        out.reason = "the learner's probe produced no numbers (see logs/e1-learner-probe.log)"
        return out
    out.detail["learner"] = result
    out.gates.append(
        G.learner_gate((result.get("timing") or {}).get("seconds_per_decision"), float(s["max_learner_s_per_decision"]))
    )
    out.gates.append(G.agreement_gate((result.get("agreement") or {}).get("mean_abs_gap"), float(s["max_logprob_gap"])))
    return out


# ---- E2 ------------------------------------------------------------------------------------------
def army_at_minute(results: Mapping[str, Any], slot: str, minute: float) -> float | None:
    row = rs.timeline_first_at_or_after(results, minute * 60_000)
    return None if row is None else float(((row.get("players") or {}).get(slot) or {}).get("army") or 0)


def armies(folder: Path, minute: float, *, trained_only: bool, last: int | None = None) -> list[float]:
    """The policy seat's army at `minute` in each finished game of `folder`, oldest first (the last `last` of them);
    with `trained_only` the games of an adapter (version 1 and up), without it v0's."""
    found: list[tuple[float, float]] = []
    for path in sorted(folder.glob("*/results.json")):
        results = read_json(path)
        if not results or results.get("status") != "finished":
            continue
        slot = rs.policy_slot(results, None)
        version = rs.policy_version(results, slot) if slot else None
        if slot is None or version is None or (version > 0) != trained_only:
            continue
        value = army_at_minute(results, slot, minute)
        if value is not None:
            found.append((path.stat().st_mtime, value))
    values = [v for _, v in sorted(found)]
    return values[-last:] if last else values


def learner_flags(s: Mapping[str, Any]) -> list[str]:
    """The learning rate and the drift and grad-norm switches a step's table sets (a default is left to the learner)."""
    flags: list[str] = []
    if float(s["lr"]) > 0:
        flags += ["--lr", f"{float(s['lr']):g}"]
    if s["drift"] != "trip":
        flags += ["--drift", str(s["drift"])]
    if s["grad_norm"] != "trip":
        flags += ["--grad-norm", str(s["grad_norm"])]
    return flags


def step_e2(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["e2"], ctx.cfg("e2")
    ctx.train_run(ctx.configs.file("e2"), e2_flags(ctx), label="control run")
    minute = float(s["minute"])
    policy = armies(cfg.root / "games", minute, trained_only=True, last=int(s["window_games"]))
    control = armies(cfg.root / "panels" / str(s["control_panel"]), minute, trained_only=False)
    trips = trips_in(cfg.root / "health.jsonl")
    gates, detail = G.positive_control_gates(
        policy,
        control,
        trips=trips,
        min_median=float(s["min_median_army"]),
        max_p=float(s["max_p"]),
        min_games=int(s["min_games"]),
    )
    updates = updates_in(cfg.root / "health.jsonl")
    out = Outcome(
        [
            G.Gate("learner updates", float(updates), ">=", float(s["min_updates"])),
            *gates,
            ended_normally(cfg.root, ("duration", "complete")),
        ],
        detail | {"updates": updates, "trips": trips},
    )
    return out


# ---- E3 ------------------------------------------------------------------------------------------
def panel_games(folder: Path, margin_scale: float | None = None, cap_weight: float = 1.0) -> list[G.PanelGame]:
    """The finished games of a panel folder, each with its start and the policy's final margin (and, given a
    `margin_scale`, its R)."""
    games = (G.panel_game(read_json(p) or {}, margin_scale, cap_weight) for p in sorted(folder.glob("*/results.json")))
    return [g for g in games if g is not None]


def training_games(folder: Path, margin_scale: float, cap_weight: float = 1.0) -> list[G.TrainingGame]:
    """The finished training games of a run's games folder with their R at `margin_scale` and `cap_weight`."""
    games = (
        G.training_game(read_json(p) or {}, margin_scale, cap_weight) for p in sorted(folder.glob("*/results.json"))
    )
    return [g for g in games if g is not None]


def e3_learner_flags(s: Mapping[str, Any]) -> list[str]:
    """E3's learner flags beyond E2's: no pull to the anchor, a short warm-up, whole same-start groups per update, the
    KL stop checked on a fraction of the batch and a few probe rows."""
    flags = [
        "--anchor-beta0", f"{float(s['anchor_beta0']):g}",
        "--warmup-updates", str(int(s["warmup_updates"])),
        "--decisions-per-update", str(int(s["decisions_per_update"])),
        "--kl-check-fraction", f"{float(s['kl_check_fraction']):g}",
        "--probe-rows", str(int(s["probe_rows"])),
    ]  # fmt: skip
    return [*flags, "--whole-groups"] if s["whole_groups"] else flags


def e3_flags(ctx: Ctx) -> list[str]:
    """E3's learner flags, also the bench's: E2's bands, lr, drift and grad norm, then E3's own."""
    s = ctx.settings["e3"]
    base = e2_baseline_flags(ctx, "e3") if s["baseline_config"] else baseline_flags(ctx)
    return [*base, *learner_flags(s), *e3_learner_flags(s)]


def render_bench_config(ctx: Ctx, kind: str) -> Path:
    """The check's config with a root, run id and prefix of its own, so each bench's bench.json is its own."""
    text = ctx.configs.file("e3_check").read_text()
    for key in ("root", "run_id"):
        text, n = re.subn(rf'^({key} = ")(.*)(")$', rf"\1\2-bench-{kind}\3", text, count=1, flags=re.M)
        if n != 1:
            raise StepFailed(f"the e3_check config has no {key} line to give the {kind} bench its own")
    text = re.sub(r'^(uri = ")(.*)(")$', rf"\1\2-bench-{kind}\3", text, count=1, flags=re.M)
    out = ctx.s2 / "generated" / f"e3-check-bench-{kind}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def run_bench(ctx: Ctx, kind: str, flags: Sequence[str]) -> dict[str, Any]:
    """One learner bench on the recorded rows: `train-run.sh --bench ROWS` in a root of its own; its bench.json."""
    s = ctx.settings["e3_check"]
    config = render_bench_config(ctx, kind)
    rc = ctx.train_run(
        config,
        ["--bench", str(s["bench_rows"]), "--bench-updates", str(int(s["bench_updates"])), *flags],
        f"{kind} bench",
    )
    result = read_json(load_config(config).root / "bench.json") or {}
    return {"exit": rc, "seconds_per_row": result.get("mean_seconds_per_row"), "bench": result}


def step_e3_check(ctx: Ctx) -> Outcome:
    """Is the end test sound and the learner fast enough: v0 played twice on identical starts must not differ under the
    paired test, every panel must finish, and the learner's seconds per row under E3's flags and E2's (reported)."""
    cfg = ctx.cfg("e3_check")
    ctx.train_run(ctx.configs.file("e3_check"), ["--play-only"], label="play-only run")
    panels = {name: panel_games(cfg.root / "panels" / name) for name in E3_CHECK_PANELS}
    null_gate, null_detail = G.null_pair_gate(panels["pair-a"], panels["pair-b"])
    gates = [null_gate, *panel_gates(cfg.root, E3_CHECK_PANELS, PANEL_FINISHED_SHARE)]
    cap = panels["petra3-cap"]
    margins = [g.margin for g in cap]
    detail: dict[str, Any] = {
        "null": null_detail,
        "petra3_cap": {
            "games": len(cap),
            "cap_win_share": sum(m > 0 for m in margins) / len(margins) if margins else None,
            "median_margin": statistics.median(margins) if margins else None,
        },
    }
    e2_bench = run_bench(ctx, "e2", [*e2_flags(ctx), "--full-prox"])
    e3_bench = run_bench(ctx, "e3", e3_flags(ctx))
    a, b = e2_bench["seconds_per_row"], e3_bench["seconds_per_row"]
    detail["bench"] = {"e2": e2_bench, "e3": e3_bench, "ratio_e3_to_e2": b / a if a and b else None}
    return Outcome(gates, detail)


def e2_flags(ctx: Ctx) -> list[str]:
    """The learner flags E2 ran with, for the bench's old-flags arm."""
    s = ctx.settings["e2"]
    return [*(e2_baseline_flags(ctx) if s["baseline_config"] else baseline_flags(ctx)), *learner_flags(s)]


def step_e3(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["e3"], ctx.cfg("e3")
    ctx.train_run(ctx.configs.file("e3"), e3_flags(ctx), label="rl run")
    final, v0 = (panel_games(cfg.root / "panels" / name) for name in reversed(E3_END_PANELS))
    paired, paired_detail = G.paired_margin_gate(final, v0, max_p=float(s["max_p"]))
    scale = float(cfg.credit.get("margin_scale", 2000.0))
    slope, slope_detail = G.in_run_slope_gate(training_games(cfg.root / "games", scale))
    updates, trips = updates_in(cfg.root / "health.jsonl"), trips_in(cfg.root / "health.jsonl")
    gates = [
        G.Gate("learner updates", float(updates), ">=", float(s["min_updates"])),
        paired,
        slope,
        G.Gate("tripwire trips", float(trips), "<=", 0.0),
        ended_normally(cfg.root, ("complete",)),
    ]
    detail = {"paired": paired_detail, "slope": slope_detail, "updates": updates, "trips": trips}
    return Outcome(gates, detail | {"cap_win_share": paired_detail["cap_win_share"]})


# ---- E4 ------------------------------------------------------------------------------------------
def e3_adapter(ctx: Ctx) -> tuple[int, Path]:
    """E3's final adapter: the version its end-final panel played and that version's folder in E3's root."""
    root = ctx.cfg("e3").root
    version = int((read_json(root / "panels" / E3_END_PANELS[1] / "summary.json") or {}).get("version") or 0)
    if not version:
        raise StepFailed(f"E3's end-final panel left no summary with the adapter version it played in {root}")
    return version, root / "adapters" / f"v{version}"


def seed_adapter(source: Path, root: Path, version: int) -> Path:
    """Publish a copy of the adapter in `source` as ROOT/adapters/v{version}, which the run's games and panels then play
    and its learner starts from. The copy is made beside it and renamed into place, and one already made from the same
    source is left alone. E3's own root is only read."""
    needed = ("adapter_config.json", "adapter_model.safetensors")
    if not all((source / name).is_file() for name in needed):
        raise StepFailed(f"{source} is not an adapter folder ({', '.join(needed)}): was it pruned?")
    stamp = {"copied_from": str(source), "source_version": (read_json(source / "manifest.json") or {}).get("version")}
    dest = root / "adapters" / f"v{version}"
    made = read_json(dest / "manifest.json") or {}
    if all(made.get(k) == v for k, v in stamp.items()) and all((dest / name).is_file() for name in needed):
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    for stale in (tmp, dest):
        shutil.rmtree(stale, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, tmp)
    manifest = {"mode": "rl", "version": version, "update": 0, **stamp, "time": time.time()}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, dest)
    return dest


def panel_report(games: Sequence[G.PanelGame]) -> dict[str, dict[str, Any]]:
    """A panel's games overall and per opponent: the games, the shares that ended in a conquest won or lost, the share
    that ended at a cap with the policy ahead on net worth (of all the games), the median net-worth margin of the cap
    games and the median wall seconds of a game."""

    def share(n: int, of: int) -> float | None:
        return n / of if of else None

    groups: dict[str, list[G.PanelGame]] = {"all": list(games)}
    for g in games:
        groups.setdefault(g.opponent, []).append(g)
    out = {}
    for name, found in groups.items():
        cap = [g for g in found if g.ended_by in rs.CAP_ENDS]
        walls = [g.wall_s for g in found if g.wall_s is not None]
        out[name] = {
            "games": len(found),
            "conquest_won_share": share(
                sum(g.ended_by == "conquest" and g.outcome == "won" for g in found), len(found)
            ),
            "conquest_lost_share": share(
                sum(g.ended_by == "conquest" and g.outcome == "lost" for g in found), len(found)
            ),
            "ahead_at_cap_share": share(sum(g.margin > 0 for g in cap), len(found)),
            "won_share": share(
                sum(g.ended_by == "conquest" and g.outcome == "won" for g in found) + sum(g.margin > 0 for g in cap),
                len(found),
            ),
            "median_cap_margin": statistics.median(g.margin for g in cap) if cap else None,
            "median_wall_s": statistics.median(walls) if walls else None,
        }
    return out


def step_e4_check(ctx: Ctx) -> Outcome:
    """Do E3's adapter and v0 play the long games: E3's adapter is copied into this root as version 1 and plays beside
    v0 on identical 30-minute starts. Gates: every panel finishes, and v0 has enough cap-ended games to measure the
    margin scale E4's reward uses (reported in the detail, with each panel per opponent and the paired comparison)."""
    cfg = ctx.cfg("e4_check")
    version, source = e3_adapter(ctx)
    copy = seed_adapter(source, cfg.root, E4_INIT_VERSION)
    ctx.say(f"e4_check: E3's v{version} ({source}) is v{E4_INIT_VERSION} in {cfg.root}")
    ctx.train_run(ctx.configs.file("e4_check"), ["--play-only"], label="play-only run")
    base, e3 = (panel_games(cfg.root / "panels" / name) for name in E4_CHECK_PANELS)
    pooled = G.pooled_margin_sd(base)
    gates = [
        *panel_gates(cfg.root, E4_CHECK_PANELS, PANEL_FINISHED_SHARE),
        G.Gate(
            f"cap-ended games of {E4_CHECK_PANELS[0]}",
            float(pooled["cap_games"]),
            ">=",
            float(E4_MIN_CAP_GAMES),
            "the margin scale's sample",
        ),
    ]
    detail: dict[str, Any] = {
        "adapter": {"source": str(source), "source_version": version, "copy": str(copy)},
        "panels": {name: panel_report(g) for name, g in zip(E4_CHECK_PANELS, (base, e3), strict=True)},
        "margin": pooled,
        "paired_e3_vs_base": G.paired_margin_test(e3, base),
    }
    if pooled["sd"] is not None and pooled["cap_games"] >= E4_MIN_CAP_GAMES:
        low, high = E4_SCALE_RANGE
        detail["margin_scale"] = min(max(float(pooled["sd"]), low), high)
    return Outcome(gates, detail)


def e4_margin_scale(ctx: Ctx) -> float:
    """The margin scale E4's check measured, or a step failure when it measured none."""
    scale = ((ctx.record.data["steps"].get("e4_check") or {}).get("detail") or {}).get("margin_scale")
    if not scale:
        raise StepFailed("e4_check recorded no margin_scale: E4's reward has no scale to take")
    return float(scale)


def render_e4(ctx: Ctx, scale: float) -> Path:
    """e4.toml with its [credit] margin_scale replaced by the check's, in ROOT/s2/generated/."""
    text, n = re.subn(
        r"^(margin_scale\s*=\s*)[0-9.eE+-]+", lambda m: f"{m.group(1)}{scale:.1f}", ctx.configs.file("e4").read_text(),
        count=1, flags=re.M,
    )  # fmt: skip
    if n != 1:
        raise StepFailed("the e4 config has no margin_scale line to set")
    out = ctx.s2 / "generated" / "e4.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def e4_flags(ctx: Ctx) -> list[str]:
    """E3's learner flags; with init_version 1, started from E3's adapter published as version 1 in E4's root, and with
    init_version 0 from the base model. No --anchor: with anchor_beta0 0 the learner does not load an anchor."""
    s, cfg = ctx.settings["e4"], ctx.cfg("e4")
    base = e2_baseline_flags(ctx, "e4") if s["baseline_config"] else baseline_flags(ctx)
    version = cfg.init_version
    init = ["--init", str(cfg.root / "adapters" / f"v{version}"), "--start-version", str(version)] if version else []
    return [*base, *learner_flags(s), *e3_learner_flags(s), *init]


def step_e4(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["e4"], ctx.cfg("e4")
    scale, weight = e4_margin_scale(ctx), float(cfg.credit["cap_weight"])
    if cfg.init_version:
        seed_adapter(e3_adapter(ctx)[1], cfg.root, cfg.init_version)
    rendered = render_e4(ctx, scale)
    ctx.train_run(rendered, e4_flags(ctx), label="rl run")
    final, start = (panel_games(cfg.root / "panels" / name, scale, weight) for name in reversed(E4_END_PANELS))
    paired, paired_detail = G.paired_reward_gate(final, start, max_p=float(s["max_p"]))
    _, slope = G.in_run_slope_gate(training_games(cfg.root / "games", scale, weight))
    check_base = panel_games(ctx.cfg("e4_check").root / "panels" / E4_CHECK_PANELS[0], scale, weight)
    updates, trips = updates_in(cfg.root / "health.jsonl"), trips_in(cfg.root / "health.jsonl")
    gates = [
        G.Gate("learner updates", float(updates), ">=", float(s["min_updates"])),
        paired,
        G.Gate("tripwire trips", float(trips), "<=", 0.0),
        ended_normally(cfg.root, ("complete",)),
    ]
    detail = {
        "margin_scale": scale,
        "cap_weight": weight,
        "paired": paired_detail,
        "paired_margin": G.paired_margin_test(final, start),
        "final_vs_check_base": G.paired_reward_gate(final, check_base)[1],
        "slope": slope,
        "panels": {name: panel_report(g) for name, g in zip(E4_END_PANELS, (start, final), strict=True)},
        "updates": updates,
        "trips": trips,
    }
    return Outcome(gates, detail)


# ---- baselines, calibration and distillation -----------------------------------------------------
def panel_gates(root: Path, names: Sequence[str], share: float) -> list[G.Gate]:
    gates = []
    for name in names:
        summary = read_json(root / "panels" / name / "summary.json") or {}
        games = int(summary.get("games") or 0)
        gates.append(
            G.Gate(
                f"panel {name}: share of its games finished",
                (summary.get("finished", 0) / games) if games else None,
                ">=",
                share,
                f"{summary.get('finished', 0)} of {games}",
            )
        )
    return gates


def brief_supply(games: Path, n: int, per_game: int, weights: str) -> dict[str, Any]:
    from agentenv_zeroad.train import distill

    supply = distill.BriefSupply(n, distill.parse_weights(weights), per_game)
    for path in sorted(games.glob("*/results.json")):
        results = read_json(path)
        if results:
            supply.add(path.parent.name, distill.briefs_of(results, path.parent.name, None, distill.LOSING_OPPONENTS))
    return supply.capacity()


def step_baselines(ctx: Ctx) -> Outcome:
    s, cfg, d = ctx.settings["baselines"], ctx.cfg("baselines"), ctx.settings["distill"]
    ctx.train_run(ctx.configs.file("baselines"), ["--play-only"], "play-only run")
    names = [p.name for p in cfg.panels]
    gates = [
        ended_normally(cfg.root, ("complete", "duration")),
        *panel_gates(cfg.root, names, float(s["min_finished_share"])),
    ]
    supply = brief_supply(cfg.root / "games", int(d["n"]), int(d["per_game"]), str(d["weights"]))
    gates.append(
        G.Gate(
            "distillation briefs available",
            float(supply["selectable"]),
            ">=",
            float(d["n"]),
            f"{supply['candidates']} candidate decisions; losing share met: {supply['shares_met']}",
        )
    )
    return Outcome(gates, {"briefs": supply})


def step_calibration(ctx: Ctx) -> Outcome:
    from agentenv_zeroad.train import report

    folder = ctx.cfg("baselines").root / "panels" / str(ctx.settings["baselines"]["calibration_panel"])
    summary = report.teacher_calibration(folder)
    name = stats.pick_teacher(summary) if summary else None
    out = Outcome(
        [G.Gate("a teacher played Petra-3 in the calibration panel", 1.0 if name else 0.0, ">=", 1.0)],
        {"calibration": summary, "teacher": name},
    )
    if name:
        ctx.say(f"calibration: distilling from {name} ({(summary or {}).get('best', {}).get('rule')})")
    return out


def baseline_records(ctx: Ctx, root: Path | None = None) -> int:
    """Health records behind a stats-only learner's baseline.json in `root` (the baselines root unless given)."""
    path = (root or ctx.cfg("baselines").root) / "baseline-records.jsonl"
    with contextlib.suppress(OSError):
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    return 0


def e2_baseline_flags(ctx: Ctx, table: str = "e2") -> list[str]:
    """`--baseline ROOT/s2/baseline-e2.json`: v0's health records on E2's own kind of games. The tripwire baseline
    comes from the 30-minute baseline games, and on E2's 12-minute games v0 itself falls outside its distinct-completion
    band (6 Nov: both of E2's trips came at its first records, KL to base 0.0000). `baseline_config` is a play-only v0
    run of those games; its games are credited with E2's credit and measured by the stats-only learner. A file already
    made with enough records is reused, so a resumed E2 does not play them again."""
    from agentenv_zeroad.train import baseline

    need = int(ctx.settings["tripwire"]["min_records"])
    path = ctx.configs.path / str(ctx.settings[table]["baseline_config"])
    cfg = load_config(path)
    produced, target = cfg.root / BASELINE_FILE, ctx.s2 / E2_BASELINE_FILE
    if not (produced.is_file() and baseline_records(ctx, cfg.root) >= need):
        if final_reason(cfg.root).split(":")[0] not in ("duration", "complete"):
            ctx.train_run(path, ["--play-only"], "E2-type v0 games")
        rows = cfg.root / "baseline-rows.jsonl"
        made = baseline.v0_rows(cfg.root / "games", rows, baseline.credit_for(ctx.cfg("e2")))
        if made["rows"] < need * DECISIONS_PER_UPDATE:
            raise StepFailed(
                f"too few v0 decisions on E2's games for its tripwire baseline: {made['rows']} of "
                f"{need * DECISIONS_PER_UPDATE} ({made['games']} games)"
            )
        for stale in (produced, cfg.root / "baseline-records.jsonl"):
            stale.unlink(missing_ok=True)
        rc = ctx.train_run(path, ["--stats-only", str(rows)], "E2-type stats-only learner")
        have = baseline_records(ctx, cfg.root)
        if rc != 0 or not produced.is_file() or have < need:
            raise StepFailed(f"no E2 tripwire baseline: exit {rc}, {have} of {need} records in {cfg.root}")
        ctx.say(f"e2: tripwire baseline from {made['rows']} v0 decisions in {made['games']} of E2's games")
    ctx.s2.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, target)
    return ["--baseline", str(target)]


def baseline_flags(ctx: Ctx) -> list[str]:
    """`--baseline ROOT/s2/baseline.json` for a training run, or a step failure: a run without it has the banded
    collapse tripwires (length, calls, exact and distinct completions, entropy) switched off."""
    need = int(ctx.settings["tripwire"]["min_records"])
    path = ctx.s2 / BASELINE_FILE
    have = baseline_records(ctx)
    if not path.is_file() or have < need:
        raise StepFailed(
            f"no tripwire baseline: {path} {'exists' if path.is_file() else 'is missing'} with {have} of {need} records"
        )
    return ["--baseline", str(path)]


def step_tripwire(ctx: Ctx) -> Outcome:
    """v0's health records without training: its decisions in the ordinary (non-panel) baseline games, credited as Run 1
    credits them, measured by the learner in batches of 256 on the base model (`learner.py --stats-only`), then
    `baseline_from` into ROOT/s2/baseline.json. The gate: that file with at least `min_records` records."""
    from agentenv_zeroad.train import baseline

    s, base = ctx.settings["tripwire"], ctx.cfg("baselines")
    need = int(s["min_records"])
    rows = base.root / "baseline-rows.jsonl"
    for stale in (rows, base.root / BASELINE_FILE, base.root / "baseline-records.jsonl"):
        stale.unlink(missing_ok=True)
    made = baseline.v0_rows(base.root / "games", rows, baseline.credit_for(ctx.cfg("run1")))
    out = Outcome(detail={"rows": made})
    out.gates.append(
        G.Gate(
            "v0 decisions for the tripwire baseline",
            float(made["rows"]),
            ">=",
            float(need * DECISIONS_PER_UPDATE),
            f"{made['rows']} rows from {made['games']} games ({made['skipped']} skipped)",
        )
    )
    if not out.gates[-1].passed:
        return out
    rc = ctx.train_run(ctx.configs.file("baselines"), ["--stats-only", str(rows)], "stats-only learner")
    produced = base.root / BASELINE_FILE
    records = baseline_records(ctx)
    out.detail["records"] = records
    out.gates.append(G.Gate("health records of v0", float(records), ">=", float(need)))
    if rc != 0 or not produced.is_file():
        out.reason = f"the stats-only learner made no baseline (exit {rc}; see logs/tripwire-stats-only-learner.log)"
        return out
    ctx.s2.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, ctx.s2 / BASELINE_FILE)
    out.detail["baseline"] = read_json(ctx.s2 / BASELINE_FILE)
    return out


def step_labels(ctx: Ctx) -> Outcome:
    d, base = ctx.settings["distill"], ctx.cfg("baselines")
    sft = ctx.cfg("sft")
    url = ctx.env.get("LITELLM_BASE_URL")
    if not url:
        raise StepFailed(
            "LITELLM_BASE_URL is not set in the unit's environment (LIVE_LITELLM_BASE_URL in the live env)"
        )
    out_dir = ctx.s2 / "distill"
    argv = [
        str(ctx.box_home / "box" / "with-model.sh"),
        ctx.python,
        "-m",
        "agentenv_zeroad.train.distill",
        "--games",
        str(base.root / "games"),
        "--out",
        str(out_dir),
        "--tokenizer",
        str(d["tokenizer"]),
        "--calibration",
        str(base.root / "panels" / str(ctx.settings["baselines"]["calibration_panel"])),
        "--base-url",
        url,
        "--api-key-env",
        "LITELLM_API_KEY",
        "--cap-usd",
        str(d["cap_usd"]),
        "--second-cap-usd",
        str(d["second_cap_usd"]),
        "--n",
        str(d["n"]),
        "--check",
        str(d["check"]),
        "--per-game",
        str(d["per_game"]),
        "--weights",
        str(d["weights"]),
        "--inbox",
        str(sft.root / "inbox"),
        "--ledger",
        str(base.ledger_path),
        "--spend-log",
        str(out_dir / "spend.jsonl"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = ctx.run(argv, "label the briefs")
    rows = 0
    with contextlib.suppress(OSError):
        rows = sum(1 for line in (out_dir / "sft.jsonl").read_text().splitlines() if line.strip())
    out = Outcome(detail={"report": read_json(out_dir / "distill.json")})
    if rc != 0:
        out.reason = f"distill.py exited {rc} (see logs/labels-label-the-briefs.log)"
    out.gates.append(G.Gate("labelled briefs", float(rows), ">=", float(d["min_labels"])))
    return out


def step_sft(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["sft"], ctx.cfg("sft")
    epochs = int(s["epochs"])
    rc = ctx.train_run(ctx.configs.file("sft"), ["--sft", str(epochs), "--lr", str(s["lr"])], "sft learner")
    made = sorted(p.parent.name for p in (cfg.root / "epochs").glob("epoch*/manifest.json"))
    done = (cfg.root / "sft-done.json").exists()
    out = Outcome(
        [G.Gate("SFT epochs published", float(len(made)), ">=", float(epochs))], {"epochs": made, "done": done}
    )
    if rc != 0 or not done:
        out.reason = f"the SFT learner did not finish (exit {rc}, sft-done.json {'present' if done else 'missing'})"
    return out


def epoch_versions(root: Path) -> dict[int, int]:
    versions = {}
    for path in (root / "epochs").glob("epoch*/manifest.json"):
        manifest = read_json(path) or {}
        if "version" in manifest:
            versions[int(path.parent.name.removeprefix("epoch"))] = int(manifest["version"])
    return versions


def step_sft_panels(ctx: Ctx) -> Outcome:
    """Each epoch plays the first `panel_games` configs of the held-out panel and the best one is chosen; the chosen
    epoch then plays the remaining configs (`SFT_REST`), so the init decision has v0's 384 paired configs."""
    from agentenv_zeroad.train import report

    s, template = ctx.settings["sft_panels"], ctx.cfg("sft")
    versions = epoch_versions(template.root)
    if not versions:
        raise StepFailed("no SFT epoch manifests to play")
    text = ctx.configs.file("sft_panels").read_text()
    rendered = ctx.s2 / "generated" / "sft-panels.toml"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text(render_sft_panels(text, versions, int(s["panel_games"])))
    cfg = load_config(rendered)
    ctx.train_run(rendered, ["--play-only"], "epoch panels")
    gates = [
        ended_normally(cfg.root, ("complete", "duration")),
        *panel_gates(cfg.root, [p.name for p in cfg.panels], 0.9),
    ]
    detail: dict[str, Any] = {"versions": versions}
    if all(g.passed for g in gates):
        selection = report.select_sft(
            template.root / "epochs",
            template.root / "panels",
            None,
            SFT_PREFIX,
            float(s["min_entropy_ratio"]),
            int(s["min_games"]),
        )
        detail["selection"] = selection
        chosen = selection.get("chosen")
        if chosen:
            first, total = int(s["panel_games"]), int(s["heldout_games"])
            rest = ctx.s2 / "generated" / "sft-rest.toml"
            clock = read_json(cfg.root / "orchestrator" / "clock.json") or {}
            rest.write_text(
                render_rest_panel(
                    text,
                    versions[int(str(chosen).removeprefix("epoch"))],
                    first,
                    total - first,
                    float(clock.get("hours", 0.0)),
                )
            )
            ctx.say(f"sft_panels: {chosen} plays the other {total - first} held-out configs")
            ctx.train_run(rest, ["--play-only"], "chosen epoch rest")
            rest_cfg = load_config(rest)
            gates += [
                ended_normally(rest_cfg.root, ("complete", "duration")),
                *panel_gates(rest_cfg.root, [SFT_REST], 0.9),
            ]
    return Outcome(gates, detail)


def played(folder: Path) -> list[Any]:
    from agentenv_zeroad.train import report

    return report.read_games(folder, None)[0] if folder.is_dir() else []


def as_stats(games: Sequence[Any]) -> list[stats.Game]:
    return [stats.Game(p.version, 0.0, p.opponent, p.result, p.margin) for p in games]


def config_ids(games: Sequence[Any], panel: str) -> set[str]:
    """The configs a panel's games played, as the panel-independent tail of the game id: 0007-petra-2."""
    return {g.game_id.removeprefix(panel + "-") for g in games}


def step_init(ctx: Ctx) -> Outcome:
    from agentenv_zeroad.train import report

    s, sft = ctx.settings["sft_panels"], ctx.cfg("sft")
    panels = sft.root / "panels"
    selection = report.select_sft(
        sft.root / "epochs", panels, None, SFT_PREFIX, float(s["min_entropy_ratio"]), int(s["min_games"])
    )
    base_panel = ctx.cfg("baselines").root / "panels" / str(ctx.settings["baselines"]["heldout_panel"])
    chosen_games, base_games = [], []
    name = selection.get("chosen")
    if name:
        panel_name = f"{SFT_PREFIX}{name.removeprefix('epoch')}"
        first, rest = played(panels / panel_name), played(panels / SFT_REST)
        chosen_games = [*first, *rest]
        ids = config_ids(first, panel_name) | config_ids(rest, SFT_REST)  # the same configs: 0007-petra-2
        base_games = [g for g in played(base_panel) if g.game_id.removeprefix(base_panel.name + "-") in ids]
    decision = G.init_decision(selection, as_stats(base_games), as_stats(chosen_games), max_p=float(s["max_p"]))
    init = str(sft.root / "epochs" / decision["init"]) if decision["init"] else None
    ctx.say(f"init: {init or 'the base model'}: {decision['reason']}")
    return Outcome(detail={"selection": selection, "decision": decision, "init_path": init})


# ---- Run 1 and the end ---------------------------------------------------------------------------
def milestone_table(games: Sequence[G.PanelGame]) -> list[dict[str, Any]]:
    """The milestones of `stats.DEFAULT_MILESTONES` on one panel's games, as a won share (a conquest won, or ahead on
    net worth at the cap) and its one-sided 97.5% Wilson lower bound against the threshold; `lower_bonferroni` is the
    bound at the level shared out across the milestones, as `stats.milestones` uses."""
    z, z_bonferroni = (stats.NORMAL.inv_cdf(1 - a) for a in (0.025, stats.ALPHA / len(stats.DEFAULT_MILESTONES)))
    out = []
    for m in stats.DEFAULT_MILESTONES:
        mine = [g for g in games if g.opponent == m.opponent]
        wins = sum(
            (g.ended_by == "conquest" and g.outcome == "won") or (g.ended_by in rs.CAP_ENDS and g.margin > 0)
            for g in mine
        )
        lower = stats.wilson_lower(wins, len(mine), z)
        out.append(
            {
                "name": m.name,
                "games": len(mine),
                "won_share": wins / len(mine) if mine else None,
                "lower": lower,
                "lower_bonferroni": stats.wilson_lower(wins, len(mine), z_bonferroni),
                "threshold": m.threshold,
                "met": bool(mine) and lower >= m.threshold,
            }
        )
    return out


def render_run1(ctx: Ctx, scale: float, init_version: int) -> Path:
    """run1.toml with its [credit] margin_scale replaced by e4_check's and its init_version by the one in use, in
    ROOT/s2/generated/."""
    text = ctx.configs.file("run1").read_text()
    for key, value in (("margin_scale", f"{scale:.1f}"), ("init_version", str(init_version))):
        text, n = re.subn(
            rf"^({key}\s*=\s*)[0-9.eE+-]+", lambda m, v=value: f"{m.group(1)}{v}", text, count=1, flags=re.M
        )
        if n != 1:
            raise StepFailed(f"the run1 config has no {key} line to set")
    out = ctx.s2 / "generated" / "run1.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def step_run1(ctx: Ctx) -> Outcome:
    """Run 1: RL from the init step's model on the documented objective, judged by the paired R of its end-final panel
    over its end-base panel, both on the baselines' held-out starts. An SFT init is published into Run 1's root as
    version 1, so the games and the base-model panel start from the right model."""
    s, cfg = ctx.settings["run1"], ctx.cfg("run1")
    init = (ctx.record.data["steps"].get("init", {}).get("detail") or {}).get("init_path")
    flags: list[str] = [*baseline_flags(ctx), *learner_flags(s), *e3_learner_flags(s)]
    version = 0
    if init:
        if not Path(init).is_relative_to(ctx.cfg("sft").root):
            raise StepFailed(f"Run 1's init {init} is not an SFT checkpoint: no E2 adapter may carry over")
        version = RUN1_INIT_VERSION
        seed_adapter(Path(init), cfg.root, version)
        flags += ["--init", str(cfg.root / "adapters" / f"v{version}"), "--start-version", str(version)]
    scale, weight = e4_margin_scale(ctx), float(cfg.credit["cap_weight"])
    ctx.say(f"run1: {'from ' + init if init else 'from the base model'}, margin scale {scale:.0f}")
    ctx.train_run(render_run1(ctx, scale, version), flags, "rl run")
    final, start = (panel_games(cfg.root / "panels" / name, scale, weight) for name in reversed(RUN1_END_PANELS))
    paired, paired_detail = G.paired_reward_gate(final, start, max_p=float(s["max_p"]))
    _, slope = G.in_run_slope_gate(training_games(cfg.root / "games", scale, weight))
    updates, trips = updates_in(cfg.root / "health.jsonl"), trips_in(cfg.root / "health.jsonl")
    gates = [
        G.Gate("learner updates", float(updates), ">=", float(s["min_updates"])),
        paired,
        G.Gate("tripwire trips", float(trips), "<=", 0.0),
        ended_normally(cfg.root, ("complete",)),
    ]
    detail = {
        "init": init,
        "init_version": version,
        "margin_scale": scale,
        "cap_weight": weight,
        "paired": paired_detail,
        "paired_margin": G.paired_margin_test(final, start),
        "slope": slope,
        "panels": {name: panel_report(g) for name, g in zip(RUN1_END_PANELS, (start, final), strict=True)},
        "milestones": {name: milestone_table(g) for name, g in zip(RUN1_END_PANELS, (start, final), strict=True)},
        "updates": updates,
        "trips": trips,
    }
    return Outcome(gates, detail)


# ---- Run 1's frontier check -----------------------------------------------------------------------
def run1_adapter(ctx: Ctx) -> tuple[int, Path]:
    """Run 1's final adapter: the version its end-final panel played (else the highest adapter in its root) and that
    version's folder in Run 1's root."""
    root = ctx.cfg("run1").root
    version = int((read_json(root / "panels" / RUN1_END_PANELS[1] / "summary.json") or {}).get("version") or 0)
    if not version:
        found = [int(p.name[1:]) for p in (root / "adapters").glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(found, default=0)
    if not version:
        raise StepFailed(f"Run 1 left no end-final summary with an adapter version and no adapters/vN in {root}")
    return version, root / "adapters" / f"v{version}"


def frontier_rows(folder: Path) -> list[dict[str, Any]]:
    """The finished games of a frontier panel folder from the policy seat's side: the teacher, the game's index j in its
    teacher's run (from the game id), the policy's outcome, how the game ended, the teacher's failed calls and spend."""
    rows = []
    for path in sorted(folder.glob("*/results.json")):
        res = read_json(path) or {}
        slot = (
            rs.policy_slot(res, None) if res.get("status") == "finished" and len(res.get("seats") or {}) == 2 else None
        )
        opp = rs.other_slot(res, slot) if slot is not None else None
        found = re.search(r"-(\d+)(?:-r\d+)?$", str((res.get("spec") or {}).get("game_id") or ""))
        if slot is None or opp is None or found is None:
            continue
        seat = res["seats"][opp] or {}
        rows.append(
            {
                "teacher": rs.player_id(res, opp),
                "j": int(found.group(1)),
                "outcome": str((res["seats"][slot] or {}).get("outcome") or ""),
                "ended_by": str(res.get("ended_by") or ""),
                "failed_calls": sum(1 for c in seat.get("brain_calls") or [] if c.get("error")),
                "usd": float(((seat.get("spend") or {}).get("brain") or {}).get("usd_est") or 0.0),
            }
        )
    return rows


def frontier_detail(
    final: Sequence[Mapping[str, Any]], v0: Sequence[Mapping[str, Any]], teachers: Sequence[str]
) -> dict[str, Any]:
    """Per teacher, plain counts from the Qwen seat's side (games, wins and losses, each by conquest or at a cap, the
    teacher's failed calls per game and spend) and, by game index, v0's result on the games both panels have."""

    def tally(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        count = {"games": len(rows), "won": 0, "lost": 0, "won_by_conquest": 0, "won_at_cap": 0}
        count |= {"lost_by_conquest": 0, "lost_at_cap": 0}
        for r in rows:
            if r["outcome"] in ("won", "lost"):
                count[r["outcome"]] += 1
                kind = (
                    "by_conquest" if r["ended_by"] == "conquest" else "at_cap" if r["ended_by"] in rs.CAP_ENDS else ""
                )
                if kind:
                    count[f"{r['outcome']}_{kind}"] += 1
        return count

    out: dict[str, Any] = {}
    for teacher in (n for n in teachers if any(r["teacher"] == n for r in final)):
        mine = sorted((r for r in final if r["teacher"] == teacher), key=lambda r: r["j"])
        theirs = {r["j"]: r for r in v0 if r["teacher"] == teacher}
        shared = [(r, theirs[r["j"]]) for r in mine if r["j"] in theirs]
        count = tally(mine)
        out[teacher] = {
            **count,
            "failed_calls_per_game": sum(r["failed_calls"] for r in mine) / len(mine),
            "spend_usd": round(sum(r["usd"] for r in mine), 4),
            "paired_with_v0": {
                "shared": len(shared),
                "qwen_won": sum(a["outcome"] == "won" for a, _ in shared),
                "qwen_lost": sum(a["outcome"] == "lost" for a, _ in shared),
                "v0_won": sum(b["outcome"] == "won" for _, b in shared),
                "v0_lost": sum(b["outcome"] == "lost" for _, b in shared),
                "games": [{"j": a["j"], "run1": a["outcome"], "v0": b["outcome"]} for a, b in shared],
            },
        }
    return out


def frontier_lines(detail: Mapping[str, Any]) -> list[str]:
    """One line per teacher: "opus: Qwen 11-5 (v0 4-4 on the shared 8)"."""
    lines = []
    for teacher, d in detail.items():
        p = d["paired_with_v0"]
        shared = f"v0 {p['v0_won']}-{p['v0_lost']} on the shared {p['shared']}" if p["shared"] else "no v0 game to pair"
        lines.append(
            f"{teacher}: Qwen {d['won']}-{d['lost']} ({d['won_by_conquest']} won by conquest, {d['won_at_cap']} at the"
            f" cap; {d['lost_by_conquest']}/{d['lost_at_cap']} lost), {d['failed_calls_per_game']:.1f} failed calls"
            f" a game, ${d['spend_usd']:.2f} ({shared})"
        )
    return lines


def step_run1_frontier(ctx: Ctx) -> Outcome:
    """Run 1's final adapter against Opus and Astra: it is copied into this root as version 1 and plays the frontier
    panel on the baselines' frontier starts. Gates: the panel finishes and the run ended normally. The detail counts
    each teacher's games from the Qwen seat's side, paired by game index with v0's frontier panel."""
    cfg, base = ctx.cfg("run1_frontier"), ctx.cfg("baselines")
    version, source = run1_adapter(ctx)
    copy = seed_adapter(source, cfg.root, RUN1_FRONTIER_VERSION)
    ctx.say(f"run1_frontier: Run 1's v{version} ({source}) is v{RUN1_FRONTIER_VERSION} in {cfg.root}")
    ctx.train_run(ctx.configs.file("run1_frontier"), ["--play-only"], label="play-only run")
    v0_panel = base.root / "panels" / str(ctx.settings["baselines"]["frontier_panel"])
    teachers = frontier_detail(
        frontier_rows(cfg.root / "panels" / RUN1_FRONTIER_PANEL),
        frontier_rows(v0_panel),
        [t.name for t in cfg.panels[0].teachers],
    )
    for line in frontier_lines(teachers):
        ctx.say(f"run1_frontier: {line}")
    gates = [
        *panel_gates(cfg.root, (RUN1_FRONTIER_PANEL,), PANEL_FINISHED_SHARE),
        ended_normally(cfg.root, ("complete",)),
    ]
    detail = {
        "adapter": {"source": str(source), "source_version": version, "copy": str(copy)},
        "teachers": teachers,
        "lines": frontier_lines(teachers),
    }
    return Outcome(gates, detail)


STEP_FUNCS: dict[str, Callable[[Ctx], Outcome]] = {
    "e1": step_e1,
    "baselines": step_baselines,
    "calibration": step_calibration,
    "tripwire": step_tripwire,
    "e2": step_e2,
    "e3_check": step_e3_check,
    "e3": step_e3,
    "e4_check": step_e4_check,
    "e4": step_e4,
    "labels": step_labels,
    "sft": step_sft,
    "sft_panels": step_sft_panels,
    "init": step_init,
    "run1": step_run1,
    "run1_frontier": step_run1_frontier,
}
QUICK = ("calibration", "init")  # reading files only: a few minutes, no box hours worth pricing
CAP_TABLE = {"labels": "distill"}


def step_hours(settings: Settings, step: str) -> float:
    """A step's wall-clock cap: its table's, or a few minutes for the steps that only read files."""
    return 0.1 if step in QUICK else settings.cap_h(CAP_TABLE.get(step, step))


def planned_usd(settings: Settings, step: str) -> float:
    """The frontier dollars a step may add to the ledger (its spend caps), which the budget guard counts ahead."""
    if step == "labels":
        return float(settings["distill"]["cap_usd"]) + float(settings["distill"]["second_cap_usd"])
    return float(settings[step].get("planned_usd", 0.0)) if step in settings.tables else 0.0


def sequence(ctx: Ctx) -> tuple[str, str | None]:
    """Run the steps; returns (verdict, reason). A step that passed in an earlier life of this record is skipped, as is
    one whose failure [sequence] accept_failed accepts (not while [sequence] rerun lists it); a step rerun lists runs
    again once, whatever its record says; the sequence stops after [sequence] stop_after."""
    for step in STEPS:
        reason = None
        if rerun_due(ctx.settings, step, ctx.record.data["steps"].get(step)):
            ctx.say(f"{step}: recorded {ctx.record.status(step)}, run again ([sequence] rerun)")
            reason = attempt(ctx, step, rerun=True)
        elif ctx.record.passed(step):
            ctx.say(f"{step}: already passed, skipped")
        elif (
            step in ctx.settings.accept_failed
            and step not in ctx.settings.rerun
            and ctx.record.status(step) in ("failed", "accepted")
        ):
            ctx.record.accept(step)
            ctx.say(f"{step}: its failure is accepted by hand ([sequence] accept_failed), skipped")
        else:
            reason = attempt(ctx, step)
        if reason:
            return "stopped", reason
        if step == ctx.settings.stop_after:
            reason = f"{step}: the sequence stops after it ([sequence] stop_after)"
            ctx.say(reason)
            return "stopped", reason
    return "complete", None


def attempt(ctx: Ctx, step: str, rerun: bool = False) -> str | None:
    """Run one step under its cap and record it; returns why the sequence must stop, or None if it passed."""
    hours = step_hours(ctx.settings, step)
    refused = ctx.budget_ok(hours, planned_usd(ctx.settings, step))
    ctx.record.begin(step, rerun)
    if refused:
        reason = f"{step}: the budget guard refuses it: {refused}"
        ctx.record.end(step, Outcome(reason=reason))
        return reason
    ctx.step, ctx.step_started, ctx.step_cap_h = step, ctx.now(), hours
    ctx.say(f"{step}: start (cap {hours:g} h)")
    try:
        outcome = STEP_FUNCS[step](ctx)
    except StepFailed as e:
        outcome = Outcome(reason=str(e))
    except KeyboardInterrupt:
        outcome = Outcome(reason=STOPPED_BY_HAND)
    except Exception as e:  # noqa: BLE001 - whatever it was, the sequence must still stop cleanly and upload
        log.exception("step %s crashed", step)
        outcome = Outcome(
            reason=f"{type(e).__name__}: {str(e)[:300]}", detail={"trace": traceback.format_exc()[-1500:]}
        )
    ctx.record.end(step, outcome)
    for gate in outcome.gates:
        ctx.say(f"{step}: {gate.line()}")
    ctx.say(f"{step}: {'PASSED' if outcome.passed else 'FAILED: ' + str(outcome.why())}")
    return None if outcome.passed else f"{step}: {outcome.why()}"


def final_report(ctx: Ctx) -> str | None:
    """s2-report.json: the report of the whole day over the folders of its steps (panels linked into one folder, as
    report.py reads them). Returns an error message, or None."""
    from agentenv_zeroad.train import report

    try:
        base, sft, run1 = ctx.cfg("baselines"), ctx.cfg("sft"), ctx.cfg("run1")
        panels = ctx.s2 / "report-panels"
        panels.mkdir(parents=True, exist_ok=True)
        b = ctx.settings["baselines"]
        links = {
            "v0-heldout": base.root / "panels" / b["heldout_panel"],
            "v0-v1-subset": base.root / "panels" / b["v1_panel"],
            "v0-frontier": base.root / "panels" / b["frontier_panel"],
            "teacher-calibration": base.root / "panels" / b["calibration_panel"],
            "run1-heldout": run1.root / "panels" / RUN1_END_PANELS[1],
            **{p.name: p for p in (sft.root / "panels").glob(f"{SFT_PREFIX}*")},
        }
        for name, target in links.items():
            link = panels / name
            if target.is_dir() and not link.exists():
                link.symlink_to(target, target_is_directory=True)
        has_run = (run1.root / "games").is_dir()
        games = run1.root / "games" if has_run else base.root / "games"
        built = report.build(
            games,
            run1.root if has_run else None,
            panels=panels,
            panel=("v0-heldout", "run1-heldout") if (panels / "run1-heldout").exists() else None,
            ledger=base.ledger_path,
            remote=run1.root / "remote" if has_run and (run1.root / "remote").is_dir() else None,
            sft_epochs=sft.root / "epochs" if (sft.root / "epochs").is_dir() else None,
        )
        report.write(ctx.s2 / "s2-report.json", built)
    except Exception as e:  # noqa: BLE001 - a report that cannot be built must not stop the upload
        log.exception("the final report failed")
        return f"{type(e).__name__}: {str(e)[:200]}"
    return None


FINAL_INCLUDES = ("gates.json", "status.json", "progress.log", "s2-report.json", "distill/*", "logs/*", "generated/*")


def upload_commands(s2: Path, uri: str, sft_epochs: Path) -> list[list[str]]:
    """`aws s3 sync` of the sequence's own files, then of the SFT checkpoints if the SFT step made any (the step roots
    upload themselves, but only manifests of adapters, and the SFT run has no orchestrator to do even that). `aws s3
    sync` fails on a missing source, which would keep a box that stopped before SFT from powering off."""
    base = uri.rstrip("/")
    own = ["aws", "s3", "sync", str(s2), base + "/s2", "--only-show-errors", "--exclude", "*"]
    for pattern in FINAL_INCLUDES:
        own += ["--include", pattern]
    if not sft_epochs.is_dir():
        return [own]
    return [own, ["aws", "s3", "sync", str(sft_epochs), base + "/s2/sft-epochs", "--only-show-errors"]]


def final_upload(ctx: Ctx, attempts: int = 3) -> bool:
    uri = ctx.settings.s3_uri
    if uri is None:
        return False
    commands = upload_commands(ctx.s2, uri, ctx.cfg("sft").root / "epochs")
    for i, argv in enumerate(commands):
        if not any(
            ctx.runner.run(argv, timeout_s=900, label=f"final-upload-{i}", grace_s=30) == 0 for _ in range(attempts)
        ):
            return False
    return True


def run(ctx: Ctx, *, shutdown: Callable[[], None] | None) -> int:
    ctx.record.save(None)
    verdict, reason = sequence(ctx)
    ctx.record.finish(verdict, reason)
    ctx.say(f"sequence {verdict}" + (f": {reason}" if reason else ""))
    error = final_report(ctx)
    ctx.say("final report: " + (error or "written"))
    ok = final_upload(ctx)
    ctx.say(f"final upload: {'done' if ok else 'FAILED or not configured'}")
    if shutdown is not None:
        if reason and STOPPED_BY_HAND in reason:
            ctx.say("a stop by hand: the box stays up")
        elif ctx.settings.s3_uri is not None and not ok:
            ctx.say("NOT powering off: the final upload failed and the files are only on this box's disk")
        else:
            ctx.say("powering off (the box stops, it is not terminated)")
            shutdown()
    return 0 if verdict == "complete" else 1


def power_off() -> None:
    poweroff.power_off()


# ---- the command line ----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentenv_zeroad.train.s2", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate a config directory and print its numbers")
    c.add_argument("dir", type=Path)
    c.add_argument("--budget", type=float)
    c.add_argument("--resume-from", type=Path, help="a sequence's folder: expect only the steps a --resume still runs")
    r = sub.add_parser("run", help="run the sequence (the zeroad-s2 unit's process)")
    r.add_argument("--config-dir", type=Path, required=True)
    r.add_argument("--budget", type=float, required=True)
    r.add_argument("--box-home", type=Path, default=Path(os.environ.get("ZEROAD_LIVE_HOME", "/opt/zeroad-live")))
    r.add_argument("--no-shutdown", action="store_true", help="end without `shutdown -h now` (tests, a dry look)")
    s = sub.add_parser("status", help="print gates.json's verdicts")
    s.add_argument("root", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "check":
        try:
            numbers = check_dir(args.dir, args.budget, args.resume_from)
        except (ValueError, OSError) as e:
            print(f"the S2 config directory is not valid: {e}", file=sys.stderr)
            return 1
        print(
            f"S2_ROOT={numbers['root']}\nS2_CAP_HOURS={numbers['cap_hours']:g}\nS2_EXPECTED_HOURS={numbers['expected_hours']:g}\n"
            f"S2_RUN1_HOURS={numbers['run1_hours']:g}\nS2_BOX_USD_PER_HOUR={numbers['rate']:g}\nS2_FRONTIER={int(numbers['frontier'])}"
        )
        return 0
    if args.cmd == "status":
        data = read_json(args.root / "gates.json")
        if not data:
            print("(no gates.json yet)")
            return 0
        print(f"verdict: {data['verdict']}" + (f" ({data['reason']})" if data.get("reason") else ""))
        for step, entry in data["steps"].items():
            print(
                f"{step}: {entry['status']}"
                + (f" in {entry.get('seconds')} s" if entry.get("seconds") is not None else "")
            )
            for gate in entry.get("gates", []):
                verdict = "PASS" if gate["passed"] else "FAIL"
                print(f"    {verdict} {gate['name']}: {gate['value']} {gate['op']} {gate['threshold']}")
            if entry.get("reason") and entry["status"] in ("failed", "accepted"):
                print(f"    reason: {entry['reason']}")
        return 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    settings = load_settings(args.config_dir / "sequence.toml")
    runner = SubprocessRunner(settings.root / "logs")
    ctx = Ctx(
        settings,
        ConfigDir(args.config_dir, settings),
        args.box_home,
        runner,
        args.budget,
        Record(settings.root, time.time),
    )
    return run(ctx, shutdown=None if args.no_shutdown else power_off)


if __name__ == "__main__":
    sys.exit(main())
