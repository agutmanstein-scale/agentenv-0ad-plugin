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
  labels      `distill.py`: the teacher relabels the briefs under a spend cap
  sft         the learner in --sft mode, a checkpoint per epoch
  sft_panels  each epoch's adapter plays the first panel_games of the held-out panel; `stats.select_sft_checkpoint`
              picks one, which then plays the remaining configs, so it has all 384 that v0 has
  init        Run 1's starting point: the chosen adapter if it beats v0 on all 384 paired configs (one-sided p < 0.05),
              else the base model
  run1        the RL run, panels at its end; then the report, the final S3 upload and (unless --no-shutdown) the box's
              own `shutdown -h now`, which stops it (it is never terminated from here)

A failed gate, a step cap, a budget refusal or a tripwire pause stops the sequence with the reason in gates.json and
status.json; the report and the final upload still run. Every step is a subprocess (a stuck one gets SIGTERM at its cap,
so an orchestrator uploads, then SIGKILL after the grace), and every step that ends is on disk, so `--resume` after a
crash or an SSM hiccup skips the steps that passed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentenv_zeroad.train import gates as G
from agentenv_zeroad.train import poweroff, stats
from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train.config import RunConfig, check_run, load_config

log = logging.getLogger("s2")

STOPPED_BY_HAND = "stopped by hand (SIGTERM)"
STEPS = ("e1", "baselines", "calibration", "tripwire", "e2", "labels", "sft", "sft_panels", "init", "run1")
GRACE_S = 2400.0  # an orchestrator told to stop uploads for up to 3 x 10 minutes (sync.SYNC_TIMEOUT_S)
SFT_PREFIX = "sft-epoch"
SFT_REST = "sft-rest"  # the chosen epoch's second panel: the held-out configs the epoch panels did not play
BASELINE_FILE = "baseline.json"  # ROOT/s2/baseline.json, the tripwire baseline; its records are in the baselines root
DECISIONS_PER_UPDATE = 256  # the learner's batch: one health record per this many decisions

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
        "cap_hours": 11.0,
        "expected_hours": 10.3,
        "heldout_panel": "end-heldout",
        "frontier_panel": "end-frontier",
        "planned_usd": 30.0,
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

    def __getitem__(self, step: str) -> Mapping[str, Any]:
        return self.tables[step]

    def cap_h(self, step: str) -> float:
        return float(self.tables[step]["cap_hours"])

    @property
    def total_cap_hours(self) -> float:
        return sum(self.cap_h(s) for s in SCHEMA)

    @property
    def total_expected_hours(self) -> float:
        return sum(float(t["expected_hours"]) for t in self.tables.values())


def load_settings(path: Path) -> Settings:
    raw = tomllib.loads(path.read_text())
    unknown = set(raw) - {"root", "s3", *SCHEMA}
    if unknown:
        raise ValueError(f"unknown sequence.toml tables: {', '.join(sorted(unknown))}")
    s3 = dict(raw.get("s3", {}))
    if set(s3) - {"uri"}:
        raise ValueError(f"unknown [s3] keys: {', '.join(sorted(set(s3) - {'uri'}))}")
    tables = {}
    for step, defaults in SCHEMA.items():
        given = dict(raw.get(step, {}))
        extra = set(given) - set(defaults)
        if extra:
            raise ValueError(f"unknown [{step}] keys: {', '.join(sorted(extra))}")
        tables[step] = {**defaults, **given}
        if float(tables[step]["cap_hours"]) <= 0:
            raise ValueError(f"[{step}] cap_hours must be positive")
    return Settings(Path(raw.get("root", DEFAULT_ROOT)), s3.get("uri"), tables)


def render_sft_panels(template: str, versions: Mapping[int, int], games: int) -> str:
    """The SFT-panels config: its template plus one held-out panel per epoch, pinned to that epoch's adapter version."""
    blocks = "".join(
        f'\n[[panels]]\nname = "{SFT_PREFIX}{epoch}"\nkind = "heldout"\nat_hours = 0\n'
        f"version = {version}\ngames = {games}\n"
        for epoch, version in sorted(versions.items())
    )
    return template.rstrip() + "\n" + blocks


def render_rest_panel(template: str, version: int, start: int, games: int) -> str:
    """The chosen epoch's second config: the template plus one held-out panel of the configs `start` .. `start + games`
    of the held-out sequence (the ones the epoch panels did not play), pinned to the chosen adapter version."""
    return (
        template.rstrip()
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


def check_dir(path: Path, budget: float | None = None) -> dict[str, Any]:
    """Validate a config directory (every step's config through `check_run`, the guards the sequence relies on) and
    return its numbers. Raises ValueError with the first problem."""
    settings = load_settings(path / "sequence.toml")
    configs = ConfigDir(path, settings)
    rates, frontier = set(), False
    for step in ("e1", "e2", "baselines", "sft", "sft_panels", "run1"):
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
    if len(rates) != 1:
        raise ValueError(f"the step configs name different box rates: {sorted(rates)}")
    _check_panel_names(configs, settings)
    if configs.load("e2").credit.get("positive_control") is not True:
        raise ValueError("e2: [credit] positive_control must be true")
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
        "expected_hours": settings.total_expected_hours,
        "run1_hours": configs.load("run1").duration_hours,
        "rate": rates.pop(),
        "frontier": frontier,
    }


def _check_panel_names(configs: ConfigDir, settings: Settings) -> None:
    wanted = {
        "baselines": ("heldout_panel", "v1_panel", "frontier_panel", "calibration_panel"),
        "run1": ("heldout_panel", "frontier_panel"),
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
        return self.data["steps"].get(step, {}).get("status") == "passed"

    def begin(self, step: str) -> None:
        self.data["steps"][step] = {
            "status": "running",
            "started": self.now(),
            "ended": None,
            "gates": [],
            "detail": {},
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


def step_e2(ctx: Ctx) -> Outcome:
    s, cfg = ctx.settings["e2"], ctx.cfg("e2")
    ctx.train_run(ctx.configs.file("e2"), baseline_flags(ctx), label="control run")
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


def baseline_records(ctx: Ctx) -> int:
    """Health records behind ROOT/s2/baseline.json (the stats-only learner's, in the baselines root); 0 if none."""
    path = ctx.cfg("baselines").root / "baseline-records.jsonl"
    with contextlib.suppress(OSError):
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    return 0


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
            rest.write_text(
                render_rest_panel(text, versions[int(str(chosen).removeprefix("epoch"))], first, total - first)
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
def step_run1(ctx: Ctx) -> Outcome:
    cfg = ctx.cfg("run1")
    init = (ctx.record.data["steps"].get("init", {}).get("detail") or {}).get("init_path")
    flags: list[str] = baseline_flags(ctx)
    if init:
        if not Path(init).is_relative_to(ctx.cfg("sft").root):
            raise StepFailed(f"Run 1's init {init} is not an SFT checkpoint: no E2 adapter may carry over")
        flags += ["--init", init, "--anchor", init]
    ctx.say(f"run1: {'from ' + init if init else 'from the base model'}")
    ctx.train_run(ctx.configs.file("run1"), flags, "rl run")
    reason = final_reason(cfg.root)
    trips = trips_in(cfg.root / "health.jsonl")
    return Outcome([ended_normally(cfg.root, ("duration",))], {"reason": reason, "trips": trips, "init": init})


STEP_FUNCS: dict[str, Callable[[Ctx], Outcome]] = {
    "e1": step_e1,
    "baselines": step_baselines,
    "calibration": step_calibration,
    "tripwire": step_tripwire,
    "e2": step_e2,
    "labels": step_labels,
    "sft": step_sft,
    "sft_panels": step_sft_panels,
    "init": step_init,
    "run1": step_run1,
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
    """Run the steps; returns (verdict, reason). A step that passed in an earlier life of this record is skipped."""
    for step in STEPS:
        if ctx.record.passed(step):
            ctx.say(f"{step}: already passed, skipped")
            continue
        hours = step_hours(ctx.settings, step)
        refused = ctx.budget_ok(hours, planned_usd(ctx.settings, step))
        ctx.record.begin(step)
        if refused:
            reason = f"{step}: the budget guard refuses it: {refused}"
            ctx.record.end(step, Outcome(reason=reason))
            return "stopped", reason
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
        if not outcome.passed:
            return "stopped", f"{step}: {outcome.why()}"
    return "complete", None


def final_report(ctx: Ctx) -> str | None:
    """s2-report.json: the report of the whole day over the folders of its steps (panels linked into one folder, as
    report.py reads them). Returns an error message, or None."""
    from agentenv_zeroad.train import report

    try:
        base, sft, run1 = ctx.cfg("baselines"), ctx.cfg("sft"), ctx.cfg("run1")
        panels = ctx.s2 / "report-panels"
        panels.mkdir(parents=True, exist_ok=True)
        b, r = ctx.settings["baselines"], ctx.settings["run1"]
        links = {
            "v0-heldout": base.root / "panels" / b["heldout_panel"],
            "v0-v1-subset": base.root / "panels" / b["v1_panel"],
            "v0-frontier": base.root / "panels" / b["frontier_panel"],
            "teacher-calibration": base.root / "panels" / b["calibration_panel"],
            "run1-heldout": run1.root / "panels" / r["heldout_panel"],
            "run1-frontier": run1.root / "panels" / r["frontier_panel"],
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
            numbers = check_dir(args.dir, args.budget)
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
            if entry.get("reason") and entry["status"] == "failed":
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
