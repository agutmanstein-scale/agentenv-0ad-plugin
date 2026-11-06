"""The run config of a post-training run (`orchestrate`): one TOML file with everything a run needs and nothing secret.

    run_id = "run1"
    root = "/var/lib/zeroad-live/train/run1"    # inbox/, adapters/, games/, health.jsonl ... (the learner's --root too)
    budget_usd = 140
    box_usd_per_hour = 10.49                    # required: a default rate once priced a box at a third of its cost
    duration_hours = 10
    shutdown = false                            # `shutdown -h now` after the final upload

    [servers]
    urls = ["http://172.17.0.1:8000", "http://172.17.0.1:8001"]
    keep_versions = [12]                        # snapshots that stay loaded; version 0 is the base model, always served

    [games]                                     # time_cap_minutes, no_cap_share, interface, score_rule, civs, maps ...
    [mix]                                       # opponent weights: bot, petra-1 .. petra-5, base (the frozen v0)
    [held_out]                                  # civ_pairs, maps, seeds: never in a training game
    [s3]
    uri = "s3://bucket/prefix/run1"
    [hard_states]                               # optional: teacher labels on all-loss opponents (off)
    [credit]                                    # optional: train.credit.CreditConfig fields (margin_scale ...)
    [credit.critic]                             # optional: train.critic.CriticConfig fields; absent: no critic
    [[panels]]
    name = "mid"
    at_hours = 5
    specs = "/path/panel.jsonl"                 # the pre-registered configs: GameSpec lines, `trainable` seats pinned
    [[panels]]                                  # the teacher calibration: frontier seats against Petra, same latency
    name = "teacher-calibration"
    kind = "teacher_calibration"
    at_hours = 0
    teachers = [{ name = "astra", model = "openai/gpt-6-astra" }, { name = "opus", model = "..." }]

Panels need no spec file when they are generated (`train/panels.py`): `kind = "heldout"` (`games`, `opponents`, `interface`,
`positions`, `time_cap_minutes`) is the policy seat against an opponent mix on held-out positions, `kind = "frontier"` has
each teacher in `teachers` play the policy seat for its own `games`. `[held_out] seed_ranges = [[low, high]]` holds out a
block of seeds for panels of hundreds of games. A baseline or panel-only run sets `panels_only` (no training games),
`stop_when_done` (end with reason "complete" once every panel is played) and `brief_target` (and until the games hold the
briefs for that many distillation labels, `brief_per_game` at most from one game).

`python -m agentenv_zeroad.train.config CONFIG` checks a config (`check_run`: the panel spec files load, and one
training game can be made from the mix) and prints `TRAIN_RUN_ID=`, `TRAIN_ROOT=`, `TRAIN_DURATION_HOURS=`,
`TRAIN_BOX_USD_PER_HOUR=` and `TRAIN_FRONTIER=` (1 when a panel has a seat on a frontier model) for the box's shell
scripts to `eval`.

The vLLM containers mount the box's `$STATE/train` at `/train`, so a run's adapters are at `/train/<root relative to
train_root>/adapters` in them. `servers.adapter_dir` is derived from `root` that way; written out, it must agree. The
box's one ledger is `$STATE/spend.jsonl` and its box hours are `$STATE/boxtime.json` (`budget.sh`'s), both found from
`train_root`'s parent when `root` is under `train_root`.

A key the config doesn't know is an error, and so is a mix or map list that would put a held-out map, seed or civ pair
into a training game. The three amounts that decide when the run stops (`budget_usd`, `box_usd_per_hour`,
`duration_hours`) have no defaults.
"""

from __future__ import annotations

import shlex
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any

from agentenv_zeroad import grading
from agentenv_zeroad.clock import DEFAULT_LATENCY_MS
from agentenv_zeroad.interface import INTERFACES, Interface
from agentenv_zeroad.pilot.games import CIVS
from agentenv_zeroad.pool import TRAIN_IMAGE, TRAIN_PATCHES, physical_cores

RUN1_MIX = {"bot": 0.10, "petra-1": 0.35, "petra-2": 0.35, "petra-3": 0.10, "base": 0.10}
TRAIN_MOUNT = "/train"  # where the vLLM containers see the box's train folder
OPPONENTS = ("bot", "petra-1", "petra-2", "petra-3", "petra-4", "petra-5", "base")


def _checked(raw: Mapping[str, Any], cls: type, where: str) -> dict[str, Any]:
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {where} keys: {', '.join(sorted(unknown))}")
    return dict(raw)


@dataclass(frozen=True)
class ServerConfig:
    urls: tuple[str, ...] = ()  # one per vLLM instance, as the orchestrator's process reaches it
    api_key_env: str = "VLLM_API_KEY"
    adapter_dir: str = ""  # where the vLLM containers see ROOT/adapters; "" derives it from the run's root
    keep_versions: tuple[int, ...] = ()
    health_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.urls:
            raise ValueError("servers.urls needs at least one vLLM instance")


@dataclass(frozen=True)
class GameConfig:
    time_cap_minutes: float = 30.0
    no_cap_share: float = 0.15
    size: int = 128
    biome: str = "generic/temperate"
    placement: str = "circle"
    latency_ms: int = DEFAULT_LATENCY_MS
    interface: Interface = "v2"  # what the LLM seats (the policy and the frozen base) play
    score_rule: grading.ScoreRule | None = None  # None: the interface's own rule, net_worth on v2
    civs: tuple[str, ...] = CIVS
    maps: tuple[str, ...] = ("mainland",)
    seed_low: int = 1
    seed_high: int = 1_000_000

    def __post_init__(self) -> None:
        if self.seed_low >= self.seed_high:
            raise ValueError("games.seed_low must be below games.seed_high")
        if not 0 <= self.no_cap_share < 1:
            raise ValueError("games.no_cap_share is a share from 0 up to (not including) 1")
        if self.interface not in INTERFACES:
            raise ValueError(f"games.interface must be one of {', '.join(INTERFACES)}, not {self.interface!r}")
        if self.score_rule is not None and self.score_rule not in grading.SCORE_KEYS:
            raise ValueError(f"games.score_rule must be one of {', '.join(grading.SCORE_KEYS)}")
        grading.resolve_score_rule(self.score_rule, [self.interface])
        if not self.maps or not self.civs:
            raise ValueError("games.maps and games.civs can't be empty")
        unknown = set(self.civs) - set(CIVS)
        if unknown:
            raise ValueError(f"unknown games.civs: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class MixConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(RUN1_MIX))

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(OPPONENTS)
        if unknown:
            raise ValueError(f"unknown opponents in the mix: {', '.join(sorted(unknown))}")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("mix weights must be non-negative and not all zero")

    def shares(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {k: w / total for k, w in self.weights.items() if w > 0}


@dataclass(frozen=True)
class HeldOut:
    civ_pairs: tuple[tuple[str, str], ...] = ()
    maps: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    seed_ranges: tuple[
        tuple[int, int], ...
    ] = ()  # [low, high) blocks of held-out seeds, for panels of hundreds of games

    def __post_init__(self) -> None:
        for low, high in self.seed_ranges:
            if low >= high:
                raise ValueError(f"held_out.seed_ranges: {low}..{high} is empty")

    def is_held_seed(self, seed: int) -> bool:
        return seed in self.seeds or any(low <= seed < high for low, high in self.seed_ranges)

    def all_seeds(self) -> list[int]:
        """Every held-out seed, the listed ones first and then each range in turn."""
        return [*self.seeds, *(s for low, high in self.seed_ranges for s in range(low, high))]


@dataclass(frozen=True)
class TeacherConfig:
    """A frontier model that plays the teacher calibration panel (and may then label briefs)."""

    name: str  # what the report calls it: astra, opus
    model: str  # the LiteLLM model name
    effort: str | None = "low"
    max_tokens: int = 1024
    games: int = 0  # a frontier baseline panel: this teacher's games against the policy seat

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith(".") or not self.model:
            raise ValueError(f"a teacher needs a plain name and a model, not {self.name!r} / {self.model!r}")


PANEL_KINDS = ("specs", "teacher_calibration", "heldout", "frontier")
PANEL_POSITIONS = ("held_out", "training")
HELDOUT_MIX = {"petra-1": 0.3125, "petra-2": 0.3125, "petra-3": 0.25, "bot": 0.0625, "base": 0.0625}
DEFAULT_CALIBRATION_GAMES = 16  # 2 teachers x Petra-3 and Petra-4: 6 and 6 against Petra-3, 2 and 2 against Petra-4
MIN_PRIMARY_GAMES = 6  # at least this many games per teacher against the rung the pick is made on (Petra-3)


@dataclass(frozen=True)
class PanelConfig:
    """A panel of games played at `at_hours`: a file of game specs (`specs`), or the teacher calibration, whose specs
    `train.calibration` makes from the `teachers`: each plays Petra at `rungs` at the policy's own lockstep latency,
    `games` in all, so distillation picks the teacher that does best against Petra-3 (ties: Petra-4)."""

    name: str
    at_hours: float
    specs: Path | None = None
    version: int | None = None  # None: whichever version is current when the panel starts
    kind: str = "specs"
    teachers: tuple[TeacherConfig, ...] = ()
    rungs: tuple[int, ...] = (3, 4)  # the Petra difficulties a teacher plays
    games: int = DEFAULT_CALIBRATION_GAMES
    latency_ms: int | None = None  # None: games.latency_ms, the policy's
    base_url: str = ""  # "": LITELLM_BASE_URL, which the box's with-model.sh puts in the environment
    api_key_env: str = "LITELLM_API_KEY"
    opponents: Mapping[str, float] = field(
        default_factory=lambda: dict(HELDOUT_MIX)
    )  # kind heldout: the policy's opponents
    interface: Interface | None = None  # kinds heldout and frontier: what the policy seat plays; None: games.interface
    positions: str = (
        "held_out"  # kinds heldout and frontier: draw maps, civs and seeds from the held-out ones or the training ones
    )
    time_cap_minutes: float | None = None  # None: games.time_cap_minutes

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(f"panel name must be a plain folder name, not {self.name!r}")
        if self.at_hours < 0:
            raise ValueError("a panel's at_hours is not negative")
        if self.kind not in PANEL_KINDS:
            raise ValueError(f"panel kind must be one of {', '.join(PANEL_KINDS)}, not {self.kind!r}")
        if self.kind == "specs" and self.specs is None:
            raise ValueError(f"panel {self.name}: specs is required")
        if self.kind == "teacher_calibration":
            self._check_calibration()
        if self.kind in ("heldout", "frontier"):
            self._check_positions()

    def _check_positions(self) -> None:
        if self.positions not in PANEL_POSITIONS:
            raise ValueError(f"panel {self.name}: positions must be one of {', '.join(PANEL_POSITIONS)}")
        if self.interface is not None and self.interface not in INTERFACES:
            raise ValueError(f"panel {self.name}: interface must be one of {', '.join(INTERFACES)}")
        if self.time_cap_minutes is not None and self.time_cap_minutes <= 0:
            raise ValueError(f"panel {self.name}: time_cap_minutes must be positive")
        if self.kind == "heldout":
            unknown = set(self.opponents) - set(OPPONENTS)
            if unknown or sum(self.opponents.values()) <= 0 or any(w < 0 for w in self.opponents.values()):
                raise ValueError(f"panel {self.name}: opponents are weights over {', '.join(OPPONENTS)}")
            if self.games < 1:
                raise ValueError(f"panel {self.name}: games is at least 1")
        else:
            if not self.teachers or any(t.games < 1 for t in self.teachers):
                raise ValueError(f"panel {self.name}: a frontier panel needs teachers, each with games of at least 1")
            if len({t.name for t in self.teachers}) != len(self.teachers):
                raise ValueError(f"panel {self.name}: teacher names must be unique")

    def _check_calibration(self) -> None:
        if not self.teachers or not self.rungs:
            raise ValueError(f"panel {self.name}: a teacher calibration needs teachers and rungs")
        if len({t.name for t in self.teachers}) != len(self.teachers):
            raise ValueError(f"panel {self.name}: teacher names must be unique")
        if not all(0 <= r <= 5 for r in self.rungs):
            raise ValueError(f"panel {self.name}: rungs are Petra difficulties, 0 to 5")
        if self.games < len(self.teachers) * len(self.rungs):
            raise ValueError(f"panel {self.name}: games is below one per teacher and rung")
        if self.latency_ms is not None and self.latency_ms <= 0:
            raise ValueError(f"panel {self.name}: latency_ms must be positive")


@dataclass(frozen=True)
class HardStateConfig:
    """Teacher labels on hard states during RL (`train.hardstate`), off by default.

    Where the policy's rolling win rate against an opponent is below `threshold` (every group of games against it is an
    all-loss group, so the policy-gradient signal there is zero), up to `per_hour` of the decisions in those games are
    relabelled by the calibrated teacher, through distillation's labelling path, and sent to the learner as rows flagged
    sft, trained on by its auxiliary cross-entropy term. Labelling is capped at `cap_usd_per_hour`, and counts toward
    the run's budget through the ledger."""

    enabled: bool = False
    window: int = 40  # the policy's last this-many games against an opponent make its rolling win rate
    min_games: int = 20  # fewer than this many in the window: not enough to call an opponent hard
    threshold: float = 0.05  # hard: rolling win rate below this (draws count half)
    per_hour: int = 200  # decisions sampled for labelling per run hour
    cap_usd_per_hour: float = 10.0
    teacher: str = "auto"  # a LiteLLM model name, or auto: the calibration panel's best (best vs Petra-3, ties Petra-4)
    calibration: str = ""  # that panel's folder; "" is ROOT/panels/teacher-calibration (S2's, when copied here)
    tokenizer: str = "Qwen/Qwen3-8B"  # the policy's: the labels become completion token ids
    base_url: str = ""  # "": LITELLM_BASE_URL, which with-model.sh puts in the environment
    api_key_env: str = "LITELLM_API_KEY"
    batch: int = 20  # decisions asked per labelling pass
    per_game: int = 8  # most decisions taken from one game
    concurrency: int = 4
    interval_s: float = 60.0  # how often a labelling pass is considered
    max_pool: int = 4000  # decisions kept waiting; the oldest go first
    alpha_floor: float = 0.01  # no labels are bought once the learner's aux_alpha has decayed below this

    def __post_init__(self) -> None:
        if self.window < 1 or not 1 <= self.min_games <= self.window:
            raise ValueError("hard_states.min_games must be from 1 to window")
        if not 0 <= self.threshold <= 1:
            raise ValueError("hard_states.threshold is a win rate, 0 to 1")
        if min(self.per_hour, self.batch, self.per_game, self.concurrency, self.max_pool) < 1:
            raise ValueError("hard_states.per_hour, batch, per_game, concurrency and max_pool are at least 1")
        if self.cap_usd_per_hour <= 0 or self.interval_s <= 0:
            raise ValueError("hard_states.cap_usd_per_hour and interval_s must be positive")
        if self.alpha_floor < 0:
            raise ValueError("hard_states.alpha_floor is not negative")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    root: Path
    budget_usd: float
    box_usd_per_hour: float
    duration_hours: float
    servers: ServerConfig
    games: GameConfig = field(default_factory=GameConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    held_out: HeldOut = field(default_factory=HeldOut)
    panels: tuple[PanelConfig, ...] = ()
    hard_states: HardStateConfig = field(default_factory=HardStateConfig)
    s3_uri: str | None = None
    shutdown: bool = False
    engines: int = 0  # engine containers; 0 = the box's physical cores
    concurrency: int = 0  # games in flight; 0 = all the engines
    engine_image: str = TRAIN_IMAGE  # the engine image: one built with the patches/ below
    engine_patches: tuple[str, ...] = (
        TRAIN_PATCHES  # its zeroad.patches label must name these; () accepts the PPA image
    )
    seed: int = 0
    ledger: Path | None = None  # frontier spend, as budget.sh's spend.jsonl; default ROOT/spend.jsonl
    sync_interval_s: float = 900.0
    sync_retry_s: float = 30.0
    budget_interval_s: float = 600.0
    poll_s: float = 10.0
    drain_timeout_s: float = 900.0
    learner_grace_s: float = 120.0
    trip_limit: int = 2
    trip_window_h: float = 3.0
    max_requeues: int = 3
    train_root: str = "/var/lib/zeroad-live/train"  # the box's $STATE/train, which the vLLM containers mount at /train
    credit: Mapping[str, Any] = field(default_factory=dict)  # train.credit.CreditConfig fields
    critic: Mapping[str, Any] | None = None  # train.critic.CriticConfig fields; None: no critic
    max_load_failures: int = 5  # consecutive failed loads of one adapter into one instance before the run stops
    learner_stall_minutes: float = 30.0  # no new health record or adapter this long, with rows waiting: stop
    stall_rows: int = 256  # rows waiting in the inbox that a live learner would have taken
    min_free_gb: float | None = None  # None: ZEROAD_MIN_FREE_GB from the box config, else 8
    game_cap_usd: float | None = None  # per frontier seat, None: ZEROAD_MATCH_CAP_USD from the box config
    panels_only: bool = False  # start no training games: the panels are the whole run
    stop_when_done: bool = False  # end with reason "complete" once every panel has finished and brief_target is met
    brief_target: int = 0  # stop_when_done also waits until this many distillation briefs could be labelled (0: none)
    brief_per_game: int = 8  # distill.select's cap per game, which the supply is counted under

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id:
            raise ValueError(f"run_id must be a plain name, not {self.run_id!r}")
        if self.budget_usd <= 0 or self.box_usd_per_hour <= 0 or self.duration_hours <= 0:
            raise ValueError("budget_usd, box_usd_per_hour and duration_hours must be positive")
        if self.s3_uri is not None and not self.s3_uri.startswith("s3://"):
            raise ValueError("s3.uri must start with s3://")
        object.__setattr__(self, "engine_patches", tuple(self.engine_patches))
        if self.engines < 0 or self.concurrency < 0:
            raise ValueError("engines and concurrency can't be negative")
        if not self.engines:
            object.__setattr__(self, "engines", physical_cores())
        if not self.concurrency:
            object.__setattr__(self, "concurrency", self.engines)
        if not 1 <= self.concurrency <= self.engines:
            raise ValueError("concurrency must be from 1 to the number of engines")
        if self.brief_target < 0 or self.brief_per_game < 1:
            raise ValueError("brief_target is not negative and brief_per_game is at least 1")
        if self.panels_only and not self.panels:
            raise ValueError("panels_only needs at least one panel")
        self._resolve_adapter_dir()
        self._check_held_out()

    def _resolve_adapter_dir(self) -> None:
        """The container path of ROOT/adapters, derived from root (or checked against it when written out)."""
        explicit = self.servers.adapter_dir
        derived = self._derived_adapter_dir()
        if explicit:
            tail = PurePosixPath(explicit).parts[-2:]
            ok = explicit.rstrip("/") == derived if derived else tail == (self.root.name, "adapters")
            if not ok:
                want = derived or f".../{self.root.name}/adapters"
                raise ValueError(f"servers.adapter_dir {explicit!r} is not where vLLM sees this run's adapters: {want}")
        elif derived is None:
            raise ValueError(
                f"servers.adapter_dir is required: root {self.root} is not under train_root {self.train_root}"
            )
        else:
            object.__setattr__(self, "servers", replace(self.servers, adapter_dir=derived))

    def _derived_adapter_dir(self) -> str | None:
        try:
            rel = self.root.relative_to(self.train_root)
        except ValueError:
            return None
        return str(PurePosixPath(TRAIN_MOUNT) / rel.as_posix() / "adapters")

    @property
    def state_dir(self) -> Path | None:
        """The box's $STATE (the parent of train_root) when root is under train_root, else None (a test or a laptop)."""
        return Path(self.train_root).parent if self._derived_adapter_dir() is not None else None

    @property
    def boxtime_path(self) -> Path | None:
        return self.state_dir / "boxtime.json" if self.state_dir else None

    def _check_held_out(self) -> None:
        clash = set(self.games.maps) & set(self.held_out.maps)
        if clash:
            raise ValueError(f"held-out maps are also training maps: {', '.join(sorted(clash))}")
        for pair in self.held_out.civ_pairs:
            if len(pair) != 2 or not set(pair) <= set(self.games.civs):
                raise ValueError(f"held-out civ pair {pair!r} must be two of games.civs")
        if not self.training_civ_pairs():
            raise ValueError("every civ pair is held out")
        low, high = self.games.seed_low, self.games.seed_high
        if high - low <= 100_000 and all(self.held_out.is_held_seed(s) for s in range(low, high)):
            raise ValueError("every seed from games.seed_low up to seed_high is held out")

    @property
    def ledger_path(self) -> Path:
        """The box's ledger (`$STATE/spend.jsonl`, the one budget.sh counts) on a box, else ROOT/spend.jsonl."""
        return self.ledger or (self.state_dir or self.root) / "spend.jsonl"

    def training_civ_pairs(self) -> list[tuple[str, str]]:
        """Every (policy civ, opponent civ) a training game may use: all but the held-out pairs, in either order."""
        held = {frozenset(pair) for pair in self.held_out.civ_pairs}
        return [(a, b) for a in self.games.civs for b in self.games.civs if frozenset((a, b)) not in held]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunConfig:
        top = dict(raw)
        servers = _checked(top.pop("servers", {}), ServerConfig, "servers")
        games = _checked(top.pop("games", {}), GameConfig, "games")
        mix = top.pop("mix", None)
        held = _checked(top.pop("held_out", {}), HeldOut, "held_out")
        panels = [_checked(p, PanelConfig, "panel") for p in top.pop("panels", ())]
        for p in panels:
            teachers = [_checked(t, TeacherConfig, "teacher") for t in p.pop("teachers", ())]
            p["teachers"] = tuple(TeacherConfig(**t) for t in teachers)
            p["rungs"] = tuple(p.get("rungs", PanelConfig.rungs))
            if "opponents" in p:
                p["opponents"] = dict(p["opponents"])
            if "specs" in p:
                p["specs"] = Path(p["specs"])
        hard = _checked(top.pop("hard_states", {}), HardStateConfig, "hard_states")
        credit = dict(top.pop("credit", {}))
        critic = credit.pop("critic", None)
        s3 = top.pop("s3", {})
        if set(s3) - {"uri"}:
            raise ValueError(f"unknown s3 keys: {', '.join(sorted(set(s3) - {'uri'}))}")
        top = _checked(top, cls, "config")
        for required in ("run_id", "root", "budget_usd", "box_usd_per_hour", "duration_hours"):
            if required not in top:
                hint = " (no default: it must be the instance type's rate)" if "usd_per_hour" in required else ""
                raise ValueError(f"{required} is required{hint}")
        for key in ("root", "ledger"):
            if key in top:
                top[key] = Path(top[key])
        s3_uri = s3.get("uri", top.pop("s3_uri", None))
        return cls(
            servers=ServerConfig(**_tuples(servers)),
            games=GameConfig(**_tuples(games)),
            mix=MixConfig(weights=dict(mix)) if mix is not None else MixConfig(),
            held_out=HeldOut(
                civ_pairs=tuple(tuple(p) for p in held.get("civ_pairs", ())),
                maps=tuple(held.get("maps", ())),
                seeds=tuple(held.get("seeds", ())),
                seed_ranges=tuple((int(a), int(b)) for a, b in held.get("seed_ranges", ())),
            ),
            panels=tuple(PanelConfig(**p) for p in panels),
            hard_states=HardStateConfig(**hard),
            s3_uri=s3_uri,
            credit=credit,
            critic=dict(critic) if critic is not None else None,
            **top,
        )


def _tuples(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}


def load_config(path: Path) -> RunConfig:
    return RunConfig.from_dict(tomllib.loads(path.read_text()))


def check_run(cfg: RunConfig) -> list[str]:
    """What load_config can't know without reading files and building a game: every panel's spec file loads, and one
    training game can be made from the mix (so a bad civ, size or interface fails now, not at the first game). Returns
    the names of the environment variables the frontier seats of the panels need."""
    from agentenv_zeroad.spend import LOCAL_PREFIX
    from agentenv_zeroad.train.calibration import panel_specs
    from agentenv_zeroad.train.mix import GameFactory

    keys: set[str] = set()
    for panel in cfg.panels:
        if panel.specs is not None and not panel.specs.is_file():
            raise ValueError(f"panel {panel.name}: no spec file at {panel.specs}")
        for spec in panel_specs(cfg, panel):
            spec.match_config()
            keys |= {
                s.api_key_env
                for s in spec.seats
                if s.kind == "llm"
                and not s.trainable
                and s.api_key_env
                and not (s.model or "").startswith(LOCAL_PREFIX)
            }
    GameFactory(cfg).next(0, cfg.servers.urls[0]).match_config()
    if cfg.hard_states.enabled:
        keys.add(cfg.hard_states.api_key_env)
    return sorted(keys)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m agentenv_zeroad.train.config CONFIG", file=sys.stderr)
        return 2
    cfg = load_config(Path(args[0]))
    frontier = check_run(cfg)
    print(
        f"TRAIN_RUN_ID={shlex.quote(cfg.run_id)}\nTRAIN_ROOT={shlex.quote(str(cfg.root))}\n"
        f"TRAIN_DURATION_HOURS={cfg.duration_hours:g}\nTRAIN_BOX_USD_PER_HOUR={cfg.box_usd_per_hour:g}\n"
        f"TRAIN_FRONTIER={int(bool(frontier))}\nTRAIN_ENGINES={cfg.engines}\nTRAIN_CONCURRENCY={cfg.concurrency}\n"
        f"TRAIN_ENGINE_IMAGE={shlex.quote(cfg.engine_image)}\n"
        f"TRAIN_ENGINE_PATCHES={shlex.quote(' '.join(cfg.engine_patches))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
