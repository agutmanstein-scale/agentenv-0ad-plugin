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
    [credit]                                    # optional: train.credit.CreditConfig fields (margin_scale ...)
    [credit.critic]                             # optional: train.critic.CriticConfig fields; absent: no critic
    [[panels]]
    name = "mid"
    at_hours = 5
    specs = "/path/panel.jsonl"                 # the pre-registered configs: GameSpec lines, `trainable` seats pinned

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


@dataclass(frozen=True)
class PanelConfig:
    name: str
    at_hours: float
    specs: Path
    version: int | None = None  # None: whichever version is current when the panel starts

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(f"panel name must be a plain folder name, not {self.name!r}")
        if self.at_hours < 0:
            raise ValueError("a panel's at_hours is not negative")


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
    s3_uri: str | None = None
    shutdown: bool = False
    engines: int = 24
    concurrency: int = 24
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

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id:
            raise ValueError(f"run_id must be a plain name, not {self.run_id!r}")
        if self.budget_usd <= 0 or self.box_usd_per_hour <= 0 or self.duration_hours <= 0:
            raise ValueError("budget_usd, box_usd_per_hour and duration_hours must be positive")
        if self.s3_uri is not None and not self.s3_uri.startswith("s3://"):
            raise ValueError("s3.uri must start with s3://")
        if not 1 <= self.concurrency <= self.engines:
            raise ValueError("concurrency must be from 1 to the number of engines")
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
        if len({s for s in self.held_out.seeds if low <= s < high}) >= high - low:
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
            ),
            panels=tuple(PanelConfig(**{**p, "specs": Path(p["specs"])}) for p in panels),
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
    from agentenv_zeroad.train.mix import GameFactory
    from agentenv_zeroad.train.spec import load_specs

    keys: set[str] = set()
    for panel in cfg.panels:
        if not panel.specs.is_file():
            raise ValueError(f"panel {panel.name}: no spec file at {panel.specs}")
        for spec in load_specs(panel.specs):
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
        f"TRAIN_FRONTIER={int(bool(frontier))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
