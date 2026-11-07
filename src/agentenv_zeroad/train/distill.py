"""The distillation probe: a frontier teacher relabels briefs from finished v2 games, the labels train an SFT adapter.

The question (docs/post-training-v2.md): is Qwen's gap to the frontier models a matter of knowing what to do, which
supervised data can fix, or of finding out through play, which needs RL? The probe puts the teacher's answers to about
3,000 briefs in front of the learner's SFT mode (`trainer/learner.py --sft`) and plays the result on the panel.

Briefs come from finished v2 games the policy already played (a game whose policy seat's recorded interface is not v2
is skipped: v1's prompt has no hold or waves, and a label written under it would not fit the v2 prompt ids it is
trained on; so is one with no recorded reference text, whose policy prompt can't be rebuilt). They are chosen on the
student's own states, DAgger-style (the teacher labels what the student reached, not what the teacher would have
reached: the literature finds that beats SFT on the teacher's games), weighted to where a good answer is worth most:
  losing    the policy was behind on net worth at decision time in a game it went on to lose to Petra-2 or Petra-3
            (`losing_opponents`): 60% by default
  rejected  turns where one of its orders was rejected: 20%
  late      turns from minute 20 on: 20%
at most `per_game` from any game; a stratum that cannot fill its share hands the rest on, and what is left over is
`other`. A turn goes in the first of the strata above, in this order (losing, rejected, late), that it fits.
`--weights losing=0.5,late=0.25,rejected=0.25` changes the shares, and the default is 3,000 labels. A turn is one
brief: the user message the policy saw, with the exact prompt token ids it was sampled from, so the label sits on
the same prompt the policy will meet. The same labelling path labels the orchestrator's hard-state turns during RL
(`hardstate.py`).

The teacher gets the policy's system prompt and tools for the brief's interface (`default_policy`: `brains.
strategist_system(interface)` and `server.tool_specs()` as `brains.offered_tools` offers it to that interface; the
CLI's `--system` and `--tools` override them), and the seat's own per-civ cost tables (the `reference` its game
recorded) after the prompt, as `Commander.system` composes it, through the same client the strategist uses (`llm.py`),
under a `SpendGuard` cap. Each call reserves the most it could cost, under a lock, before it is sent, so the calls in
flight together never take spend past the cap; labelling stops at the first call that does not fit, with what it has.
Its tool calls become the completion in Qwen's tool-call format, `<tool_call>\\n{"name": ..., "arguments":
...}\\n</tool_call>` for each call, then `<|im_end|>`, tokenized with the model's tokenizer. A teacher call that names
a tool the policy does not have, or whose arguments are not a JSON object, is left out; a brief left with no call is
dropped.

The label-noise check sends a random subset, an equal share from each stratum, to a second teacher too and reports the
Jaccard similarity of the two tool sets per brief, overall and per stratum: if two frontier models disagree on which
tools to call, the labels are noisier than the SFT can resolve.

SFT rows are `{game_id, version: 0, sft: true, prompt_token_ids, token_ids, teacher, stratum, interface}`: the ids the
learner reads (the extra fields are ignored; `sft: true` is what makes an RL learner treat the row as an auxiliary
teacher label), one JSON object per line. `write_inbox` writes them as one inbox file; the SFT learner makes its own
epochs over them (`learner.py --sft-epochs`).

The teacher is named with `--teacher`, or picked from the teacher calibration panel with `--calibration DIR` (its
folder under ROOT/panels): the frontier model with the best win rate against Petra-3, ties broken by Petra-4
(`stats.pick_teacher`), and the runner-up as the label-noise check's second teacher.

  python -m agentenv_zeroad.train.distill --games DIR --out DIR --tokenizer Qwen/Qwen3-8B \\
      --calibration ROOT/panels/teacher-calibration --base-url URL --api-key-env VAR --cap-usd 60
  python -m agentenv_zeroad.train.distill --games DIR --out DIR --tokenizer Qwen/Qwen3-8B \\
      --teacher anthropic/claude-opus-5-5 --second-teacher openai/gpt-6-astra --base-url URL --api-key-env VAR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from agentenv_zeroad import brains, spend
from agentenv_zeroad.interface import Interface
from agentenv_zeroad.llm import DEFAULT_TIMEOUT_S, Reply, StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendCapReached, SpendGuard
from agentenv_zeroad.train import results as rs
from agentenv_zeroad.train import stats

IM_END = "<|im_end|>"
INTERFACE: Interface = "v2"
STRATA = ("losing", "rejected", "late", "other")
LATE_MIN = 20.0
LOSING_OPPONENTS = ("petra-2", "petra-3")
DEFAULT_WEIGHTS = {"losing": 0.6, "rejected": 0.2, "late": 0.2}
DEFAULT_LABELS = 3000


class Teacher(Protocol):
    """What labelling needs of a model client (a `StrategistModel` is one)."""

    model: str
    guard: SpendGuard
    max_tokens: int

    def request(self, system: str, user: str, tools: list[ToolSpec]) -> tuple[str, dict[str, Any]]: ...

    async def call(self, system: str, user: str, tools: list[ToolSpec]) -> Reply: ...


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


class HubTokenizer:
    """A model's tokenizer from the `tokenizers` library (`Tokenizer.from_pretrained`), which the base install has,
    unlike transformers (`name` is a hub id, or a local tokenizer.json). It reads the same tokenizer.json
    `AutoTokenizer` does, so the ids are the same; special tokens written in the text (`<|im_end|>`) are matched as
    added tokens whether or not `add_special_tokens` is set."""

    def __init__(self, name: str):
        from tokenizers import Tokenizer as _Tokenizer

        path = Path(name)
        self._inner = _Tokenizer.from_file(str(path)) if path.is_file() else _Tokenizer.from_pretrained(name)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._inner.encode(text, add_special_tokens=add_special_tokens).ids)


@dataclass(frozen=True)
class Policy:
    """What the policy sees besides the brief: the system prompt and the tool list of one interface."""

    system: str
    tools: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class Brief:
    game_id: str
    slot: str
    began_ms: int
    user: str
    prompt_token_ids: tuple[int, ...]
    stratum: str
    interface: str = INTERFACE
    reference: str = ""  # the seat's cost tables, which its system prompt carried after the prompt text
    opponent: str = ""  # the opponent's id (petra-3, v0, ...)

    @property
    def key(self) -> str:
        return f"{self.game_id}:{self.slot}:{self.began_ms}"


# ---- choosing the briefs ----------------------------------------------------------------------------------------
def candidates(
    games: Path, learners: Sequence[str] | None = None, losing_opponents: Collection[str] | None = LOSING_OPPONENTS
) -> list[Brief]:
    """Every usable turn of every finished v2 game's policy seat, each tagged with its stratum."""
    out: list[Brief] = []
    for path in sorted(games.glob("*/results.json")):
        out += briefs_of(json.loads(path.read_text()), path.parent.name, learners, losing_opponents)
    return out


def briefs_of(
    results: Mapping[str, Any],
    name: str,
    learners: Sequence[str] | None = None,
    losing_opponents: Collection[str] | None = LOSING_OPPONENTS,
) -> list[Brief]:
    """The usable turns of one game's results (`name` is its id when the spec has none); none for a game that did not
    finish, was not played on v2 or recorded no reference text."""
    if results.get("status") != "finished":
        return []
    slot = rs.policy_slot(results, learners)
    opp = rs.other_slot(results, slot) if slot else None
    if slot is None or opp is None or results["seats"][slot].get("interface") != INTERFACE:
        return []
    reference = str(results["seats"][slot].get("reference") or "")
    if not reference:
        return []
    game_id = str((results.get("spec") or {}).get("game_id") or name)
    out = []
    for turn in results["seats"][slot].get("turns") or []:
        if not (turn.get("user") and turn.get("prompt_token_ids") and turn.get("began_ms") is not None):
            continue
        stratum = _stratum(results, slot, opp, turn, losing_opponents)
        out.append(
            Brief(
                game_id,
                slot,
                int(turn["began_ms"]),
                turn["user"],
                tuple(turn["prompt_token_ids"]),
                stratum,
                reference=reference,
                opponent=rs.player_id(results, opp),
            )
        )
    return out


def _stratum(
    results: Mapping[str, Any],
    slot: str,
    opp: str,
    turn: Mapping[str, Any],
    losing_opponents: Collection[str] | None = LOSING_OPPONENTS,
) -> str:
    """losing, else rejected, else late (from minute 20), else other.

    Losing is the DAgger stratum: the policy's net worth was below its opponent's when it decided (the official score
    when a net worth was not read) in a game it went on to lose against one of `losing_opponents` (None: any)."""
    began = turn["began_ms"]
    if results["seats"][slot].get("outcome") == "lost" and (
        losing_opponents is None or rs.player_id(results, opp) in losing_opponents
    ):
        players = (rs.timeline_at(results, began) or {}).get("players") or {}
        mine, theirs = players.get(slot) or {}, players.get(opp) or {}
        key = "net_worth" if mine.get("net_worth") is not None and theirs.get("net_worth") is not None else "total"
        if (mine.get(key) or 0) < (theirs.get(key) or 0):
            return "losing"
    if any(o.get("status") == "rejected" for o in turn.get("orders") or []):
        return "rejected"
    return "late" if began / 60_000 >= LATE_MIN else "other"


def parse_weights(text: str) -> dict[str, float]:
    """`losing=0.5,late=0.25,rejected=0.25` as stratum weights (any positive numbers: they are shares of their sum)."""
    weights: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, value = part.partition("=")
        if name not in STRATA[:-1] or not value:
            raise ValueError(f"a weight is STRATUM=NUMBER with STRATUM one of {', '.join(STRATA[:-1])}: {part!r}")
        weights[name] = float(value)
    if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    return weights


def select(
    pool: Sequence[Brief],
    n: int,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    per_game: int = 8,
    seed: int = 0,
) -> list[Brief]:
    """Up to `n` briefs, `weights` of them from each stratum, at most `per_game` from a game.

    A stratum that cannot fill its share hands the rest to the others in turn, then to the `other` stratum; fewer than
    `n` come back only when the pool is exhausted."""
    rng = random.Random(seed)
    by: dict[str, list[Brief]] = defaultdict(list)
    for b in pool:
        by[b.stratum].append(b)
    for items in by.values():
        items.sort(key=lambda b: b.key)
        rng.shuffle(items)
    taken: dict[str, int] = defaultdict(int)
    chosen: list[Brief] = []

    def take(stratum: str, count: int) -> int:
        got, left = 0, by[stratum]
        for b in list(left):
            if got >= count:
                break
            if taken[b.game_id] < per_game:
                taken[b.game_id] += 1
                chosen.append(b)
                left.remove(b)
                got += 1
        return got

    total = sum(weights.values())
    for stratum, w in weights.items():
        take(stratum, round(n * w / total))
    order = [*weights, "other"]
    while len(chosen) < n:
        before = len(chosen)
        for stratum in order:
            if len(chosen) < n:
                take(stratum, 1)
        if len(chosen) == before:
            break
    return chosen[:n]


class BriefSupply:
    """How many briefs `select` could draw from the games seen so far, without keeping the briefs: per game, how many
    turns each stratum has. The orchestrator of a baseline run adds each finished game and stops when there are enough
    (`RunConfig.brief_target`); the count follows `select` (per-stratum shares, then the leftovers handed on, at most
    `per_game` from a game), so what it promises is what the labelling pass will find."""

    def __init__(self, n: int, weights: Mapping[str, float] = DEFAULT_WEIGHTS, per_game: int = 8):
        self.n, self.weights, self.per_game = n, dict(weights), per_game
        self.by_game: dict[str, dict[str, int]] = {}

    def add(self, game_id: str, briefs: Sequence[Brief]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for b in briefs:
            counts[b.stratum] += 1
        if counts:
            self.by_game[game_id] = dict(counts)

    def decisions(self) -> int:
        return sum(sum(c.values()) for c in self.by_game.values())

    def capacity(self) -> dict[str, Any]:
        """{"selectable": briefs `select(n)` would return, "shares_met": whether every stratum filled its own share,
        "strata": what was drawn from each, "candidates": every usable decision seen}."""
        left = {g: dict(c) for g, c in sorted(self.by_game.items())}
        taken: dict[str, int] = defaultdict(int)
        drawn: dict[str, int] = defaultdict(int)

        def take(stratum: str, count: int) -> int:
            got = 0
            for game, counts in left.items():
                if got >= count:
                    break
                grab = min(counts.get(stratum, 0), count - got, self.per_game - taken[game])
                if grab > 0:
                    counts[stratum] -= grab
                    taken[game] += grab
                    got += grab
            drawn[stratum] += got
            return got

        total, shares_met = sum(self.weights.values()), True
        for stratum, w in self.weights.items():
            want = round(self.n * w / total)
            shares_met &= take(stratum, want) >= want
        order = [*self.weights, "other"]
        while sum(drawn.values()) < self.n:
            before = sum(drawn.values())
            for stratum in order:
                if sum(drawn.values()) < self.n:
                    take(stratum, 1)
            if sum(drawn.values()) == before:
                break
        return {
            "selectable": min(sum(drawn.values()), self.n), "shares_met": shares_met, "strata": dict(drawn),
            "candidates": self.decisions(),
        }  # fmt: skip


# ---- labelling --------------------------------------------------------------------------------------------------
@dataclass
class Label:
    brief: Brief
    calls: list[tuple[str, dict[str, Any]]]


def completion_text(calls: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    """The assistant turn as Qwen writes a tool-call turn, up to and including the end-of-turn token."""
    blocks = [
        "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
        for name, args in calls
    ]
    return "\n".join(blocks) + IM_END


def usable_calls(reply: Reply, tools: Sequence[ToolSpec]) -> list[tuple[str, dict[str, Any]]]:
    """The reply's calls that name a tool the policy has and carry JSON-object arguments."""
    known = {t.name for t in tools}
    return [(c.name, c.arguments) for c in reply.calls if c.error is None and c.name in known]


async def default_policy(interface: Interface) -> Policy:
    """The system prompt and order tools a seat of `interface` is shown: the prompt text and the env's tool list as
    its MCP clients and the in-process link get them (`server.tool_specs()`) through `brains.offered_tools`."""
    from agentenv_zeroad.server import tool_specs

    return Policy(brains.strategist_system(interface), tuple(brains.offered_tools(await tool_specs(), interface)))


def system_for(policy: Policy, brief: Brief) -> str:
    """The system prompt the policy had for this brief: its text, then the seat's cost tables, as `Commander.system`."""
    return f"{policy.system}\n\n{brief.reference}" if brief.reference else policy.system


async def label(
    briefs: Sequence[Brief],
    teacher: Teacher,
    policies: Mapping[str, Policy],
    *,
    concurrency: int = 8,
    on_asked: Callable[[Brief], None] | None = None,
) -> tuple[list[Label], dict[str, Any]]:
    """The teacher's usable tool calls for each brief it could be asked, and what happened (counts, spend, stop).

    Each brief is asked under its own interface's `policies` entry. A call reserves its bound (`spend.bound`) before
    it is sent and releases it once the guard has recorded its cost, so concurrent calls cannot jointly pass the
    cap; the first call that does not fit ends the labelling. `on_asked(brief)` is called for each brief the teacher
    was actually sent (answered or failed), so a caller can tell the ones the cap left unasked."""
    gate = asyncio.Semaphore(concurrency)
    reserve = asyncio.Lock()
    reserved = 0.0
    stats: dict[str, Any] = {"asked": 0, "labelled": 0, "no_calls": 0, "failed": 0, "stopped": None}
    labels: dict[int, Label] = {}

    async def one(i: int, brief: Brief) -> None:
        nonlocal reserved
        policy = policies[brief.interface]
        tools, system = list(policy.tools), system_for(policy, brief)
        _, body = teacher.request(system, brief.user, tools)
        most = spend.bound(teacher.model, body, teacher.max_tokens)
        async with gate:
            async with reserve:
                if stats["stopped"]:
                    return
                guard = teacher.guard
                if guard.cap_usd is not None and guard.spent_usd + reserved + most > guard.cap_usd:
                    stats["stopped"] = (
                        f"the next call could cost ${most:.4f}, past the ${guard.cap_usd:g} cap with "
                        f"${guard.spent_usd:.4f} spent and ${reserved:.4f} in flight"
                    )
                    return
                reserved += most
            try:
                reply = await teacher.call(system, brief.user, tools)
            except SpendCapReached as e:
                stats["stopped"] = str(e)
                return
            except Exception as e:  # noqa: BLE001 - a call that may have been sent counts as asked and failed
                stats["asked"] += 1
                if on_asked:
                    on_asked(brief)
                stats["failed"] += 1
                stats.setdefault("errors", []).append(f"{type(e).__name__}: {str(e)[:120]}")
                return
            finally:
                async with reserve:
                    reserved -= most
        stats["asked"] += 1
        if on_asked:
            on_asked(brief)
        calls = usable_calls(reply, tools)
        if calls:
            labels[i] = Label(brief, calls)
            stats["labelled"] += 1
        else:
            stats["no_calls"] += 1

    await asyncio.gather(*(one(i, b) for i, b in enumerate(briefs)))
    stats["spend"] = teacher.guard.summary()
    return [labels[i] for i in sorted(labels)], stats


def sft_rows(labels: Sequence[Label], tokenizer: Tokenizer, teacher: str) -> list[dict[str, Any]]:
    """Learner SFT rows: the policy's own prompt ids and the teacher's calls as completion tokens."""
    rows = []
    for item in labels:
        rows.append(
            {
                "game_id": item.brief.key,
                "version": 0,
                "sft": True,
                "prompt_token_ids": list(item.brief.prompt_token_ids),
                "token_ids": list(tokenizer.encode(completion_text(item.calls), add_special_tokens=False)),
                "teacher": teacher,
                "stratum": item.brief.stratum,
                "interface": item.brief.interface,
                "opponent": item.brief.opponent,
            }
        )
    return rows


def noise_check(first: Sequence[Label], second: Sequence[Label]) -> dict[str, Any]:
    """How much two teachers agree on which tools to call, over the briefs both labelled, overall and per stratum."""
    other = {item.brief.key: item for item in second}
    scores: dict[str, list[float]] = defaultdict(list)
    same: dict[str, list[bool]] = defaultdict(list)
    for item in first:
        twin = other.get(item.brief.key)
        if twin is None:
            continue
        a, b = {n for n, _ in item.calls}, {n for n, _ in twin.calls}
        for group in ("all", item.brief.stratum):
            scores[group].append(len(a & b) / len(a | b))
            same[group].append(a == b)

    def summary(group: str) -> dict[str, Any]:
        n = len(scores[group])
        return {
            "briefs": n,
            "tool_set_jaccard": statistics.fmean(scores[group]) if n else None,
            "identical_tool_sets": sum(same[group]) / n if n else None,
        }

    return {**summary("all"), "by_stratum": {s: summary(s) for s in STRATA if scores[s]}}


def check_sample(labels: Sequence[Label], n: int, *, seed: int = 0) -> list[Brief]:
    """`n` briefs drawn at random, an equal share from each stratum with labels (a short one hands on its rest)."""
    rng = random.Random(seed)
    by: dict[str, list[Brief]] = defaultdict(list)
    for item in labels:
        by[item.brief.stratum].append(item.brief)
    for items in by.values():
        items.sort(key=lambda b: b.key)
        rng.shuffle(items)
    out: list[Brief] = []
    while len(out) < n and any(by.values()):
        for stratum in [s for s in STRATA if by[s]]:
            if len(out) < n:
                out.append(by[stratum].pop())
    return out


async def distill(
    briefs: Sequence[Brief],
    teacher: Teacher,
    second: Teacher | None,
    policies: Mapping[str, Policy],
    tokenizer: Tokenizer,
    *,
    check: int = 200,
    concurrency: int = 8,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label every brief with `teacher` and a stratified random `check` of them with `second`; SFT rows and a report."""
    labels, stats = await label(briefs, teacher, policies, concurrency=concurrency)
    report: dict[str, Any] = {"teacher": teacher.model, "briefs": len(briefs), **stats}
    if second is not None and check > 0:
        sample = check_sample(labels, check, seed=seed)
        twins, twin_stats = await label(sample, second, policies, concurrency=concurrency)
        report["second_teacher"] = {"model": second.model, **twin_stats, **noise_check(labels, twins)}
    return sft_rows(labels, tokenizer, teacher.model), report


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_inbox(rows: Sequence[Mapping[str, Any]], inbox: Path, *, seed: int = 0, name: str = "sft") -> Path:
    """The rows, shuffled, as one learner inbox file `name.jsonl`, written as *.tmp and renamed into place. The SFT
    learner reads a dataset that arrives whole and makes its own epochs over it."""
    inbox.mkdir(parents=True, exist_ok=True)
    order = list(rows)
    random.Random(seed).shuffle(order)
    final = inbox / f"{name}.jsonl"
    tmp = final.with_name(final.name + ".tmp")
    write_rows(tmp, order)
    tmp.replace(final)
    return final


# ---- the command line -------------------------------------------------------------------------------------------
def teachers_from_calibration(folder: Path) -> list[str]:
    """The calibration panel's teachers' model names, best first (`stats.rank_teachers`); SystemExit when none played
    Petra-3, because there is then nothing to pick on."""
    from agentenv_zeroad.train import report

    summary = report.teacher_calibration(folder)
    ranked = stats.rank_teachers(summary) if summary else []
    if not ranked:
        raise SystemExit(f"no teacher in {folder} has finished a game against {stats.CALIBRATION_RUNG}")
    return [summary["teachers"][name]["model"] or name for name in ranked]


def _teacher(
    model: str, base_url: str, key_env: str, cap: float, http: httpx.AsyncClient, ledger: Path | None
) -> StrategistModel:
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"the environment variable {key_env} is not set")
    guard = SpendGuard(f"distill {model}", cap_usd=cap, log_path=ledger)
    return StrategistModel(model, base_url, key, guard, effort="low", max_tokens=1024, http=http)


async def _policy(args: argparse.Namespace) -> Policy:
    """The env's own policy for v2, with the system prompt or tool list replaced by the files given."""
    system = args.system.read_text() if args.system else None
    tools = tuple(ToolSpec(**t) for t in json.loads(args.tools.read_text())) if args.tools else None
    if system is not None and tools is not None:
        return Policy(system, tools)
    base = await default_policy(INTERFACE)
    return Policy(base.system if system is None else system, base.tools if tools is None else tools)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    losing = tuple(args.losing_opponents.split(","))
    pool = candidates(args.games, args.learner_slot.split(",") if args.learner_slot else None, losing)
    weights = parse_weights(args.weights) if args.weights else DEFAULT_WEIGHTS
    briefs = select(pool, args.n, weights=weights, per_game=args.per_game, seed=args.seed)
    policies = {INTERFACE: await _policy(args)}
    first, runner_up = args.teacher, args.second_teacher
    if args.calibration:
        picked = teachers_from_calibration(args.calibration)
        first, runner_up = first or picked[0], runner_up or (picked[1] if len(picked) > 1 else None)
    if not first:
        raise SystemExit("name a teacher with --teacher, or pick one with --calibration")
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as http:
        teacher = _teacher(first, args.base_url, args.api_key_env, args.cap_usd, http, args.spend_log)
        second = (
            _teacher(runner_up, args.base_url, args.api_key_env, args.second_cap_usd, http, args.spend_log)
            if runner_up
            else None
        )
        rows, report = await distill(
            briefs,
            teacher,
            second,
            policies,
            HubTokenizer(args.tokenizer),
            check=args.check,
            seed=args.seed,
        )
    report["pool"] = len(pool)
    report["weights"] = dict(weights)
    if args.calibration:
        report["calibration"] = str(args.calibration)
    report["strata"] = {s: sum(r["stratum"] == s for r in rows) for s in STRATA}
    write_rows(args.out / "sft.jsonl", rows)
    (args.out / "distill.json").write_text(json.dumps(report, indent=1))
    if args.inbox:
        write_inbox(rows, args.inbox, seed=args.seed)
    if args.ledger:
        usd = teacher.guard.spent_usd + (second.guard.spent_usd if second else 0.0)
        spend.append_ledger(args.ledger, "distill", usd)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", type=Path, required=True, help="the runner's --out: finished v2 games")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tools", type=Path, help="JSON list of tools {name, description, parameters} (default: the env's)")
    p.add_argument("--system", type=Path, help="the policy's system prompt (default: the env's v2 prompt)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--teacher", help="the teacher's model name (default: the calibration panel's best)")
    p.add_argument("--calibration", type=Path, help="the teacher calibration panel's folder: picks the teacher")
    p.add_argument("--second-teacher", help="the label-noise check's teacher (default: the calibration's runner-up)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key-env", required=True)
    p.add_argument("--cap-usd", type=float, required=True, help="the first teacher's spend cap")
    p.add_argument("--second-cap-usd", type=float, default=10.0)
    p.add_argument("--n", type=int, default=DEFAULT_LABELS)
    p.add_argument("--weights", help="stratum shares, e.g. losing=0.6,late=0.2,rejected=0.2 (the default)")
    p.add_argument("--losing-opponents", default=",".join(LOSING_OPPONENTS), help="whose wins make a turn `losing`")
    p.add_argument("--check", type=int, default=200)
    p.add_argument("--per-game", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learner-slot")
    p.add_argument("--inbox", type=Path, help="also write the SFT learner's inbox file here")
    p.add_argument("--ledger", type=Path, help="the budget ledger to append this run's spend to")
    p.add_argument("--spend-log", type=Path, help="a per-call spend log")
    args = p.parse_args(argv)
    print(json.dumps(asyncio.run(_run(args)), indent=1))


if __name__ == "__main__":
    main()
