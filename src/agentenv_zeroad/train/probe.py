"""The GPU box's E1 probe: real v2 decisions per second on each vLLM server, and rows for the learner's gates.

    VLLM_API_KEY=... python -m agentenv_zeroad.train.probe --url http://172.17.0.1:8000 --url http://172.17.0.1:8001 \\
        --out ROOT/e1 [--levels 16,32,48,64] [--window-s 60] [--rows 48]

For each in-flight level (16, 32, 48 and 64 requests at once per server) every server is kept at that many calls in
flight for `window_s` seconds, with real v2 prompts (the system prompt and order tools a training seat shows its model,
and the briefs of the recorded snapshots at 60 s, at 300 s and the busy midgame one, `tests/prompt_scenarios.py`; or any
JSON-lines file of `{system, user, tools}` given with `--prompts`) sampled the way training samples (T 1, `max_tokens`
512, token ids and log-probs on). A call counts if it finishes inside the window; a decision per second is the count
over the window. One server is one GPU (TP=1), so its rate is the per-GPU rate (`gates.per_gpu_rates`).

The first successful replies (aligned token ids and log-probs only) are kept in `--out/probe-rows.jsonl` in the
learner's inbox format: the learner's own probe (`trainer/learner.py --probe`) reads them to measure its seconds
per trained decision and how far its log-probs are from vLLM's. Results go to `--out/throughput.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agentenv_zeroad.llm import Reply, StrategistModel, ToolSpec
from agentenv_zeroad.spend import SpendGuard
from agentenv_zeroad.train import gates
from agentenv_zeroad.train.smoke import problems_in, training_body
from agentenv_zeroad.train.spec import DEFAULT_MAX_TOKENS

LEVELS = (16, 32, 48, 64)
WINDOW_S = 60.0
DEFAULT_ROWS = 48
BASE_MODEL = "local/qwen-base"


@dataclass(frozen=True)
class PromptItem:
    system: str
    user: str
    tools: list[ToolSpec]


def recorded_prompts(interface: str = "v2") -> list[PromptItem]:
    """The v2 prompts of the recorded snapshots (the repo's tests/prompt_scenarios), as a seat's model gets them."""
    try:
        from tests.prompt_scenarios import all_prompts
    except ImportError as e:  # not in the installed package: the repo checkout the box runs from has it
        raise SystemExit(
            f"the recorded prompts are the repo's tests/prompt_scenarios.py (run from the repo): {e}"
        ) from e
    return [PromptItem(p.system, p.user, p.tools) for p in all_prompts(interface)]  # type: ignore[arg-type]


def prompts_from(path: Path | None) -> list[PromptItem]:
    if path is None:
        return recorded_prompts()
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            raw = json.loads(line)
            items.append(PromptItem(raw["system"], raw["user"], [ToolSpec(**t) for t in raw["tools"]]))
    if not items:
        raise SystemExit(f"{path} has no prompts")
    return items


def row_of(reply: Reply, game_id: str) -> dict[str, Any]:
    """A reply as one learner inbox row (version 0, a placeholder advantage the learner's probe replaces)."""
    return {
        "game_id": game_id,
        "version": 0,
        "prompt_token_ids": reply.prompt_token_ids,
        "token_ids": reply.token_ids,
        "logprobs": reply.logprobs,
        "advantage": 0.0,
        "is_hold": False,
    }


@dataclass
class ServerLevel:
    url: str
    level: int
    completed: int
    failed: int
    window_s: float
    seconds: list[float]

    @property
    def rate(self) -> float:
        return self.completed / self.window_s

    def to_json(self) -> dict[str, Any]:
        ordered = sorted(self.seconds)
        median = ordered[len(ordered) // 2] if ordered else None
        return {
            "url": self.url,
            "level": self.level,
            "completed": self.completed,
            "failed": self.failed,
            "decisions_per_s": round(self.rate, 3),
            "median_call_s": median,
        }


async def drive(
    llm: StrategistModel,
    items: Sequence[PromptItem],
    level: int,
    window_s: float,
    rows: list[dict[str, Any]],
    keep: int,
) -> ServerLevel:
    """Keep `level` calls in flight on one server for `window_s`, counting those that finish inside the window."""
    deadline = time.monotonic() + window_s
    done = failed = 0
    seconds: list[float] = []
    turn = 0

    async def worker(n: int) -> None:
        nonlocal done, failed, turn
        while time.monotonic() < deadline:
            item = items[(n + turn) % len(items)]
            turn += 1
            began = time.monotonic()
            try:
                reply = await llm.call(item.system, item.user, item.tools)
            except Exception:  # noqa: BLE001 - a failed call is counted, the level goes on
                failed += 1
                continue
            if time.monotonic() > deadline:
                break
            done += 1
            seconds.append(time.monotonic() - began)
            if len(rows) < keep and not problems_in(reply):
                rows.append(row_of(reply, f"probe-{len(rows):04d}"))

    await asyncio.gather(*(worker(n) for n in range(level)))
    return ServerLevel(llm.base, level, done, failed, window_s, seconds)


async def run_probe(
    urls: Sequence[str],
    key: str,
    items: Sequence[PromptItem],
    *,
    levels: Sequence[int] = LEVELS,
    window_s: float = WINDOW_S,
    keep: int = DEFAULT_ROWS,
    model: str = BASE_MODEL,
    http: httpx.AsyncClient | None = None,
) -> tuple[list[ServerLevel], list[dict[str, Any]]]:
    """Every level on every server (the servers in parallel, the levels one after another); returns the per-server
    results and the rows kept for the learner."""
    rows: list[dict[str, Any]] = []
    out: list[ServerLevel] = []
    own = http is None
    http = http or httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=1024))
    try:
        models = [
            StrategistModel(
                model,
                url,
                key,
                SpendGuard("e1 probe"),
                max_tokens=DEFAULT_MAX_TOKENS,
                http=http,
                extra_body=training_body(),
            )
            for url in urls
        ]
        for level in levels:
            out += await asyncio.gather(*(drive(m, items, level, window_s, rows, keep) for m in models))
    finally:
        if own:
            await http.aclose()
    return out, rows


def summarise(results: Sequence[ServerLevel]) -> dict[str, Any]:
    by_level: dict[int, dict[str, float]] = defaultdict(dict)
    for r in results:
        by_level[r.level][r.url] = r.rate
    rates = gates.per_gpu_rates(by_level)
    return {
        "per_gpu": {str(k): round(v, 3) for k, v in rates.items()},
        "knee": gates.knee(rates),
        "servers": [r.to_json() for r in results],
    }


def write_outputs(out: Path, results: Sequence[ServerLevel], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    summary["rows_kept"] = len(rows)
    (out / "throughput.json").write_text(json.dumps(summary, indent=1))
    (out / "probe-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return summary


async def main_async(args: argparse.Namespace) -> int:
    items = prompts_from(args.prompts)
    results, rows = await run_probe(
        args.url, os.environ["VLLM_API_KEY"], items, levels=args.levels, window_s=args.window_s, keep=args.rows
    )
    print(json.dumps(write_outputs(args.out, results, rows)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", action="append", required=True, help="a vLLM server (repeat for each)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=list(LEVELS))
    p.add_argument("--window-s", type=float, default=WINDOW_S)
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="sampled decisions kept for the learner's probe")
    p.add_argument("--prompts", type=Path, help="JSON lines of {system, user, tools}, instead of the recorded ones")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
