"""Model spend: conservative prices, a per-run guard that stops a brain before its cap, and the loop's spend ledger.

Every brain (the strategist agent, the Luna player agent and the in-env Luna reflex) prices each call from its token
usage and keeps a `SpendGuard`. Before a call the guard is asked whether the most that call could cost still fits
under the cap (`bound`: each request byte an input token plus the output limit), so a brain stops *before* a call that
could pass the cap, not after. A call whose usage is lost (it failed, or the provider sent none) is charged at that
bound. Each call can be appended to a JSON-lines spend log, `{"what", "model", "input_tokens", "output_tokens",
"usd_est"}`.

Prices are deliberately conservative (spec §3: the runner stops before passing the cap): LiteLLM's public price map
(2026-10, as in agentenv-airops-plugin) with a 1.5x margin, and any model not listed at a high default, so an estimate
errs high. A model named `local/...` is one this box serves itself (vLLM on its own GPUs) and costs nothing.
The loop's ledger (`spend.jsonl`, one `{"what", "usd_est"}` per line) is checked with `check_budget` before
a paid run and appended with `append_ledger` after it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MARGIN = 1.5
BASE_PRICES = {  # USD per 1M tokens (input, output), LiteLLM public price map 2026-10
    "anthropic/claude-opus-5-5": (4.00, 20.00),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
    "openai/gpt-6-astra": (10.00, 50.00),
    "openai/gpt-6-sol": (10.00, 50.00),  # not in the map: priced as Astra
    "openai/gpt-6-luna": (0.10, 0.50),
    "gpt-6-luna": (0.10, 0.50),
}
DEFAULT_PRICE = (15.00, 75.00)  # anything else (Gemini, Grok, new ids): a frontier upper bound
PRICES = {model: (i * MARGIN, o * MARGIN) for model, (i, o) in BASE_PRICES.items()}
LOCAL_PREFIX = "local/"


def price(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for `model`; a `local/` model is free, unknown models at `DEFAULT_PRICE`."""
    if model.startswith(LOCAL_PREFIX):
        return (0.0, 0.0)
    return PRICES.get(model, DEFAULT_PRICE)


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def bound(model: str, request: Any, max_output_tokens: int) -> float:
    """The most one request can cost: each byte of its JSON body an input token, plus its whole output limit."""
    size = len(json.dumps(request, default=str).encode())
    return cost(model, size, max_output_tokens)


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class SpendCapReached(RuntimeError):
    """The next call could take the brain past its cap; it was not sent."""


@dataclass
class SpendGuard:
    """Tracks one brain's spend against `cap_usd` (None: no cap) and appends each call to `log_path` if given."""

    what: str
    cap_usd: float | None = None
    log_path: Path | None = None
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    unpriced_calls: int = 0
    stopped: str | None = None
    last_usd: float = 0.0  # the latest call's charge, so a failed call can be reported with what it was charged
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allows(self, most_usd: float) -> bool:
        """Whether a call that could cost `most_usd` still fits; once it doesn't, the guard stays stopped."""
        if self.stopped:
            return False
        if self.cap_usd is not None and self.spent_usd + most_usd > self.cap_usd:
            self.stopped = (
                f"the next call could cost ${most_usd:.4f}, past the ${self.cap_usd:g} cap with "
                f"${self.spent_usd:.4f} spent"
            )
            return False
        return True

    def check(self, most_usd: float) -> None:
        if not self.allows(most_usd):
            raise SpendCapReached(self.stopped or "spend cap reached")

    def record(self, model: str, input_tokens: int | None, output_tokens: int | None, *, most_usd: float) -> float:
        """Charge one call: by its usage, or at `most_usd` when usage is missing. Returns the charge."""
        with self._lock:
            if input_tokens is None or output_tokens is None:
                usd = most_usd
                self.unpriced_calls += 1
            else:
                usd = cost(model, input_tokens, output_tokens)
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
            self.spent_usd += usd
            self.last_usd = usd
            self.calls += 1
        if self.log_path is not None:
            line = {
                "what": self.what,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "usd_est": round(usd, 6),
            }
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(line) + "\n")
            except OSError:
                pass  # the log is a convenience; the guard's own totals are what stop the brain
        return usd

    def summary(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "usd_est": round(self.spent_usd, 6),
            "cap_usd": self.cap_usd,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "unpriced_calls": self.unpriced_calls,
            "stopped": self.stopped,
        }


# ---- the loop's ledger ---------------------------------------------------------------------------------------------
def ledger_total(path: Path, *, prefix: str | None = None) -> float:
    """Sum of `usd_est` in a JSON-lines ledger (only entries whose `what` starts with `prefix`, if given)."""
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if prefix is None or str(entry.get("what", "")).startswith(prefix):
            total += float(entry.get("usd_est", 0.0))
    return total


def check_budget(
    path: Path, *, planned_usd: float, cap_usd: float, unit_prefix: str | None = None, unit_cap_usd: float | None = None
) -> str | None:
    """Why a run that could spend `planned_usd` must not start (the cumulative or the unit cap), or None if it fits."""
    total = ledger_total(path)
    if total + planned_usd > cap_usd:
        return f"${total:.2f} spent + ${planned_usd:.2f} planned would pass the ${cap_usd:g} cap"
    if unit_prefix is not None and unit_cap_usd is not None:
        unit = ledger_total(path, prefix=unit_prefix)
        if unit + planned_usd > unit_cap_usd:
            return f"{unit_prefix}: ${unit:.2f} spent + ${planned_usd:.2f} planned would pass its ${unit_cap_usd:g} cap"
    return None


def append_ledger(path: Path, what: str, usd_est: float) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"what": what, "usd_est": round(usd_est, 4)}) + "\n")


def results_spend(results: dict[str, Any]) -> float:
    """A match's estimated spend from its results.json: every seat's brain and reflex spend."""
    total = 0.0
    for seat in results.get("seats", {}).values():
        for part in (seat.get("spend") or {}).values():
            if isinstance(part, dict):
                total += float(part.get("usd_est") or 0.0)
    return total


def main(argv: list[str] | None = None) -> int:
    """`python -m agentenv_zeroad.spend check|add|results` for paid runs (see the bundle README)."""
    parser = argparse.ArgumentParser(prog="agentenv_zeroad.spend")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="exit 1 if a planned run could pass the caps")
    check.add_argument("--ledger", type=Path, required=True)
    check.add_argument("--planned", type=float, required=True)
    check.add_argument("--cap", type=float, default=25.0)
    check.add_argument("--unit-prefix")
    check.add_argument("--unit-cap", type=float)
    add = sub.add_parser("add", help="append one run's estimate to the ledger")
    add.add_argument("--ledger", type=Path, required=True)
    add.add_argument("--what", required=True)
    add.add_argument("--usd", type=float, required=True)
    res = sub.add_parser("results", help="print a results.json's estimated spend")
    res.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "check":
        refusal = check_budget(
            args.ledger,
            planned_usd=args.planned,
            cap_usd=args.cap,
            unit_prefix=args.unit_prefix,
            unit_cap_usd=args.unit_cap,
        )
        print(refusal or f"ok: ${ledger_total(args.ledger):.2f} spent, ${args.planned:.2f} planned")
        return 1 if refusal else 0
    if args.cmd == "add":
        append_ledger(args.ledger, args.what, args.usd)
        print(f"ledger total ${ledger_total(args.ledger):.2f}")
        return 0
    print(f"{results_spend(json.loads(args.path.read_text())):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
