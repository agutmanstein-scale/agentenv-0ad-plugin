"""Count the prompt tokens a seat's model gets, per interface, on the recorded snapshots (the v2 gate: p90 <= 2,600).

    uv run --with tokenizers python scripts/prompt_tokens.py [--tokenizer PATH/tokenizer.json]

Renders each prompt (system with the tool schemas, user message, generation prompt) with the Qwen3 chat template that
sits beside the tokenizer (`tokenizer_config.json`), without thinking, and tokenizes it. Without a tokenizer in the
Hugging Face cache or on the command line it estimates at 3.6 characters per token and says so. Qwen3-4B and 8B share
a vocabulary, so either tokenizer gives the 8B's counts.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.prompt_scenarios import Prompt, all_prompts, chat_request, request_chars  # noqa: E402

CHARS_PER_TOKEN = 3.6


def find_tokenizer() -> Path | None:
    pattern = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-*/snapshots/*/tokenizer.json")
    return next((Path(p) for p in sorted(glob.glob(pattern))), None)


def render(prompt: Prompt, template: str) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    request = chat_request(prompt)
    return env.from_string(template).render(
        messages=request["messages"],
        tools=request["tools"],
        add_generation_prompt=True,
        enable_thinking=False,
        raise_exception=raise_exception,
    )


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=find_tokenizer())
    args = parser.parse_args()
    counter = None
    if args.tokenizer is not None and args.tokenizer.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        template = json.loads((args.tokenizer.parent / "tokenizer_config.json").read_text())["chat_template"]
        counter = lambda p: len(tokenizer.encode(render(p, template), add_special_tokens=False).ids)  # noqa: E731
        print(f"tokenizer: {args.tokenizer}")
    else:
        counter = lambda p: round(request_chars(p) / CHARS_PER_TOKEN)  # noqa: E731
        print(f"no tokenizer found: estimating at {CHARS_PER_TOKEN} characters per token")
    for interface in ("v1", "v2"):
        prompts = all_prompts(interface)
        totals = [counter(p) for p in prompts]
        fixed = counter(Prompt(prompts[0].system, "x", prompts[0].tools))
        print(
            f"{interface}: n={len(totals)} total tokens {totals}  p50={int(statistics.median(totals))} "
            f"p90={percentile(totals, 0.9)}  system+tools={fixed}"
        )


if __name__ == "__main__":
    main()
