"""One pilot training step: the next policy version's LoRA adapter, from the current one and a round's decisions.

Each decision is one strategist call: the exact prompt token ids the model saw and the completion token ids it
sampled (at temperature 1, so the batch is on-policy for the adapter that played it), with its advantage
(agentenv_zeroad.pilot.scoring). The loss is policy gradient, -advantage x the completion's mean token log-prob,
plus `kl` x a per-token KL estimate (k3) against the base model (the adapter switched off), so the policy doesn't
drift far from base Qwen and forget how to call tools. One pass over up to `samples` decisions, in optimizer steps of
`batch` decisions, gradients averaged over every GPU torchrun starts (each takes its own share of the batch).

Run by agentenv_zeroad.pilot.run inside the trainer image (vLLM's image with PEFT):
  torchrun --nproc_per_node 2 train.py --decisions D.jsonl --base Qwen/Qwen3-8B --init ADAPTER|'' --out DIR
Writes DIR (adapter_config.json, adapter_model.safetensors) and DIR/train.json (the step's numbers).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decisions", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--init", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=384)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--kl", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=8192)  # a longer decision is skipped: room on a 24 GB GPU
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def load(path: str, n: int, seed: int, max_len: int) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt, completion = row.get("prompt_token_ids") or [], row.get("token_ids") or []
        if prompt and completion and len(prompt) + len(completion) <= max_len and math.isfinite(row["advantage"]):
            rows.append({"prompt": prompt, "completion": completion, "advantage": float(row["advantage"])})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def token_logps(model: torch.nn.Module, prompt: list[int], completion: list[int], device: torch.device) -> torch.Tensor:
    """Log-probs of each completion token given everything before it (only the completion's logits are kept)."""
    ids = torch.tensor([prompt + completion], device=device)
    c = len(completion)
    logits = model(input_ids=ids, logits_to_keep=c + 1).logits[0, :-1].float()
    target = torch.tensor(completion, device=device)
    return torch.log_softmax(logits, dim=-1).gather(1, target[:, None])[:, 0]


def main() -> None:
    args = parse()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    started = time.time()

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if args.init:
        model = PeftModel.from_pretrained(base, args.init, is_trainable=True)
    else:
        config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=TARGETS, lora_dropout=0.0)
        model = get_peft_model(base, config)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    rows = load(args.decisions, args.samples, args.seed, args.max_len)
    per_rank = args.batch // world
    steps = len(rows) // (per_rank * world)
    if steps == 0:
        raise SystemExit(f"{len(rows)} usable decisions: fewer than one batch of {args.batch}")
    mine = rows[rank : steps * per_rank * world : world]
    stats = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "tokens": 0, "decisions": 0}
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for row in mine[step * per_rank : (step + 1) * per_rank]:
            with torch.no_grad(), model.disable_adapter(), torch.autocast(device.type, dtype=dtype):
                ref = token_logps(model, row["prompt"], row["completion"], device)
            with torch.autocast(device.type, dtype=dtype):
                logp = token_logps(model, row["prompt"], row["completion"], device)
            delta = ref - logp
            kl = (torch.exp(delta) - delta - 1).mean()
            pg = -row["advantage"] * logp.mean()
            loss = (pg + args.kl * kl) / per_rank
            loss.backward()
            stats["loss"] += float(loss) * per_rank
            stats["pg"] += float(pg)
            stats["kl"] += float(kl)
            stats["tokens"] += len(row["completion"])
            stats["decisions"] += 1
        if distributed:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        if rank == 0:
            print(f"step {step + 1}/{steps}: mean kl {stats['kl'] / max(1, stats['decisions']):.4f}", flush=True)
    if distributed:
        totals = torch.tensor([stats[k] for k in ("loss", "pg", "kl", "tokens", "decisions")], device=device)
        dist.all_reduce(totals)
        stats = dict(zip(("loss", "pg", "kl", "tokens", "decisions"), totals.tolist(), strict=True))
    if rank == 0:
        out = Path(args.out)
        for p in params:  # trained in float32, kept in the model's dtype: half the size for vLLM and the fetch
            p.data = p.data.to(dtype)
        model.save_pretrained(out)
        n = max(1, stats["decisions"])
        summary = {
            "steps": steps,
            "decisions": int(stats["decisions"]),
            "completion_tokens": int(stats["tokens"]),
            "mean_loss": stats["loss"] / n,
            "mean_pg": stats["pg"] / n,
            "mean_kl": stats["kl"] / n,
            "init": args.init or None,
            "seconds": round(time.time() - started, 1),
        }
        (out / "train.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
