"""Federated averaging of LoRA adapters: the pooled pilot's next version from each box's local update.

Every member trained the same version on its own games, so its adapter is that version plus its own small update; the
weighted mean of each LoRA tensor (by the decisions each member trained on, from its train.json) is the pooled update.
Averaging the A and B factors separately is the usual first-order approximation of averaging their products, close
while each step is small.

  python3 average.py --out DIR --member DIR [--member DIR ...]
Writes DIR/adapter_model.safetensors, DIR/adapter_config.json (the first member's) and DIR/train.json (the members'
numbers, summed or weighted).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--member", action="append", required=True)
    args = p.parse_args()
    members = [Path(m) for m in args.member]
    stats = [json.loads((m / "train.json").read_text()) if (m / "train.json").exists() else {} for m in members]
    weights = [max(1, int(s.get("decisions") or 1)) for s in stats]
    total = sum(weights)
    acc: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    keys = None
    for member, w in zip(members, weights, strict=True):
        tensors = load_file(str(member / "adapter_model.safetensors"))
        if keys is None:
            keys = set(tensors)
        elif set(tensors) != keys:
            raise SystemExit(f"{member}: its tensors differ from the first member's")
        for k, v in tensors.items():
            dtypes[k] = v.dtype
            acc[k] = acc.get(k, torch.zeros_like(v, dtype=torch.float32)) + v.float() * (w / total)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(dtypes[k]).contiguous() for k, v in acc.items()}, str(out / "adapter_model.safetensors"))
    shutil.copy(members[0] / "adapter_config.json", out / "adapter_config.json")
    summary = {
        "decisions": total,
        "completion_tokens": sum(int(s.get("completion_tokens") or 0) for s in stats),
        "mean_kl": sum(float(s.get("mean_kl") or 0) * w for s, w in zip(stats, weights, strict=True)) / total,
        "mean_pg": sum(float(s.get("mean_pg") or 0) * w for s, w in zip(stats, weights, strict=True)) / total,
        "seconds": max(float(s.get("seconds") or 0) for s in stats),
        "steps": max(int(s.get("steps") or 0) for s in stats),
        "members": [
            {"dir": m.name, "decisions": w, "mean_kl": s.get("mean_kl")}
            for m, w, s in zip(members, weights, stats, strict=True)
        ],
    }
    (out / "train.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    main()
