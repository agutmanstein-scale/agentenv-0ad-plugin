"""The persistent learner for post-training v2 ("Lockstep Climb"): one process that trains for the whole run.

The pilot's trainer (train.py) was rebuilt every round: a fresh AdamW, a bf16 save that threw the fp32 weights
away, and a loss of -advantage x the completion's mean log-prob, which paid short replies the most and collapsed the
policy's length. This learner keeps its LoRA weights in fp32 and its AdamW moments for the whole run, and trains on a
token-sum loss.

Input. Decision rows, one JSON object per line, in *.jsonl files dropped into ROOT/inbox. A writer creates the file
under another name (say x.jsonl.tmp) and renames it to a unique x.jsonl, so the learner never reads half a file. Fields:
game_id, version (the adapter version that sampled it, 0 = the base model), prompt_token_ids, token_ids, logprobs (the
sampler's log-prob of each completion token, same length as token_ids), advantage (already centred upstream), and
optionally loss_mask (one number per completion token) and is_hold. A row that fails validation is counted by reason
and dropped. Read files move to inbox/consumed/.

Updates. 256 decisions per update, in 2 minibatches of 128, one epoch. Rows more than `max_staleness` versions older
than the learner's current version are dropped. When more than 256 valid rows are waiting the freshest go first: the
learner is the bottleneck in the design (docs/post-training-v2.md), so queueing oldest-first would train on ever staler
data.

Loss: decoupled PPO with a truncated importance weight and a constant token divisor.
  pi_prox  the learner's weights at the start of the update; log pi_prox of every token is computed (no grad) before the
           first optimizer step, so the trust region is this update's, not the staleness of the data
  r        exp(log pi_theta - log pi_prox), with the PPO pessimistic min of r*A and clip(r, 0.8, 1.28)*A
  w        min(exp(log pi_prox - log mu), 2), mu = the sampler's stored log-prob; it carries no gradient
  L        -(1 / (B x 128)) sum_i sum_t m_it w_it min(...), B the minibatch size and 128 a constant divisor
The divisor is never the completion's length, which is what made the pilot's short replies win. m is the row's
loss_mask, or, when it has none, zero on exactly the tokens of say() tool calls. After minibatch 1, if the mean
KL(pi_after || pi_prox) over minibatch 2's trained tokens is above `kl_stop`, minibatch 2 is skipped. KL is
estimated on the sampled tokens as r log r - r + 1, which is non-negative and unbiased for data drawn from pi_prox.
The LM head and log-softmax run in fp32 whatever the model's dtype.

State. LoRA r32 alpha 64 on every linear layer, fp32 master parameters, one AdamW for the whole run, gradient clip 1.0,
lr constant after a linear warmup. The lr changes only through set_lr() and rollback(), which halves it. With torchrun
each rank takes its share of every minibatch; gradients are summed across ranks and the divisor is global, so the result
doesn't depend on the number of ranks. Rank 0 reads the inbox and broadcasts each update's rows.

Publish. After every update, ROOT/adapters/v{N}/ holds a bf16 PEFT adapter plus manifest.json, built in a hidden
directory on the same filesystem, fsynced, then renamed into place. vLLM hot-loads it from there (pilot/run.py's
load_adapter) and never sees a partial directory. N only grows: a rollback publishes the restored weights as a new
version, because vLLM caches adapters by name.

Checkpoint ring. The last `ring` states (fp32 adapter, AdamW moments, update count, lr) live in ROOT/checkpoints/ so a
rollback and a crash restart both work: rank 0 writes them atomically, every rank reads them. The ring is keyed by
version. The orchestrator (another stream) picks the version to roll back to, and a restart resumes from the newest.

Health and tripwires. Every update appends a record to ROOT/health.jsonl (see Learner._health). After each update the
learner runs evaluate_tripwires() over the newest `window` records (this one included) against an optional baseline,
the file given with --baseline (baseline_from() over the base model's health-shaped records, as JSON), and writes the
result into the record as "tripwires": {"trips": [...], "alarms": [...]}. Without a baseline only the checks that need
none run (hold share, invented tool names, rejected share against its 3% floor, clipped share, gradient norm). A trip
means roll back, an alarm means look; the learner only reports, the orchestrator acts. A rollback empties the window, so
the updates it undid cannot trip the restored weights, and a restart refills it from health.jsonl after the last
rollback.

Control. A command is a JSON object in a file in ROOT/control; the file's name only orders commands, oldest name first,
so use a sortable one such as 000017-rollback.json. Write it as a hidden or .tmp name and rename it into place: the
learner reads every *.json it finds, moves it to control/done/ whatever it held, and drops one that is not valid JSON.
The commands, handled between updates on every rank:
  {"op": "rollback", "version": N}  restore checkpoint N from the ring (weights, AdamW moments, update count), halve the
                                    lr and publish the restored weights as a new version; a version not in the ring is
                                    logged and ignored
  {"op": "set_lr", "lr": X}         the lr the warmup scales, from the next update on
  {"op": "stop"}                    leave the serve loop after the update in progress
Anything else, or a command missing its field, is logged and ignored. Each rollback also appends {"event": "rollback",
"restored": N, "version": M, "lr": X, "time": T} to health.jsonl.

SFT mode (--sft), for the distillation probe. The same process, with the loss swapped for cross-entropy on completion
tokens: L = -(1 / (B x 128)) sum_i sum_t m_it log pi_theta, with the same constant token divisor and the same masks, so
a long reply is not rewarded or punished for its length. Rows need only game_id, prompt_token_ids and token_ids (and
optionally loss_mask); version, logprobs and advantage are ignored, nothing is dropped as stale, there is no proximal
pass, no KL stop and no tripwires (the records carry empty ones), and a checkpoint remembers its mode and refuses to
resume the other. It writes its own adapters, so give it its own --root; a typical SFT lr is higher than the RL one.

Run in the trainer image (vLLM's image with PEFT), after training starts:
  torchrun --nproc_per_node 2 learner.py --base Qwen/Qwen3-8B --root /run1 [--init ADAPTER] --known-tools a,b,c
      [--baseline baseline.json]
  torchrun --nproc_per_node 2 learner.py --base Qwen/Qwen3-8B --root /sft1 --sft --lr 1e-4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("learner")

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = 151657, 151658
Decode = Callable[[list[int]], str]


@dataclass(frozen=True)
class LearnerConfig:
    base: str
    root: Path
    init: str = ""
    rank: int = 32
    alpha: int = 64
    decisions_per_update: int = 256
    minibatches: int = 2
    token_divisor: int = 128
    clip_low: float = 0.8
    clip_high: float = 1.28
    tis_cap: float = 2.0
    max_staleness: int = 6
    kl_stop: float = 0.005
    lr: float = 1e-5
    warmup_updates: int = 10
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ring: int = 20
    max_len: int = 8192  # a longer decision is rejected: room on a 24 GB GPU
    base_kl_fraction: float = 0.1
    tool_call_open: int = TOOL_CALL_OPEN
    tool_call_close: int = TOOL_CALL_CLOSE
    known_tools: tuple[str, ...] = ()
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    seed: int = 0
    sft: bool = False

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def adapters(self) -> Path:
        return self.root / "adapters"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def control(self) -> Path:
        return self.root / "control"

    @property
    def health(self) -> Path:
        return self.root / "health.jsonl"


@dataclass(frozen=True)
class ToolCall:
    name: str | None  # None when the call's JSON has no readable name
    start: int  # token span [start, end) in the completion, tags included
    end: int


@dataclass(frozen=True)
class Decision:
    game_id: str
    version: int
    prompt: list[int]
    completion: list[int]
    logprobs: list[float]
    advantage: float
    loss_mask: list[float] | None = None
    is_hold: bool = False
    mask: list[float] = field(default_factory=list)  # the resolved per-token loss mask (see annotate)
    calls: tuple[ToolCall, ...] = ()


class RowRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def parse_row(obj: Any, max_len: int, *, sft: bool = False) -> Decision:
    """A Decision from one JSON row, or RowRejected with the reason it is unusable. An SFT row needs only the ids."""
    if not isinstance(obj, dict):
        raise RowRejected("not_an_object")
    try:
        prompt, completion = list(obj["prompt_token_ids"]), list(obj["token_ids"])
        game_id = str(obj["game_id"])
        logprobs, advantage, version = (
            ([], 0.0, 0) if sft else (list(obj["logprobs"]), float(obj["advantage"]), obj["version"])
        )
    except (KeyError, TypeError, ValueError) as e:
        raise RowRejected("missing_field") from e
    if not prompt or not completion:
        raise RowRejected("empty")
    if not sft and len(logprobs) != len(completion):
        raise RowRejected("logprobs_length")
    if len(prompt) + len(completion) > max_len:
        raise RowRejected("too_long")
    if not sft and (not isinstance(version, int) or isinstance(version, bool) or version < 0):
        raise RowRejected("bad_version")
    if not math.isfinite(advantage):
        raise RowRejected("bad_advantage")
    try:
        logprobs = [float(x) for x in logprobs]
    except (TypeError, ValueError) as e:
        raise RowRejected("bad_logprobs") from e
    if not all(math.isfinite(x) for x in logprobs):
        raise RowRejected("bad_logprobs")
    loss_mask = obj.get("loss_mask")
    if loss_mask is not None:
        try:
            loss_mask = [float(x) for x in loss_mask]
        except (TypeError, ValueError) as e:
            raise RowRejected("bad_loss_mask") from e
        if len(loss_mask) != len(completion) or not all(math.isfinite(x) for x in loss_mask):
            raise RowRejected("bad_loss_mask")
    return Decision(game_id, version, prompt, completion, logprobs, advantage, loss_mask, bool(obj.get("is_hold")))


_NAME = re.compile(r'"name"\s*:\s*"([^"]*)"')


def find_tool_calls(completion: Sequence[int], decode: Decode, open_id: int, close_id: int) -> list[ToolCall]:
    """The <tool_call>{json}</tool_call> spans of a completion with each call's name (a truncated call runs on)."""
    calls: list[ToolCall] = []
    start: int | None = None
    for i, token in enumerate(completion):
        if token == open_id and start is None:
            start = i
        elif token == close_id and start is not None:
            calls.append(ToolCall(_call_name(completion[start + 1 : i], decode), start, i + 1))
            start = None
    if start is not None:
        calls.append(ToolCall(_call_name(completion[start + 1 :], decode), start, len(completion)))
    return calls


def _call_name(inner: Sequence[int], decode: Decode) -> str | None:
    text = decode(list(inner))
    try:
        name = json.loads(text).get("name")
    except (ValueError, AttributeError):
        match = _NAME.search(text)
        name = match.group(1) if match else None
    return name if isinstance(name, str) else None


def annotate(row: Decision, decode: Decode, open_id: int, close_id: int) -> Decision:
    """The row with its tool calls found and its loss mask resolved: its own loss_mask, else zero on say() calls."""
    calls = tuple(find_tool_calls(row.completion, decode, open_id, close_id))
    if row.loss_mask is not None:
        return replace(row, mask=list(row.loss_mask), calls=calls)
    mask = [1.0] * len(row.completion)
    for call in calls:
        if call.name == "say":
            mask[call.start : call.end] = [0.0] * (call.end - call.start)
    return replace(row, mask=mask, calls=calls)


class Inbox:
    """Rank 0's view of the inbox: validated rows waiting to be trained on, and counts of what was thrown away."""

    def __init__(self, cfg: LearnerConfig):
        self.cfg = cfg
        self.pool: list[Decision] = []
        self.counts: Counter[str] = Counter()
        cfg.inbox.mkdir(parents=True, exist_ok=True)
        (cfg.inbox / "consumed").mkdir(exist_ok=True)

    def poll(self) -> int:
        """Read every new *.jsonl file into the pool, moving it to consumed/. Returns the rows accepted."""
        accepted = 0
        for path in sorted(self.cfg.inbox.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    self.pool.append(parse_row(json.loads(line), self.cfg.max_len, sft=self.cfg.sft))
                    accepted += 1
                    self.counts["accepted"] += 1
                except json.JSONDecodeError:
                    self.counts["rejected:bad_json"] += 1
                except RowRejected as e:
                    self.counts[f"rejected:{e.reason}"] += 1
            os.replace(path, self.cfg.inbox / "consumed" / path.name)
        return accepted

    def take(self, current_version: int, *, flush: bool = False) -> list[Decision] | None:
        """The next batch, freshest first, or None while fewer than decisions_per_update valid rows are waiting.

        With `flush` (SFT only, once the inbox has gone quiet) the rows left over, fewer than a batch, are taken
        too: a finite SFT set would otherwise never train on its remainder."""
        floor = current_version - self.cfg.max_staleness
        keep = []
        for row in self.pool:
            if self.cfg.sft:
                keep.append(row)
            elif row.version > current_version:
                self.counts["rejected:future_version"] += 1
            elif row.version < floor:
                self.counts["stale"] += 1
            else:
                keep.append(row)
        self.pool = keep
        n = self.cfg.decisions_per_update
        if flush and self.cfg.sft and self.pool:
            n = min(n, len(self.pool))
        if len(self.pool) < n or not self.pool:
            return None
        order = sorted(range(len(self.pool)), key=lambda i: (-self.pool[i].version, i))
        chosen = set(order[:n])
        batch = [self.pool[i] for i in sorted(chosen)]
        self.pool = [row for i, row in enumerate(self.pool) if i not in chosen]
        return batch

    def drain_counts(self) -> dict[str, int]:
        """What was accepted, rejected (by reason) and dropped as stale since the last call."""
        counts = dict(self.counts)
        self.counts.clear()
        return counts


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def row_stats(rows: Sequence[Decision], current_version: int, known_tools: Sequence[str]) -> dict[str, Any]:
    """The health numbers that come from the rows alone. Shape statistics skip hold rows: a policy that has learned
    hold() answers quiet turns with one short, identical reply, and that must not read as a collapse."""
    active = [r for r in rows if not r.is_hold]
    advantages = [r.advantage for r in rows]
    stats: dict[str, Any] = {
        "decisions": len(rows),
        "hold_share": 1 - len(active) / len(rows) if rows else None,
        "adv_mean": statistics.fmean(advantages) if advantages else None,
        "adv_sd": statistics.pstdev(advantages) if advantages else None,
        "staleness": {str(k): v for k, v in sorted(Counter(current_version - r.version for r in rows).items())},
    }
    shape: dict[str, float | None] = dict.fromkeys(
        ("tokens_p10", "tokens_p50", "tokens_p90", "calls_per_decision", "exact_share", "distinct_fraction"), None
    )
    invented = None
    if active:
        lengths = [len(r.completion) for r in active]
        counts = Counter(tuple(r.completion) for r in active)
        shape = {
            "tokens_p10": percentile(lengths, 0.1),
            "tokens_p50": percentile(lengths, 0.5),
            "tokens_p90": percentile(lengths, 0.9),
            "calls_per_decision": sum(len(r.calls) for r in active) / len(active),
            "exact_share": counts.most_common(1)[0][1] / len(active),
            "distinct_fraction": len(counts) / len(active),
        }
        names = [c.name for r in active for c in r.calls]
        if known_tools and names:
            invented = sum(1 for n in names if n not in known_tools) / len(names)
    return {**stats, **shape, "invented_tool_share": invented}


@dataclass
class TokenTerms:
    loss: torch.Tensor  # sum over tokens, still carrying the gradient
    clipped: float  # trained tokens where the clipped branch of the min is the active one
    truncated: float  # trained tokens whose behaviour weight was cut at tis_cap
    tokens: float  # trained tokens


def ppo_terms(
    logp: torch.Tensor,
    logp_prox: torch.Tensor,
    logp_mu: torch.Tensor,
    advantage: float,
    mask: torch.Tensor,
    cfg: LearnerConfig,
) -> TokenTerms:
    """-sum_t m_t w_t min(r_t A, clip(r_t) A) for one completion; the caller applies 1 / (minibatch size x divisor)."""
    ratio = torch.exp(logp - logp_prox)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, cfg.clip_low, cfg.clip_high) * advantage
    objective = torch.minimum(unclipped, clipped)
    raw_weight = torch.exp(logp_prox - logp_mu).detach()
    weight = raw_weight.clamp(max=cfg.tis_cap)
    trained = mask > 0
    return TokenTerms(
        loss=-(mask * weight * objective).sum(),
        clipped=float((trained & (clipped < unclipped)).sum()),
        truncated=float((trained & (raw_weight > cfg.tis_cap)).sum()),
        tokens=float(trained.sum()),
    )


def sft_terms(logp: torch.Tensor, mask: torch.Tensor) -> TokenTerms:
    """-sum_t m_t log pi_theta for one completion: the SFT loss, still to be divided by minibatch size x divisor."""
    return TokenTerms(loss=-(mask * logp).sum(), clipped=0.0, truncated=0.0, tokens=float(mask.sum()))


def kl_estimate(logp_after: torch.Tensor, logp_before: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """(sum, count) of r log r - r + 1 over the trained tokens, r = pi_after / pi_before."""
    log_r = logp_after - logp_before
    per_token = torch.exp(log_r) * log_r - torch.exp(log_r) + 1
    return float((per_token * mask).sum()), float(mask.sum())


class Comm:
    """The few collectives the learner needs; with one process they are no-ops."""

    def __init__(self, rank: int = 0, world: int = 1, backend: str | None = None):
        self.rank, self.world = rank, world
        if world > 1:
            dist.init_process_group(
                backend or ("nccl" if torch.cuda.is_available() else "gloo"), rank=rank, world_size=world
            )

    @classmethod
    def from_env(cls) -> Comm:
        return cls(int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")))

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor

    def broadcast(self, obj: Any) -> Any:
        if self.world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def sync_params(self, params: Sequence[torch.Tensor]) -> None:
        """Make every rank's parameters equal to rank 0's."""
        if self.world > 1:
            for p in params:
                dist.broadcast(p.data, src=0)

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def close(self) -> None:
        if self.world > 1:
            dist.destroy_process_group()


def build_model(cfg: LearnerConfig, device: torch.device) -> PeftModel:
    """The base model plus a trainable LoRA (a fresh one, or cfg.init's), with the LoRA parameters in fp32."""
    dtype = getattr(torch, cfg.dtype)
    base = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    if cfg.init:
        model = PeftModel.from_pretrained(base, cfg.init, is_trainable=True)
    else:
        config = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, target_modules=TARGETS, lora_dropout=0.0)
        model = get_peft_model(base, config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()  # from_pretrained returns eval mode, and HF only checkpoints layers while training; dropout is 0
    return model


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_dir(parent: Path, name: str, populate: Callable[[Path], None]) -> Path:
    """parent/name, filled by populate(tmp) in a hidden sibling directory, fsynced, then renamed into place."""
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=parent))
    try:
        tmp.chmod(0o755)
        populate(tmp)
        for f in tmp.iterdir():
            fsync_path(f)
        fsync_path(tmp)
        final = parent / name
        os.rename(tmp, final)
        fsync_path(parent)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def write_file_atomically(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    fsync_path(tmp)
    os.replace(tmp, path)
    fsync_path(path.parent)


class Learner:
    _KEYS = (
        "loss",
        "clipped",
        "truncated",
        "tokens",
        "entropy_sum",
        "entropy_n",
        "base_kl_sum",
        "base_kl_n",
        "nonfinite_steps",
    )

    def __init__(
        self,
        cfg: LearnerConfig,
        model: PeftModel,
        decode: Decode,
        comm: Comm | None = None,
        *,
        baseline: dict[str, dict[str, float]] | None = None,
        tripwires: dict[str, Any] | None = None,
    ):
        self.cfg, self.model, self.decode = cfg, model, decode
        self.comm = comm or Comm()
        if not cfg.known_tools and not cfg.sft:
            log.error("known_tools is empty: the invented-tool tripwire is off")
        self.baseline = baseline or {}
        self.tripwire_config = tripwires
        self.recent: deque[dict[str, Any]] = deque(maxlen={**DEFAULTS, **(tripwires or {})}["window"])
        self.mode = "sft" if cfg.sft else "rl"
        self.device = next(model.parameters()).device
        self.dtype = getattr(torch, cfg.dtype)
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [p for _, p in self.params], lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
        self.lr = cfg.lr
        self.version = 0
        self.updates = 0
        self.decisions_total = 0
        backbone = model.get_base_model()
        self.backbone = backbone.model
        self.head_weight = backbone.lm_head.weight.detach().float()
        cfg.checkpoints.mkdir(parents=True, exist_ok=True)

    def effective_lr(self) -> float:
        return self.lr * min(1.0, (self.updates + 1) / self.cfg.warmup_updates)

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def logps(
        self, row: Decision, *, grad: bool, adapter: bool = True, entropy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """fp32 log-probs of the completion's tokens (and each position's entropy); head and softmax in fp32."""
        c = len(row.completion)
        ids = torch.tensor([row.prompt + row.completion], device=self.device)
        target = torch.tensor(row.completion, device=self.device)
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad() if grad else torch.no_grad())
            if not adapter:
                stack.enter_context(self.model.disable_adapter())
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                hidden = self.backbone(input_ids=ids).last_hidden_state[0, -(c + 1) : -1]
            log_probs = torch.log_softmax(hidden.float() @ self.head_weight.T, dim=-1)
            logp = log_probs.gather(1, target[:, None])[:, 0]
            ent = -(log_probs.exp() * log_probs).sum(-1) if entropy else None
        return logp, ent

    def update(self, rows: list[Decision], intake: dict[str, int] | None = None) -> dict[str, Any]:
        """One update on `rows`: publishes the next version and returns its health record."""
        cfg, comm = self.cfg, self.comm
        started, base_version, lr_used = time.time(), self.version, self.effective_lr()
        rows = [annotate(r, self.decode, cfg.tool_call_open, cfg.tool_call_close) for r in rows]
        batches = self._minibatches(rows)
        owned = [[r for i, r in enumerate(mb) if i % comm.world == comm.rank] for mb in batches]
        totals: Counter[str] = Counter()
        prox = {} if cfg.sft else self._prox_pass(rows, owned, totals)
        stepped, grad_norms, kl_check = 0, [], None
        for j, mb in enumerate(owned):
            if j > 0 and not cfg.sft:
                kl_check = self._kl_after(mb, prox)
                if kl_check > cfg.kl_stop:
                    break
            grad_norms.append(self._train_minibatch(mb, len(batches[j]), prox, totals))
            stepped += 1
        reduced = self._reduce(totals)
        self.updates += 1
        self.version += 1
        trained = sum(len(batches[j]) for j in range(stepped))
        self.decisions_total += trained
        health = self._health(
            rows, reduced, (stepped, trained, grad_norms, kl_check), intake or {}, (started, base_version, lr_used)
        )
        health["tripwires"] = self._tripwires(health)
        self._persist(health)
        return health

    def _tripwires(self, health: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """This update's trips and alarms over the newest records; an SFT run samples nothing, so it has none."""
        if self.cfg.sft:
            return {"trips": [], "alarms": []}
        self.recent.append(health)
        return evaluate_tripwires(list(self.recent), self.baseline, self.tripwire_config)

    def _minibatches(self, rows: list[Decision]) -> list[list[Decision]]:
        order = list(rows)
        random.Random(self.cfg.seed * 1_000_003 + self.version).shuffle(order)
        m, n = self.cfg.minibatches, len(order)
        bounds = [n * k // m for k in range(m + 1)]
        return [order[bounds[k] : bounds[k + 1]] for k in range(m)]

    def _prox_pass(
        self, rows: list[Decision], owned: list[list[Decision]], totals: Counter[str]
    ) -> dict[int, torch.Tensor]:
        """log pi_prox of every owned row, before any step, plus the entropy and KL-to-base health samples."""
        sample = set(random.Random(self.cfg.seed + self.version).sample(
            range(len(rows)), max(1, math.ceil(self.cfg.base_kl_fraction * len(rows)))
        ))  # fmt: skip
        position = {id(r): i for i, r in enumerate(rows)}
        prox: dict[int, torch.Tensor] = {}
        for row in (r for mb in owned for r in mb):
            logp, ent = self.logps(row, grad=False, entropy=not row.is_hold)
            prox[id(row)] = logp
            mask = torch.tensor(row.mask, device=self.device)
            if ent is not None:
                totals["entropy_sum"] += float((ent * mask).sum())
                totals["entropy_n"] += float(mask.sum())
            if position[id(row)] in sample:
                base_logp, _ = self.logps(row, grad=False, adapter=False)
                delta = base_logp - logp
                totals["base_kl_sum"] += float(((torch.exp(delta) - delta - 1) * mask).sum())
                totals["base_kl_n"] += float(mask.sum())
        return prox

    def _kl_after(self, mb: list[Decision], prox: dict[int, torch.Tensor]) -> float:
        """Mean KL(pi_after || pi_prox) over the trained tokens of every rank's share of the minibatch."""
        acc = torch.zeros(2, device=self.device)
        for row in mb:
            logp, _ = self.logps(row, grad=False)
            s, n = kl_estimate(logp, prox[id(row)], torch.tensor(row.mask, device=self.device))
            acc += torch.tensor([s, n], device=self.device)
        total, count = self.comm.sum(acc).tolist()
        return total / max(count, 1.0)

    def _train_minibatch(
        self, mb: list[Decision], size: int, prox: dict[int, torch.Tensor], totals: Counter[str]
    ) -> float:
        """Accumulate the loss gradient over this rank's rows, sum it across ranks, clip, step. Returns the norm."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        for row in mb:
            logp, _ = self.logps(row, grad=True)
            mask = torch.tensor(row.mask, device=self.device)
            if cfg.sft:
                terms = sft_terms(logp, mask)
            else:
                terms = ppo_terms(
                    logp, prox[id(row)], torch.tensor(row.logprobs, device=self.device), row.advantage, mask, cfg
                )
            scaled = terms.loss / (size * cfg.token_divisor)
            scaled.backward()
            totals["loss"] += float(scaled.detach())
            totals["clipped"] += terms.clipped
            totals["truncated"] += terms.truncated
            totals["tokens"] += terms.tokens
        self._sum_grads()
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in self.params], cfg.max_grad_norm))
        if not math.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=True)
            totals["nonfinite_steps"] += 1.0 / self.comm.world
            return norm
        for group in self.optimizer.param_groups:
            group["lr"] = self.effective_lr()
        self.optimizer.step()
        return norm

    def _sum_grads(self) -> None:
        if self.comm.world == 1:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for _, p in self.params])
        self.comm.sum(flat)
        offset = 0
        for _, p in self.params:
            p.grad = flat[offset : offset + p.numel()].view_as(p)
            offset += p.numel()

    def _reduce(self, totals: Counter[str]) -> dict[str, float]:
        vec = torch.tensor([totals[k] for k in self._KEYS], dtype=torch.float64, device=self.device)
        return dict(zip(self._KEYS, self.comm.sum(vec).tolist(), strict=True))

    def _health(
        self,
        rows: list[Decision],
        t: dict[str, float],
        steps: tuple[int, int, list[float], float | None],
        intake: dict[str, int],
        context: tuple[float, int, float],
    ) -> dict[str, Any]:
        stepped, trained, grad_norms, kl_check = steps
        started, base_version, lr_used = context
        received = sum(v for k, v in intake.items() if k == "accepted" or k.startswith("rejected:"))
        rejected = sum(v for k, v in intake.items() if k.startswith("rejected:"))
        finite = [g for g in grad_norms if math.isfinite(g)]
        stats = row_stats(rows, base_version, self.cfg.known_tools)
        return {
            **stats,
            "version": self.version,
            "update": self.updates,
            "time": time.time(),
            "seconds": round(time.time() - started, 2),
            "trained": trained,
            "minibatches_run": stepped,
            "kl_after_mb": kl_check,
            "entropy": t["entropy_sum"] / t["entropy_n"] if t["entropy_n"] else None,
            "kl_to_base": t["base_kl_sum"] / t["base_kl_n"] if t["base_kl_n"] else None,
            "clipped_share": t["clipped"] / t["tokens"] if t["tokens"] else None,
            "tis_truncation_share": t["truncated"] / t["tokens"] if t["tokens"] else None,
            "grad_norm": statistics.fmean(finite) if finite else None,
            "nonfinite_steps": int(round(t["nonfinite_steps"])),
            "loss": t["loss"],
            "lr": lr_used,
            "rejected_share": rejected / received if received else 0.0,
            "stale_dropped": intake.get("stale", 0),
            "intake": intake,
        }

    def _persist(self, health: dict[str, Any]) -> None:
        """Checkpoint, publish and log the update just made (rank 0 writes, everyone waits)."""
        if self.comm.rank == 0:
            self.save_checkpoint()
            self.publish(health)
            with self.cfg.health.open("a") as f:
                f.write(json.dumps(health) + "\n")
                f.flush()
                os.fsync(f.fileno())
        self.comm.barrier()

    def publish(self, health: dict[str, Any]) -> Path:
        """Write adapters/v{version}/: the master weights rounded to bf16, the adapter config and manifest.json."""
        state = {
            k: v.detach().to(torch.bfloat16).contiguous().cpu()
            for k, v in get_peft_model_state_dict(self.model).items()
        }
        manifest = {
            "mode": self.mode,
            "version": self.version,
            "update": self.updates,
            "decisions": health.get("decisions"),
            "trained": health.get("trained"),
            "decisions_total": self.decisions_total,
            "time": time.time(),
            "health": health,
        }

        def populate(tmp: Path) -> None:
            self.model.peft_config["default"].save_pretrained(str(tmp))
            save_file(state, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))

        return publish_dir(self.cfg.adapters, f"v{self.version}", populate)

    def checkpoint_versions(self) -> list[int]:
        return sorted(int(p.stem.removeprefix("ckpt-")) for p in self.cfg.checkpoints.glob("ckpt-*.pt"))

    def save_checkpoint(self) -> None:
        state = {
            "params": {n: p.detach().cpu() for n, p in self.params},
            "optimizer": self.optimizer.state_dict(),
            "version": self.version,
            "update": self.updates,
            "lr": self.lr,
            "decisions_total": self.decisions_total,
            "mode": self.mode,
        }
        path = self.cfg.checkpoints / f"ckpt-{self.version:06d}.pt"
        write_file_atomically(path, lambda tmp: torch.save(state, tmp))
        for old in self.checkpoint_versions()[: -self.cfg.ring]:
            (self.cfg.checkpoints / f"ckpt-{old:06d}.pt").unlink()

    def _load_checkpoint(self, version: int) -> dict[str, Any]:
        path = self.cfg.checkpoints / f"ckpt-{version:06d}.pt"
        if not path.exists():
            raise KeyError(f"no checkpoint for version {version}; the ring holds {self.checkpoint_versions()}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("mode", "rl") != self.mode:
            raise ValueError(
                f"checkpoint {version} was written by a {state.get('mode', 'rl')} run, this one is {self.mode}"
            )
        with torch.no_grad():
            for n, p in self.params:
                p.copy_(state["params"][n])
        self.optimizer.load_state_dict(state["optimizer"])
        return state

    def rollback(self, version: int) -> dict[str, Any]:
        """Restore checkpoint `version` (weights, AdamW moments, update count), halve the lr, publish a new version."""
        state = self._load_checkpoint(version)
        self.updates = state["update"]
        self.decisions_total = state["decisions_total"]
        self.lr = self.lr / 2
        self.recent.clear()
        self.version = self.checkpoint_versions()[-1] + 1
        record = {"event": "rollback", "restored": version, "version": self.version, "lr": self.lr, "time": time.time()}
        self._persist(record)
        return record

    def resume(self) -> bool:
        """Continue from the newest checkpoint, if any (a restart): same weights, moments, lr and version numbering."""
        versions = self.checkpoint_versions()
        if not versions:
            return False
        state = self._load_checkpoint(versions[-1])
        self.version, self.updates = state["version"], state["update"]
        self.lr, self.decisions_total = state["lr"], state["decisions_total"]
        self._refill_window()
        if self.comm.rank == 0 and not (self.cfg.adapters / f"v{self.version}").exists():
            self.publish({"event": "resume"})
        self.comm.barrier()
        return True

    def _refill_window(self) -> None:
        """After a restart, the tripwire window is the update records since the last rollback."""
        self.recent.clear()
        if self.cfg.sft or not self.cfg.health.exists():
            return
        for line in self.cfg.health.read_text().splitlines():
            record = json.loads(line) if line.strip() else {}
            if record.get("event") == "rollback":
                self.recent.clear()
            elif "update" in record:
                self.recent.append(record)

    def pop_command(self) -> dict[str, Any] | None:
        """The oldest file in control/ as a command (moved to control/done/), or None."""
        self.cfg.control.mkdir(parents=True, exist_ok=True)
        (self.cfg.control / "done").mkdir(exist_ok=True)
        for path in sorted(self.cfg.control.glob("*.json")):
            try:
                command = json.loads(path.read_text())
            except json.JSONDecodeError:
                log.warning("ignoring unreadable command file %s", path.name)
                command = None
            os.replace(path, self.cfg.control / "done" / path.name)
            if isinstance(command, dict):
                return command
        return None

    def apply(self, command: dict[str, Any]) -> None:
        """Run one control command on every rank; a bad command is logged, never fatal."""
        try:
            if command.get("op") == "rollback":
                self.rollback(int(command["version"]))
            elif command.get("op") == "set_lr":
                self.set_lr(float(command["lr"]))
            else:
                log.warning("unknown command %s", command)
        except (KeyError, ValueError) as e:
            log.warning("command %s failed: %r", command, e)


DEFAULTS: dict[str, Any] = {
    "window": 10,
    "min_records": 5,
    "band_sd": 4.0,
    "band_floor": {"length": 5.0, "calls": 0.1, "exact_share": 0.02, "distinct_fraction": 0.02, "entropy": 0.05},
    "banded": {
        "length": "tokens_p50",
        "calls": "calls_per_decision",
        "exact_share": "exact_share",
        "distinct_fraction": "distinct_fraction",
        "entropy": "entropy",
    },
    "hold_alarm": 0.6,
    "hold_trip": 0.8,
    "invented_trip": 0.005,
    "rejected_factor": 2.0,
    "rejected_floor": 0.03,
    "clipped_trip": 0.2,
    "grad_norm_factor": 5.0,
    "kl_to_base_alarm": 0.10,
    "tis_truncation_alarm": 0.05,
}


def evaluate_tripwires(
    window: Sequence[dict[str, Any]], baseline: dict[str, dict[str, float]], config: dict[str, Any] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Health records in, {"trips": [...], "alarms": [...]} out. A trip means roll back; an alarm means look.

    window    the newest health records, oldest first; the last `window` of them are used, and nothing is judged
              before `min_records` have arrived
    baseline  {metric: {"mean": m, "sd": s}} from the reference policy (v0): for the banded metrics sd is the SD of
              that metric's mean over a window of updates, so a band is baseline mean +- band_sd SD, never narrower than
              band_floor[name] (a reference policy whose sd is 0 must not trip on one duplicate completion); for
              rejected_share only the mean is read
    config    overrides for DEFAULTS

    Trips: window mean outside the band for length (tokens_p50), calls (calls_per_decision), exact_share,
    distinct_fraction or entropy; hold share above 0.8; invented tool names above 0.5%; rejected share above
    max(2 x baseline, 3%); this update's clipped share above clipped_trip; the newest grad norm above 5x the window's
    median. Alarms: hold share above 0.6; KL to base or TIS truncation above their limits.
    """
    cfg = {**DEFAULTS, **(config or {})}
    recent = list(window)[-cfg["window"] :]
    trips: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    if len(recent) < cfg["min_records"]:
        return {"trips": trips, "alarms": alarms}

    def mean(metric: str) -> float | None:
        values = [r[metric] for r in recent if r.get(metric) is not None]
        return statistics.fmean(values) if values else None

    def flag(into: list, name: str, metric: str, value: float, limit: Any, message: str) -> None:
        into.append({"name": name, "metric": metric, "value": value, "limit": limit, "message": message})

    for name, metric in cfg["banded"].items():
        value, ref = mean(metric), baseline.get(metric)
        if value is None or ref is None:
            continue
        spread = max(cfg["band_sd"] * ref["sd"], cfg["band_floor"].get(name, 0.0))
        low, high = ref["mean"] - spread, ref["mean"] + spread
        if not low <= value <= high:
            flag(trips, name, metric, value, [low, high], f"{metric} {value:.4g} outside {low:.4g}..{high:.4g}")

    hold = mean("hold_share")
    if hold is not None and hold > cfg["hold_trip"]:
        flag(
            trips, "hold_share", "hold_share", hold, cfg["hold_trip"], f"hold share {hold:.2f} above {cfg['hold_trip']}"
        )
    elif hold is not None and hold > cfg["hold_alarm"]:
        flag(
            alarms,
            "hold_share",
            "hold_share",
            hold,
            cfg["hold_alarm"],
            f"hold share {hold:.2f} above {cfg['hold_alarm']}",
        )

    invented = mean("invented_tool_share")
    if invented is not None and invented > cfg["invented_trip"]:
        flag(
            trips,
            "invented_tools",
            "invented_tool_share",
            invented,
            cfg["invented_trip"],
            f"invented tool names {invented:.4f}",
        )

    rejected = mean("rejected_share")
    if rejected is not None:
        limit = max(cfg["rejected_factor"] * baseline.get("rejected_share", {}).get("mean", 0.0), cfg["rejected_floor"])
        if rejected > limit:
            flag(
                trips,
                "rejected_share",
                "rejected_share",
                rejected,
                limit,
                f"rejected share {rejected:.3f} above {limit:.3f}",
            )

    clipped = recent[-1].get("clipped_share")
    if clipped is not None and clipped > cfg["clipped_trip"]:
        flag(trips, "clipped_share", "clipped_share", clipped, cfg["clipped_trip"], f"clipped share {clipped:.3f}")

    norms = [r["grad_norm"] for r in recent if r.get("grad_norm") is not None]
    if len(norms) >= 2 and norms[-1] > cfg["grad_norm_factor"] * statistics.median(norms):
        flag(
            trips,
            "grad_norm",
            "grad_norm",
            norms[-1],
            cfg["grad_norm_factor"] * statistics.median(norms),
            "grad norm spike",
        )

    for name, metric in (("kl_to_base", "kl_to_base"), ("tis_truncation", "tis_truncation_share")):
        value = mean(metric)
        if value is not None and value > cfg[f"{name}_alarm"]:
            flag(
                alarms, name, metric, value, cfg[f"{name}_alarm"], f"{metric} {value:.4g} above {cfg[f'{name}_alarm']}"
            )
    return {"trips": trips, "alarms": alarms}


def baseline_from(records: Sequence[dict[str, Any]], window: int = 10) -> dict[str, dict[str, float]]:
    """A baseline for evaluate_tripwires from health-shaped records of the reference policy (one per 256 decisions).

    mean is the metric's mean; sd is the SD of its mean over `window` records, estimated from the non-overlapping
    windows, or from the per-record SD divided by sqrt(window) when there are fewer than two windows.
    """
    baseline: dict[str, dict[str, float]] = {}
    metrics = [*DEFAULTS["banded"].values(), "rejected_share"]
    for metric in metrics:
        values = [r[metric] for r in records if r.get(metric) is not None]
        if len(values) < 2:
            continue
        chunks = [statistics.fmean(values[i : i + window]) for i in range(0, len(values) - window + 1, window)]
        sd = statistics.stdev(chunks) if len(chunks) >= 2 else statistics.stdev(values) / math.sqrt(window)
        baseline[metric] = {"mean": statistics.fmean(values), "sd": sd}
    return baseline


def serve(learner: Learner, inbox: Inbox | None, *, poll: float = 2.0, max_updates: int | None = None) -> None:
    """Run until a stop command (or max_updates): rank 0 decides each step and every rank carries it out."""
    done = quiet = 0
    while True:
        action: dict[str, Any] = {"op": "idle"}
        if learner.comm.rank == 0:
            command = learner.pop_command()
            if command is not None:
                action = {"op": "command", "command": command}
            elif inbox is not None:
                quiet = 0 if inbox.poll() else quiet + 1
                batch = inbox.take(learner.version, flush=quiet >= 2)
                if batch is not None:
                    action = {"op": "update", "rows": batch, "intake": inbox.drain_counts()}
        action = learner.comm.broadcast(action)
        if action["op"] == "command":
            if action["command"].get("op") == "stop":
                return
            learner.apply(action["command"])
        elif action["op"] == "update":
            health = learner.update(action["rows"], action["intake"])
            done += 1
            log.info(
                "version %d: %s", health["version"], {k: health[k] for k in ("trained", "kl_after_mb", "grad_norm")}
            )
            if max_updates is not None and done >= max_updates:
                return
        else:
            time.sleep(poll)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument(
        "--root", required=True, help="the run's directory: inbox/, adapters/, checkpoints/, control/, health.jsonl"
    )
    p.add_argument("--init", default="", help="a PEFT adapter to start from (ignored when the root has checkpoints)")
    p.add_argument(
        "--known-tools", default="", help="comma-separated tool names; any other name a policy emits is invented"
    )
    p.add_argument("--baseline", default="", help="the base model's tripwire baseline (JSON from baseline_from)")
    p.add_argument("--sft", action="store_true", help="cross-entropy on completion tokens; write adapters to --root")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--poll", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse()
    known_tools = tuple(t for t in args.known_tools.split(",") if t)
    if not known_tools and not args.sft:
        sys.exit("--known-tools names no tools, which would disable the invented-tool tripwire")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    comm = Comm.from_env()
    torch.manual_seed(args.seed)
    cfg = LearnerConfig(
        base=args.base, root=Path(args.root), init=args.init, lr=args.lr, seed=args.seed, dtype=args.dtype,
        known_tools=known_tools, sft=args.sft,
    )  # fmt: skip
    model = build_model(cfg, device)
    comm.sync_params([p for p in model.parameters() if p.requires_grad])
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    learner = Learner(cfg, model, lambda ids: tokenizer.decode(ids, skip_special_tokens=False), comm, baseline=baseline)
    if learner.resume():
        log.info("resumed at version %d", learner.version)
    serve(learner, Inbox(cfg) if comm.rank == 0 else None, poll=args.poll)
    comm.barrier()
    comm.close()


if __name__ == "__main__":
    main()
