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
and dropped. Read files move to inbox/consumed/. A row flagged `"sft": true` is a teacher label, not a sample: it needs
only game_id, prompt_token_ids and token_ids (and optionally loss_mask), carries no advantage, and is trained on by the
auxiliary cross-entropy term below; with that term off (aux_alpha0 = 0) such rows are dropped and counted.

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

Two terms are added to the minibatch's loss, each over the same trained tokens (the masks) and divided by the same
B x 128, so a weight reads as "per token, relative to the policy-gradient term":
  anchor   + beta_t sum_i sum_t m_it k3_it, the KL to a frozen reference policy held on the sampled tokens by the k3
           estimator r - 1 - log r, r = pi_anchor / pi_theta. It is evaluated on the rows' tokens, which the sampler mu
           drew (a stale pi_theta, so only approximately pi_theta's own samples), so it is a per-token penalty that
           pulls log pi_theta toward the anchor's (its gradient on log pi_theta is 1 - r: negative where the anchor is
           the likelier, positive where it is not) rather than an unbiased KL estimate; it is not weighted by the TIS
           weight w. The reference is the adapter given with --anchor (the frozen distilled SFT adapter: RL's pull back
           to the imitation policy, which made imitation pay off in AlphaStar and VPT); without --anchor it is --init,
           so a run started from the SFT adapter is held to it, and `--anchor base` is the base model. beta_t = beta0 x
           decay^update, beta0 0.1 and decay 0.999 by default, 0 turns it off. The update count is what a checkpoint
           and a rollback restore, so the schedule comes back with them. The anchor's log-probs are computed once per
           update, with no gradient, before the first step; the health record logs kl_to_anchor (its mean k3 over the
           update's rows), anchor_beta and the anchor in use. kl_to_base, the alarm, stays.
  aux      + alpha_t sum_i sum_t m_it (-log pi_theta) over the sft-flagged rows, up to aux_rows_per_update beside
           an update's 256 policy-gradient rows, all in minibatch 1 (so a KL stop of minibatch 2 never drops them).
           alpha_t = alpha0 x decay^aux_updates, alpha0 0.2 and decay 0.995 by default, aux_updates the updates that
           trained on any such rows (saved in a checkpoint and restored by a rollback), so the weight does not decay
           while no label arrives. These rows are never in the PG term (no advantage, no ratio, no KL stop or
           tripwire statistics); they are the teacher's answer on states the policy lost from (the orchestrator's hard-
           state labels), added only where RL has no signal. The health record logs aux_alpha, aux_rows and aux_ce. Its
`loss` is the policy-gradient and anchor terms (the SFT cross-entropy in SFT mode); pg_loss and anchor_loss are its
two parts and aux_loss is the auxiliary term, each already divided by B x 128.

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
It trains `sft_epochs` passes over the dataset (default 2, reshuffled each pass; the dataset is complete once the
inbox has gone quiet, so drop it as one file). After each pass it publishes ROOT/epochs/epoch{N}/ (the adapter and a
manifest with the version and the mean token entropy of the checkpoint and of the base model on a fixed sample of the
teacher's tokens). Pick the checkpoint to keep by held-out win rate and that entropy ratio, never by the SFT loss:
`select_sft_checkpoint` in train/stats.py. The entropy is measured on a held-out tenth of the dataset (chosen by a
hash of each row's game_id, never trained on). A checkpoint records the epoch and how many of its rows were trained, and
a restart rebuilds the dataset from inbox/consumed/ and carries on from there; an epoch directory that already exists
is reused when it holds this version's weights and refused otherwise. After the last epoch the learner writes
ROOT/sft-done.json and exits.

Run in the trainer image (vLLM's image with PEFT), after training starts:
  torchrun --nproc_per_node 2 learner.py --base Qwen/Qwen3-8B --root /run1 [--init ADAPTER] --known-tools a,b,c
      [--baseline baseline.json]
  torchrun --nproc_per_node 2 learner.py --base Qwen/Qwen3-8B --root /sft1 --sft --lr 1e-4 [--sft-epochs 2]
  torchrun --nproc_per_node 2 learner.py --base Qwen/Qwen3-8B --root /run1 --init /sft1/epochs/epoch1
      --anchor /sft1/epochs/epoch1 --known-tools a,b,c
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
import zlib
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
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
    anchor: str = ""  # a frozen PEFT adapter the policy is held near; "" anchors to the base model
    anchor_beta0: float = 0.1  # KL-to-anchor weight at update 0; 0 turns the anchor off
    anchor_decay: float = 0.999  # beta_t = anchor_beta0 * anchor_decay ** update
    aux_alpha0: float = 0.2  # weight of the cross-entropy on sft-flagged rows at update 0; 0 drops those rows
    aux_decay: float = 0.995  # alpha_t = aux_alpha0 * aux_decay ** update
    aux_rows_per_update: int = 32  # most sft-flagged rows trained on beside one update's policy-gradient rows
    sft_epochs: int = 2  # SFT mode: passes over the dataset; a checkpoint is published after each
    sft_entropy_rows: int = 128  # SFT mode: rows the epoch-end entropy is measured on
    sft_holdout: float = 0.1  # SFT mode: the share of the dataset (by a hash of game_id) held out of training for it

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

    @property
    def epochs(self) -> Path:
        return self.root / "epochs"

    @property
    def sft_done(self) -> Path:
        return self.root / "sft-done.json"


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
    aux: bool = False  # an sft-flagged row in an RL run: a teacher label, trained on by cross-entropy, no advantage


class RowRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def parse_row(obj: Any, max_len: int, *, sft: bool = False) -> Decision:
    """A Decision from one JSON row, or RowRejected with the reason it is unusable. An SFT row needs only the ids, and
    so does a row flagged `"sft": true` in an RL run (an auxiliary teacher label: `aux`)."""
    if not isinstance(obj, dict):
        raise RowRejected("not_an_object")
    aux = not sft and obj.get("sft") is True
    ids_only = sft or aux
    try:
        prompt, completion = list(obj["prompt_token_ids"]), list(obj["token_ids"])
        game_id = str(obj["game_id"])
        logprobs, advantage, version = (
            ([], 0.0, 0) if ids_only else (list(obj["logprobs"]), float(obj["advantage"]), obj["version"])
        )
    except (KeyError, TypeError, ValueError) as e:
        raise RowRejected("missing_field") from e
    if not prompt or not completion:
        raise RowRejected("empty")
    if not ids_only and len(logprobs) != len(completion):
        raise RowRejected("logprobs_length")
    if len(prompt) + len(completion) > max_len:
        raise RowRejected("too_long")
    if not ids_only and (not isinstance(version, int) or isinstance(version, bool) or version < 0):
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
    return Decision(
        game_id, version, prompt, completion, logprobs, advantage, loss_mask, bool(obj.get("is_hold")), aux=aux
    )


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
        self.aux: list[Decision] = []  # sft-flagged teacher labels, oldest first
        self.dataset: list[Decision] = []  # SFT mode: every row to train on, replayed by each further epoch
        self.holdout: list[Decision] = []  # SFT mode: rows never trained on, for the epoch-end entropy
        self.epoch, self.epoch_trained, self.finished = 1, 0, False
        self.counts: Counter[str] = Counter()
        cfg.inbox.mkdir(parents=True, exist_ok=True)
        (cfg.inbox / "consumed").mkdir(exist_ok=True)

    def poll(self) -> int:
        """Read every new *.jsonl file into the pool, moving it to consumed/. Returns the rows accepted."""
        accepted = 0
        for path in sorted(self.cfg.inbox.glob("*.jsonl")):
            accepted += self._read(path)
            os.replace(path, self.cfg.inbox / "consumed" / path.name)
        return accepted

    def _read(self, path: Path) -> int:
        accepted = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self.counts["rejected:bad_json"] += 1
                continue
            # a malformed teacher label is counted apart: it must not feed the rejected share the tripwire reads
            kind = (
                "aux_rejected" if not self.cfg.sft and isinstance(obj, dict) and obj.get("sft") is True else "rejected"
            )
            try:
                row = parse_row(obj, self.cfg.max_len, sft=self.cfg.sft)
            except RowRejected as e:
                self.counts[f"{kind}:{e.reason}"] += 1
                continue
            if row.aux:
                self._keep_aux(row)
                continue
            if self.cfg.sft and self.held_out(row):
                self.holdout.append(row)
            else:
                self.pool.append(row)
                if self.cfg.sft:
                    self.dataset.append(row)
            accepted += 1
            self.counts["accepted"] += 1
        return accepted

    def held_out(self, row: Decision) -> bool:
        """Whether an SFT row is in the held-out slice: a hash of its game_id, so the same rows are held out however
        the dataset arrives, and again after a restart."""
        key = zlib.crc32(f"{self.cfg.seed}:{row.game_id}".encode())
        return key % 10_000 < self.cfg.sft_holdout * 10_000

    def _keep_aux(self, row: Decision) -> None:
        """An sft-flagged row waits for take_aux; with the auxiliary term off it is dropped, and counted apart from the
        policy-gradient rows so it never shows in the rejected share."""
        if self.cfg.aux_alpha0 > 0:
            self.aux.append(row)
            self.counts["aux_accepted"] += 1
        else:
            self.counts["aux_dropped"] += 1

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
        if self.cfg.sft:
            self.epoch_trained += len(batch)
        return batch

    def take_aux(self) -> list[Decision]:
        """Up to aux_rows_per_update waiting sft-flagged rows, oldest first (a label does not go stale)."""
        n = self.cfg.aux_rows_per_update
        batch, self.aux = self.aux[:n], self.aux[n:]
        return batch

    def epoch_done(self) -> bool:
        """SFT only, to be asked once the inbox is quiet: the epoch in progress has trained on all its rows."""
        return self.cfg.sft and not self.finished and self.epoch_trained > 0 and not self.pool

    def entropy_sample(self) -> list[Decision]:
        """The held-out rows for the epoch-end entropy (at most sft_entropy_rows, the same ones at every epoch)."""
        rows = list(self.holdout)
        random.Random(self.cfg.seed).shuffle(rows)
        return rows[: self.cfg.sft_entropy_rows]

    def progress(self) -> tuple[int, int]:
        """(epoch, rows of it trained so far): what a checkpoint records so a restart can carry on."""
        return self.epoch, self.epoch_trained

    def _shuffled(self, epoch: int) -> list[Decision]:
        rows = list(self.dataset)
        if epoch > 1:
            random.Random(self.cfg.seed * 1_000_003 + epoch).shuffle(rows)
        return rows

    def next_epoch(self) -> None:
        """Close the epoch just published: refill the pool with the dataset reshuffled, or finish after the last."""
        if self.epoch >= self.cfg.sft_epochs:
            self.finished = True
            return
        self.epoch += 1
        self.epoch_trained = 0
        self.pool = self._shuffled(self.epoch)

    def restore(self, epoch: int, trained: int) -> None:
        """SFT restart: the rows already read are in inbox/consumed/, so rebuild the dataset (and the held-out slice)
        from there and carry on in `epoch` after the first `trained` rows of it. Files still in the inbox are read as
        usual afterwards."""
        self.pool, self.dataset, self.holdout = [], [], []
        for path in sorted((self.cfg.inbox / "consumed").glob("*.jsonl")):
            self._read(path)
        self.counts.clear()
        self.epoch, self.epoch_trained, self.finished = epoch, trained, False
        self.pool = self._shuffled(epoch)[trained:]

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


def anchor_k3(logp: torch.Tensor, logp_anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum_t m_t (r_t - 1 - log r_t), r_t = pi_anchor / pi_theta on the rows' tokens: the k3 estimator, non-negative.
    It is unbiased for KL(pi_theta || pi_anchor) only on tokens drawn from pi_theta; these were drawn by the sampler mu
    (a stale pi_theta), so as a loss it is a penalty that pulls log pi_theta toward the anchor's: its gradient on
    log pi_theta is 1 - r_t. It is not weighted by the TIS weight. The anchor's log-probs carry no grad."""
    delta = logp_anchor.detach() - logp
    return ((torch.exp(delta) - delta - 1) * mask).sum()


def kl_estimate(logp_after: torch.Tensor, logp_before: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """(sum, count) of r log r - r + 1 over the trained tokens, r = pi_after / pi_before."""
    log_r = logp_after - logp_before
    per_token = torch.exp(log_r) * log_r - torch.exp(log_r) + 1
    return float((per_token * mask).sum()), float(mask.sum())


@dataclass(frozen=True)
class Schedule:
    """The weights one update uses: beta on the KL to the anchor, alpha on the cross-entropy of sft-flagged rows."""

    beta: float
    alpha: float


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
        "anchor_kl_sum",
        "anchor_kl_n",
        "pg_loss",
        "anchor_loss",
        "aux_loss",
        "aux_ce_sum",
        "aux_tokens",
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
        self.aux_updates = 0  # updates that trained on sft-flagged rows: what the aux weight decays over
        self.sft_progress: tuple[int, int] = (1, 0)  # SFT: (epoch, rows of it trained), saved in each checkpoint
        self.resumed = False
        self.decisions_total = 0
        backbone = model.get_base_model()
        self.backbone = backbone.model
        self.head_weight = backbone.lm_head.weight.detach().float()
        self.anchor_weights = self._load_anchor() if cfg.anchor and self.anchor_enabled else {}
        cfg.checkpoints.mkdir(parents=True, exist_ok=True)

    @property
    def anchor_enabled(self) -> bool:
        return not self.cfg.sft and self.cfg.anchor_beta0 > 0

    def anchor_label(self) -> str | None:
        """What the KL penalty holds the policy to: the anchor adapter's path, "base", or None when it is off."""
        return None if not self.anchor_enabled else self.cfg.anchor or "base"

    def anchor_beta(self) -> float:
        """beta_t = beta0 * decay ** update: the weight of the KL to the anchor in the update about to run. It is a
        function of the update count, which a checkpoint and a rollback restore."""
        return self.cfg.anchor_beta0 * self.cfg.anchor_decay**self.updates if self.anchor_enabled else 0.0

    def aux_alpha(self) -> float:
        """alpha_t = alpha0 * decay ** aux_updates: the weight of the cross-entropy on sft-flagged rows in this update.
        It decays only over updates that had such rows, so a run that gets labels late is not already at the floor."""
        return self.cfg.aux_alpha0 * self.cfg.aux_decay**self.aux_updates if not self.cfg.sft else 0.0

    def _load_anchor(self) -> dict[str, torch.Tensor]:
        """The frozen anchor adapter's weights, by the trainable parameters' names. They are swapped in for a no-grad
        pass (anchor_logps), never trained and never published. The adapter must be the learner's LoRA: the same r,
        lora_alpha and target modules (read from its adapter_config.json, since the same shapes can carry another
        scaling), and every parameter present with the same shape."""
        folder = Path(self.cfg.anchor)
        try:
            saved = json.loads((folder / "adapter_config.json").read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"anchor adapter {folder} has no readable adapter_config.json: {e}") from e
        mine = self.model.peft_config["default"]
        want = {"r": mine.r, "lora_alpha": mine.lora_alpha, "target_modules": sorted(mine.target_modules)}
        got = {"r": saved.get("r"), "lora_alpha": saved.get("lora_alpha"), "target_modules": None}
        modules = saved.get("target_modules")
        got["target_modules"] = sorted(modules) if isinstance(modules, list | set | tuple) else modules
        if got != want:
            raise ValueError(
                f"anchor adapter {folder} does not fit the learner's LoRA: it is {got}, the learner's is {want}"
            )
        state = load_file(str(folder / "adapter_model.safetensors"), device=str(self.device))
        weights = {}
        for name, p in self.params:
            key = name.replace(".default", "")
            if key not in state or state[key].shape != p.shape:
                raise ValueError(f"anchor adapter {self.cfg.anchor} does not fit the learner's LoRA at {key}")
            weights[name] = state[key].float().contiguous()
        return weights

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

    def anchor_logps(self, row: Decision) -> torch.Tensor:
        """No-grad log-probs of one row's tokens under the anchor (see anchor_logps_many)."""
        return self.anchor_logps_many([row])[0]

    def anchor_logps_many(self, rows: Sequence[Decision]) -> list[torch.Tensor]:
        """No-grad log-probs of the rows' tokens under the anchor: the anchor adapter's weights swapped into the LoRA
        once for all the rows, or the base model (adapter off) when no anchor adapter was given. The policy's weights
        are put back whatever happens, a failure part-way through the copy-in included."""
        if not self.anchor_weights:
            return [self.logps(row, grad=False, adapter=False)[0] for row in rows]
        held = {n: p.detach().clone() for n, p in self.params}
        try:
            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(self.anchor_weights[n])
            return [self.logps(row, grad=False)[0] for row in rows]
        finally:
            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(held[n])

    def update(
        self, rows: list[Decision], intake: dict[str, int] | None = None, aux: Sequence[Decision] = ()
    ) -> dict[str, Any]:
        """One update on `rows` (and the sft-flagged `aux` rows beside them): publishes the next version and returns its
        health record."""
        cfg, comm = self.cfg, self.comm
        started, base_version, lr_used = time.time(), self.version, self.effective_lr()
        rows = [annotate(r, self.decode, cfg.tool_call_open, cfg.tool_call_close) for r in rows]
        batches = self._minibatches(rows)
        owned = [[r for i, r in enumerate(mb) if i % comm.world == comm.rank] for mb in batches]
        schedule = Schedule(self.anchor_beta(), self.aux_alpha())
        aux_rows = (
            [annotate(r, self.decode, cfg.tool_call_open, cfg.tool_call_close) for r in aux] if schedule.alpha else []
        )
        # every auxiliary row goes in minibatch 1, which always runs: a KL stop of the later ones cannot drop them
        aux_owned = [[r for i, r in enumerate(aux_rows) if i % comm.world == comm.rank]] + [[] for _ in batches[1:]]
        totals: Counter[str] = Counter()
        prox, anchor = ({}, {}) if cfg.sft else self._prox_pass(rows, owned, totals, schedule.beta > 0)
        stepped, grad_norms, kl_check = 0, [], None
        for j, mb in enumerate(owned):
            if j > 0 and not cfg.sft:
                kl_check = self._kl_after(mb, prox)
                if kl_check > cfg.kl_stop:
                    break
            grad_norms.append(self._train_minibatch(mb, len(batches[j]), prox, totals, anchor, aux_owned[j], schedule))
            stepped += 1
        reduced = self._reduce(totals)
        self.updates += 1
        self.aux_updates += bool(aux_rows)
        self.version += 1
        trained = sum(len(batches[j]) for j in range(stepped))
        self.decisions_total += trained
        health = self._health(
            rows, reduced, (stepped, trained, grad_norms, kl_check), intake or {}, (started, base_version, lr_used)
        )
        health |= self._aux_health(reduced, schedule, len(aux_rows))
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
        self, rows: list[Decision], owned: list[list[Decision]], totals: Counter[str], with_anchor: bool = False
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """log pi_prox of every owned row, before any step, plus the entropy and KL-to-base health samples; with
        `with_anchor` also the anchor's log-prob of every owned row (and the KL to it, over all of them)."""
        sample = set(random.Random(self.cfg.seed + self.version).sample(
            range(len(rows)), max(1, math.ceil(self.cfg.base_kl_fraction * len(rows)))
        ))  # fmt: skip
        position = {id(r): i for i, r in enumerate(rows)}
        mine = [r for mb in owned for r in mb]
        prox: dict[int, torch.Tensor] = {}
        anchor: dict[int, torch.Tensor] = {}
        masks = {id(r): torch.tensor(r.mask, device=self.device) for r in mine}
        for row in mine:
            logp, ent = self.logps(row, grad=False, entropy=not row.is_hold)
            prox[id(row)] = logp
            if ent is not None:
                totals["entropy_sum"] += float((ent * masks[id(row)]).sum())
                totals["entropy_n"] += float(masks[id(row)].sum())
        if with_anchor:  # one swap of the anchor's weights for every row
            for row, logp_anchor in zip(mine, self.anchor_logps_many(mine), strict=True):
                anchor[id(row)] = logp_anchor
                totals["anchor_kl_sum"] += float(anchor_k3(prox[id(row)], logp_anchor, masks[id(row)]))
                totals["anchor_kl_n"] += float(masks[id(row)].sum())
        for row in (r for r in mine if position[id(r)] in sample):
            # with the base model as the anchor its log-probs are the base's: no second pass
            base_logp = (
                anchor[id(row)]
                if with_anchor and not self.anchor_weights
                else self.logps(row, grad=False, adapter=False)[0]
            )
            delta = base_logp - prox[id(row)]
            totals["base_kl_sum"] += float(((torch.exp(delta) - delta - 1) * masks[id(row)]).sum())
            totals["base_kl_n"] += float(masks[id(row)].sum())
        return prox, anchor

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
        self,
        mb: list[Decision],
        size: int,
        prox: dict[int, torch.Tensor],
        totals: Counter[str],
        anchor: dict[int, torch.Tensor] | None = None,
        aux: Sequence[Decision] = (),
        schedule: Schedule | None = None,
    ) -> float:
        """Accumulate the loss gradient over this rank's rows, sum it across ranks, clip, step. Returns the norm.

        `size` is the minibatch's policy-gradient row count and the divisor of every term, so beta and alpha weigh the
        anchor and the auxiliary cross-entropy against the PG term token for token."""
        cfg = self.cfg
        schedule = schedule or Schedule(0.0, 0.0)
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
            divisor = size * cfg.token_divisor
            scaled = pg = terms.loss / divisor
            if schedule.beta > 0 and anchor:
                pull = schedule.beta * anchor_k3(logp, anchor[id(row)], mask) / divisor
                scaled = scaled + pull
                totals["anchor_loss"] += float(pull.detach())
            scaled.backward()
            totals["pg_loss"] += float(pg.detach())
            totals["loss"] += float(scaled.detach())
            totals["clipped"] += terms.clipped
            totals["truncated"] += terms.truncated
            totals["tokens"] += terms.tokens
        for row in aux:
            logp, _ = self.logps(row, grad=True)
            terms = sft_terms(logp, torch.tensor(row.mask, device=self.device))
            scaled = schedule.alpha * terms.loss / (size * cfg.token_divisor)
            scaled.backward()
            totals["aux_loss"] += float(scaled.detach())
            totals["aux_ce_sum"] += float(terms.loss.detach())
            totals["aux_tokens"] += terms.tokens
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
            "loss": t["loss"],  # pg_loss + anchor_loss (the SFT cross-entropy in SFT mode); aux_loss is apart
            "pg_loss": t["pg_loss"],
            "anchor_loss": t["anchor_loss"],
            "lr": lr_used,
            "rejected_share": rejected / received if received else 0.0,
            "stale_dropped": intake.get("stale", 0),
            "intake": intake,
        }

    def _aux_health(self, t: dict[str, float], schedule: Schedule, aux_rows: int) -> dict[str, Any]:
        """The anchor's and the auxiliary term's numbers: the KL to the anchor over the update's rows (measured before
        its first step) and the beta, alpha and mean per-token cross-entropy they were weighed with."""
        return {
            "kl_to_anchor": t["anchor_kl_sum"] / t["anchor_kl_n"] if t["anchor_kl_n"] else None,
            "anchor_beta": schedule.beta,
            "anchor": self.anchor_label(),
            "aux_alpha": schedule.alpha,
            "aux_rows": aux_rows,
            "aux_ce": t["aux_ce_sum"] / t["aux_tokens"] if t["aux_tokens"] else None,
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
        manifest = {
            "mode": self.mode,
            "version": self.version,
            "update": self.updates,
            "decisions": health.get("decisions"),
            "trained": health.get("trained"),
            "decisions_total": self.decisions_total,
            "anchor": self.anchor_label(),
            "init": self.cfg.init or None,
            "time": time.time(),
            "health": health,
        }
        return self._publish_adapter(self.cfg.adapters, f"v{self.version}", manifest)

    def _publish_adapter(self, parent: Path, name: str, manifest: dict[str, Any]) -> Path:
        state = {
            k: v.detach().to(torch.bfloat16).contiguous().cpu()
            for k, v in get_peft_model_state_dict(self.model).items()
        }

        def populate(tmp: Path) -> None:
            self.model.peft_config["default"].save_pretrained(str(tmp))
            save_file(state, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))

        return publish_dir(parent, name, populate)

    def finish_epoch(self, epoch: int, sample: Sequence[Decision]) -> dict[str, Any]:
        """SFT mode, after the last update of an epoch: publish epochs/epoch{N}/ (the adapter and a manifest with the
        epoch's version and the mean token entropy over `sample`, with the adapter and with the base model alone).

        The entropy is measured on the teacher's tokens at every completion position, so the pair says how much
        sharper than the base the checkpoint has become on that text; `select_sft_checkpoint` (train/stats.py) reads it.
        Every rank measures its share and the sums are reduced; rank 0 writes. An epoch directory that already exists
        (a restart between its publication and the next checkpoint) is reused if it holds this version's weights and
        refused otherwise; every rank decides that alike, before any collective, so none waits at a barrier."""
        existing = self.cfg.epochs / f"epoch{epoch}"
        if existing.exists():
            try:
                record = json.loads((existing / "manifest.json").read_text())
            except (OSError, ValueError):
                record = {}
            if record.get("version") != self.version:
                raise RuntimeError(
                    f"{existing} already exists and is not version {self.version}'s (it is {record.get('version')}): "
                    "move it away, or use a fresh --root"
                )
            log.warning("%s is already published for version %d: reusing it", existing, self.version)
            return record
        rows = [annotate(r, self.decode, self.cfg.tool_call_open, self.cfg.tool_call_close) for r in sample]
        acc = torch.zeros(4, dtype=torch.float64, device=self.device)
        for row in (r for i, r in enumerate(rows) if i % self.comm.world == self.comm.rank):
            mask = torch.tensor(row.mask, device=self.device)
            for k, adapter in ((0, True), (2, False)):
                _, ent = self.logps(row, grad=False, adapter=adapter, entropy=True)
                assert ent is not None
                acc[k] += float((ent * mask).sum())
                acc[k + 1] += float(mask.sum())
        total = self.comm.sum(acc).tolist()
        entropy = total[0] / total[1] if total[1] else None
        base = total[2] / total[3] if total[3] else None
        record = {
            "mode": self.mode,
            "epoch": epoch,
            "version": self.version,
            "update": self.updates,
            "decisions_total": self.decisions_total,
            "entropy": entropy,
            "base_entropy": base,
            "entropy_ratio": entropy / base if entropy is not None and base else None,
            "entropy_rows": len(rows),
            "time": time.time(),
        }
        if self.comm.rank == 0:
            self._publish_adapter(self.cfg.epochs, f"epoch{epoch}", record)
        self.comm.barrier()
        return record

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
            "aux_updates": self.aux_updates,
            "sft_progress": list(self.sft_progress),
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
        self.aux_updates = state.get("aux_updates", 0)
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
        self.aux_updates = state.get("aux_updates", 0)
        epoch, trained = state.get("sft_progress", (1, 0))
        self.sft_progress = (int(epoch), int(trained))
        self.resumed = True
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


def serve(
    learner: Learner,
    inbox: Inbox | None,
    *,
    poll: float = 2.0,
    max_updates: int | None = None,
    stop_when: Callable[[], bool] | None = None,
) -> None:
    """Run until a stop command (or max_updates, or `stop_when()` on rank 0): rank 0 decides each step and every rank
    carries it out. In SFT mode, once the inbox has gone quiet and an epoch has trained on all its rows, the epoch's
    checkpoint is published and the next epoch's pass over the dataset begins, up to `sft_epochs`."""
    done = quiet = 0
    if inbox is not None and learner.cfg.sft and learner.resumed:
        inbox.restore(*learner.sft_progress)
    while True:
        action: dict[str, Any] = {"op": "idle"}
        if learner.comm.rank == 0:
            command = learner.pop_command()
            if command is None and stop_when is not None and stop_when():
                command = {"op": "stop"}
            if command is not None:
                action = {"op": "command", "command": command}
            elif inbox is not None:
                quiet = 0 if inbox.poll() else quiet + 1
                batch = inbox.take(learner.version, flush=quiet >= 2)
                if batch is not None:
                    action = {
                        "op": "update", "rows": batch, "intake": inbox.drain_counts(), "aux": inbox.take_aux(),
                        "progress": inbox.progress(),
                    }  # fmt: skip
                elif quiet >= 2 and inbox.epoch_done():
                    action = {
                        "op": "epoch_end", "epoch": inbox.epoch, "sample": inbox.entropy_sample(),
                        "last": inbox.epoch >= learner.cfg.sft_epochs,
                    }  # fmt: skip
        action = learner.comm.broadcast(action)
        if action["op"] == "command":
            if action["command"].get("op") == "stop":
                return
            learner.apply(action["command"])
        elif action["op"] == "epoch_end":
            record = learner.finish_epoch(action["epoch"], action["sample"])
            log.info("sft epoch %d done: %s", record["epoch"], {k: record[k] for k in ("version", "entropy_ratio")})
            if inbox is not None:
                inbox.next_epoch()
                learner.sft_progress = inbox.progress()
            if learner.comm.rank == 0:
                learner.save_checkpoint()  # so a restart begins the next epoch, not this one again
                if action["last"]:
                    learner.cfg.sft_done.write_text(json.dumps({"epochs": action["epoch"], "version": learner.version}))
            learner.comm.barrier()
            if action["last"]:
                return
        elif action["op"] == "update":
            if "progress" in action:
                learner.sft_progress = (int(action["progress"][0]), int(action["progress"][1]))
            health = learner.update(action["rows"], action["intake"], action["aux"])
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
    p.add_argument("--sft-epochs", type=int, default=2, help="SFT: passes over the dataset, a checkpoint after each")
    p.add_argument(
        "--anchor",
        default=None,
        help="RL: the frozen adapter the KL penalty holds the policy near; default: --init (the distilled SFT "
        "adapter), or the base model when there is no --init; `base` forces the base model",
    )
    p.add_argument(
        "--anchor-beta0", type=float, default=0.1, help="RL: KL-to-anchor weight at update 0; 0 turns it off"
    )
    p.add_argument("--anchor-decay", type=float, default=0.999, help="RL: beta_t = beta0 * decay ** update")
    p.add_argument("--aux-alpha0", type=float, default=0.2, help="RL: weight of the CE on sft rows; 0 drops those rows")
    p.add_argument("--aux-decay", type=float, default=0.995, help="RL: alpha_t = alpha0 * decay ** update")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--poll", type=float, default=2.0)
    return p.parse_args()


def resolve_anchor(init: str, anchor: str | None) -> str:
    """The anchor adapter's path, "" for the base model: the one given, else --init (a run started from the distilled
    adapter is held near it, not silently near the base), and `base` or an empty value mean the base model."""
    if anchor is None:
        return init
    return "" if anchor in ("", "base") else anchor


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
        known_tools=known_tools, sft=args.sft, sft_epochs=args.sft_epochs,
        anchor=resolve_anchor(args.init, args.anchor),
        anchor_beta0=args.anchor_beta0, anchor_decay=args.anchor_decay, aux_alpha0=args.aux_alpha0,
        aux_decay=args.aux_decay,
    )  # fmt: skip
    log.info("init: %s; anchor: %s", cfg.init or "none", cfg.anchor or "base model")
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
