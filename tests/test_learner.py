"""The persistent learner on a tiny random Qwen3 on the CPU: the loss, the update loop, publish and rollback, tripwires.

Run with: uv run --with torch --with transformers --with peft pytest tests/test_learner.py -q
"""

import json
import math
import os
import random
import socket
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import learner as L  # noqa: E402

OPEN, CLOSE = 100, 101
PIECES = {
    10: '{"name": "say", "arguments": {"text": "hello"}}',
    11: '{"name": "army", "arguments": {"action": "attack"}}',
    12: '{"name": "hold", "arguments": {}}',
    13: '{"name": "say", "arguments": {"te',
    14: '{"name": "teleport", "arguments": {}}',
    15: "not json at all",
}


def decode(ids):
    return "".join(PIECES.get(i, "") for i in ids)


def tiny_model(path: Path) -> None:
    config = transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
    )  # fmt: skip
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)


def config(tmp_path: Path, **overrides) -> L.LearnerConfig:
    values = dict(
        base=str(tmp_path / "base"), root=tmp_path / "run", dtype="float32", rank=4, alpha=8, decisions_per_update=4,
        tool_call_open=OPEN, tool_call_close=CLOSE, known_tools=("army", "hold", "say"), lr=1e-2, warmup_updates=1,
        kl_stop=1e9, gradient_checkpointing=False,
    )  # fmt: skip
    return L.LearnerConfig(**{**values, **overrides})


def make_learner(tmp_path: Path, comm=None, **overrides) -> L.Learner:
    if not (tmp_path / "base").exists():
        tiny_model(tmp_path / "base")
    cfg = config(tmp_path, **overrides)
    torch.manual_seed(0)
    return L.Learner(cfg, L.build_model(cfg, torch.device("cpu")), decode, comm)


def row(seed=0, version=0, advantage=1.0, completion=None, hold=False, loss_mask=None, logprobs=None) -> L.Decision:
    rng = random.Random(seed)
    completion = completion or [rng.randrange(20, 90) for _ in range(rng.randrange(3, 7))]
    return L.Decision(
        game_id=f"g{seed}", version=version, prompt=[rng.randrange(20, 90) for _ in range(4)], completion=completion,
        logprobs=logprobs or [-3.0] * len(completion), advantage=advantage, loss_mask=loss_mask, is_hold=hold,
    )  # fmt: skip


def rows(n=4, **kwargs) -> list[L.Decision]:
    return [row(seed=i, advantage=(-1) ** i * (1 + i / 4), **kwargs) for i in range(n)]


def weights(learner: L.Learner) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in learner.params}


def terms(logp, prox, mu, advantage, mask=None):
    mask = torch.ones_like(logp) if mask is None else mask
    return L.ppo_terms(logp, prox, mu, advantage, mask, config(Path("unused")))


# --- loss ---


def test_a_completion_twice_as_long_gets_twice_the_gradient_from_the_token_sum():
    grads = []
    for n in (4, 8):
        logp = torch.zeros(n, requires_grad=True)
        t = terms(logp, torch.zeros(n), torch.zeros(n), 1.0)
        (t.loss / (1 * 128)).backward()
        grads.append(logp.grad.abs().sum().item())
    assert grads[1] / grads[0] == pytest.approx(2.0)
    assert grads[0] == pytest.approx(4 / 128)  # 1/128 a token: no division by the completion's length


def test_the_gradient_through_the_model_grows_with_the_completion(tmp_path):
    learner = make_learner(tmp_path)
    norms = []
    for n in (4, 8):
        learner.optimizer.zero_grad()
        r = row(completion=[5] * n)
        logp, _ = learner.logps(r, grad=True)
        t = terms(logp, logp.detach(), logp.detach(), 1.0)
        (t.loss / 128).backward()
        norms.append(torch.sqrt(sum((p.grad**2).sum() for _, p in learner.params if p.grad is not None)).item())
    assert 1.4 < norms[1] / norms[0] < 2.6


@pytest.mark.parametrize(
    ("advantage", "ratio", "gradient", "clipped"),
    [
        (1.0, 1.5, False, True),  # already above the band: nothing more to gain
        (1.0, 1.1, True, False),
        (1.0, 0.5, True, False),  # below the band with A > 0: pessimistic min keeps the unclipped term
        (-1.0, 0.5, False, True),
        (-1.0, 1.5, True, False),  # above the band with A < 0: the unclipped term is the worse one
        (-1.0, 0.9, True, False),
    ],
)
def test_the_ppo_clip_and_pessimistic_min(advantage, ratio, gradient, clipped):
    logp = torch.full((3,), math.log(ratio), requires_grad=True)
    t = terms(logp, torch.zeros(3), torch.zeros(3), advantage)
    t.loss.backward()
    assert (logp.grad.abs().sum().item() > 0) is gradient
    assert t.clipped == (3 if clipped else 0)


def test_the_clip_band_is_0_8_to_1_28():
    assert terms(torch.tensor([math.log(1.27)]), torch.zeros(1), torch.zeros(1), 1.0).clipped == 0
    assert terms(torch.tensor([math.log(1.29)]), torch.zeros(1), torch.zeros(1), 1.0).clipped == 1
    assert terms(torch.tensor([math.log(0.81)]), torch.zeros(1), torch.zeros(1), -1.0).clipped == 0
    assert terms(torch.tensor([math.log(0.79)]), torch.zeros(1), torch.zeros(1), -1.0).clipped == 1


def test_the_behaviour_weight_is_capped_at_two_and_carries_no_gradient():
    logp = torch.zeros(1, requires_grad=True)
    mu = torch.full((1,), -math.log(5.0), requires_grad=True)  # pi_prox / mu = 5
    t = terms(logp, torch.zeros(1), mu, 1.0)
    assert t.loss.item() == pytest.approx(-2.0) and t.truncated == 1
    t.loss.backward()
    assert mu.grad is None and logp.grad.item() == pytest.approx(-2.0)  # w scales the gradient; it has none itself
    mild = terms(torch.zeros(1), torch.zeros(1), torch.full((1,), -math.log(1.5)), 1.0)
    assert mild.loss.item() == pytest.approx(-1.5) and mild.truncated == 0


def test_masked_tokens_contribute_nothing():
    logp = torch.zeros(4, requires_grad=True)
    t = terms(logp, torch.zeros(4), torch.zeros(4), 1.0, mask=torch.tensor([1.0, 0.0, 0.0, 1.0]))
    t.loss.backward()
    assert t.tokens == 2 and logp.grad.tolist() == [-1.0, 0.0, 0.0, -1.0]


# --- tool calls and say masking ---


def test_the_tokens_of_say_calls_are_masked_and_other_calls_are_not():
    r = L.annotate(row(completion=[5, OPEN, 10, CLOSE, 6, OPEN, 11, CLOSE]), decode, OPEN, CLOSE)
    assert r.mask == [1, 0, 0, 0, 1, 1, 1, 1]
    assert [c.name for c in r.calls] == ["say", "army"]


def test_a_truncated_or_unparseable_say_is_still_found_by_name_and_other_junk_is_not_masked():
    cut = L.annotate(row(completion=[5, OPEN, 13]), decode, OPEN, CLOSE)
    assert cut.mask == [1, 0, 0] and cut.calls[0].name == "say"
    junk = L.annotate(row(completion=[5, OPEN, 15, CLOSE]), decode, OPEN, CLOSE)
    assert junk.mask == [1, 1, 1, 1] and junk.calls[0].name is None


def test_an_explicit_loss_mask_replaces_say_detection():
    r = row(completion=[5, OPEN, 10, CLOSE], loss_mask=[0, 1, 1, 1])
    assert L.annotate(r, decode, OPEN, CLOSE).mask == [0, 1, 1, 1]


# --- intake ---


def drop(inbox: Path, name: str, rows: list[dict] | list[str]) -> None:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    (inbox / f"{name}.tmp").write_text("\n".join(lines))
    (inbox / f"{name}.tmp").rename(inbox / name)


def raw(i=0, **overrides) -> dict:
    base = {"game_id": f"g{i}", "version": 0, "prompt_token_ids": [1, 2], "token_ids": [3, 4, 5],
            "logprobs": [-1.0, -2.0, -3.0], "advantage": 0.5}  # fmt: skip
    return {**base, **overrides}


def test_rows_whose_logprobs_do_not_match_the_tokens_are_rejected_and_counted(tmp_path):
    inbox = L.Inbox(config(tmp_path))
    bad = [raw(1, logprobs=[-1.0]), raw(2, token_ids=[]), raw(3, advantage=float("nan")), "{nope", raw(4, version="x")]
    drop(inbox.cfg.inbox, "a.jsonl", [raw(0), *bad, {"game_id": "x"}, raw(5, loss_mask=[1, 1])])
    (inbox.cfg.inbox / "half.jsonl.tmp").write_text(json.dumps(raw(9)))
    assert inbox.poll() == 1
    counts = inbox.drain_counts()
    assert counts == {
        "accepted": 1, "rejected:logprobs_length": 1, "rejected:empty": 1, "rejected:bad_advantage": 1,
        "rejected:bad_json": 1, "rejected:bad_version": 1, "rejected:missing_field": 1, "rejected:bad_loss_mask": 1,
    }  # fmt: skip
    assert not list(inbox.cfg.inbox.glob("*.jsonl")) and (inbox.cfg.inbox / "consumed" / "a.jsonl").exists()
    assert (inbox.cfg.inbox / "half.jsonl.tmp").exists()


def test_rows_more_than_six_versions_old_are_dropped(tmp_path):
    inbox = L.Inbox(config(tmp_path, decisions_per_update=2))
    inbox.pool = [row(seed=v, version=v) for v in (0, 1, 2, 7, 8)] + [row(seed=99, version=9)]
    batch = inbox.take(current_version=8)
    assert sorted(r.version for r in batch) == [7, 8]
    assert inbox.counts["stale"] == 2 and inbox.counts["rejected:future_version"] == 1  # v0, v1 stale; v9 impossible
    assert [r.version for r in inbox.pool] == [2]  # six versions old is still allowed


def test_the_freshest_rows_go_first_and_a_short_pool_waits(tmp_path):
    inbox = L.Inbox(config(tmp_path, decisions_per_update=2))
    inbox.pool = [row(seed=i, version=v) for i, v in enumerate((3, 5, 4, 5))]
    assert [r.version for r in inbox.take(5)] == [5, 5]
    assert [r.version for r in inbox.pool] == [3, 4]
    assert inbox.take(5) is not None and inbox.take(5) is None


# --- update loop ---


def test_an_update_publishes_a_version_and_logs_its_health(tmp_path):
    learner = make_learner(tmp_path)
    before = weights(learner)
    health = learner.update(rows(), {"accepted": 4})
    assert learner.version == 1 and health["version"] == 1 and health["minibatches_run"] == 2 and health["trained"] == 4
    assert any(not torch.equal(before[n], p) for n, p in learner.params)
    adapter = learner.cfg.adapters / "v1"
    assert sorted(f.name for f in adapter.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "manifest.json",
    ]
    manifest = json.loads((adapter / "manifest.json").read_text())
    assert manifest["version"] == 1 and manifest["update"] == 1 and manifest["decisions"] == 4
    assert manifest["health"]["tokens_p50"] == health["tokens_p50"] and manifest["time"] > 0
    logged = [json.loads(line) for line in learner.cfg.health.read_text().splitlines()]
    assert len(logged) == 1 and logged[0]["grad_norm"] > 0 and logged[0]["kl_to_base"] == pytest.approx(0, abs=1e-9)
    assert logged[0]["entropy"] > 0 and logged[0]["clipped_share"] == 0 and logged[0]["rejected_share"] == 0


def test_kl_to_base_is_zero_on_a_fresh_adapter_and_positive_after_it_has_moved(tmp_path):
    learner = make_learner(tmp_path)
    first = learner.update(rows())
    second = learner.update(rows())
    assert first["kl_to_base"] == pytest.approx(0, abs=1e-9)
    assert second["kl_to_base"] > 1e-6


def test_the_adapter_off_pass_gives_the_plain_base_models_logprobs_and_only_a_tenth_of_rows_are_sampled(tmp_path):
    learner = make_learner(tmp_path, decisions_per_update=20)
    learner.update(rows(20))
    base = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path / "base"), torch_dtype=torch.float32)
    sample = row(seed=7)
    ids = torch.tensor([sample.prompt + sample.completion])
    with torch.no_grad():
        logits = base(ids).logits[0, len(sample.prompt) - 1 : -1].float()
    expected = torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(sample.completion)[:, None])[:, 0]
    off, _ = learner.logps(sample, grad=False, adapter=False)
    on, _ = learner.logps(sample, grad=False)
    assert torch.allclose(off, expected, atol=1e-5) and not torch.allclose(on, expected, atol=1e-5)

    spy = []
    real = learner.logps
    learner.logps = lambda r, **kw: spy.append(kw.get("adapter", True)) or real(r, **kw)
    learner.update(rows(20))
    assert spy.count(False) == 2  # ceil(10% of 20 rows)


def test_the_second_minibatch_is_skipped_when_the_first_step_moved_the_policy_too_far(tmp_path):
    steps = {}
    for name, kl_stop in (("tight", 1e-9), ("loose", 1e9)):
        learner = make_learner(tmp_path / name, kl_stop=kl_stop)
        health = learner.update(rows())
        steps[name] = (
            health["minibatches_run"],
            health["trained"],
            next(iter(learner.optimizer.state.values()))["step"],
        )
        assert health["kl_after_mb"] is not None
    assert steps["tight"][:2] == (1, 2) and float(steps["tight"][2]) == 1
    assert steps["loose"][:2] == (2, 4) and float(steps["loose"][2]) == 2


def test_adamw_moments_and_step_count_carry_across_updates(tmp_path):
    learner = make_learner(tmp_path)
    optimizer = learner.optimizer
    learner.update(rows())
    first = {n: optimizer.state[p]["exp_avg"].clone() for n, p in learner.params}
    assert any(t.abs().sum() > 0 for t in first.values())
    learner.update(rows())
    assert learner.optimizer is optimizer
    states = [optimizer.state[p] for _, p in learner.params]
    assert all(float(s["step"]) == 4 for s in states)  # 2 updates x 2 minibatches, never reset
    assert any(not torch.equal(first[n], optimizer.state[p]["exp_avg"]) for n, p in learner.params)


def test_master_weights_stay_fp32_under_a_bf16_base_and_the_published_adapter_is_bf16(tmp_path):
    from safetensors.torch import load_file

    learner = make_learner(tmp_path, dtype="bfloat16")
    assert next(learner.backbone.parameters()).dtype == torch.bfloat16
    learner.update(rows())
    assert all(p.dtype == torch.float32 for _, p in learner.params)
    saved = load_file(str(learner.cfg.adapters / "v1" / "adapter_model.safetensors"))
    assert saved and all(t.dtype == torch.bfloat16 for t in saved.values())
    base = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path / "base"), torch_dtype=torch.float32)
    loaded = peft.PeftModel.from_pretrained(base, str(learner.cfg.adapters / "v1"))
    for key, value in peft.get_peft_model_state_dict(loaded).items():
        master = peft.get_peft_model_state_dict(learner.model)[key]
        assert torch.equal(value, master.to(torch.bfloat16).float())


def test_an_update_really_checkpoints_the_layers_when_gradient_checkpointing_is_on(tmp_path):
    learner = make_learner(tmp_path, gradient_checkpointing=True)
    calls = []
    for layer in learner.backbone.layers:
        real = layer._gradient_checkpointing_func
        layer._gradient_checkpointing_func = lambda *a, _real=real, **k: calls.append(1) or _real(*a, **k)
    assert learner.backbone.training
    health = learner.update(rows())
    assert health["minibatches_run"] == 2 and health["grad_norm"] > 0
    assert len(calls) >= len(learner.backbone.layers) * 4  # every layer, in every row's forward and backward pass


def test_no_checkpointing_happens_when_it_is_off(tmp_path):
    learner = make_learner(tmp_path, gradient_checkpointing=False)
    assert not any(getattr(layer, "gradient_checkpointing", False) for layer in learner.backbone.layers)


def test_the_lm_head_runs_in_fp32_even_with_a_bf16_model(tmp_path):
    learner = make_learner(tmp_path, dtype="bfloat16")
    assert learner.head_weight.dtype == torch.float32
    logp, ent = learner.logps(row(), grad=False, entropy=True)
    assert logp.dtype == torch.float32 and ent.dtype == torch.float32 and (logp <= 0).all()


def test_the_lr_warms_up_over_ten_updates_then_holds_and_changes_only_through_set_lr(tmp_path):
    learner = make_learner(tmp_path, warmup_updates=4, lr=1e-3, kl_stop=1e9)
    lrs = [learner.update(rows())["lr"] for _ in range(5)]
    assert lrs == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3])
    learner.set_lr(2e-3)
    assert learner.update(rows())["lr"] == pytest.approx(2e-3)


# --- publish ---


def test_publish_is_atomic_no_partial_version_directory_is_ever_visible(tmp_path, monkeypatch):
    learner = make_learner(tmp_path)
    seen = []
    real = L.save_file

    def spying(*args, **kwargs):
        real(*args, **kwargs)
        seen.append(sorted(p.name for p in learner.cfg.adapters.iterdir()))

    monkeypatch.setattr(L, "save_file", spying)
    learner.update(rows())
    assert seen and all(not any(n == "v1" for n in names) for names in seen)  # only a hidden temp dir while writing
    assert any(n.startswith(".v1.") for n in seen[0])
    assert [p.name for p in learner.cfg.adapters.iterdir()] == ["v1"]  # temp dir renamed away, nothing left behind


def test_a_failed_publish_leaves_no_directory_behind(tmp_path, monkeypatch):
    learner = make_learner(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(L, "save_file", boom)
    with pytest.raises(OSError, match="disk full"):
        learner.update(rows())
    assert list(learner.cfg.adapters.iterdir()) == []


# --- checkpoint ring, rollback, resume ---


def run_updates(learner: L.Learner, n: int) -> dict[int, dict]:
    snapshots = {}
    for k in range(n):
        learner.update([row(seed=10 * k + i, advantage=(-1) ** i, version=learner.version) for i in range(4)])
        snapshots[learner.version] = (
            weights(learner),
            {n: learner.optimizer.state[p]["exp_avg"].clone() for n, p in learner.params},
        )
    return snapshots


def test_the_ring_keeps_the_last_checkpoints_and_rollback_restores_one(tmp_path):
    learner = make_learner(tmp_path, ring=3)
    snaps = run_updates(learner, 5)
    assert learner.checkpoint_versions() == [3, 4, 5]
    lr = learner.lr
    record = learner.rollback(3)
    restored_weights, restored_moments = snaps[3]
    assert all(torch.equal(restored_weights[n], p) for n, p in learner.params)
    assert all(torch.equal(restored_moments[n], learner.optimizer.state[p]["exp_avg"]) for n, p in learner.params)
    assert learner.updates == 3 and learner.lr == lr / 2 and record["restored"] == 3
    assert learner.version == 6 and (learner.cfg.adapters / "v6").is_dir()  # restored weights go out as a new version
    assert (learner.cfg.adapters / "v5").is_dir()  # the abandoned versions are never overwritten
    learner.rollback(4)
    assert learner.lr == lr / 4 and learner.version == 7
    with pytest.raises(KeyError, match="ring holds"):
        learner.rollback(1)


def test_training_continues_from_a_rollback(tmp_path):
    learner = make_learner(tmp_path, ring=5)
    run_updates(learner, 3)
    learner.rollback(2)
    health = learner.update(rows(version=learner.version))
    assert health["version"] == 5 and health["update"] == 3 and (learner.cfg.adapters / "v5").is_dir()


def test_a_restart_resumes_from_the_newest_checkpoint(tmp_path):
    learner = make_learner(tmp_path)
    run_updates(learner, 3)
    learner.set_lr(5e-3)
    learner.save_checkpoint()
    snap = weights(learner)
    (learner.cfg.adapters / "v3").rename(tmp_path / "lost")  # crashed between checkpoint and publish
    fresh = make_learner(tmp_path)
    assert not any(torch.equal(snap[n], p) for n, p in fresh.params)
    assert fresh.resume() and fresh.version == 3 and fresh.updates == 3 and fresh.lr == 5e-3
    assert all(torch.equal(snap[n], p) for n, p in fresh.params)
    assert (fresh.cfg.adapters / "v3").is_dir()
    assert fresh.update(rows(version=3))["version"] == 4
    assert make_learner(tmp_path / "empty").resume() is False


# --- health numbers ---


def annotated(*rows_: L.Decision) -> list[L.Decision]:
    return [L.annotate(r, decode, OPEN, CLOSE) for r in rows_]


def test_shape_statistics_skip_hold_rows_and_tool_names_are_checked(tmp_path):
    batch = annotated(
        row(1, version=5, advantage=1.0, completion=[5, 6, OPEN, 11, CLOSE, OPEN, 10, CLOSE]),
        row(2, version=5, advantage=-1.0, completion=[5, 6, OPEN, 11, CLOSE, OPEN, 10, CLOSE]),
        row(3, version=3, advantage=3.0, completion=[7] * 4 + [OPEN, 14, CLOSE]),
        row(4, version=5, advantage=0.0, completion=[OPEN, 12, CLOSE], hold=True),
        row(5, version=5, advantage=0.0, completion=[OPEN, 12, CLOSE], hold=True),
    )
    s = L.row_stats(batch, current_version=5, known_tools=("army", "hold", "say"))
    assert s["hold_share"] == pytest.approx(0.4)
    assert s["tokens_p50"] == 8 and s["tokens_p10"] == pytest.approx(7.2) and s["tokens_p90"] == pytest.approx(8.0)
    assert s["calls_per_decision"] == pytest.approx((2 + 2 + 1) / 3)
    assert s["exact_share"] == pytest.approx(2 / 3) and s["distinct_fraction"] == pytest.approx(2 / 3)
    assert s["invented_tool_share"] == pytest.approx(1 / 5)  # teleport
    assert s["adv_mean"] == pytest.approx(0.6) and s["adv_sd"] == pytest.approx(math.sqrt(1.84))
    assert s["staleness"] == {"0": 4, "2": 1}


def test_health_of_an_all_hold_batch_has_no_shape_numbers():
    s = L.row_stats(annotated(row(1, hold=True), row(2, hold=True)), 0, ())
    assert s["hold_share"] == 1 and s["tokens_p50"] is None and s["exact_share"] is None
    assert s["invented_tool_share"] is None


def test_an_update_reports_the_batch_staleness_and_the_rejected_share(tmp_path):
    learner = make_learner(tmp_path)
    learner.version = 3
    batch = [row(i, version=v, advantage=(-1) ** i) for i, v in enumerate((3, 3, 2, 1))]
    health = learner.update(batch, {"accepted": 6, "rejected:logprobs_length": 2, "stale": 5})
    assert health["staleness"] == {"0": 2, "1": 1, "2": 1}
    assert health["rejected_share"] == pytest.approx(2 / 8) and health["stale_dropped"] == 5


# --- control ---


def test_control_commands_run_oldest_first_and_a_bad_one_is_not_fatal(tmp_path, caplog):
    learner = make_learner(tmp_path)
    run_updates(learner, 2)
    control = learner.cfg.control
    control.mkdir(parents=True, exist_ok=True)
    (control / "001.json").write_text(json.dumps({"op": "set_lr", "lr": 3e-3}))
    (control / "002.json").write_text("{broken")
    (control / "003.json").write_text(json.dumps({"op": "rollback", "version": 99}))
    (control / "004.json").write_text(json.dumps({"op": "rollback", "version": 1}))
    commands = []
    while (c := learner.pop_command()) is not None:
        commands.append(c)
        learner.apply(c)
    assert [c["op"] for c in commands] == ["set_lr", "rollback", "rollback"]
    assert learner.version == 3 and learner.lr == pytest.approx(1.5e-3)  # set to 3e-3, halved once by the good rollback
    assert sorted(p.name for p in (control / "done").iterdir()) == ["001.json", "002.json", "003.json", "004.json"]


def test_serve_trains_on_inbox_files_until_told_to_stop(tmp_path):
    learner = make_learner(tmp_path)
    inbox = L.Inbox(learner.cfg)
    drop(learner.cfg.inbox, "a.jsonl", [raw(i, advantage=(-1) ** i) for i in range(4)])
    drop(learner.cfg.inbox, "b.jsonl", [raw(i, advantage=(-1) ** i) for i in range(5)])
    L.serve(learner, inbox, poll=0, max_updates=1)
    assert learner.version == 1 and len(inbox.pool) == 5
    (learner.cfg.control).mkdir(parents=True, exist_ok=True)
    (learner.cfg.control / "1.json").write_text(json.dumps({"op": "stop"}))
    L.serve(learner, inbox, poll=0)
    assert learner.version == 1
    health = json.loads(learner.cfg.health.read_text().splitlines()[0])
    assert health["intake"]["accepted"] == 9


# --- two ranks ---


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rank_worker(rank: int, world: int, tmp: str, port: int) -> None:
    loopback = "lo0" if sys.platform == "darwin" else "lo"  # gloo otherwise resolves the hostname, which can hang
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME=loopback)
    torch.set_num_threads(1)
    comm = L.Comm(rank, world, backend="gloo")
    learner = make_learner(Path(tmp) / f"r{world}", comm=comm)
    comm.sync_params([p for _, p in learner.params])
    learner.update(rows())
    if rank == 0:
        torch.save(weights(learner), Path(tmp) / f"w{world}.pt")
    comm.barrier()
    comm.close()


def test_two_ranks_reach_the_same_weights_as_one(tmp_path):
    for name in ("r1", "r2"):
        tiny_model(tmp_path / name / "base")
    rank_worker(0, 1, str(tmp_path), 0)
    torch.multiprocessing.spawn(rank_worker, args=(2, str(tmp_path), free_port()), nprocs=2, join=True)
    one, two = torch.load(tmp_path / "w1.pt"), torch.load(tmp_path / "w2.pt")
    assert one.keys() == two.keys() and any(t.abs().sum() > 0 for t in one.values())
    for n in one:
        assert torch.allclose(one[n], two[n], atol=1e-6), n
    assert (tmp_path / "r2" / "run" / "adapters" / "v1").is_dir()


# --- tripwires ---

BASELINE = {
    "tokens_p50": {"mean": 100.0, "sd": 5.0},
    "calls_per_decision": {"mean": 3.3, "sd": 0.2},
    "exact_share": {"mean": 0.02, "sd": 0.005},
    "distinct_fraction": {"mean": 0.95, "sd": 0.02},
    "entropy": {"mean": 0.8, "sd": 0.05},
    "rejected_share": {"mean": 0.01, "sd": 0.01},
}


def healthy(**overrides) -> dict:
    record = {
        "tokens_p50": 100.0, "calls_per_decision": 3.3, "exact_share": 0.02, "distinct_fraction": 0.95, "entropy": 0.8,
        "hold_share": 0.3, "invented_tool_share": 0.0, "rejected_share": 0.01, "clipped_share": 0.02, "grad_norm": 1.0,
        "kl_to_base": 0.01, "tis_truncation_share": 0.01,
    }  # fmt: skip
    return {**record, **overrides}


def names(result: dict, kind: str) -> list[str]:
    return sorted(t["name"] for t in result[kind])


def test_a_healthy_window_trips_nothing():
    assert L.evaluate_tripwires([healthy()] * 10, BASELINE) == {"trips": [], "alarms": []}


@pytest.mark.parametrize(
    ("metric", "bad", "name"),
    [
        ("tokens_p50", 50.0, "length"),
        ("tokens_p50", 160.0, "length"),
        ("calls_per_decision", 1.5, "calls"),
        ("exact_share", 0.15, "exact_share"),
        ("distinct_fraction", 0.5, "distinct_fraction"),
        ("entropy", 0.2, "entropy"),
        ("entropy", 1.5, "entropy"),
    ],
)
def test_a_window_mean_more_than_four_sd_from_the_baseline_trips_that_band(metric, bad, name):
    assert names(L.evaluate_tripwires([healthy(**{metric: bad})] * 10, BASELINE), "trips") == [name]


@pytest.mark.parametrize("flags", [[], ["--known-tools", ""], ["--known-tools", ","]])
def test_the_cli_refuses_to_start_without_known_tool_names(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["learner.py", "--base", "b", "--root", "r", *flags])
    with pytest.raises(SystemExit):
        L.main()


def test_a_zero_variance_baseline_does_not_trip_on_one_duplicate_completion():
    flat = {
        **BASELINE,
        "distinct_fraction": {"mean": 1.0, "sd": 0.0},
        "exact_share": {"mean": 0.004, "sd": 0.0},
        "entropy": {"mean": 0.8, "sd": 0.0},
    }
    one_duplicate = healthy(distinct_fraction=0.996, exact_share=0.008, entropy=0.81)
    assert L.evaluate_tripwires([one_duplicate] * 10, flat)["trips"] == []
    collapsed = healthy(distinct_fraction=0.5, exact_share=0.3, entropy=0.2)
    assert names(L.evaluate_tripwires([collapsed] * 10, flat), "trips") == [
        "distinct_fraction",
        "entropy",
        "exact_share",
    ]
    assert names(L.evaluate_tripwires([one_duplicate] * 10, flat, {"band_floor": {}}), "trips") != []


def test_a_shift_inside_four_sd_is_not_a_trip_and_the_band_uses_the_window_mean():
    assert L.evaluate_tripwires([healthy(tokens_p50=118.0)] * 10, BASELINE)["trips"] == []
    mixed = [healthy(tokens_p50=40.0)] + [healthy()] * 9  # one bad update among ten: mean 94, inside 80..120
    assert L.evaluate_tripwires(mixed, BASELINE)["trips"] == []
    old_then_new = [healthy(tokens_p50=40.0)] * 10 + [healthy()] * 10  # only the last ten count
    assert L.evaluate_tripwires(old_then_new, BASELINE)["trips"] == []


def test_nothing_is_judged_before_enough_updates_have_arrived():
    assert L.evaluate_tripwires([healthy(tokens_p50=1.0, hold_share=0.99)] * 4, BASELINE) == {"trips": [], "alarms": []}
    assert L.evaluate_tripwires([healthy(tokens_p50=1.0)] * 4, BASELINE, {"min_records": 3})["trips"]


def test_hold_share_alarms_above_sixty_percent_and_trips_above_eighty():
    assert L.evaluate_tripwires([healthy(hold_share=0.5)] * 10, BASELINE) == {"trips": [], "alarms": []}
    mid = L.evaluate_tripwires([healthy(hold_share=0.7)] * 10, BASELINE)
    assert names(mid, "alarms") == ["hold_share"] and mid["trips"] == []
    high = L.evaluate_tripwires([healthy(hold_share=0.85)] * 10, BASELINE)
    assert names(high, "trips") == ["hold_share"] and high["alarms"] == []


def test_invented_tool_names_trip_above_half_a_percent():
    assert L.evaluate_tripwires([healthy(invented_tool_share=0.004)] * 10, BASELINE)["trips"] == []
    assert names(L.evaluate_tripwires([healthy(invented_tool_share=0.006)] * 10, BASELINE), "trips") == [
        "invented_tools"
    ]


@pytest.mark.parametrize(
    ("baseline_mean", "share", "trips"),
    [(0.01, 0.025, False), (0.01, 0.04, True), (0.05, 0.09, False), (0.05, 0.11, True), (0.0, 0.031, True)],
)
def test_rejected_share_trips_above_the_larger_of_twice_baseline_and_three_percent(baseline_mean, share, trips):
    baseline = {**BASELINE, "rejected_share": {"mean": baseline_mean, "sd": 0.01}}
    result = L.evaluate_tripwires([healthy(rejected_share=share)] * 10, baseline)
    assert bool(result["trips"]) is trips


def test_kl_to_base_and_tis_truncation_are_alarms_never_trips():
    result = L.evaluate_tripwires([healthy(kl_to_base=0.3, tis_truncation_share=0.2)] * 10, BASELINE)
    assert result["trips"] == [] and names(result, "alarms") == ["kl_to_base", "tis_truncation"]


def test_a_clipped_share_spike_and_a_gradient_norm_spike_trip():
    window = [healthy()] * 9 + [healthy(clipped_share=0.4, grad_norm=9.0)]
    assert names(L.evaluate_tripwires(window, BASELINE), "trips") == ["clipped_share", "grad_norm"]


def test_missing_metrics_are_skipped_not_judged():
    record = healthy(tokens_p50=None, calls_per_decision=None, exact_share=None, distinct_fraction=None, entropy=None)
    assert L.evaluate_tripwires([record] * 10, BASELINE)["trips"] == []
    assert L.evaluate_tripwires([healthy(tokens_p50=1.0)] * 10, {})["trips"] == []  # no baseline row, no band


def test_a_baseline_comes_from_reference_records_as_the_sd_of_window_means():
    rng = random.Random(0)
    records = [{"tokens_p50": rng.gauss(100, 10), "rejected_share": 0.01} for _ in range(100)]
    baseline = L.baseline_from(records, window=10)
    assert baseline["tokens_p50"]["mean"] == pytest.approx(100, abs=3)
    assert baseline["tokens_p50"]["sd"] == pytest.approx(10 / math.sqrt(10), abs=1.2)
    short = L.baseline_from(records[:15], window=10)  # under two windows: per-record SD / sqrt(window)
    assert short["tokens_p50"]["sd"] == pytest.approx(10 / math.sqrt(10), abs=1.5)
    assert "entropy" not in baseline


# --- tripwires in the health record, and the window across rollbacks and restarts ---


def test_every_update_writes_its_trips_and_alarms_into_the_health_record(tmp_path):
    learner = make_learner(tmp_path)
    learner.update(rows())
    learner.update(rows())
    logged = [json.loads(line) for line in learner.cfg.health.read_text().splitlines()]
    assert [r["tripwires"] for r in logged] == [{"trips": [], "alarms": []}] * 2  # too few records to judge yet
    manifest = json.loads((learner.cfg.adapters / "v2" / "manifest.json").read_text())
    assert manifest["health"]["tripwires"] == {"trips": [], "alarms": []}


def test_the_hold_trip_needs_the_window_to_be_mostly_hold(tmp_path):
    learner = make_learner(tmp_path)
    learner.tripwire_config = {"min_records": 2}
    holding = [row(seed=i, advantage=(-1) ** i, hold=True) for i in range(4)]
    learner.update(holding)
    health = learner.update(holding)
    assert names(health["tripwires"], "trips") == ["hold_share"] and health["tripwires"]["trips"][0]["value"] == 1.0


def test_a_baseline_file_sets_the_bands_the_update_is_judged_against(tmp_path):
    reference = [healthy(tokens_p50=50.0 + i % 3, entropy=0.8) for i in range(30)]
    learner = make_learner(tmp_path, ring=5)
    learner.baseline = L.baseline_from(reference)
    learner.tripwire_config = {"min_records": 1}
    health = learner.update(rows())  # completions of 3 to 6 tokens, a baseline median of 50
    assert "length" in names(health["tripwires"], "trips")
    json.dumps(health)  # the record stays plain JSON


def test_a_rollback_empties_the_tripwire_window_and_a_restart_refills_it_from_the_last_rollback(tmp_path):
    learner = make_learner(tmp_path, ring=5)
    run_updates(learner, 3)
    assert len(learner.recent) == 3
    learner.rollback(2)
    assert len(learner.recent) == 0
    run_updates(learner, 2)
    fresh = make_learner(tmp_path, ring=5)
    assert fresh.resume() and len(fresh.recent) == 2  # the two updates since the rollback, not the three before it
    assert [r["version"] for r in fresh.recent] == [5, 6]


# --- control files ---


def test_a_command_still_being_written_is_not_read_and_files_run_in_name_order(tmp_path):
    learner = make_learner(tmp_path)
    learner.cfg.control.mkdir(parents=True, exist_ok=True)
    (learner.cfg.control / ".000002-set_lr.json.tmp").write_text('{"op": "set_lr"')
    (learner.cfg.control / "000002-set_lr.json").write_text(json.dumps({"op": "set_lr", "lr": 2e-3}))
    (learner.cfg.control / "000001-set_lr.json").write_text(json.dumps({"op": "set_lr", "lr": 1e-3}))
    assert [learner.pop_command()["lr"] for _ in range(2)] == [1e-3, 2e-3]
    assert learner.pop_command() is None
    assert (learner.cfg.control / ".000002-set_lr.json.tmp").exists()


def test_the_documented_commands_are_the_ones_apply_runs():
    for op in ("rollback", "set_lr", "stop"):
        assert f'"op": "{op}"' in L.__doc__


# --- SFT mode ---


def sft_raw(i=0, **overrides) -> dict:
    return {"game_id": f"t{i}", "prompt_token_ids": [1, 2, 3], "token_ids": [30 + i % 3, 31, 32, 33], **overrides}


def sft_rows(n=4) -> list[L.Decision]:
    return [L.parse_row(sft_raw(i), 100, sft=True) for i in range(n)]


def test_an_sft_row_needs_only_the_ids_and_an_rl_row_still_needs_the_rest():
    parsed = L.parse_row(sft_raw(), 100, sft=True)
    assert parsed.logprobs == [] and parsed.advantage == 0.0 and parsed.version == 0
    with pytest.raises(L.RowRejected) as e:
        L.parse_row(sft_raw(), 100)
    assert e.value.reason == "missing_field"
    with pytest.raises(L.RowRejected) as e:
        L.parse_row(sft_raw(token_ids=[]), 100, sft=True)
    assert e.value.reason == "empty"
    with pytest.raises(L.RowRejected) as e:
        L.parse_row(sft_raw(loss_mask=[1.0]), 100, sft=True)
    assert e.value.reason == "bad_loss_mask"


def test_the_sft_loss_is_cross_entropy_summed_over_unmasked_tokens_with_no_length_normalisation():
    logp = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    t = L.sft_terms(logp, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert t.loss.item() == pytest.approx(6.0) and t.tokens == 3
    t.loss.backward()
    assert logp.grad.tolist() == [-1.0, -1.0, -1.0, 0.0]


def test_sft_training_lowers_the_loss_on_its_rows_writes_its_own_adapter_and_has_no_rl_machinery(tmp_path):
    learner = make_learner(tmp_path, sft=True, ring=5, lr=5e-2)
    before = [float(learner.logps(r, grad=False)[0].sum()) for r in sft_rows()]
    healths = [learner.update(sft_rows(), {"accepted": 4}) for _ in range(4)]
    after = [float(learner.logps(r, grad=False)[0].sum()) for r in sft_rows()]
    assert sum(after) > sum(before)
    assert healths[0]["loss"] > healths[-1]["loss"] > 0
    assert healths[0]["tripwires"] == {"trips": [], "alarms": []} and healths[0]["kl_to_base"] is None
    assert healths[0]["clipped_share"] == 0 and healths[0]["kl_after_mb"] is None and healths[0]["minibatches_run"] == 2
    manifest = json.loads((learner.cfg.adapters / "v4" / "manifest.json").read_text())
    assert manifest["mode"] == "sft"


def test_sft_rows_are_never_stale_whatever_the_version_counter_says(tmp_path):
    inbox = L.Inbox(config(tmp_path, sft=True, decisions_per_update=2))
    inbox.pool = sft_rows(3)
    assert len(inbox.take(current_version=50)) == 2 and inbox.counts == {}


def test_an_sft_remainder_smaller_than_a_batch_is_taken_once_the_inbox_is_quiet(tmp_path):
    inbox = L.Inbox(config(tmp_path, sft=True, decisions_per_update=2))
    inbox.pool = sft_rows(3)
    assert len(inbox.take(0)) == 2 and inbox.take(0) is None
    assert len(inbox.take(0, flush=True)) == 1 and inbox.take(0, flush=True) is None
    rl = L.Inbox(config(tmp_path, decisions_per_update=4))
    assert rl.take(0, flush=True) is None


def test_a_checkpoint_refuses_to_resume_in_the_other_mode(tmp_path):
    learner = make_learner(tmp_path, sft=True)
    learner.update(sft_rows())
    with pytest.raises(ValueError, match="sft run"):
        make_learner(tmp_path).resume()
    assert make_learner(tmp_path, sft=True).resume()


def test_the_cli_needs_no_tool_names_for_sft(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["learner.py", "--base", "b", "--root", "r", "--sft"])
    monkeypatch.setattr(L, "build_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached the model")))
    with pytest.raises(RuntimeError, match="reached the model"):
        L.main()


# --- rows from the other modules ---


def test_credit_rows_parse_as_rl_rows_and_distill_rows_as_sft_rows():
    from agentenv_zeroad.train import distill
    from agentenv_zeroad.train.credit import Credit
    from tests.train_fakes import results, turns_every

    rl = Credit().rows(results("g", turns={"1": turns_every(25, tools=("hold",))}))
    parsed = [L.parse_row(json.loads(json.dumps(r)), 8192) for r in rl]
    assert parsed and all(p.is_hold and p.version == 3 and len(p.logprobs) == len(p.completion) for p in parsed)
    brief = distill.Brief("g", "1", 0, "b", (1, 2, 3), "other")
    sft = distill.sft_rows([distill.Label(brief, [("hold", {})])], types_tokenizer(), "opus")
    assert L.parse_row(sft[0], 8192, sft=True).completion == [ord(c) for c in distill.completion_text([("hold", {})])]
    with pytest.raises(L.RowRejected):
        L.parse_row(sft[0], 8192)


def types_tokenizer():
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]

    return Tokenizer()
