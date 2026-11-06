"""The learner's KL anchor, its auxiliary cross-entropy on sft-flagged rows and its SFT epochs, on a tiny random Qwen3.

Run with: uv run --with torch --with transformers --with peft pytest tests/test_learner_anchor.py -q
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import learner as L  # noqa: E402

from tests.test_learner import config, drop, make_learner, row, rows, sft_raw, sft_rows, weights  # noqa: E402


def aux_raw(i=0, **overrides) -> dict:
    return {**sft_raw(i), "sft": True, **overrides}


def aux_rows(n=3) -> list[L.Decision]:
    return [L.parse_row(aux_raw(i), 100) for i in range(n)]


def trained_adapter(tmp_path: Path, updates: int = 2) -> Path:
    """A distilled-adapter stand-in: a learner that has moved off the base for a few updates, its last version."""
    learner = make_learner(tmp_path, root=tmp_path / "sft")
    for _ in range(updates):
        learner.update(rows())
    return learner.cfg.adapters / f"v{updates}"


def ce(learner: L.Learner, batch: list[L.Decision]) -> float:
    return sum(float(learner.logps(L.annotate(r, learner.decode, 100, 101), grad=False)[0].sum()) for r in batch)


# --- the anchor ---


def test_the_k3_estimate_is_zero_and_flat_at_the_anchor_and_positive_away_from_it():
    logp = torch.tensor([-1.0, -2.0], requires_grad=True)
    mask = torch.ones(2)
    at = L.anchor_k3(logp, logp.detach().clone(), mask)
    at.backward()
    assert at.item() == 0 and logp.grad.abs().sum().item() == 0
    away = torch.tensor([-1.5, -1.0], requires_grad=True)
    moved = L.anchor_k3(away, torch.tensor([-1.0, -2.0]), mask)
    moved.backward()
    assert moved.item() > 0 and away.grad.abs().sum().item() > 0
    assert L.anchor_k3(away, torch.tensor([-1.0, -2.0]), torch.zeros(2)).item() == 0  # masked tokens add nothing


def test_the_kl_to_the_anchor_adapter_is_zero_when_the_policy_equals_it_and_positive_after_an_update(tmp_path):
    anchor = trained_adapter(tmp_path)
    learner = make_learner(
        tmp_path, root=tmp_path / "rl", init=str(anchor), anchor=str(anchor), anchor_beta0=0.1, lr=5e-2
    )
    first = learner.update(rows())
    second = learner.update(rows())
    assert first["kl_to_anchor"] == pytest.approx(0, abs=1e-8)
    assert second["kl_to_anchor"] > 1e-6
    assert first["kl_to_base"] > 1e-6  # the anchor is not the base here: both numbers are kept, apart


def test_without_an_anchor_adapter_the_anchor_is_the_base_model(tmp_path):
    learner = make_learner(tmp_path, anchor_beta0=0.1, lr=5e-2, base_kl_fraction=1.0)
    assert learner.anchor_weights == {}
    first, second = learner.update(rows()), learner.update(rows())
    assert first["kl_to_anchor"] == pytest.approx(0, abs=1e-8) and second["kl_to_anchor"] > 1e-6
    assert second["kl_to_anchor"] == pytest.approx(second["kl_to_base"], rel=1e-6)  # the same rows, the same formula


def test_the_anchor_adds_a_frozen_copy_that_is_never_trained_or_published_and_the_swap_leaves_the_policy_alone(
    tmp_path,
):
    anchor = trained_adapter(tmp_path)
    learner = make_learner(tmp_path, root=tmp_path / "rl", anchor=str(anchor), anchor_beta0=0.1, lr=5e-2)
    assert set(learner.anchor_weights) == {n for n, _ in learner.params}
    held = {n: w.clone() for n, w in learner.anchor_weights.items()}
    before = weights(learner)
    learner.anchor_logps(row())
    assert all(torch.equal(before[n], p) for n, p in learner.params)
    learner.update(rows())
    learner.update(rows())
    assert all(torch.equal(held[n], w) for n, w in learner.anchor_weights.items())
    published = json.loads((learner.cfg.adapters / "v2" / "manifest.json").read_text())
    assert published["health"]["anchor_beta"] > 0
    assert sorted(p.name for p in (learner.cfg.adapters / "v2").iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "manifest.json",
    ]


def test_a_strong_anchor_keeps_the_policy_nearer_the_anchor_than_a_negligible_one(tmp_path):
    anchor = trained_adapter(tmp_path)
    final = {}
    for beta0 in (1e-9, 20.0):
        learner = make_learner(
            tmp_path, root=tmp_path / f"rl{beta0}", init=str(anchor), anchor=str(anchor), anchor_beta0=beta0, lr=5e-2
        )
        final[beta0] = [learner.update(rows(8))["kl_to_anchor"] for _ in range(4)][-1]
    assert final[20.0] < 0.5 * final[1e-9]


def test_beta_decays_per_update_and_zero_turns_the_anchor_off(tmp_path):
    learner = make_learner(tmp_path, anchor_beta0=0.1, anchor_decay=0.5)
    betas = [learner.update(rows())["anchor_beta"] for _ in range(3)]
    assert betas == pytest.approx([0.1, 0.05, 0.025])
    off = make_learner(tmp_path, root=tmp_path / "off", anchor_beta0=0.0).update(rows())
    assert off["anchor_beta"] == 0 and off["kl_to_anchor"] is None
    sft = make_learner(tmp_path, root=tmp_path / "sft2", sft=True, anchor_beta0=0.1)
    assert sft.anchor_beta() == 0 and not sft.anchor_enabled


def test_a_rollback_and_a_restart_restore_the_anchor_schedule(tmp_path):
    learner = make_learner(tmp_path, anchor_beta0=0.1, anchor_decay=0.5)
    for _ in range(3):
        learner.update(rows())
    assert learner.anchor_beta() == pytest.approx(0.1 * 0.5**3)
    learner.rollback(1)
    assert learner.updates == 1 and learner.anchor_beta() == pytest.approx(0.05)
    assert learner.update(rows())["anchor_beta"] == pytest.approx(0.05)
    restarted = make_learner(tmp_path, anchor_beta0=0.1, anchor_decay=0.5)
    assert restarted.resume() and restarted.anchor_beta() == pytest.approx(0.1 * 0.5**restarted.updates)


def test_an_anchor_adapter_that_does_not_fit_the_learners_lora_is_refused(tmp_path):
    anchor = trained_adapter(tmp_path)
    with pytest.raises(ValueError, match="does not fit"):
        make_learner(tmp_path, root=tmp_path / "rl", anchor=str(anchor), anchor_beta0=0.1, rank=2, alpha=4)


# --- sft-flagged rows ---


def test_an_sft_flagged_row_needs_only_the_ids_in_an_rl_run_and_an_unflagged_one_still_needs_the_rest():
    parsed = L.parse_row(aux_raw(), 100)
    assert parsed.aux and parsed.logprobs == [] and parsed.advantage == 0.0
    for flag in (False, None, "yes", 1):
        with pytest.raises(L.RowRejected):
            L.parse_row({**sft_raw(), "sft": flag}, 100)
    with pytest.raises(L.RowRejected) as e:
        L.parse_row(aux_raw(loss_mask=[1.0]), 100)
    assert e.value.reason == "bad_loss_mask"


def test_sft_flagged_rows_wait_apart_from_the_pg_pool_and_are_dropped_when_the_term_is_off(tmp_path):
    inbox = L.Inbox(config(tmp_path, aux_rows_per_update=2))
    pg = {
        "game_id": "g",
        "version": 0,
        "prompt_token_ids": [1, 2],
        "token_ids": [3, 4],
        "logprobs": [-1, -1],
        "advantage": 1,
    }
    drop(inbox.cfg.inbox, "a.jsonl", [pg, aux_raw(0), aux_raw(1), aux_raw(2)])
    assert inbox.poll() == 1
    assert len(inbox.pool) == 1 and len(inbox.aux) == 3
    assert inbox.counts == {"accepted": 1, "aux_accepted": 3}
    assert [r.game_id for r in inbox.take_aux()] == ["t0", "t1"] and [r.game_id for r in inbox.take_aux()] == ["t2"]
    off = L.Inbox(config(tmp_path, root=tmp_path / "off", aux_alpha0=0.0))
    drop(off.cfg.inbox, "a.jsonl", [aux_raw(0)])
    assert off.poll() == 0 and not off.aux and off.counts == {"aux_dropped": 1}


def test_sft_rows_do_not_enter_the_pg_term_and_only_the_auxiliary_term_moves_the_policy_on_zero_advantage(tmp_path):
    learner = make_learner(tmp_path, aux_alpha0=1.0, lr=5e-2)
    flat = [row(seed=i, advantage=0.0) for i in range(4)]
    targets = aux_rows(3)
    before = ce(learner, targets)
    healths = [learner.update(flat, aux=targets) for _ in range(5)]
    assert ce(learner, targets) > before  # the log-likelihood of the teacher's tokens went up
    assert all(h["decisions"] == 4 and h["trained"] == 4 and h["aux_rows"] == 3 for h in healths)
    assert healths[0]["adv_mean"] == 0 and healths[0]["aux_ce"] > healths[-1]["aux_ce"] > 0
    plain = make_learner(tmp_path, root=tmp_path / "plain", aux_alpha0=1.0, lr=5e-2)
    start = ce(plain, targets)
    for _ in range(5):
        plain.update(flat)
    assert ce(plain, targets) == pytest.approx(start, abs=1e-5)  # zero advantage and no aux rows: nothing moves


def test_alpha_decays_per_update_and_zero_ignores_the_rows_it_is_given(tmp_path):
    learner = make_learner(tmp_path, aux_alpha0=0.2, aux_decay=0.5)
    alphas = [learner.update(rows(), aux=aux_rows())["aux_alpha"] for _ in range(3)]
    assert alphas == pytest.approx([0.2, 0.1, 0.05])
    off = make_learner(tmp_path, root=tmp_path / "off", aux_alpha0=0.0).update(rows(), aux=aux_rows())
    assert off["aux_alpha"] == 0 and off["aux_rows"] == 0 and off["aux_ce"] is None
    none = make_learner(tmp_path, root=tmp_path / "none").update(rows())
    assert none["aux_rows"] == 0 and none["aux_ce"] is None


def test_the_auxiliary_term_uses_the_constant_divisor_and_the_masks_of_the_pg_term(tmp_path):
    learner = make_learner(tmp_path, aux_alpha0=0.5)
    pg = [L.annotate(r, learner.decode, 100, 101) for r in rows(2)]
    short = L.annotate(L.parse_row(aux_raw(0, token_ids=[30, 31]), 100), learner.decode, 100, 101)
    long = L.annotate(
        L.parse_row(aux_raw(1, token_ids=[30, 31, 32, 33], loss_mask=[1, 1, 0, 1]), 100), learner.decode, 100, 101
    )
    prox, _ = learner._prox_pass(pg, [pg], Counter())
    totals = Counter()
    expected = 0.0
    for r in (short, long):
        logp = learner.logps(r, grad=False)[0]
        expected += float(-(torch.tensor(r.mask) * logp).sum())
    learner._train_minibatch(pg, 2, prox, totals, {}, [short, long], L.Schedule(0.0, 0.5))
    assert totals["aux_ce_sum"] == pytest.approx(expected, rel=1e-5)
    assert totals["aux_loss"] == pytest.approx(0.5 * expected / (2 * 128), rel=1e-5)  # alpha, B = 2 and the divisor
    assert totals["aux_tokens"] == 2 + 3  # the masked token is not trained


# --- SFT epochs ---


def run_sft(tmp_path: Path, n_rows: int, **overrides) -> L.Learner:
    # sft_holdout 0.3 holds out t1 of t0..t6 (a hash of the game_id): 7 rows are 6 to train on and 1 for the entropy
    cfg_kwargs = {
        "sft": True, "lr": 5e-2, "sft_epochs": 2, "sft_entropy_rows": 4, "ring": 20, "sft_holdout": 0.3, **overrides
    }  # fmt: skip
    learner = make_learner(tmp_path, **cfg_kwargs)
    inbox = L.Inbox(learner.cfg)
    drop(inbox.cfg.inbox, "sft.jsonl", [sft_raw(i) for i in range(n_rows)])
    ticks = {"n": 0}

    def stop_when() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 40

    L.serve(learner, inbox, poll=0, stop_when=stop_when)
    return learner


def test_sft_trains_epochs_over_the_dataset_and_publishes_a_checkpoint_after_each(tmp_path):
    learner = run_sft(tmp_path, 7)  # 6 to train on: a batch of 4, then the quiet remainder of 2, an epoch
    manifests = [json.loads((learner.cfg.epochs / f"epoch{e}" / "manifest.json").read_text()) for e in (1, 2)]
    assert [m["epoch"] for m in manifests] == [1, 2] and [m["version"] for m in manifests] == [2, 4]
    assert learner.version == 4 and learner.decisions_total == 12  # 2 x 6 rows: nothing more after the last epoch
    assert not (learner.cfg.epochs / "epoch3").exists()
    for m in manifests:
        assert m["mode"] == "sft" and m["entropy"] > 0 and m["base_entropy"] > 0 and m["entropy_rows"] == 1
        assert m["entropy_ratio"] == pytest.approx(m["entropy"] / m["base_entropy"])
        assert sorted(p.name for p in (learner.cfg.epochs / f"epoch{m['epoch']}").iterdir()) == [
            "adapter_config.json", "adapter_model.safetensors", "manifest.json",
        ]  # fmt: skip
    assert all(m["entropy_ratio"] < 1 for m in manifests)  # fitting the teacher's tokens sharpens the policy


def test_one_sft_epoch_stops_after_one_checkpoint_and_the_epoch_count_is_configurable(tmp_path):
    learner = run_sft(tmp_path, 7, sft_epochs=1)
    assert (learner.cfg.epochs / "epoch1").exists() and not (learner.cfg.epochs / "epoch2").exists()
    assert learner.version == 2 and learner.decisions_total == 6


def test_each_epoch_reshuffles_the_dataset_and_the_entropy_sample_is_fixed(tmp_path):
    inbox = L.Inbox(config(tmp_path, sft=True, sft_epochs=3, sft_entropy_rows=3, decisions_per_update=2, seed=3))
    inbox.dataset = inbox.pool = sft_rows(6)
    inbox.holdout = sft_rows(6)
    first = [r.game_id for r in inbox.pool]
    while inbox.take(0, flush=True) is not None:
        pass
    assert inbox.epoch_done()
    sample = [r.game_id for r in inbox.entropy_sample()]
    inbox.next_epoch()
    second = [r.game_id for r in inbox.pool]
    assert sorted(second) == sorted(first) and second != first and inbox.epoch == 2 and not inbox.epoch_done()
    assert [r.game_id for r in inbox.entropy_sample()] == sample and len(sample) == 3
    while inbox.take(0, flush=True) is not None:
        pass
    inbox.next_epoch()
    while inbox.take(0, flush=True) is not None:
        pass
    inbox.next_epoch()
    assert inbox.finished and not inbox.epoch_done()


def test_the_orchestrators_hard_state_rows_are_auxiliary_rows_to_an_rl_learner():
    from agentenv_zeroad.train import distill

    class Chars:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) % 90 for c in text]

    brief = distill.Brief("g1", "1", 1000, "brief", (1, 2, 3), "losing", opponent="petra-3")
    (row,) = distill.sft_rows([distill.Label(brief, [("army", {"action": "attack"})])], Chars(), "opus")
    parsed = L.parse_row(json.loads(json.dumps(row)), 10_000)
    assert parsed.aux and parsed.prompt == [1, 2, 3] and parsed.advantage == 0.0 and parsed.completion
    assert not L.parse_row(json.loads(json.dumps(row)), 10_000, sft=True).aux  # the SFT learner just reads the ids


# --- review fixes: the aux weight, the SFT restart, the anchor defaults ---


def test_the_aux_weight_decays_only_over_updates_that_had_labels_and_a_checkpoint_and_rollback_restore_it(tmp_path):
    learner = make_learner(tmp_path, aux_alpha0=0.2, aux_decay=0.5)
    for _ in range(3):
        health = learner.update(rows())  # no teacher labels yet: the weight must not decay
    assert health["aux_alpha"] == 0.2 and learner.aux_updates == 0
    assert learner.update(rows(), aux=aux_rows(2))["aux_alpha"] == 0.2 and learner.aux_updates == 1
    assert learner.update(rows(), aux=aux_rows(2))["aux_alpha"] == pytest.approx(0.1)
    assert learner.update(rows())["aux_alpha"] == pytest.approx(0.05) and learner.aux_updates == 2
    again = make_learner(tmp_path, aux_alpha0=0.2, aux_decay=0.5)
    assert again.resume() and again.aux_updates == 2 and again.aux_alpha() == pytest.approx(0.05)
    learner.rollback(4)  # the checkpoint after the first labelled update
    assert learner.aux_updates == 1 and learner.aux_alpha() == pytest.approx(0.1)


def test_every_aux_row_is_trained_in_minibatch_one_even_when_the_kl_stop_skips_the_second(tmp_path):
    learner = make_learner(tmp_path, aux_alpha0=0.2, kl_stop=-1.0)  # any KL is above it: minibatch 2 never runs
    aux = aux_rows(3)
    want = -ce(learner, aux) / sum(sum(L.annotate(r, learner.decode, 100, 101).mask) for r in aux)
    health = learner.update(rows(), aux=aux)
    assert health["minibatches_run"] == 1 and health["aux_rows"] == 3
    assert health["aux_ce"] == pytest.approx(want, rel=1e-4)  # all three rows' tokens, none dropped with minibatch 2


def test_a_malformed_teacher_label_is_counted_apart_from_the_rejected_share(tmp_path):
    cfg = config(tmp_path, aux_alpha0=0.2)
    inbox = L.Inbox(cfg)
    drop(cfg.inbox, "a.jsonl", [{"sft": True, "game_id": "x"}, {"sft": True, **sft_raw(1), "token_ids": []}])
    drop(cfg.inbox, "b.jsonl", [{"game_id": "y"}])
    inbox.poll()
    assert dict(inbox.counts) == {"aux_rejected:missing_field": 1, "aux_rejected:empty": 1, "rejected:missing_field": 1}
    learner = make_learner(tmp_path / "l")
    health = learner.update(rows(), {"accepted": 4, "aux_rejected:missing_field": 50})
    assert health["rejected_share"] == 0.0  # fifty bad labels do not feed the tripwire's rejected share
    assert learner.update(rows(), {"accepted": 4, "rejected:bad_json": 4})["rejected_share"] == 0.5


def test_the_anchor_pass_swaps_the_weights_once_per_update_and_restores_them_when_the_copy_in_fails(tmp_path):
    anchor = trained_adapter(tmp_path)
    learner = make_learner(tmp_path, root=tmp_path / "rl", init=str(anchor), anchor=str(anchor), anchor_beta0=0.1)
    swaps = []
    many = learner.anchor_logps_many
    learner.anchor_logps_many = lambda batch: swaps.append(len(batch)) or many(batch)
    learner.anchor_logps = lambda row: pytest.fail("one swap per row")
    learner.update(rows())
    assert swaps == [4]
    before = weights(learner)
    names = [n for n, _ in learner.params]
    learner.anchor_weights[names[-1]] = torch.zeros(1, 1, 1)  # a copy-in that fails after the earlier ones were done
    with pytest.raises(RuntimeError):
        many(rows())
    assert all(torch.equal(before[n], p) for n, p in learner.params)


def test_with_the_base_as_the_anchor_the_base_kl_sample_reuses_the_anchor_log_probs(tmp_path):
    learner = make_learner(tmp_path, anchor_beta0=0.1, base_kl_fraction=1.0, lr=5e-2)
    base_passes = []
    real = learner.logps
    learner.logps = lambda row, *, grad, adapter=True, entropy=False: (
        base_passes.append(row) if not adapter else None,
        real(row, grad=grad, adapter=adapter, entropy=entropy),
    )[1]
    learner.update(rows())
    second = learner.update(rows())
    assert len(base_passes) == 8  # one adapter-off pass per row per update (the anchor's), not one more for the base KL
    assert second["kl_to_base"] == pytest.approx(second["kl_to_anchor"], rel=1e-6)


def test_an_anchor_adapter_with_the_right_shapes_but_another_scaling_or_target_set_is_refused(tmp_path):
    anchor = trained_adapter(tmp_path)
    for field, value in (("lora_alpha", 16), ("target_modules", ["q_proj", "k_proj"]), ("r", 8)):
        other = tmp_path / f"anchor-{field}"
        other.mkdir()
        for f in anchor.iterdir():
            (other / f.name).write_bytes(f.read_bytes())
        saved = json.loads((other / "adapter_config.json").read_text())
        (other / "adapter_config.json").write_text(json.dumps({**saved, field: value}))
        with pytest.raises(ValueError, match="does not fit"):
            make_learner(tmp_path, root=tmp_path / f"rl-{field}", anchor=str(other), anchor_beta0=0.1)
    ok = make_learner(tmp_path, root=tmp_path / "rl-ok", anchor=str(anchor), anchor_beta0=0.1)
    assert ok.anchor_weights


def test_the_anchor_defaults_to_init_and_base_is_chosen_explicitly(tmp_path, monkeypatch):
    assert L.resolve_anchor("/train/sft", None) == "/train/sft"
    assert L.resolve_anchor("/train/sft", "base") == "" and L.resolve_anchor("/train/sft", "") == ""
    assert L.resolve_anchor("/train/sft", "/train/other") == "/train/other" and L.resolve_anchor("", None) == ""
    monkeypatch.setattr(sys, "argv", ["learner.py", "--base", "b", "--root", "r", "--init", "/train/sft"])
    assert L.parse().anchor is None  # not given: main anchors to --init, not silently to the base
    anchor = trained_adapter(tmp_path)
    learner = make_learner(
        tmp_path, root=tmp_path / "rl", init=str(anchor), anchor=L.resolve_anchor(str(anchor), None), anchor_beta0=0.1
    )
    health = learner.update(rows())
    manifest = json.loads((learner.cfg.adapters / "v1" / "manifest.json").read_text())
    assert health["anchor"] == manifest["anchor"] == str(anchor) and manifest["init"] == str(anchor)
    off = make_learner(tmp_path, root=tmp_path / "off", anchor_beta0=0.0)
    assert off.update(rows())["anchor"] is None


# --- SFT restart ---

SFT = {"sft": True, "lr": 5e-2, "sft_epochs": 2, "sft_entropy_rows": 4, "sft_holdout": 0.3, "ring": 20}


def sft_inbox(tmp_path: Path, n: int = 7, **overrides):
    learner = make_learner(tmp_path, **{**SFT, **overrides})
    inbox = L.Inbox(learner.cfg)
    drop(inbox.cfg.inbox, "sft.jsonl", [sft_raw(i) for i in range(n)])
    return learner, inbox


def finishes_by_itself(learner, inbox) -> None:
    """Serve until the learner returns on its own: a stop_when that fires at 300 polls is a failure to finish."""
    ticks = {"n": 0}

    def stop_when() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 300

    L.serve(learner, inbox, poll=0, stop_when=stop_when)
    assert ticks["n"] < 100


def restart(tmp_path: Path, **overrides):
    learner = make_learner(tmp_path, **{**SFT, **overrides})
    assert learner.resume()
    return learner, L.Inbox(learner.cfg)


def test_a_restarted_sft_learner_rebuilds_its_dataset_and_publishes_the_epochs_it_still_owes(tmp_path):
    learner, inbox = sft_inbox(tmp_path)
    L.serve(learner, inbox, poll=0, stop_when=lambda: (learner.cfg.epochs / "epoch1").exists())
    assert not (learner.cfg.epochs / "epoch2").exists()
    again, inbox2 = restart(tmp_path)
    assert again.sft_progress == (2, 0)
    finishes_by_itself(again, inbox2)
    assert len(inbox2.dataset) == 6 and len(inbox2.holdout) == 1  # rebuilt from inbox/consumed
    epoch2 = json.loads((again.cfg.epochs / "epoch2" / "manifest.json").read_text())
    assert epoch2["version"] == again.version == 4 and again.decisions_total == 12 and epoch2["entropy"] > 0
    assert (again.cfg.sft_done).exists() and not (again.cfg.epochs / "epoch3").exists()


def test_a_restart_in_the_middle_of_an_epoch_trains_only_the_rows_that_were_left(tmp_path):
    learner, inbox = sft_inbox(tmp_path)
    L.serve(learner, inbox, poll=0, stop_when=lambda: learner.version >= 1)  # 4 of the epoch's 6 rows trained
    assert learner.sft_progress == (1, 4) and not (learner.cfg.epochs / "epoch1").exists()
    again, inbox2 = restart(tmp_path)
    finishes_by_itself(again, inbox2)
    assert again.version == 4 and again.decisions_total == 12  # 4 before, then 2 left in epoch 1, then all 6 of epoch 2
    assert (again.cfg.epochs / "epoch1").exists() and (again.cfg.epochs / "epoch2").exists()


def test_a_restart_after_the_last_epoch_neither_retrains_nor_fails_on_the_existing_directory(tmp_path):
    learner, inbox = sft_inbox(tmp_path)
    L.serve(learner, inbox, poll=0)
    assert learner.cfg.sft_done.exists() and learner.version == 4
    again, inbox2 = restart(tmp_path)
    L.serve(again, inbox2, poll=0)
    assert again.version == 4 and again.decisions_total == 12  # epoch 2 was reused as published, not trained again


def test_an_existing_epoch_directory_is_reused_for_the_same_version_and_refused_for_another(tmp_path):
    learner, inbox = sft_inbox(tmp_path, sft_epochs=1)
    L.serve(learner, inbox, poll=0)
    assert learner.finish_epoch(1, [])["version"] == learner.version  # reused: no rename, no barrier left hanging
    learner.version += 1
    with pytest.raises(RuntimeError, match="already exists"):
        learner.finish_epoch(1, [])


def test_the_sft_learner_exits_by_itself_after_its_last_epoch_and_leaves_a_done_marker(tmp_path):
    learner, inbox = sft_inbox(tmp_path)
    ticks = {"n": 0}

    def stop_when() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 500

    L.serve(learner, inbox, poll=0, stop_when=stop_when)
    assert ticks["n"] < 60 and json.loads(learner.cfg.sft_done.read_text()) == {"epochs": 2, "version": 4}


def test_the_entropy_is_measured_on_held_out_rows_that_are_never_trained_on(tmp_path):
    learner, inbox = sft_inbox(tmp_path, n=40, sft_holdout=0.25, sft_entropy_rows=100)
    inbox.poll()
    trained, held = {r.game_id for r in inbox.dataset}, {r.game_id for r in inbox.holdout}
    assert held and trained and not trained & held and len(trained | held) == 40
    assert {r.game_id for r in inbox.entropy_sample()} == held
    again = L.Inbox(learner.cfg)
    again.restore(1, 0)  # a restart holds out the same rows
    assert {r.game_id for r in again.holdout} == held and {r.game_id for r in again.dataset} == trained
    ratio = learner.finish_epoch(1, inbox.entropy_sample())
    assert ratio["entropy_rows"] == len(held)


# --- the anchor swap under the box's settings (bf16 autocast, gradient checkpointing) ---
BOX = {"dtype": "bfloat16", "gradient_checkpointing": True}


def test_anchor_log_probs_match_an_independent_model_loaded_with_the_anchor_under_the_box_settings(tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    sft = make_learner(tmp_path, root=tmp_path / "sft", lr=5e-2, **BOX)
    for _ in range(3):
        sft.update(rows())
    anchor = sft.cfg.adapters / "v3"
    learner = make_learner(tmp_path, root=tmp_path / "rl", anchor=str(anchor), anchor_beta0=0.1, lr=5e-2, **BOX)
    for _ in range(2):
        learner.update(rows())  # move the policy off the base and off the anchor
    before = weights(learner)
    r = L.annotate(row(seed=7), learner.decode, 100, 101)
    got = learner.anchor_logps(r)
    policy = learner.logps(r, grad=False)[0]
    assert all(torch.equal(before[n], p) for n, p in learner.params)
    ref = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(learner.cfg.base, torch_dtype=torch.bfloat16), str(anchor)
    )
    for n, p in ref.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()  # LoRA in fp32 as the learner keeps it
    ref.eval()
    ids = torch.tensor([r.prompt + r.completion])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = ref.get_base_model().model(input_ids=ids).last_hidden_state[0, -(len(r.completion) + 1) : -1]
    lp = torch.log_softmax(hidden.float() @ ref.get_base_model().lm_head.weight.float().T, -1)
    want = lp.gather(1, torch.tensor(r.completion)[:, None])[:, 0]
    assert torch.allclose(got, want, atol=1e-4) and (got - policy).abs().max().item() > 1e-4


def test_an_exception_inside_the_anchor_pass_restores_the_policy_under_the_box_settings(tmp_path):
    sft = make_learner(tmp_path, root=tmp_path / "sft", lr=5e-2, **BOX)
    sft.update(rows())
    learner = make_learner(tmp_path, root=tmp_path / "rl", anchor=str(sft.cfg.adapters / "v1"), anchor_beta0=0.1, **BOX)
    learner.update(rows())
    before = weights(learner)
    real = learner.logps

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulated)")

    learner.logps = boom
    with pytest.raises(RuntimeError):
        learner.anchor_logps(L.annotate(row(seed=1), learner.decode, 100, 101))
    learner.logps = real
    assert all(torch.equal(before[n], p) for n, p in learner.params)


def test_init_equal_to_the_anchor_gives_zero_kl_with_the_aux_term_in_the_box_settings(tmp_path):
    sft = make_learner(tmp_path, root=tmp_path / "sft", lr=5e-2, **BOX)
    sft.update(rows())
    anchor = str(sft.cfg.adapters / "v1")
    learner = make_learner(
        tmp_path, root=tmp_path / "rl", init=anchor, anchor=anchor, anchor_beta0=0.1, aux_alpha0=0.2, **BOX
    )
    health = learner.update(rows(), aux=aux_rows(3))
    assert health["kl_to_anchor"] == pytest.approx(0, abs=1e-8) and health["aux_rows"] == 3
