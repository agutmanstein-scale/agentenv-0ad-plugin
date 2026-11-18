"""The learner's lean update (no redundant proximal forward), whole-group batching, the held-out step KL and the bench
mode, on a tiny random Qwen3.

Run with: uv run --with torch --with transformers --with peft pytest tests/test_learner_lean.py -q
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))
import learner as L  # noqa: E402

from tests.test_learner import config, drop, make_learner, raw, row, rows  # noqa: E402


def grouped(n: int, size: int, **kwargs) -> list[L.Decision]:
    return [replace(r, group=f"grp{i // size}") for i, r in enumerate(rows(n, **kwargs))]


def capture_first_step(learner: L.Learner) -> list[dict[str, torch.Tensor]]:
    """The gradients each optimizer step is about to apply, in step order."""
    seen: list[dict[str, torch.Tensor]] = []
    real = learner.optimizer.step

    def step(*args, **kwargs):
        seen.append({n: p.grad.detach().clone() for n, p in learner.params if p.grad is not None})
        return real(*args, **kwargs)

    learner.optimizer.step = step
    return seen


# --- the lean proximal pass ---


@pytest.mark.parametrize("beta0", [0.0, 0.1])
def test_the_lean_prox_path_gives_the_same_loss_gradients_and_measurements_as_the_full_pass(tmp_path, beta0):
    out = {}
    for name, full in (("lean", False), ("full", True)):
        learner = make_learner(tmp_path / name, full_prox=full, anchor_beta0=beta0, base_kl_fraction=0.5, lr=5e-2)
        grads = capture_first_step(learner)
        health = learner.update(rows(8))
        out[name] = (health, grads, {n: p.detach().clone() for n, p in learner.params})
    (lean, lean_grads, lean_weights), (full, full_grads, full_weights) = out["lean"], out["full"]
    assert len(lean_grads) == len(full_grads) == 2
    for a, b in zip(lean_grads, full_grads, strict=True):
        assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-7, rtol=1e-5) for k in a)
    assert all(torch.allclose(lean_weights[k], full_weights[k], atol=1e-7) for k in lean_weights)
    for key in (
        "loss",
        "pg_loss",
        "anchor_loss",
        "entropy",
        "kl_to_base",
        "kl_to_anchor",
        "kl_after_mb",
        "clipped_share",
    ):
        assert lean[key] == pytest.approx(full[key], rel=1e-5, abs=1e-8), key


def test_the_lean_path_runs_no_proximal_forward_for_minibatch_one(tmp_path):
    counts = {}
    for name, full in (("lean", False), ("full", True)):
        learner = make_learner(tmp_path / name, full_prox=full, base_kl_fraction=0.01)
        calls: list[tuple[bool, bool]] = []
        real = learner.logps
        learner.logps = lambda r, *, grad, adapter=True, entropy=False, _real=real, _calls=calls: (
            _calls.append((grad, adapter)) or _real(r, grad=grad, adapter=adapter, entropy=entropy)
        )
        learner.update(rows(8))
        counts[name] = calls
    no_grad_on = lambda calls: sum(1 for grad, adapter in calls if not grad and adapter)  # noqa: E731
    # 8 rows, 2 minibatches: the proximal pass is 8 forwards in full and 4 (minibatch 2's) when lean; the KL-after
    # check adds 4 to both
    assert no_grad_on(counts["full"]) == 12 and no_grad_on(counts["lean"]) == 8
    assert sum(1 for grad, _ in counts["lean"] if grad) == sum(1 for grad, _ in counts["full"] if grad) == 8


def test_an_anchor_beta0_of_zero_never_runs_the_anchor_pass_or_loads_the_anchor(tmp_path):
    learner = make_learner(tmp_path, anchor_beta0=0.0, anchor=str(tmp_path / "no-such-adapter"))
    assert learner.anchor_weights == {} and learner.anchor_label() is None

    def boom(*args, **kwargs):
        raise AssertionError("the anchor's pass ran with beta0 = 0")

    learner.anchor_logps_many = boom
    health = learner.update(rows(8))
    assert health["anchor_beta"] == 0 and health["kl_to_anchor"] is None and health["anchor_loss"] == 0
    assert learner.version == 1


def test_the_kl_check_runs_on_the_given_fraction_of_minibatch_two(tmp_path):
    seen = {}
    for fraction in (1.0, 0.25):
        learner = make_learner(tmp_path / str(fraction), kl_check_fraction=fraction, decisions_per_update=16)
        checked: list[int] = []
        real = learner._kl_after
        learner._kl_after = lambda mb, prox, _real=real, _c=checked: (_c.append(len(mb)), _real(mb, prox))[1]
        calls: list[bool] = []
        real_logps = learner.logps
        learner.logps = lambda r, *, grad, adapter=True, entropy=False, _rl=real_logps, _cl=calls: (
            _cl.append(grad) or _rl(r, grad=grad, adapter=adapter, entropy=entropy)
        )
        health = learner.update(rows(16))
        seen[fraction] = (checked, calls.count(False), health["kl_after_mb"])
    assert seen[1.0][0] == seen[0.25][0] == [8]  # the minibatch handed to the check is whole
    assert seen[1.0][1] - seen[0.25][1] == 8 - 2  # 8 forwards for the check in full, ceil(25% of 8) = 2
    assert seen[0.25][2] is not None


# --- whole start groups ---


def pool_of(tmp_path: Path, versions_and_groups: list[tuple[int, str]], **overrides) -> L.Inbox:
    inbox = L.Inbox(config(tmp_path, **overrides))
    inbox.pool = [replace(row(seed=i, version=v), group=g) for i, (v, g) in enumerate(versions_and_groups)]
    return inbox


def test_whole_groups_are_taken_freshest_first_until_the_batch_is_full_and_may_overshoot(tmp_path):
    spec = [(3, "a")] * 3 + [(5, "b")] * 3 + [(4, "c")] * 3 + [(5, "d")] * 3
    inbox = pool_of(tmp_path, spec, decisions_per_update=4, whole_groups=True)
    batch = inbox.take(5)
    # b and d hold the newest rows (5), the earlier one first; two groups = 6 rows >= 4, so it overshoots by 2
    assert sorted({r.group for r in batch}) == ["b", "d"] and len(batch) == 6
    assert sorted({r.group for r in inbox.pool}) == ["a", "c"]
    assert inbox.take(5) is not None and inbox.take(5) is None  # c (4, 3 rows) + a (3 rows): 6, then 0


def test_whole_groups_never_split_a_group_and_a_short_pool_waits(tmp_path):
    inbox = pool_of(tmp_path, [(5, "a")] * 2 + [(5, "b")] * 2, decisions_per_update=5, whole_groups=True)
    assert inbox.take(5) is None


def test_rows_without_a_group_batch_as_before_with_whole_groups_on(tmp_path):
    spec = [(v, "") for v in (3, 5, 4, 5)]
    plain = pool_of(tmp_path / "p", spec, decisions_per_update=2).take(5)
    whole = pool_of(tmp_path / "w", spec, decisions_per_update=2, whole_groups=True).take(5)
    assert [r.game_id for r in whole] == [r.game_id for r in plain] and [r.version for r in whole] == [5, 5]


def test_staleness_still_drops_rows_of_a_group_with_whole_groups(tmp_path):
    inbox = pool_of(tmp_path, [(0, "a"), (9, "a"), (9, "b")], decisions_per_update=2, whole_groups=True)
    batch = inbox.take(9)
    assert sorted(r.group for r in batch) == ["a", "b"] and inbox.counts["stale"] == 1


def test_the_group_field_is_read_from_a_row_and_empty_without_one():
    assert L.parse_row(raw(group="s1"), 100).group == "s1" and L.parse_row(raw(), 100).group == ""
    assert L.parse_row(raw(group=7), 100).group == ""


def test_the_minibatch_split_keeps_a_group_inside_one_minibatch(tmp_path):
    learner = make_learner(tmp_path, whole_groups=True)
    batch = grouped(12, 3)  # four groups of three
    for version in range(6):
        learner.version = version
        first, second = learner._minibatches(batch)
        assert not {r.group for r in first} & {r.group for r in second}
        assert sorted(r.game_id for r in first + second) == sorted(r.game_id for r in batch) and first and second


def test_the_minibatch_split_falls_back_to_rows_when_there_is_one_group(tmp_path):
    learner = make_learner(tmp_path, whole_groups=True)
    first, second = learner._minibatches(grouped(8, 8))
    assert len(first) == len(second) == 4


def test_an_update_on_whole_groups_trains_every_row_of_the_overshot_batch(tmp_path):
    learner = make_learner(tmp_path, whole_groups=True, decisions_per_update=4)
    health = learner.update(grouped(6, 3))
    assert health["trained"] == 6 and health["minibatches_run"] == 2


# --- the held-out step KL ---


def probe_pool(tmp_path: Path, n_rows: int, games: int = 40) -> L.Inbox:
    inbox = L.Inbox(config(tmp_path, probe_rows=n_rows))
    drop(inbox.cfg.inbox, "a.jsonl", [raw(i % games, token_ids=[3 + i % 5, 4, 5]) for i in range(games * 2)])
    inbox.poll()
    return inbox


def test_the_probe_rows_are_cut_by_game_and_never_enter_the_pool(tmp_path):
    inbox = probe_pool(tmp_path, 6)
    assert len(inbox.probe) == 6 and inbox.probe_games
    assert {r.game_id for r in inbox.probe} <= inbox.probe_games
    assert not inbox.probe_games & {r.game_id for r in inbox.pool}  # no row of a probe game trains
    assert len(inbox.pool) + len(inbox.probe) + inbox.counts["probe_dropped"] == 80
    assert inbox.take_probe() is not None and inbox.take_probe() is None  # handed over once
    again = probe_pool(tmp_path / "again", 6)
    assert [r.game_id for r in again.probe] == [
        r.game_id for r in inbox.probe
    ]  # stable under a restart of the same rows


def test_the_probe_is_off_by_default(tmp_path):
    inbox = probe_pool(tmp_path, 0)
    assert inbox.probe == [] and inbox.take_probe() is None and len(inbox.pool) == 80


def test_the_held_out_step_kl_is_zero_when_the_weights_do_not_move_and_positive_after_a_step(tmp_path):
    learner = make_learner(tmp_path, lr=0.0, probe_rows=4)
    learner.set_probe([row(seed=100 + i) for i in range(4)])
    still = learner.update(rows(8))
    assert still["kl_step_heldout"] == pytest.approx(0, abs=1e-9)
    learner.lr = 5e-2
    moved = learner.update(rows(8))
    assert moved["kl_step_heldout"] > 1e-6
    assert learner._probe_step() == pytest.approx(0, abs=1e-9)  # the last readout is the baseline: nothing moved since


def test_the_health_record_has_no_step_kl_without_a_probe(tmp_path):
    assert make_learner(tmp_path).update(rows(8))["kl_step_heldout"] is None


def test_a_rollback_refreshes_the_probe_baseline(tmp_path):
    learner = make_learner(tmp_path, lr=5e-2)
    learner.set_probe([row(seed=100 + i) for i in range(4)])
    learner.update(rows(8))
    learner.update(rows(8))
    learner.apply({"op": "rollback", "version": 1})
    assert learner._probe_step() == pytest.approx(0, abs=1e-9)


# --- bench mode ---


def bench_dir(tmp_path: Path, n: int) -> Path:
    out = tmp_path / "bench-rows"
    out.mkdir()
    for part in range(2):
        drop(
            out,
            f"part{part}.jsonl",
            [
                raw(i, version=99, token_ids=[20 + i % 7, 21, 22 + i % 3], logprobs=[-3.0] * 3, advantage=(-1) ** i)
                for i in range(part * n // 2, (part + 1) * n // 2)
            ],
        )
    return out


def test_bench_times_real_updates_and_writes_bench_json_but_publishes_nothing(tmp_path):
    learner = make_learner(tmp_path, bench=True, decisions_per_update=4)
    before = {n: p.detach().clone() for n, p in learner.params}
    result = L.bench(learner, bench_dir(tmp_path, 10), 3)
    written = json.loads(learner.cfg.bench_result.read_text())
    assert written == json.loads(json.dumps(result))
    assert [u["update"] for u in written["updates"]] == [1, 2, 3]
    assert all(u["rows"] == 4 and u["seconds"] > 0 for u in written["updates"])
    assert all(u["seconds_per_row"] == pytest.approx(u["seconds"] / 4, rel=1e-2) for u in written["updates"])
    warm = [u["seconds_per_row"] for u in written["updates"][1:]]
    assert written["mean_seconds_per_row"] == pytest.approx(sum(warm) / 2)  # update 1 is the warm-up
    assert learner.version == 3 and any(not torch.equal(before[n], p) for n, p in learner.params)
    assert not any(learner.cfg.adapters.glob("*")) if learner.cfg.adapters.exists() else True
    assert not list(learner.cfg.checkpoints.glob("*.pt")) and not learner.cfg.health.exists()
    assert written["flags"]["full_prox"] is False


def test_bench_ignores_the_rows_versions_and_wraps_around_a_short_directory(tmp_path):
    learner = make_learner(tmp_path, bench=True, decisions_per_update=4)  # the rows say version 99; the learner is at 0
    result = L.bench(learner, bench_dir(tmp_path, 6), 4)  # 4 updates x 4 rows from 6 rows
    assert len(result["updates"]) == 4 and all(u["rows"] == 4 for u in result["updates"])


def test_bench_takes_the_probe_from_the_first_rows_and_refuses_a_directory_under_one_batch(tmp_path):
    learner = make_learner(tmp_path, bench=True, decisions_per_update=4, probe_rows=2, whole_groups=True)
    result = L.bench(learner, bench_dir(tmp_path, 30), 2)
    assert [u["kl_step_heldout"] is not None for u in result["updates"]] == [True, True]
    small = make_learner(tmp_path / "small", bench=True, decisions_per_update=40)
    with pytest.raises(SystemExit, match="at least one batch of 40"):
        L.bench(small, bench_dir(tmp_path / "small", 6), 1)


# --- an init adapter with a version of its own ---


def test_a_learner_started_from_an_init_adapter_counts_on_from_its_version_and_never_publishes_over_it(tmp_path):
    learner = make_learner(tmp_path, start_version=3)
    assert learner.version == 3
    published = learner.cfg.adapters
    published.mkdir(parents=True, exist_ok=True)
    (published / "v3").mkdir()
    (published / "v3" / "marker").write_text("the init adapter")
    learner.update(rows(8))
    assert learner.version == 4 and (published / "v4" / "manifest.json").exists()
    assert (published / "v3" / "marker").read_text() == "the init adapter"
    assert json.loads((published / "v4" / "manifest.json").read_text())["version"] == 4


def test_the_default_start_version_is_zero(tmp_path):
    assert make_learner(tmp_path).version == 0
