"""The pilot's trainer on a tiny random Qwen3 on the CPU: it trains a fresh adapter, then continues one."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

TRAIN = Path(__file__).resolve().parents[1] / "trainer" / "train.py"


def tiny_model(path: Path) -> None:
    config = transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
    )  # fmt: skip
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)


def decisions(path: Path, n: int) -> None:
    rows = [
        {"prompt_token_ids": [1 + (i % 50), 2, 3, 4], "token_ids": [5, 6 + (i % 7), 7], "advantage": (-1) ** i * 1.5}
        for i in range(n)
    ]
    rows.append({"prompt_token_ids": [], "token_ids": [], "advantage": 0.0})  # no token ids: skipped
    path.write_text("\n".join(json.dumps(r) for r in rows))


def train(tmp_path: Path, init: str, out: Path) -> dict:
    cmd = [sys.executable, str(TRAIN), "--decisions", str(tmp_path / "d.jsonl"), "--base", str(tmp_path / "base")]
    cmd += ["--init", init, "--out", str(out), "--samples", "8", "--batch", "4", "--dtype", "float32", "--rank", "4"]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads((out / "train.json").read_text())


def test_the_trainer_writes_an_adapter_and_can_continue_it(tmp_path):
    tiny_model(tmp_path / "base")
    decisions(tmp_path / "d.jsonl", 10)
    first = train(tmp_path, "", tmp_path / "v1")
    assert first["steps"] == 2 and first["decisions"] == 8 and (tmp_path / "v1" / "adapter_config.json").exists()
    assert first["mean_kl"] >= 0
    second = train(tmp_path, str(tmp_path / "v1"), tmp_path / "v2")
    assert second["init"] == str(tmp_path / "v1") and (tmp_path / "v2" / "adapter_model.safetensors").exists()
    assert second["mean_kl"] > 0  # v1 already moved away from the base


def test_averaging_weights_each_members_tensors_by_the_decisions_it_trained_on(tmp_path):
    from safetensors.torch import load_file, save_file

    for name, value, decisions in (("a", 1.0, 1), ("b", 4.0, 3)):
        d = tmp_path / name
        d.mkdir()
        save_file({"w": torch.full((2, 2), value, dtype=torch.bfloat16)}, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text("{}")
        (d / "train.json").write_text(json.dumps({"decisions": decisions, "mean_kl": value / 100}))
    cmd = [sys.executable, str(TRAIN.with_name("average.py")), "--out", str(tmp_path / "v2")]
    cmd += ["--member", str(tmp_path / "a"), "--member", str(tmp_path / "b")]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    w = load_file(str(tmp_path / "v2" / "adapter_model.safetensors"))["w"]
    assert w.dtype == torch.bfloat16 and float(w[0, 0]) == (1.0 * 1 + 4.0 * 3) / 4
    stats = json.loads((tmp_path / "v2" / "train.json").read_text())
    assert stats["decisions"] == 4 and [m["decisions"] for m in stats["members"]] == [1, 3]
