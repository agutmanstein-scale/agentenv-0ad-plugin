"""The training box's orchestrator must import and run on the base install: the box venv is `uv sync --frozen --no-dev`,
without the `train` extra, for a play-only or S1 run, and the learner's libraries (torch, transformers, peft) are only
in the trainer image. Each case runs in a fresh interpreter with those packages made unimportable."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BLOCKED = ("numpy", "scipy", "sklearn", "torch", "transformers", "peft", "safetensors", "lightgbm")
ROOT = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent(
    """
    import importlib, runpy, sys

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {blocked!r}:
                raise ImportError(f"blocked (not in the base install): {{name}}")

    sys.meta_path.insert(0, Blocker())
    mode, *rest = sys.argv[1:]
    if mode == "import":
        for module in rest:
            importlib.import_module(module)
        print("imported", *rest)
    elif mode == "code":
        exec(rest[0])
    else:
        sys.argv = ["orchestrate", *rest]
        runpy.run_module("agentenv_zeroad.train.orchestrate", run_name="__main__")
    """
).format(blocked=set(BLOCKED))


def run_blocked(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", DRIVER, *args], capture_output=True, text=True, cwd=ROOT, timeout=120, check=False
    )


@pytest.mark.parametrize(
    "module",
    [
        "agentenv_zeroad.train.orchestrate",
        "agentenv_zeroad.train.hardstate",
        "agentenv_zeroad.train.distill",
        "agentenv_zeroad.train.report",
        "agentenv_zeroad.train.stats",
        "agentenv_zeroad.train.calibration",
        "agentenv_zeroad.train.credit",
    ],
)
def test_the_box_side_modules_import_without_numpy_scikit_learn_torch_or_transformers(module):
    done = run_blocked("import", module)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_orchestrator_prints_its_tools_without_the_train_extra():
    done = run_blocked("run-tools", "tools")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "hold" in done.stdout and "," in done.stdout


def test_the_teacher_pick_and_wilson_interval_need_no_numpy():
    code = (
        "from agentenv_zeroad.train import stats\n"
        "s = stats.teacher_calibration([stats.TeacherGame('a', 'petra-3', 1.0)])\n"
        "print(stats.pick_teacher(s), stats.wilson_interval(1, 2))"
    )
    done = run_blocked("code", code)
    assert done.returncode == 0 and done.stdout.startswith("a (0."), done.stderr[-1500:]


def test_the_numeric_board_still_needs_numpy_and_says_so_when_it_is_called_without_it():
    code = (
        "from agentenv_zeroad.train import stats\n"
        "try:\n"
        "    stats.trend_test([stats.Game(1, 1.0, 'petra-3', 1.0), stats.Game(2, 2.0, 'petra-3', 0.0)])\n"
        "except ImportError as e:\n"
        "    print('needs numpy:', e)\n"
    )
    done = run_blocked("code", code)
    assert done.returncode == 0 and "needs numpy" in done.stdout, done.stderr[-1500:]
