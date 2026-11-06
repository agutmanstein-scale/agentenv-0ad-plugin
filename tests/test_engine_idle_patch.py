"""The engine patches (patches/rl-realtime.patch, patches/rl-idle-wait.patch) against the pristine 0.28.0 sources."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from agentenv_zeroad.pool import TRAIN_IMAGE, TRAIN_PATCHES

ROOT = Path(__file__).resolve().parent.parent
PRISTINE = ROOT / "tests/fixtures/0ad-0.28.0"
SHA256 = {
    "source/main.cpp": "ca2c4d60adada93cd81aa91c5d96d616fbae3e1a84f2b7a92cf52a700377fde5",
    "source/rlinterface/RLInterface.cpp": "27900e2c3cbb632e3bb73e29c188f678e8b1a369252765e718f393b6eed66ecf",
    "source/rlinterface/RLInterface.h": "9a133be97c91093f364fea02db9273627a37ade8daa798f3a49b028250090793",
}
PATCHES = ("rl-realtime.patch", "rl-idle-wait.patch")
needs_patch = pytest.mark.skipif(shutil.which("patch") is None, reason="`patch` is not installed")


def test_the_pristine_copies_are_the_0_28_0_files():
    for name, digest in SHA256.items():
        assert hashlib.sha256((PRISTINE / name).read_bytes()).hexdigest() == digest, name


def _tree(tmp_path: Path) -> Path:
    tree = tmp_path / "0ad"
    for name in SHA256:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PRISTINE / name, tree / name)
    return tree


def _patch(tree: Path, name: str, *flags: str) -> subprocess.CompletedProcess[str]:
    cmd = ["patch", "-p1", "-F0", "--no-backup-if-mismatch", *flags, "-i", str(ROOT / "patches" / name)]
    return subprocess.run(cmd, cwd=tree, capture_output=True, text=True)


@needs_patch
def test_both_patches_apply_in_order_to_the_pristine_sources_with_no_fuzz(tmp_path):
    tree = _tree(tmp_path)
    for name in PATCHES:
        done = _patch(tree, name, "--dry-run")
        assert done.returncode == 0, f"{name}: {done.stdout}{done.stderr}"
        assert "fuzz" not in done.stdout and "offset" not in done.stdout.replace("offset 0", ""), done.stdout
        done = _patch(tree, name)  # the second patch is checked against the tree the first one made
        assert done.returncode == 0, f"{name}: {done.stdout}{done.stderr}"


@needs_patch
def test_the_idle_patch_does_not_apply_without_the_realtime_patch_first(tmp_path):
    assert _patch(_tree(tmp_path), "rl-idle-wait.patch", "--dry-run").returncode != 0


@needs_patch
def test_the_patched_sources_wake_on_post_clear_on_take_and_skip_the_state_on_request(tmp_path):
    tree = _tree(tmp_path)
    for name in PATCHES:
        assert _patch(tree, name).returncode == 0
    cpp = (tree / "source/rlinterface/RLInterface.cpp").read_text()
    take = cpp.split("bool Interface::TryGetGameMessage")[1].split("\n}\n")[0]
    wait = cpp.split("void Interface::WaitForMessage")[1].split("\n}\n")[0]
    assert "m_Posted = false;" in take and "m_Posted = false" not in wait
    assert 'm_WantState ? GetGameState() : std::string("{}")' in cpp
    assert '"state"' in cpp
    assert "WaitForMessage(std::chrono::milliseconds{2})" in (tree / "source/main.cpp").read_text()


def test_the_rt_image_builds_both_patches_in_order_and_says_so_in_its_label():
    dockerfile = (ROOT / "engine/Dockerfile.rt").read_text()
    assert "COPY rl-realtime.patch rl-idle-wait.patch /src/" in dockerfile
    assert dockerfile.index("patch -p1 -F0 < /src/rl-realtime.patch") < dockerfile.index(
        "patch -p1 -F0 < /src/rl-idle-wait.patch"
    )
    assert f'LABEL zeroad.patches="{" ".join(TRAIN_PATCHES)}"' in dockerfile
    assert TRAIN_PATCHES == ("rl-realtime", "rl-idle-wait")
    assert TRAIN_IMAGE == "zeroad-engine:0.28-rt"
