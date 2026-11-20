"""scripts/build-engine.sh decides from the image's labels whether to rebuild the rt/show images: the names of the
patches are not enough, a changed patch must be built too (docker is a stub on PATH)."""

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHOW_PATCHES = ("rl-realtime", "rl-idle-wait", "rl-paced-render")
FAKE_DOCKER = """#!/bin/sh
printf '%s\\n' "$*" >> "$DOCKER_LOG"
case "$*" in
  *zeroad.patches.sha256*) [ "$1 $2" = "image inspect" ] && echo "$INSPECT_SHA" ;;
  *zeroad.patches*) [ "$1 $2" = "image inspect" ] && echo "$INSPECT_NAMES" ;;
esac
exit 0
"""
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def _sha(names: tuple[str, ...]) -> str:
    data = b"".join((ROOT / "patches" / f"{name}.patch").read_bytes() for name in names)
    return hashlib.sha256(data).hexdigest()[:16]


def _build(tmp_path: Path, target: str, *, names: str, sha: str, **env: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "docker.log"
    log.write_text("")
    environ = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_LOG": str(log),
        "INSPECT_NAMES": names,
        "INSPECT_SHA": sha,
        **env,
    }
    done = subprocess.run(
        ["bash", str(ROOT / "scripts/build-engine.sh"), target], env=environ, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return [line for line in log.read_text().splitlines() if line.startswith("build ")]


@needs_bash
@pytest.mark.parametrize("sha", ["", "0123456789abcdef"])
def test_build_engine_rebuilds_the_show_image_when_a_patch_changes(tmp_path, sha):
    builds = _build(tmp_path, "show", names=" ".join(SHOW_PATCHES), sha=sha)
    [show] = [b for b in builds if "-f engine/Dockerfile.show" in b]
    assert f"--label zeroad.patches.sha256={_sha(SHOW_PATCHES)}" in show


@needs_bash
def test_build_engine_skips_the_show_image_when_names_and_hash_match(tmp_path):
    builds = _build(tmp_path, "show", names=" ".join(SHOW_PATCHES), sha=_sha(SHOW_PATCHES))
    assert not [b for b in builds if "Dockerfile.show" in b]


@needs_bash
def test_build_engine_rebuilds_the_show_image_when_asked_even_if_nothing_changed(tmp_path):
    builds = _build(tmp_path, "show", names=" ".join(SHOW_PATCHES), sha=_sha(SHOW_PATCHES), REBUILD="1")
    assert [b for b in builds if "Dockerfile.show" in b]


@needs_bash
def test_build_engine_hashes_the_rt_patches_for_the_rt_image(tmp_path):
    rt = ("rl-realtime", "rl-idle-wait")
    [build] = [b for b in _build(tmp_path, "rt", names="rl-realtime", sha="") if "Dockerfile.rt" in b]
    assert f"--label zeroad.patches.sha256={_sha(rt)}" in build
    assert not [b for b in _build(tmp_path, "rt", names=" ".join(rt), sha=_sha(rt)) if "Dockerfile.rt" in b]
