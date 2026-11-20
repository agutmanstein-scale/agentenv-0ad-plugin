"""patches/rl-paced-render.patch against the pristine 0.28.0 sources, on top of the two patches under it.

CI cannot build the engine, so this is what catches a patch that has drifted from the real source; the compile and
tests/engine/test_paced_engine.py run on the image build (scripts/build-engine.sh show).
"""

import re
import shutil

import pytest

from tests.test_engine_idle_patch import PATCHES, ROOT, _patch, _tree

needs_patch = pytest.mark.skipif(shutil.which("patch") is None, reason="`patch` is not installed")
PACED = "rl-paced-render.patch"


def _paced_tree(tmp_path):
    tree = _tree(tmp_path)
    for name in PATCHES:
        assert _patch(tree, name).returncode == 0
    return tree


@needs_patch
def test_the_paced_patch_applies_on_top_of_the_two_patches_with_no_fuzz_or_offset(tmp_path):
    tree = _paced_tree(tmp_path)
    for flags in (("--dry-run",), ()):
        done = _patch(tree, PACED, *flags)
        assert done.returncode == 0, f"{done.stdout}{done.stderr}"
        assert "fuzz" not in done.stdout and not re.search(r"offset [1-9]", done.stdout), done.stdout


@needs_patch
def test_the_paced_patch_does_not_apply_without_the_patches_under_it(tmp_path):
    assert _patch(_tree(tmp_path), PACED, "--dry-run").returncode != 0


@needs_patch
def test_the_patched_frame_loop_runs_one_turn_per_step_and_replies_after_it(tmp_path):
    tree = _paced_tree(tmp_path)
    assert _patch(tree, PACED).returncode == 0
    cpp = (tree / "source/rlinterface/RLInterface.cpp").read_text()
    assert 'g_CmdLineArgs.Has("rl-paced")' in cpp
    update = cpp.split("void Interface::PacedUpdate")[1].split("\n}\n")[0]
    assert "GetSimulation2()->Interpolate(" in update, "the idle frame must keep interpolating"
    no_clock = update.replace("Interpolate(", "").replace("GetTimeManager().Update(", "")
    assert "Update(" not in no_clock, "the frame loop must not feed sim time (only the renderer's clock moves)"
    step = cpp.split("case GameMessageType::Commands")[1].split("case GameMessageType::Evaluate")[0]
    assert "GetTurnManager()->Update(deltaSimTime, maxTurns)" in step.split("else if (m_Paced)")[1]
    assert step.index("else if (m_Paced)") < step.index("g_Game->Update(deltaRealTime);")
    main = (tree / "source/main.cpp").read_text()
    assert main.index("rlInterface->PacedUpdate(") < main.index("rlInterface->AdvancesInRealTime()")


@needs_patch
def test_the_paced_step_runs_the_turn_without_advancing_the_renderer_clock(tmp_path):
    tree = _paced_tree(tmp_path)
    assert _patch(tree, PACED).returncode == 0
    cpp = (tree / "source/rlinterface/RLInterface.cpp").read_text()
    paced = cpp.split("else if (m_Paced)")[1].split("else if (!m_RealTime)")[0]
    assert "GetTurnManager()->Update(" in paced
    assert '"SimulationUpdate"' in paced and "MakeDirty()" in paced
    assert "g_Game->Update(" not in paced and "GetTimeManager()" not in paced


@needs_patch
def test_paced_update_advances_the_renderer_clock_with_the_blend(tmp_path):
    tree = _paced_tree(tmp_path)
    assert _patch(tree, PACED).returncode == 0
    cpp = (tree / "source/rlinterface/RLInterface.cpp").read_text()
    update = cpp.split("void Interface::PacedUpdate")[1].split("\n}\n")[0]
    assert "CRenderer::IsInitialised()" in update
    clock = update.split("GetTimeManager().Update(")[1]
    assert "m_PacedBlend - " in clock.split(";")[0]


def test_the_show_image_builds_all_three_patches_in_order_and_says_so_in_its_label():
    dockerfile = (ROOT / "engine/Dockerfile.show").read_text()
    assert "COPY rl-realtime.patch rl-idle-wait.patch rl-paced-render.patch /src/" in dockerfile
    order = [dockerfile.index(f"patch -p1 -F0 < /src/{name}") for name in (*PATCHES, PACED)]
    assert order == sorted(order)
    assert 'LABEL zeroad.patches="rl-realtime rl-idle-wait rl-paced-render"' in dockerfile
    assert "zeroad-engine:0.28-show" in (ROOT / "scripts/build-engine.sh").read_text()
