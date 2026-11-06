"""The engine patch that stops an idle nonvisual engine spinning a core (patches/rl-idle-wait.patch)."""

import re
from pathlib import Path

from agentenv_zeroad.pool import TRAIN_IMAGE

ROOT = Path(__file__).resolve().parent.parent
IDLE = (ROOT / "patches/rl-idle-wait.patch").read_text()


def hunks(patch: str) -> dict[str, str]:
    """The added lines of each file the patch changes."""
    out: dict[str, list[str]] = {}
    current = ""
    for line in patch.splitlines():
        if line.startswith("+++ "):
            current = line[6:]
            out[current] = []
        elif current and line.startswith("+") and not line.startswith("+++"):
            out[current].append(line[1:])
    return {name: "\n".join(lines) for name, lines in out.items()}


def test_the_idle_patch_touches_the_rl_interface_and_the_nonvisual_loop_only():
    added = hunks(IDLE)
    assert set(added) == {"source/main.cpp", "source/rlinterface/RLInterface.cpp", "source/rlinterface/RLInterface.h"}
    assert "WaitForMessage(std::chrono::milliseconds{2})" in added["source/main.cpp"]
    assert "Frame(" not in added["source/main.cpp"]
    main_hunks = [h for h in re.split(r"^@@.*$", IDLE.split("+++ b/source/main.cpp")[1].split("--- a/")[0], flags=re.M)]
    assert len(main_hunks) == 2  # one hunk, in the `else if (rlInterface)` branch below `if (isVisual)`
    assert "else if(rlInterface)" in main_hunks[1] and "if (isVisual)" in main_hunks[1]


def test_posting_a_message_wakes_the_waiting_main_thread_and_the_wait_has_a_timeout():
    cpp = hunks(IDLE)["source/rlinterface/RLInterface.cpp"]
    assert re.search(r"m_Posted = true;.*m_Wake\.notify_one\(\);", cpp, re.S)
    assert "m_Wake.wait_for(wakeLock, timeout" in cpp
    assert "std::atomic<bool> m_Posted" in hunks(IDLE)["source/rlinterface/RLInterface.h"]


def test_the_rt_image_builds_both_patches_in_order_and_the_build_script_says_who_it_is_for():
    dockerfile = (ROOT / "engine/Dockerfile.rt").read_text()
    assert "COPY rl-realtime.patch rl-idle-wait.patch /src/" in dockerfile
    assert dockerfile.index("patch -p1 < /src/rl-realtime.patch") < dockerfile.index(
        "patch -p1 < /src/rl-idle-wait.patch"
    )
    assert "rl-idle-wait" in (ROOT / "scripts/build-engine.sh").read_text()
    assert TRAIN_IMAGE == "zeroad-engine:0.28-rt"
