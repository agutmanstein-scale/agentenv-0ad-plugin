"""The box's own power-off at the end of a run. The box scripts run with PATH=/usr/local/bin:/usr/bin:/bin, which
lacks shutdown's /usr/sbin, so a bare `shutdown` was not found and the box stayed up; the command is looked up in the
sbin directories, with `systemctl poweroff` as the fallback."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable

log = logging.getLogger(__name__)

SEARCH_PATH = "/usr/sbin:/sbin:/usr/bin:/bin"
COMMANDS = (("shutdown", "-h", "now"), ("systemctl", "poweroff"))


def power_off(run: Callable[[list[str]], int] = lambda argv: subprocess.run(argv, check=False).returncode) -> bool:
    for name, *args in COMMANDS:
        exe = shutil.which(name, path=SEARCH_PATH)
        if exe is not None and run([exe, *args]) == 0:
            return True
    log.error("could not power off: neither shutdown nor systemctl poweroff worked")
    return False
