"""Session hygiene: before any test starts an engine, remove engine containers left by runs that were killed.

Each `zeroad-engine-*` container carries its owner's pid and start time (engine.owner_labels), so only containers
whose owner process is gone are removed; containers of a test run still going, and unlabeled ones, are left alone.
"""

import asyncio
import shutil

import pytest

from agentenv_zeroad.engine import reap_orphans


@pytest.fixture(scope="session", autouse=True)
def reap_orphaned_engines():
    if shutil.which("docker"):
        asyncio.run(reap_orphans())
    yield


@pytest.fixture
def anyio_backend():
    return "asyncio"
