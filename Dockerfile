# The 0 A.D. env: the engine image (zeroad-engine:0.28, 0 A.D. 0.28 from ppa:wfg/0ad on ubuntu:24.04) with the env's
# MCP server beside it, so the server starts pyrogenesis from PATH (the engine's `local` launcher).
# Build with `agent-env zeroad setup`, which builds the engine image first and passes GAME_ENV; by hand:
#   scripts/build-engine.sh && docker build --build-arg GAME_ENV=<agentenv-game-env archive URL> -t mcp-server-zeroad .
ARG ENGINE_IMAGE=zeroad-engine:0.28
FROM ${ENGINE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/games:$PATH
# The server's dependencies from pyproject.toml, without agentenv-framework (the control plane, which the CLI plugin
# and the task steps use), and with the protocol pinned to the version the supported agentenv-framework pins.
RUN pip install "agentenv-framework-protocol==0.1.370" "mcp>=1.25,<2" "pydantic>=2.7,<3" "httpx>=0.27,<1"
# agentenv-game-env (the lobby and the match), its env side alone, from the archive pyproject.toml pins.
ARG GAME_ENV
RUN test -n "$GAME_ENV" || { echo "build with --build-arg GAME_ENV=<agentenv-game-env archive URL>" >&2; exit 1; } \
 && pip install --no-deps "$GAME_ENV"
COPY pyproject.toml README.md /tmp/pkg/
COPY src /tmp/pkg/src
RUN pip install --no-deps /tmp/pkg && rm -rf /tmp/pkg \
 && command -v pyrogenesis >/dev/null
# 0 A.D. refuses to run as root; ubuntu (uid 1000) is the image's own user.
USER ubuntu
ENV HOME=/home/ubuntu ZEROAD_ENGINE=local ZEROAD_OUT=/home/ubuntu/matches
EXPOSE 18765
CMD ["python", "-m", "agentenv_zeroad.server"]
