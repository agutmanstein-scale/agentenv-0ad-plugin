# The live env image, mcp-server-zeroad-live: the env image built on the stream engine (zeroad-engine:0.28-rt, see
# engine/Dockerfile.rt), plus Chrome for the overlay and the director mod. Xvfb and ffmpeg come with the engine image.
# A lobby with `render: true` renders, frames and records its match in here (src/agentenv_zeroad/live.py).
# Build with `agent-env zeroad setup --live`; by hand, from the repo root:
#   scripts/build-engine.sh rt
#   docker build --build-arg GAME_ENV=<archive> --build-arg ENGINE_IMAGE=zeroad-engine:0.28-rt -t mcp-server-zeroad-rt .
#   docker build -f streamer/Dockerfile -t mcp-server-zeroad-live .
ARG ENV_IMAGE=mcp-server-zeroad-rt
FROM ${ENV_IMAGE}
USER root
RUN apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends fonts-dejavu-core fonts-noto-core xdotool \
 && curl -fsSL -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends /tmp/chrome.deb \
 && rm -f /tmp/chrome.deb && rm -rf /var/lib/apt/lists/* \
 && HOME=/root google-chrome --version \
 && test ! -e /home/ubuntu/.local && test ! -e /home/ubuntu/.config
COPY mod/zeroad-director /usr/share/games/0ad/mods/zeroad-director
USER ubuntu
ENV ZEROAD_LIVE=1
