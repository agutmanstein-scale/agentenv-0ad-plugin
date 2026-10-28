# zeroad-luna: GPT-6 Luna alone playing one 0 A.D. seat through the Decisions API (agent.py), as an A2A agent.
# Build from the repo root: agent-env zeroad setup --agents, or
#   docker build -t a2a-agent-zeroad-luna -f agents/zeroad-luna/Dockerfile .
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install "agentenv-framework-protocol[agent]==0.1.370" "mcp==1.30.0" "httpx==0.28.1" "pydantic>=2.7,<3" \
 && useradd --uid 10001 --create-home agent
# The brains live in the agentenv_zeroad package (brains.py, llm.py, decisions.py, spend.py); none of its env-side
# dependencies are needed here.
COPY pyproject.toml README.md /tmp/pkg/
COPY src /tmp/pkg/src
RUN pip install --no-deps /tmp/pkg && rm -rf /tmp/pkg
COPY agents/zeroad-luna/agent.py /app/agent.py
USER agent
EXPOSE 8000
CMD ["python", "/app/agent.py"]
