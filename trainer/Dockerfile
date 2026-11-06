# zeroad-trainer: the pilot's trainer (train.py) and v2's persistent learner (learner.py) on vLLM's own image, so
# torch, CUDA and transformers match the server that loads its adapters. Built on the pilot box by
# box/pilot-bootstrap.sh:
#   docker build -t zeroad-trainer --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.11.0 trainer/
ARG VLLM_IMAGE=vllm/vllm-openai:v0.11.0
FROM ${VLLM_IMAGE}
RUN pip install --no-cache-dir "peft==0.17.1"
COPY train.py average.py learner.py /app/
ENTRYPOINT []
CMD ["python3", "/app/train.py", "--help"]
