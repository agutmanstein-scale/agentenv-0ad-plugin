"""The post-training pilot: an open model (Qwen, served by vLLM on the box) plays 30-minute 1v1 games and is trained
on them with RL, round after round, against an opponent ladder that gets harder as it improves.

One round (`run.py`): play the round's training games (opponents picked by `ladder.py`) and the fixed probe panel at
once, each its own agent-env task (`games.py`); read every game's results.json (`scoring.py`): the outcome, the score
margin, and every decision the policy made with the token ids it saw and sampled; judge each decision against what a
value estimate expected from its position; train a LoRA adapter on them (`trainer/train.py`, in its own container on
the box's other GPUs); load the adapter into vLLM; rate every version against every opponent (`ratings.py`).
"""
