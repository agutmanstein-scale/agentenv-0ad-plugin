# Post-training pilot

Qwen3-8B, served by vLLM on the box, plays 30-minute 1v1 games at 2x speed and is trained on them with RL, round
after round. Every script reads `LIVE_ENV=live/pilot.env` (git-ignored: `live.env`'s account, network and bucket, with
the pilot's own box, `zeroad-pilot`, a g6e.12xlarge at $10.49/h). Each script that changes anything prints its plan
and stops unless given `--yes`.

## One round

1. **Play.** 16 training games against opponents picked where the policy wins about half the time (Petra 1 to 5 and
   snapshots of earlier versions; `pilot/ladder.py`), plus the probe panel: Petra 1 to 5, the previous version and
   Sonnet, on fixed seeds no training game uses. The policy samples at temperature 1 in training games (on-policy)
   and 0.7 in probes; it calls every 25 game-seconds. Training games log every decision's exact prompt and
   completion token ids (`ZEROAD_TURN_LOG`), and every game's results.json keeps a timeline of both players' scores.
2. **Score.** Two ridge regressions over the observer's view of the position predict how the score margin moves over
   the next 2 minutes and to the end. A decision's advantage is how much better than predicted it did, less a
   penalty for rejected orders (`pilot/scoring.py`). The estimates see both players unfogged; they only score.
3. **Train.** One pass of policy gradient with a KL penalty to base Qwen over 384 of the round's decisions, on a LoRA
   adapter (rank 32), on GPUs 2 and 3 (`trainer/train.py`, in the `zeroad-trainer` image).
4. **Swap.** The adapter loads into vLLM as `local/policy-vN`; every second version joins the training pool.
5. **Rate.** A Bradley-Terry fit over every game rates each version and each opponent, with Petra medium at 1000.
   `progress.md` in the pilot's folder has the ratings and every round's games.

## Run it

1. `LIVE_ENV=live/pilot.env live/launch.sh --yes` (prints the instance id; it powers off after 11 hours).
2. `LIVE_ENV=live/pilot.env live/bootstrap.sh ID --pilot --yes` (about 25 minutes: the env image, vLLM and the model,
   the trainer image; it ends with one tool call through vLLM that must return token ids).
3. For the Sonnet probe: `live/put-secret.sh litellm-key --yes` (the key is read without echo and kept in SSM).
4. `LIVE_ENV=live/pilot.env live/pilot.sh ID start --baseline --yes`: one round of the base model, nothing trained
   (`--speed 4` tries the engine faster).
5. `LIVE_ENV=live/pilot.env live/pilot.sh ID start --yes`: the round loop (14 rounds by default). It asks the budget
   guard before every round and stops before one that would pass `LIVE_BUDGET_USD`.
6. `LIVE_ENV=live/pilot.env live/pilot.sh ID status` any time; `live/pilot.sh ID stop --yes` ends it after the round
   in play (`--now` at once).
7. `LIVE_ENV=live/pilot.env live/fetch.sh ID --only pilot --yes`, then `LIVE_ENV=live/pilot.env live/teardown.sh ID
   --yes`.
