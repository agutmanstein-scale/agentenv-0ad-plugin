"""Lockstep training games (post-training v2): many games at once on a pool of engines, in one process per worker.

A game (`spec.GameSpec`) is played by `game.play_game`: a `LockstepMatch` steps its engine as fast as it answers, each
LLM seat is a `Commander` run as a library (`brains.CommanderSeat` over an `InProcessSeatLink`), and the match writes
the same results.json the pilot reads, plus the sampler's per-token log-probs and the commands that were applied to
the engine, turn by turn. `runner` plays a list of specs on an `EnginePool`, in several worker processes if asked.
"""
