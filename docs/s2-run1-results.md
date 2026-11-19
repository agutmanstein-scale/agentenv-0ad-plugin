# Run 1: train to win, passed

Run 1 trained base Qwen3-8B (rank-32 LoRA on all seven projections, about 43.7M trainable parameters) for 8 hours on the
documented objective and mix (`live/s2/run1.toml`, `[run1]` in `live/s2/sequence.toml`), from 7:33 pm ET on 18 Nov. Its end
panels played the baselines' 384 `heldout` starts with base Qwen (`end-base`) and with the final adapter, version 50
(`end-final`): the same seed, positions, civs and opponent for each pair, 30-minute cap. It finished at 7:41 am ET on 19 Nov
and the box powered itself off.

Under the plan's rule (GO if the test and the trend pass with no recent trip) Run 1 is a GO.

## Gates

| Gate | Result | Needed |
|---|---|---|
| Final beats base on the paired documented reward (Wilcoxon, one-sided) | p < 1e-15 (z 10.2, 384 pairs; median +0.164, mean +0.615) | p < 0.05 |
| Learner updates | 50 | >= 5 |
| Tripwire trips | 0 | 0 |
| Ended normally (complete) | yes | yes |

The in-run trend, the gate E3 missed, also clears: +0.010 reward per version, 95% interval +0.0056 to +0.0146, over 1,462
trained games and versions 0 to 49.

## End panels

Won = a conquest, or ahead in net worth at 30:00.

| Opponent | Games | Base won | Final won | Base conquests won / lost | Final conquests won / lost |
|---|---|---|---|---|---|
| Petra-1 | 120 | 55.0% | 85.8% | 20.0% / 0.8% | 38.3% / 0.0% |
| Petra-2 | 120 | 40.0% | 78.3% | 6.7% / 20.0% | 37.5% / 3.3% |
| Petra-3 | 96 | 12.5% | 45.8% | 1.0% / 61.5% | 17.7% / 18.8% |
| Scripted bot | 24 | 25.0% | 87.5% | 8.3% / 50.0% | 45.8% / 0.0% |
| Frozen base | 24 | 54.2% | 66.7% | 33.3% / 20.8% | 37.5% / 25.0% |
| All | 384 | 37.8% | 72.4% | 11.2% / 26.3% | 33.3% / 7.3% |

The gain in games won is +34.6 points (95% interval +29.2 to +40.1). Conquest wins went from 43 to 128 and conquest losses
from 101 to 28, so the gain is in real wins, not only a lead at the cap. The median net-worth margin at the cap rose by
6,379. Base's rates match the baselines' own on these starts (Petra-1 54%, Petra-2 37%, Petra-3 8%).

## Milestones

A milestone passes when the 97.5% lower bound of the win rate clears it.

| Milestone | Base won (lower bound) | Final won (lower bound) |
|---|---|---|
| Petra-1 >= 50%, >= 65% | 55.0% (46.1%): neither | 85.8% (78.5%): both |
| Petra-2 >= 15%, 30%, 45%, 50% | 40.0% (31.7%): 15% and 30% | 78.3% (70.1%): all four |
| Petra-3 >= 15% | 12.5% (7.3%): no | 45.8% (36.2%): yes |

All seven pass with a Bonferroni correction too (the lowest corrected bound is 32.9%, Petra-3's).

## Training

- Learning rate 3e-5, 512 decisions per update, 16 to 24 decisions trained per seat-game: 50 updates in 8 hours, against
  E4's 12 in 3.5.
- KL to base 0.05 at the end (peak 0.11); entropy down 11% (0.0130 to 0.0115); no trips or rollbacks.
- 2,230 games started, 1,462 trained, 768 end-panel games.

## Limits

One run with one seed, on the training map (mainland): civ pairs and seeds were held out, but no map was. Against Petra-3 the
adapter still trails at the cap on the median (-10,858).

## Spend

The box's counter was reset at 10:25 pm ET on 18 Nov (`live/budget-reset.sh`; $782.29 before). Run 1 cost about $129 of box
time (12.3 h at $10.49/h) and no model API; $98 of it falls after the reset.

## Next

Run 2 per the plan: 24 hours on one box or two replicate runs, an adaptive mix, frozen snapshots of the policy (at most 20%),
about 30 frontier games a day. Worth adding: a held-out map, a replicate seed, more weight on Petra-3 and above.
