# E3: what it showed, and why E4 is different

E3 trained from v0 on the game outcome (conquest, or the net-worth margin at the 12-minute cap) with same-start credit, for
22 updates, and was judged on 240 paired held-out Petra-2 starts (`live/s2/e3.toml`, `[e3]` in `live/s2/sequence.toml`,
which stay as they were run). It is accepted in `[sequence] accept_failed`: its paired held-out gate passed and its in-run
slope gate missed narrowly.

## What happened

- The final adapter (version 22) led on net worth at the cap in 73.3% of the 240 games, against 54.2% for v0. The paired
  Wilcoxon test on the margin gives p 1.9e-5, with a median paired difference of +656.
- The in-run reward slope was +0.0076 per policy version, 95% interval -0.0026 to +0.0179 (starts resampled): the gate
  needs the lower bound above 0, so it missed.
- 22 learner updates and 0 tripwire trips.
- The check beforehand: v0 against v0 on the same starts gave p 0.40 (the paired test is quiet when nothing differs); v0 led
  Petra-3 at the cap in 17% of games; the lean learner took 0.87 s per decision against 1.14 s for E2's flags.
- At the end of the games the medians differ in the same direction as the margin: final net worth 11,636 against 9,235 for
  v0. The in-game military score fell, mean 59 against 104, and the median enemy units killed fell from 5 to 3.
- Every game ended at the 12-minute cap. There were no conquests.
- Spend $685.35 over 31.1 box hours.

## So E4

E3's policy learned to avoid fights and keep its value, which wins a 12-minute cap and wins nothing else. Two things change.
The games are 30 minutes, so Petra's attacks land and a policy that avoids fights has to survive them. And the reward is
`cap_weight` 0.5 times the cap term, so any lead at the cap is worth at most half a conquest. E4 starts from E3's final
adapter and is judged against it on the same starts. A check first plays v0 and E3's adapter on the long games and measures
the margin scale for the reward. See "E4: the long-game check and E4" in `docs/s2-box-day.md`.
