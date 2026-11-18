# E2: what it showed, and why E3 is different

E2 was the positive control: reward the policy for its standing army at game minute 10, and see whether the loop learns.
The last two runs, E2c and E2d, are the record (`live/s2/e2.toml` and the `[e2]` table in `live/s2/sequence.toml` stay
as they were run). E2 is not run again.

## What happened

- The loop learns. E2c and E2d both climbed by about 0.8 to 1.0 soldiers at minute 10 per 10 updates, and the intervals
  exclude 0. They ended in the same place: the last blocks of E2d minus E2c came to +0.13 soldiers, interval [-0.66, 0.91].
- The old gate could not see it. It compared the last 40 policy games with a frozen-v0 arm of 60 games. The frozen arm was
  played in the first 0.27 hours of the run and the policy games it was compared with at 2.9 to 3.1 hours, so the two arms
  differ by 2.8 hours of box time as well as by the policy. E2d's last 40 games were also only 10 starts. For a policy that
  really improved, that gate passes 9 to 33% of the time.
- The gain was one lever: earlier, larger soldier batches from the bank. Of the 3.1 extra soldiers, 1.3 were lost under
  the standing attack order.
- It did not move the real objective. The share of 12-minute games in which the policy leads at the cap was 71.5% for v0
  and 73.5% and 69.6% for the two final policies.
- Updates were signal-limited. The KL stop and the gradient clip were 13 to 19 times above what the updates used. An update
  saw about 26 games from about 8 distinct starts in E2d, and 36 to 40% of the rows were never trained on.
- Per-turn credit (GAE, lambda 0.8) put 72% of its weight on minutes 6 to 10. It was not the bottleneck.

## So E3

E3 trains on the outcome the games already record (conquest, or the net-worth margin at the 12-minute cap) and judges
the result with a paired end-of-run panel instead of a comparison across time. Credit's baseline is the mean of the same
start's other games, and the learner takes whole same-start groups, so an update sees more distinct starts and less
noise. The end panels play v0 and the final adapter on identical starts, so the margins pair game by game. See "E3 and
its check" in `docs/s2-box-day.md`.
