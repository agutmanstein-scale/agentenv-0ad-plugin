# E4: what it showed, and why Run 1 follows

E4 trained from the base model on 30-minute games against Petra-2 and Petra-3, on a reward where a conquest outweighs a lead
at the cap (`cap_weight` 0.5), and was judged on 100 paired held-out starts (`live/s2/e4.toml`, `[e4]` in
`live/s2/sequence.toml`, which stay as they were run). It is accepted in `[sequence] accept_failed`: it did not beat base.

## The check

`e4_check` played v0 and E3's final adapter (version 22) on 100 identical 30-minute starts.

- v0 won 26 games: 5 by conquest and 21 ahead at the cap. It lost 34 by conquest.
- E3's adapter won 13 (4 by conquest, 9 ahead at the cap) and lost 52 by conquest.

So E3's gain at the 12-minute cap did not carry to long games, and E4 starts from the base model.

## The runs

- The first E4, from E3's adapter (root `e4`), was stopped 12 minutes in once the check showed that.
- E4 from base (root `e4b`) made 12 learner updates in 3.5 hours of training, with the learner busy 49% of the time. The
  default `decisions` range (8 to 12 per seat-game) starved the learner on long games.
- End panels, 100 games each on the check's starts: base won 25 (7 by conquest, 18 ahead at the cap) and lost 33; the E4 adapter
  won 21 (9 by conquest, 12 ahead at the cap) and lost 46. The paired R test gave p 0.94.
- `margin_scale` measured by the check: 18,084.
- Spend $749.71 over 37.2 box hours.

## So Run 1

E4 trained too little (12 updates) and its policy lost more games by conquest than base. Run 1 is the run that trains Qwen to
win, and it changes what starved E4: `decisions = [16, 24]` per seat-game, 8 hours of training instead of 3.5, and the
documented reward (`cap_weight` 1, so a lead at the cap counts as a win) at the check's measured margin scale. It is judged on
384 held-out starts that pair with the baselines' own. See "Run 1: train to win" in `docs/s2-box-day.md`.
