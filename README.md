# Champions League Predictor

Predicts football match results (home win / draw / away win) using XGBoost.
Trained on European club football 2013–2026 from Transfermarkt data.

## Result

**52.9%** accuracy on the 2025/26 Champions League season (189 games).

Guessing "home win" every time gets **49.7%**, so it's a small real edge. Not amazing. Football is hard.

## How it works

The model looks at:

- **Elo rating** — goes up when you win, down when you lose
- **Recent form** — points, goals scored and conceded over the last 5 and 10 games
- **Squad value** — market value of the players who actually started
- **Lineup stability** — how much the starting 11 changed from last game
- **Manager Elo** — same as club Elo but for managers
- **Rest days**, formations, competition, round

The main rule everywhere: a game can only use information from **before** it was played.

## Bugs I found along the way

These are the interesting part, honestly.

**1. My best feature was cheating.** League table position was ranked way above everything else. Two problems: it was the *end of season* position, so September games were being predicted with May information. And it's missing for every Champions League game anyway. Removing it dropped accuracy from 56% to 49% — which means 56% was never real.

**2. Scaled vs unscaled data.** I trained on data squashed to 0–1, then tested on raw numbers (Elo ~1800, squad values in the millions). The model predicted away wins correctly **zero times** out of 61. No error message, just confident nonsense. Fixed by transforming the test set in the same place as the training set.

**3. `season` as a feature.** Training was old seasons, testing was new ones, so the model saw season values it had never encountered. Dropped it.

## The 2026 Final

PSG vs Arsenal. The model can't decide:

```
run 1:  home 0.498 | draw 0.027 | away 0.475   -> PSG
run 2:  home 0.466 | draw 0.027 | away 0.507   -> Arsenal
```

Same code, same seed, different runs. XGBoost adds up gradients across threads in whatever
order they finish, and floating point addition isn't exactly associative, so the numbers wobble
in the last decimals. Usually you'd never notice. Here the two teams are 0.03 apart, so it's
enough to flip the answer.

I left it as is, because honestly the instability *is* the result. The game finished 1-1 and went
to penalties (PSG won 4-3). A model that can't separate the two teams is telling you the truth
about that match.

It does get confident when it should:

| Home | Away | Actual | P(home) | P(draw) | P(away) |
|---|---|---|---|---|---|
| Barcelona | Sevilla | home_win | **0.676** | 0.249 | 0.075 |
| Sevilla | Barcelona | away_win | 0.108 | 0.321 | **0.571** |
| Barcelona | Sevilla | home_win | **0.702** | 0.244 | 0.054 |

So it's not just hedging everything at 50/50.

## Known problems

- Draws are predicted badly (F1 ~0.15). They're rare and genuinely unpredictable.
- Two-legged knockout features I added had basically no effect.
- Can't predict penalty shootouts (obviously).

## Running it

```
cl_predictor_2026_eda.ipynb       <- run this first (builds the features)
cl_predictor_2026_machine.ipynb   <- then this (trains and tests)
```

Data from [Transfermarkt on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) — put the CSVs in `Data_Sources/`.

Install with `pip install -r requirements.txt`
