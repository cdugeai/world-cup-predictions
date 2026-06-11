## Setup

Parse html files with odds to csv:

```sh
uv run python3 parser_odds_unibet.py
uv run python3 parser_odds_checker.py
uv run python3 parser_odds_mpp.py
```

## Compute result predictions

To compute match predictions, run:

```sh
uv run python3 main.py
```

The predictions are available in [predictions.csv](data/out/predictions.csv).

## Betting advices

To get advices about on the edges and when to bet against the odds to have positive Expected value:

```sh
uv run python3 betting_optimizer.py
```

The recommendations are available in [betting_recommendations.csv](data/out/betting_recommendations.csv).
A cleaner version of this file is available in [betting_recommendations_clean.csv](data/out/betting_recommendations_clean.csv).
