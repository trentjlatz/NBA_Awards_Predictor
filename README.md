# NBA Awards Predictor

A multi-label classification model that predicts which awards, if any, an NBA player will receive in a given season based on their performance statistics.

## Problem

Can player performance metrics be used to predict NBA award outcomes? We frame this as a multi-label classification problem across five award categories: MVP vote, All-Star, All-NBA, DPOY vote, and All-Defensive — a player can be predicted to receive more than one.

## Data

Player-season performance data (basic + advanced stats) across five seasons, 2021–2025, filtered to players with meaningful playing time (40+ games, 10+ minutes per game). Awards were parsed out of a raw awards-text column into five binary target labels.

## Approach

- **Model:** Random Forest Classifier wrapped in a `ClassifierChain`, so predictions for one award (e.g. an MVP vote) inform the model's predictions for related awards (e.g. All-NBA) rather than treating each label independently.
- **Class imbalance:** award-winning seasons are rare relative to the full player pool, so we applied class weighting to keep the model from defaulting to "no award" predictions.
- **Feature selection:** after an initial pass with the full feature set, we used Random Forest feature importances to drop low-signal features and refit on a leaner, more interpretable set.
- **Threshold tuning:** rather than using a default 0.5 decision threshold, we swept a range of thresholds against validation data and selected the one that maximized F1 score, since the "no award" class dominates the data and a naive threshold underpredicts awards.

## Results

- **0.68 micro-averaged F1** on unseen test data across all five award categories
- **0.81 F1** on All-Star prediction specifically, our strongest-performing category
- As expected going in, defensive awards (DPOY, All-Defensive) were harder to predict than offensive/overall awards, since defensive impact is less fully captured by standard box-score metrics

## Try it

The `make_prediction()` function at the end of the notebook takes a player's season stats and returns the awards the model predicts for them — the notebook includes example predictions for Stephen Curry (2023) and Victor Wembanyama (2025) checked against their actual award outcomes that season.

## Tech Stack

Python, pandas, scikit-learn (`RandomForestClassifier`, `ClassifierChain`), matplotlib

## References

- [scikit-learn: Classifier Chain example](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html)
- [scikit-learn: ClassifierChain docs](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)
- [scikit-learn: RandomForestClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Contributors

Trent Latz, Henry Blake
