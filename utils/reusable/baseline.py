"""
baseline.py
-----------
Score the trivial predictor every real model has to beat.

Run once, centrally, before the models fan out: the workers all need the same
floor, and five workers each computing it would be duplicated work that can
disagree. Scored out-of-fold on the training set with the run's own
cross-validator, so the number sits on the same scale as the model scores the
judge will see.

    run_baseline(train, target, task_type, plan, config) -> BaselineResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from models import BaselineResult, Configs, SplitPlan
from utils.reusable.metrics import PROBABILITY_METRICS, make_cv, score

_DUMMY_STRATEGY = {
    "mean": "mean",
    "median": "median",
    "most_frequent": "most_frequent",
    "prior": "prior",
}


def run_baseline(
    train: pd.DataFrame,
    target: str,
    task_type: str,
    plan: SplitPlan,
    config: Configs,
) -> BaselineResult:
    strategy = config.baseline.strategy

    if not config.baseline.applicable or strategy == "none":
        return BaselineResult(strategy=strategy, applicable=False, note="no baseline for this task")

    y = train[target]
    X = np.zeros((len(train), 1))  # dummy predictors ignore the features
    groups = train[plan.group_column] if plan.cv_strategy == "group_kfold" else None

    try:
        folds = list(make_cv(plan, len(train)).split(X, y, groups))
    except ValueError as error:
        return BaselineResult(strategy=strategy, applicable=False, note=f"cv failed: {error}")

    is_classification = task_type.endswith("classification")
    predictions = np.empty(len(train), dtype=object)
    probabilities = np.zeros((len(train), y.nunique())) if is_classification else None

    for train_positions, test_positions in folds:
        y_fit = y.iloc[train_positions]

        if strategy in ("naive_last", "seasonal_naive"):
            # the last value the fold could have seen; seasonal periods are not
            # in the profile, so a seasonal request degrades to the same thing
            constant = y_fit.iloc[-1]
            predictions[test_positions] = constant
            continue

        if strategy not in _DUMMY_STRATEGY:
            return BaselineResult(strategy=strategy, applicable=False, note=f"unknown strategy: {strategy}")

        dummy = (DummyClassifier if is_classification else DummyRegressor)(
            strategy=_DUMMY_STRATEGY[strategy]
        )
        dummy.fit(X[train_positions], y_fit)
        predictions[test_positions] = dummy.predict(X[test_positions])
        if is_classification:
            probabilities[test_positions] = dummy.predict_proba(X[test_positions])

    # TimeSeriesSplit never predicts the first fold, so score only what was covered
    covered = np.array([p is not None for p in predictions])
    y_true = y[covered]
    y_pred = pd.Series(predictions[covered]).astype(y.dtype)
    y_proba = probabilities[covered] if is_classification else None

    scores = {}
    for metric in config.eval_matrics:
        if metric in PROBABILITY_METRICS and not is_classification:
            continue
        try:
            scores[metric] = score(metric, y_true, y_pred, y_proba)
        except Exception as error:
            scores[metric] = float("nan")
            print(f"baseline: {metric} failed -> {error}")

    note = f"{strategy} scored out-of-fold on {covered.sum()} of {len(train)} training rows"
    return BaselineResult(strategy=strategy, applicable=True, cv_scores=scores, note=note)
