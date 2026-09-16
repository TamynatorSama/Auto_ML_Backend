"""
baseline.py
-----------
Score the trivial predictor every real model has to beat.

Run once, centrally, before the models fan out: the workers all need the same
floor, and five workers each computing it would be duplicated work that can
disagree. Scored out-of-fold on the run's frozen folds and with the same
scoring function the evaluator uses, so the floor and the model scores are the
same kind of number: pooled over every training row a fold predicted.

    run_baseline(train, target, task_type, plan, config, folds) -> BaselineResult
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from automl_runtime.folds import build_folds
from automl_runtime.metrics import PROBABILITY_METRICS, is_classification, score_predictions
from models import BaselineResult, Configs, SplitPlan

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
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> BaselineResult:
    strategy = config.baseline.strategy

    if not config.baseline.applicable or strategy == "none":
        return BaselineResult(strategy=strategy, applicable=False, note="no baseline for this task")

    y = train[target]
    valid = y.notna().to_numpy()
    classification = is_classification(task_type)

    if folds is None:
        groups = train[plan.group_column] if plan.cv_strategy == "group_kfold" else None
        try:
            folds = build_folds(plan.cv_strategy, plan.cv_folds, plan.random_seed, y, groups)
        except ValueError as error:
            return BaselineResult(strategy=strategy, applicable=False, note=f"cv failed: {error}")

    classes = list(np.unique(y[valid].to_numpy())) if classification else None
    index = {label: position for position, label in enumerate(classes or [])}
    X = np.zeros((len(train), 1))  # dummy predictors ignore the features
    predictions = np.empty(len(train), dtype=object)
    probabilities = np.zeros((len(train), len(classes))) if classification else None
    covered = np.zeros(len(train), dtype=bool)

    for train_positions, test_positions in folds:
        train_positions = train_positions[valid[train_positions]]
        test_positions = test_positions[valid[test_positions]]
        if len(train_positions) == 0 or len(test_positions) == 0:
            continue
        y_fit = y.iloc[train_positions]

        if strategy in ("naive_last", "seasonal_naive"):
            # the last value the fold could have seen; seasonal periods are not
            # in the profile, so a seasonal request degrades to the same thing
            predictions[test_positions] = y_fit.iloc[-1]
            covered[test_positions] = True
            continue

        if strategy not in _DUMMY_STRATEGY:
            return BaselineResult(strategy=strategy, applicable=False, note=f"unknown strategy: {strategy}")

        dummy = (DummyClassifier if classification else DummyRegressor)(strategy=_DUMMY_STRATEGY[strategy])
        dummy.fit(X[train_positions], y_fit)
        predictions[test_positions] = dummy.predict(X[test_positions])
        if classification:
            # a fold missing a class yields fewer probability columns; line them up
            raw = dummy.predict_proba(X[test_positions])
            for column, label in enumerate(dummy.classes_):
                probabilities[test_positions, index[label]] = raw[:, column]
        covered[test_positions] = True

    # TimeSeriesSplit never predicts the first block, so score only what was covered
    rows = np.flatnonzero(covered)
    y_true = y.iloc[rows]
    y_pred = predictions[rows] if classification else predictions[rows].astype(float)
    y_proba = probabilities[rows] if classification else None
    wanted = [m for m in config.eval_matrics if classification or m not in PROBABILITY_METRICS]

    scores, problems = score_predictions(wanted, task_type, y_true, y_pred, y_proba, classes)
    for problem in problems:
        print(f"baseline: {problem}")

    note = f"{strategy} scored out-of-fold on {len(rows)} of {len(train)} training rows"
    return BaselineResult(strategy=strategy, applicable=True, cv_scores=scores, note=note)
