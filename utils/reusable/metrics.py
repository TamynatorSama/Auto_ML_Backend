"""
metrics.py
----------
Facts about metrics, and the rule that decides whether an attempt improved.

None of this is asked of an LLM. Whether a lower rmse is better is not a
judgement call, and the stopping rule has to give the same answer every time it
is asked or the loop is not reproducible.

    METRIC_DIRECTION[metric]                    -> "higher" | "lower"
    score(metric, y_true, y_pred, y_proba)      -> float
    is_improvement(best, candidate, ...)        -> bool
    make_cv(plan, n_rows)                       -> sklearn splitter
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn import metrics as skm
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from models import SplitPlan

METRIC_DIRECTION = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "f1_macro": "higher",
    "roc_auc": "higher",
    "pr_auc": "higher",
    "log_loss": "lower",
    "rmse": "lower",
    "mae": "lower",
    "mape": "lower",
    "r2": "higher",
}

# metrics that need predicted probabilities rather than predicted labels
PROBABILITY_METRICS = {"roc_auc", "pr_auc", "log_loss"}


def direction(metric: str) -> str:
    if metric not in METRIC_DIRECTION:
        raise ValueError(f"unknown metric: {metric}")
    return METRIC_DIRECTION[metric]


def score(metric: str, y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> float:
    """One metric, one number. y_proba is required for the probability metrics."""
    if metric in PROBABILITY_METRICS and y_proba is None:
        raise ValueError(f"{metric} needs y_proba")

    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "r2":
        return float(skm.r2_score(y_true, y_pred))

    if metric == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y_true, y_pred))
    if metric == "precision":
        return float(skm.precision_score(y_true, y_pred, average="binary", zero_division=0))
    if metric == "recall":
        return float(skm.recall_score(y_true, y_pred, average="binary", zero_division=0))
    if metric == "f1":
        return float(skm.f1_score(y_true, y_pred, average="binary", zero_division=0))
    if metric == "f1_macro":
        return float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))

    if metric == "log_loss":
        return float(skm.log_loss(y_true, y_proba))
    if metric == "roc_auc":
        if y_proba.ndim > 1 and y_proba.shape[1] > 2:
            return float(skm.roc_auc_score(y_true, y_proba, multi_class="ovr"))
        return float(skm.roc_auc_score(y_true, _positive_column(y_proba)))
    if metric == "pr_auc":
        return float(skm.average_precision_score(y_true, _positive_column(y_proba)))

    raise ValueError(f"unknown metric: {metric}")


def _positive_column(y_proba: np.ndarray) -> np.ndarray:
    y_proba = np.asarray(y_proba)
    return y_proba[:, 1] if y_proba.ndim > 1 else y_proba


def is_improvement(best: float, candidate: float, metric: str, delta: float, mode: str) -> bool:
    """Did `candidate` beat `best` by enough to count as real progress?

    `delta` is always positive and always means improvement in the correct
    direction for the metric. `mode` is "absolute" or "relative"; relative is
    measured against the current best, which is why unbounded error metrics use
    it — a 1% cut in rmse means something without knowing the target's scale.
    """
    gain = (best - candidate) if direction(metric) == "lower" else (candidate - best)

    if mode == "relative":
        if best == 0:
            return gain > 0
        gain = gain / abs(best)
    elif mode != "absolute":
        raise ValueError(f"unknown improvement mode: {mode}")

    return gain >= delta


def make_cv(plan: SplitPlan, n_rows: int):
    """The cross-validator the whole run shares, built from the split plan.

    Every model folds the training set the same way or their scores are not
    comparable, so this is derived from the plan and never chosen per model.
    """
    folds = max(2, min(plan.cv_folds, n_rows))

    if plan.cv_strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=folds)
    if plan.cv_strategy == "group_kfold":
        return GroupKFold(n_splits=folds)
    if plan.cv_strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=plan.random_seed)
    if plan.cv_strategy == "kfold":
        return KFold(n_splits=folds, shuffle=True, random_state=plan.random_seed)

    raise ValueError(f"unknown cv strategy: {plan.cv_strategy}")
