"""
metrics.py
----------
Every score the run reports is computed here, by the harness, from predictions.

A candidate never scores itself. When scripts printed their own numbers, the
leaderboard mixed pooled out-of-fold scores with averages of per-fold scores,
and nothing could tell them apart. One function, used by the evaluator, the
baseline and the guards, is the only way to make the numbers comparable.

"Cross-validated score" means one thing: the metric computed once, pooled over
every training row that was in a test fold. A splitter that never predicts some
rows (TimeSeriesSplit leaves out the earliest block) scores only what it covered.

    METRIC_DIRECTION[metric]                                    -> "higher" | "lower"
    score_predictions(metrics, task, y_true, y_pred, y_score, classes)
                                                                -> (scores, problems)
    score(metric, y_true, y_pred, y_proba)                      -> float
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize

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

# metrics computed from a continuous score (probabilities or a decision
# function) rather than from predicted labels
PROBABILITY_METRICS = {"roc_auc", "pr_auc", "log_loss"}

# log_loss needs calibrated probabilities; the other two only need a ranking,
# so a binary decision_function is enough for them
NEEDS_PROBABILITIES = {"log_loss"}

REGRESSION_METRICS = {"rmse", "mae", "mape", "r2"}


def direction(metric: str) -> str:
    if metric not in METRIC_DIRECTION:
        raise ValueError(f"unknown metric: {metric}")
    return METRIC_DIRECTION[metric]


def is_classification(task_type: str) -> bool:
    return task_type.endswith("classification")


def _regression(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "r2":
        return float(skm.r2_score(y_true, y_pred))
    raise ValueError(f"{metric} is not a regression metric")


def _encode(values: Sequence, index: Dict) -> np.ndarray:
    # a label the training rows never contained cannot be a correct prediction;
    # -1 keeps it countable as wrong instead of raising
    return np.array([index.get(value, -1) for value in values], dtype=int)


def _positive_score(y_score: np.ndarray) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=float)
    return y_score[:, 1] if y_score.ndim > 1 else y_score


def _classification(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    n_classes: int,
) -> float:
    binary = n_classes == 2
    averaging = "binary" if binary else "macro"

    if metric == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y_true, y_pred))
    if metric == "precision":
        return float(skm.precision_score(y_true, y_pred, average=averaging, pos_label=1, zero_division=0))
    if metric == "recall":
        return float(skm.recall_score(y_true, y_pred, average=averaging, pos_label=1, zero_division=0))
    if metric == "f1":
        return float(skm.f1_score(y_true, y_pred, average=averaging, pos_label=1, zero_division=0))
    if metric == "f1_macro":
        return float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))

    if y_score is None:
        raise ValueError(f"{metric} needs predicted probabilities or scores, and the model gave none")
    y_score = np.asarray(y_score, dtype=float)
    labels = list(range(n_classes))

    if metric == "roc_auc":
        if binary:
            return float(skm.roc_auc_score(y_true, _positive_score(y_score)))
        return float(skm.roc_auc_score(y_true, y_score, multi_class="ovr", labels=labels))
    if metric == "pr_auc":
        if binary:
            return float(skm.average_precision_score(y_true, _positive_score(y_score)))
        return float(skm.average_precision_score(label_binarize(y_true, classes=labels), y_score, average="macro"))
    if metric == "log_loss":
        if y_score.ndim == 1:
            y_score = np.column_stack([1.0 - y_score, y_score])
        return float(skm.log_loss(y_true, y_score, labels=labels))

    raise ValueError(f"{metric} is not a classification metric")


def score_predictions(
    metrics: Sequence[str],
    task_type: str,
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Optional[np.ndarray] = None,
    classes: Optional[Sequence] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Every requested metric for one set of predictions.

    Classification labels are encoded against `classes`, the sorted labels of
    the training rows. That is the order scikit-learn uses for `predict_proba`
    columns, so the positive class of a binary task is `classes[1]` whatever the
    labels are called, and string labels score exactly like integer ones.

    `y_score` is an (n, n_classes) probability matrix, or for a binary task an
    (n,) positive-class score. Returns the scores and, separately, a sentence
    for each metric that could not be computed, so one undefined metric (an AUC
    on a fold holding a single class) does not hide the others.
    """
    scores: Dict[str, float] = {}
    problems: List[str] = []

    if is_classification(task_type):
        if classes is None:
            classes = np.unique(np.asarray(y_true))
        index = {label: position for position, label in enumerate(classes)}
        encoded_true = _encode(y_true, index)
        encoded_pred = _encode(y_pred, index)
        for metric in metrics:
            try:
                scores[metric] = _classification(metric, encoded_true, encoded_pred, y_score, len(classes))
            except Exception as error:
                problems.append(f"{metric} could not be computed: {error}")
        return scores, problems

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    for metric in metrics:
        try:
            scores[metric] = _regression(metric, true, pred)
        except Exception as error:
            problems.append(f"{metric} could not be computed: {error}")
    return scores, problems


def score(metric: str, y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> float:
    """One metric, one number, for callers that already hold encoded labels."""
    if metric in REGRESSION_METRICS:
        return _regression(metric, np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    classes = np.unique(np.asarray(y_true))
    scores, problems = score_predictions([metric], "classification", y_true, y_pred, y_proba, classes)
    if metric not in scores:
        raise ValueError(problems[0] if problems else f"unknown metric: {metric}")
    return scores[metric]
