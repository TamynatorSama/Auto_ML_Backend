"""
metrics.py
----------
Facts about metrics, and the rule that decides whether an attempt improved.

None of this is asked of an LLM. Whether a lower rmse is better is not a
judgement call, and the stopping rule has to give the same answer every time it
is asked or the loop is not reproducible.

Scoring itself lives in automl_runtime.metrics, which the evaluator, the
baseline and the guards all import, so there is exactly one definition of every
number the run reports.

    METRIC_DIRECTION[metric]                    -> "higher" | "lower"
    REGRESSION_METRICS                          the ones that suit a regression
    score(metric, y_true, y_pred, y_proba)      -> float
    is_improvement(best, candidate, ...)        -> bool
    make_cv(plan, n_rows)                       -> sklearn splitter
"""

from __future__ import annotations

from automl_runtime.folds import make_splitter
from automl_runtime.metrics import (  # noqa: F401  (re-exported)
    METRIC_DIRECTION,
    PROBABILITY_METRICS,
    REGRESSION_METRICS,
    direction,
    score,
    score_predictions,
)
from models import SplitPlan


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
    """The cross-validator the whole run shares, built from the split plan."""
    return make_splitter(plan.cv_strategy, plan.cv_folds, plan.random_seed, n_rows)
