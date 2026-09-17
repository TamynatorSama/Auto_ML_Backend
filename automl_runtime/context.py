"""
context.py
----------
What a candidate may know when it builds and fits its estimator.

    BuildContext  - passed to build_pipeline(columns, task, ctx)
    FitContext    - passed to fit_params(ctx), when a candidate defines it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class BuildContext:
    task: str                    # regression | binary_classification | multiclass_classification
    n_jobs: int                  # parallelism cap for this candidate; never use -1
    seed: int                    # use it on every estimator and sampler
    n_rows: int                  # training rows in one fold
    metrics: List[str]
    primary_metric: str
    time_budget_seconds: int     # for the whole evaluation, every fold included
    classes: Optional[List[Any]] = None


class FitContext:
    """Rows for model-specific fit arguments, such as early stopping.

    The validation rows are carved out of the fold's own training rows, never
    from its test rows: the latest rows when the data is ordered in time, whole
    groups when the data is grouped, a seeded random slice otherwise. X_train /
    y_train are the rows left after that carve-out; when fit_params returns
    anything, the estimator is fitted on them alone.

    X_valid / y_valid are the validation rows as `final_estimator` receives them:
    through every Pipeline step before it, fitted on X_train, and through the
    target transform of any TransformedTargetRegressor around it. They go
    straight into an eval_set. They are prepared on first use, since preparing
    them fits the preprocessing once more. X_valid_raw / y_valid_raw are the
    same rows as the estimator's own input.
    """

    def __init__(
        self,
        task: str,
        n_jobs: int,
        seed: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid_raw: pd.DataFrame,
        y_valid_raw: pd.Series,
        final_estimator: str = "",
        classes: Optional[List[Any]] = None,
        prepare: Optional[Callable[[], Tuple[Any, Any]]] = None,
    ):
        self.task = task
        self.n_jobs = n_jobs
        self.seed = seed
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid_raw = X_valid_raw
        self.y_valid_raw = y_valid_raw
        self.final_estimator = final_estimator
        self.classes = classes
        self.extra: dict = {}
        self._prepare = prepare
        self._prepared: Optional[Tuple[Any, Any]] = None

    def _validation(self) -> Tuple[Any, Any]:
        if self._prepared is None:
            self._prepared = self._prepare() if self._prepare else (self.X_valid_raw, self.y_valid_raw)
        return self._prepared

    @property
    def X_valid(self):
        return self._validation()[0]

    @property
    def y_valid(self):
        return self._validation()[1]
