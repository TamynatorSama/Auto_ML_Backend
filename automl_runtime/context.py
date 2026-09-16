"""
context.py
----------
What a candidate may know when it builds and fits its estimator.

    BuildContext  - passed to build_pipeline(columns, task, ctx)
    FitContext    - passed to fit_params(ctx), when a candidate defines it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

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


@dataclass
class FitContext:
    """Rows for model-specific fit arguments, such as early stopping.

    X_valid / y_valid are carved out of the fold's own training rows, never from
    its test rows: the latest rows when the data is ordered in time, whole
    groups when the data is grouped, a seeded random slice otherwise. X_train /
    y_train are the rows left after that carve-out. Fit on X_train when you use
    the validation rows, or leakage from them into the fit is yours.
    """

    task: str
    n_jobs: int
    seed: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    classes: Optional[List[Any]] = None
    extra: dict = field(default_factory=dict)
