"""
automl_runtime
--------------
The harness that evaluates generated candidates, and the small API a candidate
may import from it.

A candidate is a module defining

    build_pipeline(columns, task, ctx) -> unfitted estimator
    fit_params(ctx) -> dict                      # optional

and everything else (loading data, folds, fitting, scoring, the test set,
saving the model) happens here, identically for every model in a run.
"""

from automl_runtime.candidate import load_model
from automl_runtime.columns import NUMERIC_KINDS, ColumnInfo, names
from automl_runtime.context import BuildContext, FitContext
from automl_runtime.features import CategoryCleaner, DateParts, FrequencyEncoder

__all__ = [
    "BuildContext",
    "CategoryCleaner",
    "ColumnInfo",
    "DateParts",
    "FitContext",
    "FrequencyEncoder",
    "NUMERIC_KINDS",
    "load_model",
    "names",
]
