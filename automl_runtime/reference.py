"""
reference.py
------------
A plain, robust candidate the harness owns, for measurements that must not
depend on what a generated candidate happens to do.

A leak check normally re-runs the suspicious candidate without the suspect
columns. A candidate that names those columns explicitly cannot run without
them, and then the same comparison is made with this model instead: gradient
boosted trees on every column the kind information says is usable, with no
tuning. It works for any tabular dataset and any of the supported tasks.
"""

REFERENCE_CANDIDATE = '''# model: reference
# attempt: 1
# changes: harness reference model for a leak check
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from automl_runtime import DateParts, FrequencyEncoder, names


def build_pipeline(columns, task, ctx):
    numeric = names(columns, "numeric", "discrete_numeric")
    categories = [c for c in columns if c.kind in ("categorical", "binary")]
    low = [c.name for c in categories if (c.n_unique or 0) <= 50]
    high = [c.name for c in categories if (c.n_unique or 0) > 50]
    dates = names(columns, "datetime")

    steps = []
    if numeric:
        steps.append(("numeric", "passthrough", numeric))
    if low:
        steps.append(("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                            encoded_missing_value=-1), low))
    if high:
        steps.append(("high", FrequencyEncoder(), high))
    if dates:
        steps.append(("dates", DateParts(parts=("year", "month", "dayofweek")), dates))

    estimator = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    return Pipeline([("prep", ColumnTransformer(steps)), ("model", estimator(random_state=ctx.seed))])
'''
