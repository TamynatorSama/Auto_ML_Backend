"""
validation.py
-------------
Validation rows as the model at the end of a pipeline receives them.

Early stopping scores an eval_set with the model being fitted, so the eval_set
has to look exactly like that model's training input: through every
preprocessing step before it and, under a TransformedTargetRegressor, with the
target transformed too. A candidate cannot build that itself. fit_params runs
before anything is fitted, so it was handed raw rows, passed date strings to a
booster that expected numbers, and the repair that stuck was deleting early
stopping altogether.

The harness walks the estimator instead, on a copy: a Pipeline's steps before
the last are fitted on the training rows and transform the validation rows, a
TransformedTargetRegressor's target transform is fitted on the training target
and applied to the validation target, and the walk goes on into the last step
or the regressor until it reaches an estimator that is neither. That is the
final estimator, the one fit parameters are meant for, and the same walk
yields the prefix that routes them to it.

    route(estimator)                                          -> (prefix, final estimator name)
    for_final_estimator(estimator, X_train, y_train, X_valid, y_valid) -> (X_valid, y_valid)
    route_params(params, prefix)                              -> dict
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _skipped(step) -> bool:
    return step is None or (isinstance(step, str) and step == "passthrough")


def _regressor(estimator: TransformedTargetRegressor):
    # the default sklearn itself uses when no regressor is given
    return estimator.regressor if estimator.regressor is not None else LinearRegression()


def route(estimator) -> Tuple[str, str]:
    """The fit-parameter prefix that reaches the final estimator, and its class name.

    A Pipeline routes `step__key` to its step; a TransformedTargetRegressor hands
    every fit parameter to its regressor unchanged, so it adds nothing.
    """
    prefix = ""
    while True:
        if isinstance(estimator, Pipeline) and estimator.steps:
            name, estimator = estimator.steps[-1]
            prefix += f"{name}__"
        elif isinstance(estimator, TransformedTargetRegressor):
            estimator = _regressor(estimator)
        else:
            return prefix, "passthrough" if _skipped(estimator) else type(estimator).__name__


def route_params(params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Unprefixed keys go to the final estimator; a key that already names a step is kept as written."""
    return {key if "__" in key else f"{prefix}{key}": value for key, value in params.items()}


def _needs_features(estimator) -> bool:
    """Whether a Pipeline further down has steps to fit on the transformed training rows."""
    if isinstance(estimator, Pipeline):
        return True
    if isinstance(estimator, TransformedTargetRegressor):
        return _needs_features(_regressor(estimator))
    return False


def _target_transformer(estimator: TransformedTargetRegressor):
    """The transform TransformedTargetRegressor.fit builds for the target, unfitted."""
    if estimator.transformer is not None:
        return clone(estimator.transformer)
    transformer = FunctionTransformer(
        func=estimator.func, inverse_func=estimator.inverse_func, validate=True, check_inverse=False
    )
    return transformer.set_output(transform="default")


def _transform_target(transformer, y_train, y_valid):
    train = np.asarray(y_train, dtype=float)
    one_dimensional = train.ndim == 1

    def as_fitted(values):
        return values.reshape(-1, 1) if one_dimensional else values

    def apply(values):
        out = np.asarray(transformer.transform(as_fitted(np.asarray(values, dtype=float))))
        # as TransformedTargetRegressor does: a 1-d target stays 1-d
        return out.ravel() if one_dimensional and out.ndim == 2 and out.shape[1] == 1 else out

    transformer.fit(as_fitted(train))
    return apply(train), apply(y_valid)


def for_final_estimator(estimator, X_train, y_train, X_valid, y_valid):
    """X_valid and y_valid as the final estimator receives them when `estimator` is fitted on X_train, y_train.

    Nothing on `estimator` itself is fitted: the walk runs on a copy, and the
    real fit starts from scratch as it always does. The validation rows must not
    be empty; transformers reject zero rows.
    """
    try:
        estimator = clone(estimator)
    except Exception:
        estimator = copy.deepcopy(estimator)

    while True:
        if isinstance(estimator, Pipeline) and estimator.steps:
            head = [(name, step) for name, step in estimator.steps[:-1] if not _skipped(step)]
            estimator = estimator.steps[-1][1]
            if head:
                steps = Pipeline(head)
                if _needs_features(estimator):
                    X_train = steps.fit_transform(X_train, y_train)
                else:
                    # the transformed training rows would only be a second copy held for nothing
                    steps.fit(X_train, y_train)
                X_valid = steps.transform(X_valid)
        elif isinstance(estimator, TransformedTargetRegressor):
            y_train, y_valid = _transform_target(_target_transformer(estimator), y_train, y_valid)
            estimator = _regressor(estimator)
        else:
            return X_valid, y_valid
