"""
evaluate.py
-----------
Evaluate one candidate. The harness, not the candidate, owns every step that
decides what a score means.

    python -m automl_runtime.evaluate --attempt-dir DIR
    python -m automl_runtime.evaluate --attempt-dir DIR --final --test TEST_CSV
    python -m automl_runtime.evaluate --attempt-dir DIR --load-check --test TEST_CSV

Loop mode climbs a ladder of increasingly expensive checks and stops at the
first failure, so a mechanical mistake costs seconds instead of a full
cross-validation:

    import     candidate.py imports
    data       the training file loads; excluded columns are removed
    build      build_pipeline(columns, task, ctx) returns something
    estimator  it has fit and predict, and probabilities when a metric needs them
    smoke      it fits and predicts on a small sample of one fold
    pickle     the fitted estimator survives a pickle round trip
    cv         every frozen fold: fit on its training rows, predict its test rows
    scoring    pooled out-of-fold scores for every metric
    importance permutation importance on raw columns, when the budget allows

Final mode fits on every training row, predicts the test file once, scores it,
and saves the model. Load-check mode, run by the runner once final mode has
exited, opens that model in a fresh process and predicts five test rows.

Nothing escapes as an exception: every outcome is written to result.json (loop),
result_final.json (final) or load_check.json, which is what the runner reads.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from automl_runtime.candidate import (
    call_build,
    call_fit_params,
    import_candidate,
    load_model,
    wants_fit_params,
)
from automl_runtime.columns import ColumnInfo, infer_kind
from automl_runtime.context import BuildContext, FitContext
from automl_runtime.contract import Contract
from automl_runtime.folds import load_folds
from automl_runtime.metrics import (
    METRIC_DIRECTION,
    NEEDS_PROBABILITIES,
    PROBABILITY_METRICS,
    is_classification,
    score_predictions,
)
from automl_runtime.validation import for_final_estimator, route, route_params

RESULT_FILE = "result.json"
FINAL_RESULT_FILE = "result_final.json"
CONTRACT_FILE = "contract.json"
OOF_FILE = "oof_predictions.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
IMPORTANCE_FILE = "feature_importance.csv"
MODEL_FILE = "model.joblib"

SMOKE_TRAIN_ROWS = 1000          # enough rows that estimators with internal CV still fit
SMOKE_PREDICT_ROWS = 200
SMOKE_MIN_PER_CLASS = 20
VALIDATION_SHARE = 0.15          # of a fold's training rows, for fit_params
IMPORTANCE_ROWS = 1000
IMPORTANCE_REPEATS = 2
IMPORTANCE_BUDGET_SHARE = 0.25   # of the time still left
BUDGET_MARGIN = 0.95             # a projection past this share of the budget stops the run
ERROR_CHARS = 6000
LOAD_CHECK_TIMEOUT = 300
LOAD_CHECK_FILE = "load_check.json"

PICKLE_HINT = (
    "the fitted estimator cannot be pickled, so it could never be saved as a model. "
    "Define helper functions and transformer classes at the top level of candidate.py "
    "(or import them); never use lambdas or functions defined inside build_pipeline."
)


class StageFailure(Exception):
    def __init__(self, stage: str, message: str, status: str = "error"):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.status = status


class PredictionProblem(ValueError):
    """Predictions that cannot be scored; the enclosing stage names where it happened."""


def _describe(error: BaseException) -> str:
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return text[-ERROR_CHARS:]


def _out_of_memory(error: BaseException) -> bool:
    """A MemoryError, or an exception raised while handling one."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, MemoryError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


@contextmanager
def stage(name: str, timings: Dict[str, float]):
    # announced before it runs: a process stopped from outside, for its memory
    # or its time, writes no result, and this line is how the runner learns
    # how far it got
    print(f"[stage: {name}]", flush=True)
    started = time.perf_counter()
    try:
        yield
    except StageFailure:
        raise
    except PredictionProblem as problem:
        raise StageFailure(name, str(problem)) from problem
    except Exception as error:
        status = "out_of_memory" if _out_of_memory(error) else "error"
        raise StageFailure(name, _describe(error), status=status) from error
    finally:
        timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - started, 3)


def _plain(value):
    """JSON-safe: numpy scalars to python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Data:
    """The training rows as the candidate sees them."""

    def __init__(self, contract: Contract):
        frame = pd.read_csv(contract.train_path, low_memory=False)
        if contract.target not in frame.columns:
            raise StageFailure("data", f"target column {contract.target!r} is not in {contract.train_path}")

        self.y = frame[contract.target]
        self.valid = self.y.notna().to_numpy()
        self.groups = (
            frame[contract.group_column]
            if contract.group_column and contract.group_column in frame.columns
            else None
        )

        self.excluded = [
            column for column in contract.excluded_columns
            if column in frame.columns and column != contract.target
        ]
        self.X = frame.drop(columns=[contract.target, *self.excluded])

        described = {column["name"]: column for column in contract.columns}
        self.columns: List[ColumnInfo] = []
        for name in self.X.columns:
            dtype = str(self.X[name].dtype)
            if name in described:
                self.columns.append(ColumnInfo.from_dict({**described[name], "dtype": dtype}))
            else:
                self.columns.append(ColumnInfo(name=name, kind=infer_kind(self.X[name]), dtype=dtype))

        self.classes = (
            list(np.unique(self.y[self.valid].to_numpy()))
            if is_classification(contract.task_type)
            else None
        )


def _check_folds(folds, n_rows: int) -> None:
    if not folds:
        raise StageFailure("data", "the frozen folds file holds no folds")
    largest = max(int(max(train.max(initial=0), test.max(initial=0))) for train, test in folds)
    if largest >= n_rows:
        raise StageFailure(
            "data",
            f"the frozen folds index row {largest} but the training file has {n_rows} rows; "
            "the folds were built for a different file",
        )


# ---------------------------------------------------------------------------
# estimator behaviour
# ---------------------------------------------------------------------------

def _check_estimator(estimator, contract: Contract, data: Data) -> bool:
    """Fail fast on an estimator that cannot be scored; return whether to collect scores."""
    missing = [name for name in ("fit", "predict") if not callable(getattr(estimator, name, None))]
    if missing:
        raise StageFailure(
            "estimator",
            f"build_pipeline returned a {type(estimator).__name__}, which has no "
            f"{' or '.join(missing)} method. Return an unfitted scikit-learn-style estimator or Pipeline.",
        )

    if not is_classification(contract.task_type):
        return False

    has_proba = hasattr(estimator, "predict_proba")
    has_decision = hasattr(estimator, "decision_function")
    wanted = [metric for metric in contract.metrics if metric in PROBABILITY_METRICS]
    needs_probabilities = any(metric in NEEDS_PROBABILITIES for metric in wanted) or (
        wanted and len(data.classes) > 2
    )

    if needs_probabilities and not has_proba:
        raise StageFailure(
            "estimator",
            f"the metrics {', '.join(wanted)} need predicted probabilities, but the estimator has no "
            "predict_proba. SVC needs probability=True; other margin classifiers can be wrapped in "
            "CalibratedClassifierCV.",
        )
    if wanted and not (has_proba or has_decision):
        raise StageFailure(
            "estimator",
            f"the metrics {', '.join(wanted)} need a continuous score, but the estimator has neither "
            "predict_proba nor decision_function.",
        )
    return has_proba or has_decision


def _predict(estimator, X: pd.DataFrame, classes, want_scores: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    predictions = np.asarray(estimator.predict(X))
    if predictions.ndim > 1:
        if predictions.shape[1] != 1:
            raise ValueError(f"predict returned shape {predictions.shape}; expected one value per row")
        predictions = predictions.ravel()

    if not want_scores:
        return predictions, None

    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(X), dtype=float)
        own = getattr(estimator, "classes_", None)
        if own is None:
            if raw.shape[1] != len(classes):
                raise ValueError(
                    f"predict_proba returned {raw.shape[1]} columns for {len(classes)} classes, and the "
                    "estimator exposes no classes_ to line them up"
                )
            return predictions, raw
        # a fold missing a rare class yields fewer columns; place each where it belongs
        index = {label: position for position, label in enumerate(classes)}
        aligned = np.zeros((len(X), len(classes)))
        for column, label in enumerate(own):
            if label in index:
                aligned[:, index[label]] = raw[:, column]
        return predictions, aligned

    decision = np.asarray(estimator.decision_function(X), dtype=float)
    return predictions, decision


def _check_predictions(where: str, predictions, scores, expected: int, contract: Contract, data: Data) -> List[str]:
    if len(predictions) != expected:
        raise PredictionProblem(f"{where}: predict returned {len(predictions)} values for {expected} rows")

    notes: List[str] = []
    if is_classification(contract.task_type):
        known = set(data.classes)
        unknown = sum(1 for value in predictions if value not in known)
        if unknown:
            notes.append(f"{where}: {unknown} predicted labels are not classes seen in training")
        if scores is not None and not np.all(np.isfinite(scores)):
            raise PredictionProblem(f"{where}: predicted probabilities contain NaN or infinite values")
        return notes

    try:
        numeric = predictions.astype(float)
    except (TypeError, ValueError):
        raise PredictionProblem(f"{where}: regression predictions are not numeric")
    bad = int((~np.isfinite(numeric)).sum())
    if bad:
        raise PredictionProblem(f"{where}: {bad} of {expected} predictions are NaN or infinite")
    return notes


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _validation_split(rows: np.ndarray, contract: Contract, data: Data, rng) -> Tuple[np.ndarray, np.ndarray]:
    n_valid = int(round(len(rows) * VALIDATION_SHARE))
    if n_valid < 1 or len(rows) - n_valid < 2:
        return rows, rows[:0]

    if contract.ordered:
        return rows[:-n_valid], rows[-n_valid:]

    if data.groups is not None:
        groups = data.groups.to_numpy()[rows]
        chosen, taken = [], 0
        for group in rng.permutation(pd.unique(groups)):
            if taken >= n_valid:
                break
            chosen.append(group)
            taken += int((groups == group).sum())
        mask = np.isin(groups, chosen)
        return rows[~mask], rows[mask]

    shuffled = rng.permutation(rows)
    return np.sort(shuffled[n_valid:]), np.sort(shuffled[:n_valid])


def _fit(estimator, module, rows: np.ndarray, contract: Contract, data: Data, rng) -> None:
    if not wants_fit_params(module):
        estimator.fit(data.X.iloc[rows], data.y.iloc[rows])
        return

    fit_rows, valid_rows = _validation_split(rows, contract, data, rng)
    X_train, y_train = data.X.iloc[fit_rows], data.y.iloc[fit_rows]
    X_valid, y_valid = data.X.iloc[valid_rows], data.y.iloc[valid_rows]
    prefix, final_estimator = route(estimator)
    ctx = FitContext(
        task=contract.task_type,
        n_jobs=contract.n_jobs,
        seed=contract.seed,
        X_train=X_train,
        y_train=y_train,
        X_valid_raw=X_valid,
        y_valid_raw=y_valid,
        final_estimator=final_estimator,
        classes=data.classes,
        # transformers reject zero rows, and there is nothing to transform
        prepare=(lambda: for_final_estimator(estimator, X_train, y_train, X_valid, y_valid))
        if len(valid_rows) else None,
    )
    # the candidate names arguments for the final estimator; the pipeline around it decides the prefix
    params = route_params(call_fit_params(module, ctx), prefix)
    if params and len(valid_rows):
        # the candidate uses the validation rows, so they must stay out of the fit
        estimator.fit(ctx.X_train, ctx.y_train, **params)
    else:
        estimator.fit(data.X.iloc[rows], data.y.iloc[rows], **params)


def _smoke_rows(rows: np.ndarray, data: Data, classification: bool, rng) -> np.ndarray:
    rows = rows[data.valid[rows]]
    if len(rows) <= SMOKE_TRAIN_ROWS:
        return rows
    if not classification:
        return np.sort(rng.choice(rows, SMOKE_TRAIN_ROWS, replace=False))

    # every class present, so a classifier's own checks see the real label set
    labels = data.y.to_numpy()[rows]
    picked = []
    for label in pd.unique(labels):
        members = rows[labels == label]
        picked.append(rng.choice(members, min(len(members), SMOKE_MIN_PER_CLASS), replace=False))
    picked = np.unique(np.concatenate(picked))
    rest = np.setdiff1d(rows, picked)
    extra = min(len(rest), max(0, SMOKE_TRAIN_ROWS - len(picked)))
    filler = rng.choice(rest, extra, replace=False) if extra else rest[:0]
    return np.sort(np.concatenate([picked, filler]))


# ---------------------------------------------------------------------------
# loop mode
# ---------------------------------------------------------------------------

def _prediction_frame(rows, y_true, predictions, scores, classes) -> pd.DataFrame:
    frame = pd.DataFrame({"row_index": rows, "y_true": np.asarray(y_true), "prediction": predictions})
    if scores is not None:
        if scores.ndim == 1:
            frame["score"] = scores
        else:
            for position, label in enumerate(classes):
                frame[f"proba_{label}"] = scores[:, position]
    return frame


def _importance(estimator, rows, contract, data, want_scores, started, rng, result) -> None:
    metric = contract.primary_metric
    sample = np.sort(rng.choice(rows, min(IMPORTANCE_ROWS, len(rows)), replace=False))
    X_sample = data.X.iloc[sample].reset_index(drop=True)
    y_sample = data.y.iloc[sample].reset_index(drop=True)

    clock = time.perf_counter()
    predictions, scores = _predict(estimator, X_sample, data.classes, want_scores)
    predict_seconds = time.perf_counter() - clock
    base, problems = score_predictions([metric], contract.task_type, y_sample, predictions, scores, data.classes)
    if metric not in base:
        result["warnings"].append(f"feature importance skipped: {'; '.join(problems)}")
        return

    projected = predict_seconds * len(X_sample.columns) * IMPORTANCE_REPEATS
    remaining = contract.time_budget_seconds - (time.perf_counter() - started)
    if projected > IMPORTANCE_BUDGET_SHARE * max(remaining, 0.0):
        result["warnings"].append(
            f"feature importance skipped: about {projected:.0f}s of predictions against {remaining:.0f}s left"
        )
        return

    sign = 1.0 if METRIC_DIRECTION.get(metric, "higher") == "higher" else -1.0
    table = []
    for column in X_sample.columns:
        drops = []
        for _ in range(IMPORTANCE_REPEATS):
            shuffled = X_sample.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            predictions, scores = _predict(estimator, shuffled, data.classes, want_scores)
            value, _ = score_predictions([metric], contract.task_type, y_sample, predictions, scores, data.classes)
            if metric in value:
                drops.append(sign * (base[metric] - value[metric]))
        table.append((column, float(np.mean(drops)) if drops else float("nan")))

    frame = pd.DataFrame(table, columns=["feature", "importance"]).sort_values("importance", ascending=False)
    frame.to_csv(result["_dir"] / IMPORTANCE_FILE, index=False)
    result["artifacts"]["feature_importance"] = IMPORTANCE_FILE


def _loop(attempt_dir: Path, contract: Contract, result: dict, started: float) -> None:
    timings = result["timings"]
    classification = is_classification(contract.task_type)
    primary = contract.primary_metric

    with stage("import", timings):
        module = import_candidate(attempt_dir)

    with stage("data", timings):
        data = Data(contract)
        folds = load_folds(contract.folds_path)
        _check_folds(folds, len(data.X))
    result["n_rows"] = int(data.valid.sum())
    result["excluded_columns"] = data.excluded

    build_ctx = BuildContext(
        task=contract.task_type,
        n_jobs=contract.n_jobs,
        seed=contract.seed,
        n_rows=int(len(folds[0][0])),
        metrics=list(contract.metrics),
        primary_metric=primary,
        time_budget_seconds=contract.time_budget_seconds,
        classes=_plain(data.classes),
    )

    with stage("build", timings):
        estimator = call_build(module, data.columns, contract.task_type, build_ctx)
    with stage("estimator", timings):
        want_scores = _check_estimator(estimator, contract, data)

    rng = np.random.default_rng(contract.seed)
    train_zero, test_zero = folds[0]

    with stage("smoke", timings):
        fit_rows = _smoke_rows(train_zero, data, classification, rng)
        predict_rows = test_zero[data.valid[test_zero]][:SMOKE_PREDICT_ROWS]
        smoke = call_build(module, data.columns, contract.task_type, build_ctx)
        _fit(smoke, module, fit_rows, contract, data, rng)
        predictions, scores = _predict(smoke, data.X.iloc[predict_rows], data.classes, want_scores)
        _check_predictions("smoke test", predictions, scores, len(predict_rows), contract, data)
    print(f"smoke test passed on {len(fit_rows)} rows", flush=True)

    with stage("pickle", timings):
        try:
            restored = pickle.loads(pickle.dumps(smoke))
        except Exception as error:
            raise StageFailure("pickle", f"{PICKLE_HINT}\n\n{_describe(error)}")
        restored.predict(data.X.iloc[predict_rows[:5]])

    n = len(data.X)
    oof_predictions = np.empty(n, dtype=object)
    oof_scores: Optional[np.ndarray] = None
    covered = np.zeros(n, dtype=bool)
    last_model, last_rows = None, None

    with stage("cv", timings):
        for number, (train_rows, test_rows) in enumerate(folds, start=1):
            train_rows = train_rows[data.valid[train_rows]]
            test_rows = test_rows[data.valid[test_rows]]
            if len(test_rows) == 0 or len(train_rows) == 0:
                continue

            fold_started = time.perf_counter()
            model = call_build(module, data.columns, contract.task_type, build_ctx)
            _fit(model, module, train_rows, contract, data, rng)
            result["fit_seconds"] += time.perf_counter() - fold_started

            predictions, scores = _predict(model, data.X.iloc[test_rows], data.classes, want_scores)
            result["warnings"] += _check_predictions(
                f"fold {number}", predictions, scores, len(test_rows), contract, data
            )
            oof_predictions[test_rows] = predictions
            if scores is not None:
                if oof_scores is None:
                    oof_scores = np.full((n,) + scores.shape[1:], np.nan)
                oof_scores[test_rows] = scores
            covered[test_rows] = True

            fold_score, _ = score_predictions(
                [primary], contract.task_type, data.y.iloc[test_rows], predictions, scores, data.classes
            )
            result["fold_scores"].append(fold_score.get(primary))
            fold_seconds = time.perf_counter() - fold_started
            print(f"fold {number}/{len(folds)}: {fold_seconds:.1f}s {primary} {fold_score.get(primary)}", flush=True)

            if number == 1 and len(folds) > 1:
                projected = (time.perf_counter() - started) + fold_seconds * (len(folds) - 1)
                if projected > contract.time_budget_seconds * BUDGET_MARGIN:
                    raise StageFailure(
                        "budget",
                        f"fold 1 took {fold_seconds:.0f}s, so all {len(folds)} folds would take about "
                        f"{projected:.0f}s against a {contract.time_budget_seconds}s budget. Make the model "
                        "cheaper (fewer estimators or iterations, a subsample, a cheaper encoding of wide "
                        "categoricals); more time is not available.",
                        status="timeout",
                    )
            last_model, last_rows = model, test_rows

    rows = np.flatnonzero(covered)
    result["n_covered"] = int(len(rows))

    with stage("scoring", timings):
        if len(rows) == 0:
            raise StageFailure("scoring", "no fold produced predictions, so there is nothing to score")
        y_true = data.y.iloc[rows]
        predictions = oof_predictions[rows]
        if not classification:
            predictions = predictions.astype(float)
        scores = oof_scores[rows] if oof_scores is not None else None

        cv_scores, problems = score_predictions(
            contract.metrics, contract.task_type, y_true, predictions, scores, data.classes
        )
        result["warnings"] += problems
        finite = {name: value for name, value in cv_scores.items() if math.isfinite(value)}
        if primary not in finite:
            raise StageFailure(
                "scoring",
                f"the primary metric {primary} could not be computed from the out-of-fold predictions: "
                + ("; ".join(problems) or f"it came out as {cv_scores.get(primary)}"),
            )
        result["cv_scores"] = finite

        # scale-free readings, recorded whatever the configured metrics are
        diagnostic = ["r2"] if not classification else (["roc_auc"] if scores is not None else [])
        extra, _ = score_predictions(diagnostic, contract.task_type, y_true, predictions, scores, data.classes)
        result["diagnostics"] = extra

        _prediction_frame(rows, y_true, predictions, scores, data.classes).to_csv(
            attempt_dir / OOF_FILE, index=False
        )
        result["artifacts"]["oof_predictions"] = OOF_FILE

    try:
        with stage("importance", timings):
            _importance(last_model, last_rows, contract, data, want_scores, started, rng, result)
    except StageFailure as failure:
        tail = failure.message.strip().splitlines()
        result["warnings"].append(f"feature importance failed: {tail[-1] if tail else 'unknown error'}")


# ---------------------------------------------------------------------------
# final mode
# ---------------------------------------------------------------------------

def check_model(attempt_dir: str | Path, test_path: str) -> dict:
    """Open the saved model in a process that never imported the candidate, and
    predict five test rows with it. Writes load_check.json.

    It runs as a process of its own after the final evaluation has exited. Run
    from inside the evaluator, it was a second interpreter held against the same
    memory ceiling as the one that had just fitted the model: a 131 MB random
    forest that fitted and scored within 383 MB was stopped there, and its test
    scores, computed but not yet written, were lost with it.
    """
    from joblib import parallel_config

    attempt_dir = Path(attempt_dir).resolve()
    result = {"status": "ok", "error": ""}
    started = time.perf_counter()
    try:
        contract = Contract.load(attempt_dir / CONTRACT_FILE)
        header = pd.read_csv(contract.train_path, nrows=0).columns
        columns = [c for c in header if c != contract.target and c not in contract.excluded_columns]
        model = load_model(attempt_dir / MODEL_FILE)
        frame = pd.read_csv(test_path, nrows=5, low_memory=False)
        # five rows need no pool of workers, whatever n_jobs the model was built with
        with parallel_config(backend="sequential"):
            model.predict(frame[columns])
    except Exception as error:
        tail = _describe(error).strip().splitlines()
        result.update(status="error", error=tail[-1][:240] if tail else "unknown error")
    result["wall_seconds"] = round(time.perf_counter() - started, 3)
    (attempt_dir / LOAD_CHECK_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _final(attempt_dir: Path, contract: Contract, test_path: str, result: dict) -> None:
    timings = result["timings"]
    classification = is_classification(contract.task_type)

    with stage("import", timings):
        module = import_candidate(attempt_dir)
    with stage("data", timings):
        data = Data(contract)
    result["n_rows"] = int(data.valid.sum())
    result["excluded_columns"] = data.excluded

    rows = np.flatnonzero(data.valid)
    build_ctx = BuildContext(
        task=contract.task_type,
        n_jobs=contract.n_jobs,
        seed=contract.seed,
        n_rows=int(len(rows)),
        metrics=list(contract.metrics),
        primary_metric=contract.primary_metric,
        time_budget_seconds=contract.time_budget_seconds,
        classes=_plain(data.classes),
    )

    with stage("build", timings):
        estimator = call_build(module, data.columns, contract.task_type, build_ctx)
    with stage("estimator", timings):
        want_scores = _check_estimator(estimator, contract, data)

    with stage("final_fit", timings):
        clock = time.perf_counter()
        _fit(estimator, module, rows, contract, data, np.random.default_rng(contract.seed))
        result["fit_seconds"] = time.perf_counter() - clock

    with stage("test_data", timings):
        test = pd.read_csv(test_path, low_memory=False)
        if contract.target not in test.columns:
            raise StageFailure("test_data", f"target column {contract.target!r} is not in the test file")
        missing = [column for column in data.X.columns if column not in test.columns]
        if missing:
            raise StageFailure("test_data", f"the test file lacks training columns: {', '.join(missing)}")
        y_test = test[contract.target]
        test_rows = np.flatnonzero(y_test.notna().to_numpy())
        X_test = test[list(data.X.columns)].iloc[test_rows]

    with stage("test_predict", timings):
        predictions, scores = _predict(estimator, X_test, data.classes, want_scores)
        result["warnings"] += _check_predictions("test set", predictions, scores, len(test_rows), contract, data)
        if not classification:
            predictions = predictions.astype(float)

    y_true = y_test.iloc[test_rows]
    test_scores, problems = score_predictions(
        contract.metrics, contract.task_type, y_true, predictions, scores, data.classes
    )
    result["warnings"] += problems
    result["test_scores"] = {name: value for name, value in test_scores.items() if math.isfinite(value)}
    _prediction_frame(test_rows, y_true, predictions, scores, data.classes).to_csv(
        attempt_dir / TEST_PREDICTIONS_FILE, index=False
    )
    result["artifacts"]["test_predictions"] = TEST_PREDICTIONS_FILE

    # the scores stand whether or not the model can be kept; a model that
    # scored but cannot be saved is still a result, just not a deliverable
    try:
        import joblib

        joblib.dump(estimator, attempt_dir / MODEL_FILE)
    except Exception as error:
        result["warnings"].append(
            f"{MODEL_FILE} could not be saved ({type(error).__name__}: {str(error)[:200]}); "
            "the test scores stand, but there is no deliverable model"
        )
        return

    result["artifacts"]["model"] = MODEL_FILE


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def evaluate(attempt_dir: str | Path, final: bool = False, test_path: Optional[str] = None) -> dict:
    attempt_dir = Path(attempt_dir).resolve()
    started = time.perf_counter()
    result = {
        "status": "ok",
        "stage": "",
        "error": "",
        "mode": "final" if final else "loop",
        "cv_scores": {},
        "fold_scores": [],
        "diagnostics": {},
        "test_scores": {},
        "warnings": [],
        "timings": {},
        "artifacts": {},
        "n_rows": 0,
        "n_covered": 0,
        "fit_seconds": 0.0,
        "excluded_columns": [],
        "_dir": attempt_dir,
    }

    try:
        with stage("contract", result["timings"]):
            contract = Contract.load(attempt_dir / CONTRACT_FILE)
        if final:
            if not test_path:
                raise StageFailure("contract", "final mode needs the test file (--test)")
            _final(attempt_dir, contract, test_path, result)
        else:
            _loop(attempt_dir, contract, result, started)
    except StageFailure as failure:
        result.update(status=failure.status, stage=failure.stage, error=failure.message)
    except Exception as error:
        # a fault in the harness itself, not in the candidate
        result.update(status="error", stage="harness", error=_describe(error))

    result.pop("_dir", None)
    result["fit_seconds"] = round(result["fit_seconds"], 3)
    result["wall_seconds"] = round(time.perf_counter() - started, 3)
    name = FINAL_RESULT_FILE if final else RESULT_FILE
    (attempt_dir / name).write_text(json.dumps(_plain(result), indent=2), encoding="utf-8")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one AutoML candidate.")
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--test", default=None)
    parser.add_argument("--load-check", action="store_true", help="open the saved model and predict five test rows")
    args = parser.parse_args(argv)

    if args.load_check:
        checked = check_model(args.attempt_dir, args.test)
        print(f"load check {checked['status']}", flush=True)
        return 0

    result = evaluate(args.attempt_dir, final=args.final, test_path=args.test)
    outcome = result["status"] if not result["stage"] else f"{result['status']} at {result['stage']}"
    print(f"evaluation {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
