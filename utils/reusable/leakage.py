"""
leakage.py
----------
Decide whether a suspiciously good attempt is exploiting a leak, and keep the
run-wide record of columns removed because of one.

A score alone cannot tell a leak from a strong legitimate feature. What can be
measured is how much of the score depends on a few columns: re-run the same
candidate without the columns carrying most of its importance, and compare.
A leak is confirmed when the attempt was suspicious to begin with (a signal
that does not depend on the target's scale) AND most of its gain over the
baseline disappears without those columns.

Whether a confirmed leak is removed is a separate question only a person can
answer: a column known at prediction time is a legitimate input however well it
reconstructs the target. The schema declares that; the run's policy covers the
columns it does not.

The same question is asked once more cheaply, before any model is fitted: does
a simple formula of one or two columns reproduce the target on held-out rows?
A leak that exact is visible in the data itself, and catching it there means no
model ever trains on it.

    relation_screen(frame, target, task_type, columns, seed) -> list[hit]
    render_screen(hits)                                  -> str
    columns_to_ablate(attempt_dir)                       -> list[str]
    normalized_gain(score, floor, direction)             -> float | None
    judge_ablation(full, ablated, floor, direction)      -> (confirmed, retained)
    read_exclusions(run_dir) / add_exclusions(run_dir, columns, evidence)
"""

from __future__ import annotations

import json
import os
import threading
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

IMPORTANCE_FILE = "feature_importance.csv"
EXCLUSIONS_FILE = "exclusions.json"

IMPORTANCE_COVERAGE = 0.8     # ablate the fewest columns carrying this share of importance
MAX_ABLATED = 3               # a leak spread wider than this is not what this check is for
RETAINED_GAIN_LIMIT = 0.5     # keeping at most this share of the gain confirms the leak

SCREEN_ROWS = 20_000          # a sample is plenty to see an exact relation
SCREEN_TOP_K = 12             # columns most associated with the target; pairs grow as k^2
SCREEN_HOLDOUT = 0.3
SCREEN_R2 = 0.99              # a formula this exact on held-out rows reconstructs the target
SCREEN_BALANCED_ACCURACY = 0.995
SCREEN_TREE_DEPTH = 8
SCREEN_KINDS = ("numeric", "discrete_numeric", "binary")

_lock = threading.Lock()      # workers are threads in one process


# ---------------------------------------------------------------------------
# before any model: does a simple formula reproduce the target?
# ---------------------------------------------------------------------------

def _numeric_candidates(frame: pd.DataFrame, target: str, columns: List[dict]) -> List[str]:
    kinds = {column["name"]: column.get("kind") for column in columns}
    return [
        name for name in frame.columns
        if name != target
        and kinds.get(name, "numeric") in SCREEN_KINDS
        and pd.api.types.is_numeric_dtype(frame[name])
        and frame[name].notna().mean() > 0.5
    ]


def _association(values: pd.Series, target: pd.Series, classification: bool) -> float:
    if classification:
        # how much of the column's variance the class means explain
        grand = values.mean()
        total = float(((values - grand) ** 2).sum())
        if total == 0:
            return 0.0
        groups = values.groupby(target).agg(["mean", "size"])
        between = float((groups["size"] * (groups["mean"] - grand) ** 2).sum())
        return between / total
    correlation = values.corr(target, method="spearman")
    return 0.0 if pd.isna(correlation) else abs(float(correlation))


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return 0.0 if total == 0 else 1.0 - residual / total


def _linear_formula(names: List[str], coefficients, intercept) -> str:
    terms = " ".join(
        f"{'+' if value >= 0 else '-'} {abs(value):.4g}*{name}" for name, value in zip(names, coefficients)
    )
    return f"target = {terms.lstrip('+ ')} {'+' if intercept >= 0 else '-'} {abs(intercept):.4g}"


def relation_screen(
    frame: pd.DataFrame, target: str, task_type: str, columns: List[dict], seed: int = 0
) -> List[dict]:
    """Columns that reproduce the target almost exactly, alone or in pairs.

    Rank correlation sees one column at a time, so a target that is the product
    or difference of two columns can hide from it (units 0.72 and unit_price
    0.14 against revenue = units * unit_price). A simple fit on the pair cannot
    miss it. Every relation is scored on held-out rows, so a formula that merely
    memorises the sample does not count.

    Returns hits, strongest first: {"columns", "relation", "score", "measure", "formula"}.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if target not in frame.columns:
        return []
    classification = task_type.endswith("classification")
    data = frame[frame[target].notna()]
    if len(data) > SCREEN_ROWS:
        data = data.sample(SCREEN_ROWS, random_state=seed)
    names = _numeric_candidates(data, target, columns)
    if not names or len(data) < 50:
        return []

    y = data[target]
    if not classification:
        y = pd.to_numeric(y, errors="coerce")
        data, y = data[y.notna()], y[y.notna()]
    features = data[names].apply(lambda column: column.fillna(column.median()))

    ranked = sorted(names, key=lambda name: _association(features[name], y, classification), reverse=True)
    top = ranked[:SCREEN_TOP_K]

    rng = np.random.default_rng(seed)
    holdout = rng.random(len(data)) < SCREEN_HOLDOUT
    fit, check = ~holdout, holdout
    y_values = y.to_numpy()
    hits: List[dict] = []

    # one column: any function of it, as a shallow tree
    exact_alone = set()
    for name in top:
        X = features[[name]].to_numpy()
        if classification:
            tree = DecisionTreeClassifier(max_depth=SCREEN_TREE_DEPTH, random_state=seed).fit(X[fit], y_values[fit])
            score = balanced_accuracy_score(y_values[check], tree.predict(X[check]))
            passed, measure = score >= SCREEN_BALANCED_ACCURACY, "balanced accuracy"
        else:
            tree = DecisionTreeRegressor(max_depth=SCREEN_TREE_DEPTH, random_state=seed).fit(X[fit], y_values[fit])
            score = _r2(y_values[check], tree.predict(X[check]))
            passed, measure = score >= SCREEN_R2, "r2"
        if passed:
            exact_alone.add(name)
            hits.append({"columns": [name], "relation": "a function of this one column", "score": float(score),
                         "measure": measure, "formula": f"target determined by {name}"})

    # two columns: a weighted sum or difference, or a product or ratio
    for a, b in combinations([name for name in top if name not in exact_alone], 2):
        X = features[[a, b]].to_numpy(dtype=float)
        if classification:
            tree = DecisionTreeClassifier(max_depth=SCREEN_TREE_DEPTH, random_state=seed).fit(X[fit], y_values[fit])
            score = balanced_accuracy_score(y_values[check], tree.predict(X[check]))
            if score >= SCREEN_BALANCED_ACCURACY:
                hits.append({"columns": [a, b], "relation": "a function of these two columns", "score": float(score),
                             "measure": "balanced accuracy", "formula": f"target determined by {a} and {b}"})
            continue

        linear = LinearRegression().fit(X[fit], y_values[fit])
        best = (_r2(y_values[check], linear.predict(X[check])), "a weighted sum or difference",
                _linear_formula([a, b], linear.coef_, linear.intercept_))

        if (X > 0).all() and (y_values > 0).all():
            logs = np.log(X)
            log_fit = LinearRegression().fit(logs[fit], np.log(y_values[fit]))
            predicted = np.exp(log_fit.predict(logs[check]))
            powers = log_fit.coef_
            formula = (f"target = {np.exp(log_fit.intercept_):.4g} * {a}^{powers[0]:.3g} * {b}^{powers[1]:.3g}")
            candidate = (_r2(y_values[check], predicted), "a product or ratio", formula)
            if candidate[0] > best[0]:
                best = candidate

        if best[0] >= SCREEN_R2:
            hits.append({"columns": [a, b], "relation": best[1], "score": float(best[0]),
                         "measure": "r2", "formula": best[2]})

    return sorted(hits, key=lambda hit: hit["score"], reverse=True)


def render_screen(hits: List[dict]) -> str:
    """The screen's result as a profile section the planners and generators read."""
    if not hits:
        return ""
    lines = ["", "## LEAKAGE SCREEN (held-out rows, before any model)"]
    for hit in hits:
        outcome = hit.get("outcome", "")
        lines.append(
            f"- {', '.join(hit['columns'])}: {hit['formula']} ({hit['measure']} {hit['score']:.4f} on held-out "
            f"rows){' -> ' + outcome if outcome else ''}"
        )
    return "\n".join(lines)


def columns_to_ablate(attempt_dir: str | Path) -> List[str]:
    """The fewest raw columns carrying most of the attempt's permutation importance."""
    path = Path(attempt_dir) / IMPORTANCE_FILE
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if "feature" not in frame or "importance" not in frame:
        return []

    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    positive = frame[frame["importance"] > 0].sort_values("importance", ascending=False)
    total = float(positive["importance"].sum())
    if total <= 0:
        return []

    chosen, covered = [], 0.0
    for row in positive.itertuples():
        chosen.append(str(row.feature))
        covered += float(row.importance) / total
        if covered >= IMPORTANCE_COVERAGE or len(chosen) == MAX_ABLATED:
            break
    return chosen


def normalized_gain(score: Optional[float], floor: Optional[float], direction: str) -> Optional[float]:
    """How much of the possible improvement over the baseline a score achieved.

    1.0 is perfect, 0.0 is the baseline. For an error metric the best possible
    score is 0; every higher-is-better metric the run uses is bounded by 1.
    Scale-free, so the same rule holds for any target and any metric.
    """
    if score is None or floor is None:
        return None
    if direction == "lower":
        return None if floor <= 0 else (floor - score) / floor
    headroom = 1.0 - floor
    return None if headroom <= 0 else (score - floor) / headroom


def judge_ablation(
    full: Optional[float], ablated: Optional[float], floor: Optional[float], direction: str
) -> Tuple[Optional[bool], Optional[float]]:
    """(confirmed, retained share of the gain), or (None, None) when it cannot be measured."""
    full_gain = normalized_gain(full, floor, direction)
    ablated_gain = normalized_gain(ablated, floor, direction)
    if full_gain is None or ablated_gain is None or full_gain <= 0:
        return None, None
    retained = max(ablated_gain, 0.0) / full_gain
    return retained <= RETAINED_GAIN_LIMIT, retained


# ---------------------------------------------------------------------------
# the run-wide record
# ---------------------------------------------------------------------------

def read_exclusions(run_dir: str | Path) -> Dict[str, dict]:
    path = Path(run_dir) / EXCLUSIONS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def add_exclusions(run_dir: str | Path, columns: List[str], evidence: dict) -> Dict[str, dict]:
    """Record columns as excluded for the rest of the run; every worker reads this file."""
    path = Path(run_dir) / EXCLUSIONS_FILE
    with _lock:
        current = read_exclusions(run_dir)
        for column in columns:
            current.setdefault(column, dict(evidence, columns=list(columns)))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return current
