"""
guards.py
---------
Sanity checks on a finished attempt.

These catch the failures that do not raise: a script that exits cleanly, prints
a well-formed result, and reports a score that is wrong. The worst of them look
like success — a target transform that is never inverted reports an error metric
in log units, which reads as a twenty-fold improvement and stops the loop early.

Nothing here decides anything. Every check returns a sentence for the judge and
the report to read, because a warning can also be a genuinely good model.

    check_attempt(record, context) -> list[str]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from models import AttemptRecord, RunContext
from utils.reusable.metrics import METRIC_DIRECTION, score

SCALE_TOLERANCE = 10.0        # predictions this many times off the target's scale
SCORE_MISMATCH = 0.02         # reported vs recomputed, as a fraction
SUSPICIOUS_FACTOR = 20.0      # error metric this many times better than baseline
NEAR_PERFECT = 0.999          # bounded metric above this on real data


def _load_oof(record: AttemptRecord) -> Optional[pd.DataFrame]:
    # the artifact key is whatever the script chose to call it, so match on the
    # filename rather than insisting on one spelling of the key
    directory = Path(record.script_path).parent
    candidates = [
        value for key, value in record.artifacts.items()
        if "oof" in str(key).lower() or "oof" in str(value).lower()
    ]
    candidates += [p.name for p in sorted(directory.glob("*oof*.csv"))]

    path = next((directory / name for name in candidates if (directory / name).exists()), None)
    if path is None:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if "y_true" not in frame or "oof_prediction" not in frame:
        return None
    return frame


def check_attempt(record: AttemptRecord, context: RunContext) -> List[str]:
    if record.status not in ("ok", "cached"):
        return []

    warnings: List[str] = []
    metric = context.primary_metric
    reported = record.cv_scores.get(metric)

    if reported is None:
        warnings.append(f"no {metric} in cv_scores, so this attempt cannot be ranked")

    for name, value in record.cv_scores.items():
        if not np.isfinite(value):
            warnings.append(f"cv {name} is {value}, which is not a usable score")

    oof = _load_oof(record)
    if oof is None:
        warnings.append(
            "no readable oof_predictions.csv, so the reported scores cannot be checked "
            "against the predictions they came from"
        )
    else:
        warnings += _check_predictions(oof, record, context, reported)

    warnings += _check_against_baseline(record, context, reported)
    warnings += _check_cv_versus_test(record, context)
    warnings += _check_model_loads(record)
    return warnings


def _check_model_loads(record: AttemptRecord) -> List[str]:
    """Open the saved model in an interpreter that never saw the script.

    Pickle stores functions by reference, so a pipeline holding a transformer the
    script defined saves without complaint and cannot be loaded by anything else.
    Checking it here, in a clean process, is the only way to find out — and the
    model is what a run is ultimately for.
    """
    name = next(
        (v for k, v in record.artifacts.items() if str(v).endswith(".joblib")),
        None,
    )
    if not name or not record.script_path:
        return []

    path = Path(record.script_path).parent / name
    if not path.exists():
        return [f"{name} was reported as an artifact but is not on disk"]

    probe = f"import joblib; joblib.load(r'{path}')"
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(Path.cwd())
    )
    if finished.returncode == 0:
        return []

    tail = (finished.stderr or "").strip().splitlines()
    return [
        f"{name} cannot be loaded outside the script that wrote it "
        f"({tail[-1][:160] if tail else 'unknown error'}); it holds something defined in the "
        "script, so it is not usable as a deliverable"
    ]


def _check_predictions(
    oof: pd.DataFrame, record: AttemptRecord, context: RunContext, reported: Optional[float]
) -> List[str]:
    warnings: List[str] = []
    y_true = pd.to_numeric(oof["y_true"], errors="coerce")
    y_pred = pd.to_numeric(oof["oof_prediction"], errors="coerce")

    bad = int((~np.isfinite(y_pred)).sum())
    if bad:
        warnings.append(f"{bad} of {len(y_pred)} out-of-fold predictions are NaN or infinite")

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return warnings + ["every out-of-fold prediction is unusable"]

    y_true, y_pred = y_true[finite], y_pred[finite]

    if y_pred.nunique() == 1:
        warnings.append(
            f"every out-of-fold prediction is the same value ({y_pred.iloc[0]:.4g}); "
            "the model learned nothing and is a baseline in disguise"
        )

    # the target transform check: y_true in the file should live on the same
    # scale as the target in the profile. If a script reassigned y, both columns
    # here are in the transformed space and agree with each other, so only the
    # comparison against the real target catches it.
    target_median = context.target_stats.get("median")
    if target_median not in (None, 0) and context.task_type == "regression":
        file_median = float(np.median(np.abs(y_true)))
        if file_median > 0:
            ratio = file_median / abs(target_median)
            if ratio > SCALE_TOLERANCE or ratio < 1 / SCALE_TOLERANCE:
                warnings.append(
                    f"out-of-fold y_true has median |{file_median:.4g}| but the training target's "
                    f"median is {target_median:.4g} ({ratio:.3g}x apart): the target looks "
                    "transformed and never inverted, so these scores are not in the target's units"
                )

    # the strongest check available: recompute the metric from the predictions
    # the script itself saved
    if reported is not None and context.primary_metric in METRIC_DIRECTION:
        try:
            recomputed = score(context.primary_metric, y_true, y_pred)
        except Exception:
            return warnings
        spread = abs(recomputed - reported) / max(abs(reported), 1e-9)
        if spread > SCORE_MISMATCH:
            warnings.append(
                f"reported cv {context.primary_metric} is {reported:.4g} but its own out-of-fold "
                f"predictions give {recomputed:.4g}: the score was not computed from these rows"
            )

    return warnings


def _check_against_baseline(
    record: AttemptRecord, context: RunContext, reported: Optional[float]
) -> List[str]:
    metric = context.primary_metric
    floor = context.baseline.cv_scores.get(metric)
    if reported is None or floor in (None, 0) or not context.baseline.applicable:
        return []

    if context.metric_direction == "lower":
        if reported > floor:
            return [
                f"cv {metric} {reported:.4g} is worse than the {context.baseline.strategy} "
                f"baseline ({floor:.4g})"
            ]
        if reported * SUSPICIOUS_FACTOR < floor:
            return [
                f"cv {metric} {reported:.4g} beats the baseline ({floor:.4g}) by "
                f"{floor / max(reported, 1e-9):.0f}x, which is large enough to suspect leakage "
                "or a unit mismatch rather than a better model"
            ]
        return []

    if reported < floor:
        return [
            f"cv {metric} {reported:.4g} is worse than the {context.baseline.strategy} "
            f"baseline ({floor:.4g})"
        ]
    if reported > NEAR_PERFECT:
        return [
            f"cv {metric} {reported:.4g} is near perfect, which on real data usually means a "
            "leaked feature rather than a good model"
        ]
    return []


def _check_cv_versus_test(record: AttemptRecord, context: RunContext) -> List[str]:
    metric = context.primary_metric
    cv = record.cv_scores.get(metric)
    test = record.test_scores.get(metric)
    if cv is None or test is None or cv == 0:
        return []

    spread = abs(test - cv) / abs(cv)
    if spread > 0.5:
        better = "better" if (
            (test < cv) if context.metric_direction == "lower" else (test > cv)
        ) else "worse"
        return [
            f"test {metric} {test:.4g} is {spread * 100:.0f}% {better} than cv {cv:.4g}; "
            "the two estimates disagree enough that one of them is not measuring what it claims"
        ]
    return []
