"""
guards.py
---------
Sanity checks on a finished attempt.

The harness computes every score from the candidate's predictions, so a score
can no longer be misreported. What is left are predictions that score but mean
something else: a target transform that was never inverted, a model that
predicts one constant, an improvement too large to be real, a cross-validation
estimate the test set contradicts.

Every check returns a Finding with a severity the selection rules act on:

    blocking   the attempt cannot be selected (its predictions mean something else)
    suspect    too good to be true; the loop runs a leak check before trusting it
    warning    worth reading, changes nothing on its own

    check_attempt(record, context) -> list[Finding]
    check_final(cv_record, test_record, context) -> list[str]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from models import AttemptRecord, Finding, RunContext
from utils.reusable.leakage import normalized_gain

SCALE_TOLERANCE = 10.0        # predictions this many times off the target's scale
SUSPICIOUS_FACTOR = 20.0      # error metric this many times better than baseline
NEAR_PERFECT = 0.999          # bounded primary metric above this on real data
NEAR_PERFECT_R2 = 0.99        # regression explained this completely by the features
NEAR_PERFECT_AUC = 0.995      # classes separated this completely
# a very strong score that almost entirely rests on one or two columns: the
# signature of a target rebuilt from its own components
CONCENTRATED_GAIN = 0.9       # share of the possible gain over the baseline
CONCENTRATED_SHARE = 0.9      # share of the importance carried by...
CONCENTRATED_COLUMNS = 2      # ...at most this many raw columns
OOF_FILE = "oof_predictions.csv"
IMPORTANCE_FILE = "feature_importance.csv"


def _load_oof(record: AttemptRecord) -> Optional[pd.DataFrame]:
    path = Path(record.script_path).parent / OOF_FILE
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "y_true" not in frame or "prediction" not in frame:
        return None
    return frame


def check_attempt(record: AttemptRecord, context: RunContext) -> List[Finding]:
    if record.status not in ("ok", "cached"):
        return []

    findings: List[Finding] = []
    metric = context.primary_metric
    cv = record.cv_scores.get(metric)

    if cv is None:
        findings.append(Finding(
            kind="no_primary_score", severity="blocking",
            message=f"no {metric} in cv_scores, so this attempt cannot be ranked",
        ))

    oof = _load_oof(record)
    if oof is None:
        findings.append(Finding(
            kind="missing_predictions", severity="blocking",
            message=f"no readable {OOF_FILE}, so this attempt's predictions cannot be inspected",
        ))
    else:
        findings += _check_predictions(oof, context)

    findings += _check_against_baseline(context, cv)
    findings += _check_scale_free(record, context)
    findings += _check_concentration(record, context, cv)
    return _one_suspicion(findings)


def _one_suspicion(findings: List[Finding]) -> List[Finding]:
    """Several signals of the same suspicion are one suspicion with more evidence."""
    suspects = [f for f in findings if f.kind == "suspect_leakage"]
    if len(suspects) <= 1:
        return findings
    merged = Finding(
        kind="suspect_leakage", severity="suspect",
        message="; ".join(f.message for f in suspects),
        evidence={key: value for f in suspects for key, value in f.evidence.items()},
    )
    return [f for f in findings if f.kind != "suspect_leakage"] + [merged]


def _check_predictions(oof: pd.DataFrame, context: RunContext) -> List[Finding]:
    findings: List[Finding] = []
    predictions = oof["prediction"]

    if predictions.nunique(dropna=True) <= 1:
        findings.append(Finding(
            kind="constant_predictions", severity="blocking",
            message=(
                f"every out-of-fold prediction is the same value ({predictions.iloc[0]}); "
                "the model learned nothing and is a baseline in disguise"
            ),
        ))

    if context.task_type != "regression":
        return findings

    # a target transformed inside the model and never inverted produces
    # predictions on a different scale from the target they are scored against
    y_true = pd.to_numeric(oof["y_true"], errors="coerce")
    y_pred = pd.to_numeric(predictions, errors="coerce")
    usable = np.isfinite(y_true) & np.isfinite(y_pred)
    true_median = float(np.median(np.abs(y_true[usable]))) if usable.any() else 0.0
    pred_median = float(np.median(np.abs(y_pred[usable]))) if usable.any() else 0.0
    if true_median > 0 and pred_median > 0:
        ratio = pred_median / true_median
        if ratio > SCALE_TOLERANCE or ratio < 1 / SCALE_TOLERANCE:
            findings.append(Finding(
                kind="untransformed_target", severity="blocking",
                message=(
                    f"out-of-fold predictions have median |{pred_median:.4g}| but the target's is "
                    f"{true_median:.4g} ({ratio:.3g}x apart): the target looks transformed and never "
                    "inverted, so the predictions are not in the target's units"
                ),
                evidence={"scale_ratio": ratio},
            ))
    return findings


def _check_against_baseline(context: RunContext, cv: Optional[float]) -> List[Finding]:
    metric = context.primary_metric
    floor = context.baseline.cv_scores.get(metric)
    if cv is None or floor in (None, 0) or not context.baseline.applicable:
        return []

    if context.metric_direction == "lower":
        if cv > floor:
            return [Finding(
                kind="worse_than_baseline", severity="warning",
                message=f"cv {metric} {cv:.4g} is worse than the {context.baseline.strategy} baseline ({floor:.4g})",
            )]
        if cv * SUSPICIOUS_FACTOR < floor:
            factor = floor / max(cv, 1e-12)
            return [Finding(
                kind="suspect_leakage", severity="suspect",
                message=(
                    f"cv {metric} {cv:.4g} beats the baseline ({floor:.4g}) by {factor:.0f}x, which is "
                    "large enough to suspect leakage or a unit mismatch rather than a better model"
                ),
                evidence={"baseline_factor": factor},
            )]
        return []

    if cv < floor:
        return [Finding(
            kind="worse_than_baseline", severity="warning",
            message=f"cv {metric} {cv:.4g} is worse than the {context.baseline.strategy} baseline ({floor:.4g})",
        )]
    if cv > NEAR_PERFECT:
        return [Finding(
            kind="suspect_leakage", severity="suspect",
            message=(
                f"cv {metric} {cv:.4g} is near perfect, which on real data usually means a leaked "
                "feature (suspect leakage) rather than a good model"
            ),
            evidence={"near_perfect": cv},
        )]
    return []


def _check_scale_free(record: AttemptRecord, context: RunContext) -> List[Finding]:
    """Suspicion that does not depend on the target's scale or the baseline.

    A baseline ratio only fires when the baseline happens to be weak enough;
    a model explaining nearly all of the target's variance is suspicious on any
    dataset.
    """
    r2 = record.diagnostics.get("r2")
    if context.task_type == "regression" and r2 is not None and r2 >= NEAR_PERFECT_R2:
        return [Finding(
            kind="suspect_leakage", severity="suspect",
            message=(
                f"cv r2 {r2:.4f}: the features explain almost all of the target's variance, "
                "so suspect leakage before trusting it"
            ),
            evidence={"r2": r2},
        )]
    auc = record.diagnostics.get("roc_auc")
    if context.task_type != "regression" and auc is not None and auc >= NEAR_PERFECT_AUC:
        return [Finding(
            kind="suspect_leakage", severity="suspect",
            message=f"cv roc_auc {auc:.4f}: the classes are separated almost perfectly, so suspect leakage",
            evidence={"roc_auc": auc},
        )]
    return []


def _check_concentration(record: AttemptRecord, context: RunContext, cv: Optional[float]) -> List[Finding]:
    """A near-complete gain over the baseline carried by one or two columns.

    Tree models approximate an exact product or difference in steps, so a leak
    that a linear model would turn into a perfect score can leave a booster just
    under every ratio and r2 threshold. What they cannot hide is where the score
    comes from: nearly all of the importance in the columns the target is built
    from. Both conditions are scale-free, so the rule holds for any dataset.
    """
    floor = context.baseline.cv_scores.get(context.primary_metric) if context.baseline.applicable else None
    gain = normalized_gain(cv, floor, context.metric_direction)
    if gain is None or gain < CONCENTRATED_GAIN:
        return []

    path = Path(record.script_path).parent / IMPORTANCE_FILE
    try:
        frame = pd.read_csv(path)
        importance = pd.to_numeric(frame["importance"], errors="coerce")
    except Exception:
        return []
    positive = frame[importance > 0].assign(importance=importance[importance > 0])
    total = float(positive["importance"].sum())
    if total <= 0:
        return []

    top = positive.sort_values("importance", ascending=False).head(CONCENTRATED_COLUMNS)
    share = float(top["importance"].sum()) / total
    if share < CONCENTRATED_SHARE:
        return []

    columns = ", ".join(str(name) for name in top["feature"])
    return [Finding(
        kind="suspect_leakage", severity="suspect",
        message=(
            f"cv {context.primary_metric} {cv:.4g} captures {gain:.0%} of the possible gain over the "
            f"baseline, and {columns} carry {share:.0%} of the importance: a target rebuilt from a "
            "few of its own inputs looks like this, so suspect leakage"
        ),
        evidence={"normalized_gain": gain, "top_share": share},
    )]


def check_final(cv_record: AttemptRecord, test_record: AttemptRecord, context: RunContext) -> List[str]:
    """Checks that need both halves: the loop attempt's cross-validation and the
    final run's test scores, which are two separate records."""
    metric = context.primary_metric
    cv = cv_record.cv_scores.get(metric)
    test = test_record.test_scores.get(metric)
    if cv is None or test is None or cv == 0:
        return []

    spread = abs(test - cv) / abs(cv)
    if spread > 0.5:
        better = "better" if ((test < cv) if context.metric_direction == "lower" else (test > cv)) else "worse"
        return [
            f"test {metric} {test:.4g} is {spread * 100:.0f}% {better} than cv {cv:.4g}; "
            "the two estimates disagree enough that one of them is not measuring what it claims"
        ]
    return []
