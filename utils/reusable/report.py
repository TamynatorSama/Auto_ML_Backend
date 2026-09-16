"""
report.py
---------
Turn the finished workers into one JSON document describing the run.

Everything here is derived from what actually happened: the attempt records, the
prediction files the scripts wrote, and the context the run was measured under.
Nothing is estimated, and the wording that interprets the numbers is a separate,
optional field rather than something baked into the data.

It is written for a consumer that was not present for the run. A leaderboard
alone cannot be checked: how the data was split, what floor the winner beat,
which attempt produced it, where it errs, and whether its saved model can even
be opened are all part of whether the number means anything, so all of it is in
the document.

    build_report(results, context) -> RunReport
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models import (
    AttemptRecord,
    ConfusionCell,
    DatasetInfo,
    ErrorBand,
    FeatureWeight,
    GenerationStep,
    MetricSpec,
    ModelResult,
    Protocol,
    RunContext,
    RunReport,
    RunTotals,
    ScoreRow,
    Warning,
)
from utils.reusable.leakage import read_exclusions
from utils.reusable.metrics import METRIC_DIRECTION

TOP_FEATURES = 15
ERROR_BANDS = 5

# rank groups: a result that may be selected always ranks above one that may not
_GROUPS = {"clean": 0, "unverified": 1}


def result_group(result: ModelResult, score: Optional[float]) -> int:
    """0 clean with a score, 1 unverified with a score, 2 blocked, 3 failed, 4 unavailable."""
    if result.eligibility == "blocked":
        return 2
    if score is not None and np.isfinite(score):
        return _GROUPS.get(result.eligibility, 1)
    # tried and failed still ranks above never tried at all
    return 4 if result.status == "unavailable" else 3

# phrases that mean a number should not be trusted, as opposed to a number that
# is merely disappointing
CRITICAL_MARKS = (
    "cannot be loaded",
    "not measuring what it claims",
    "is not a usable score",
    "was not computed from these rows",
    "never inverted",
    "suspect leakage",
    "learned nothing",
    "cannot be ranked",
    "unusable",
)


# ---------------------------------------------------------------------------
# rows and traces
# ---------------------------------------------------------------------------

def _cv(row: ScoreRow, metric: str) -> Optional[float]:
    """The score models are selected on. The test set is scored once per finalist
    to estimate how the chosen model generalises; choosing between models by it
    would turn that estimate into one more thing the run was tuned to."""
    value = row.cv_scores.get(metric)
    return value if value is not None and np.isfinite(value) else None


def _score(row: ScoreRow, metric: str) -> Optional[float]:
    value = row.test_scores.get(metric, row.cv_scores.get(metric))
    return value if value is not None and np.isfinite(value) else None


def _winner_record(result: ModelResult) -> Optional[AttemptRecord]:
    return next((r for r in result.attempts if r.attempt == result.best_attempt), None)


def _artifacts(record: Optional[AttemptRecord]) -> Dict[str, str]:
    if record is None or not record.script_path:
        return {}
    directory = Path(record.script_path).parent
    found = {"candidate": str(directory / "candidate.py")}
    for name in ("contract.json", "model.joblib", "oof_predictions.csv", "test_predictions.csv",
                 "feature_importance.csv"):
        if (directory / name).exists():
            found[name.split(".")[0]] = str(directory / name)
    return found


def _row(result: ModelResult) -> ScoreRow:
    generations = {r.generation for r in result.attempts if r.kind == "generate"}
    note = result.error
    if result.eligibility != "clean" and result.eligibility_note:
        note = f"{result.eligibility}: {result.eligibility_note}" + (f"; {note}" if note else "")
    return ScoreRow(
        model=result.model,
        status=result.status,
        test_scores=result.test_scores,
        cv_scores=result.best_cv_scores,
        generations=len(generations),
        repairs=sum(1 for r in result.attempts if r.kind == "repair"),
        # a leak check that ran nothing is a verdict, not an execution
        executions=sum(1 for r in result.attempts if r.status != "skipped"),
        best_attempt=result.best_attempt,
        wall_seconds=round(sum(r.wall_seconds or 0.0 for r in result.attempts), 1),
        artifacts=_artifacts(_winner_record(result)),
        note=note,
        eligibility=result.eligibility,
    )


def _trace(result: ModelResult, context: RunContext) -> List[GenerationStep]:
    """Generation by generation: what changed, and what it bought.

    Repairs fold into the generation they belong to rather than appearing as
    steps of their own — they were making one idea run, not trying a new one.
    """
    metric = context.improvement_metric
    lower_is_better = METRIC_DIRECTION.get(metric, "lower") == "lower"
    steps: List[GenerationStep] = []
    best: Optional[float] = None

    for record in sorted(result.attempts, key=lambda r: r.attempt):
        if record.kind != "generate":
            continue

        family = [r for r in result.attempts if r.generation == record.generation]
        scored = [r for r in family if r.cv_scores.get(metric) is not None]
        chosen = (
            min(scored, key=lambda r: r.cv_scores[metric] * (1 if lower_is_better else -1))
            if scored else None
        )
        score = chosen.cv_scores[metric] if chosen else None

        delta = None
        if score is not None and best is not None:
            delta = (best - score) if lower_is_better else (score - best)
        if score is not None and (best is None or (score < best if lower_is_better else score > best)):
            best = score

        steps.append(
            GenerationStep(
                generation=record.generation,
                attempt=(chosen or record).attempt,
                status=(chosen or family[-1]).status,
                changes=record.changes,
                score=score,
                delta=delta,
                repairs=sum(1 for r in family if r.kind == "repair"),
                judge_notes=(chosen or family[-1]).judge_notes,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# what the winner's own files say about it
# ---------------------------------------------------------------------------

def _read(record: Optional[AttemptRecord], filename: str) -> Optional[pd.DataFrame]:
    if record is None or not record.script_path:
        return None
    path = Path(record.script_path).parent / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _importance(record: Optional[AttemptRecord]) -> List[FeatureWeight]:
    frame = _read(record, "feature_importance.csv")
    if frame is None or "feature" not in frame or "importance" not in frame:
        return []

    frame = frame.copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").abs()
    frame = frame.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    total = float(frame["importance"].sum()) or 1.0
    return [
        FeatureWeight(
            feature=str(row.feature),
            importance=float(row.importance),
            share=round(float(row.importance) / total, 6),
        )
        for row in frame.head(TOP_FEATURES).itertuples()
    ]


def _error_bands(record: Optional[AttemptRecord]) -> List[ErrorBand]:
    """Where the winner errs across the range of the target.

    One error number hides whether a model is uniformly decent or excellent on
    the common rows and hopeless on the expensive ones.
    """
    frame = _read(record, "test_predictions.csv")
    if frame is None or "y_true" not in frame or "prediction" not in frame:
        return []

    y_true = pd.to_numeric(frame["y_true"], errors="coerce")
    y_pred = pd.to_numeric(frame["prediction"], errors="coerce")
    usable = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[usable], y_pred[usable]
    if len(y_true) == 0:
        return []

    try:
        buckets = pd.qcut(y_true, ERROR_BANDS, duplicates="drop")
    except ValueError:
        return []

    bands: List[ErrorBand] = []
    for interval, index in y_true.groupby(buckets, observed=True).groups.items():
        actual, predicted = y_true.loc[index], y_pred.loc[index]
        bands.append(
            ErrorBand(
                band=f"{interval.left:,.4g} to {interval.right:,.4g}",
                lower=float(interval.left),
                upper=float(interval.right),
                rows=len(index),
                mean_absolute_error=float(np.mean(np.abs(actual - predicted))),
                mean_signed_error=float(np.mean(predicted - actual)),
            )
        )
    return bands


def _confusion(record: Optional[AttemptRecord]) -> List[ConfusionCell]:
    frame = _read(record, "test_predictions.csv")
    if frame is None or "y_true" not in frame or "prediction" not in frame:
        return []
    pairs = frame.groupby(["y_true", "prediction"], observed=True).size()
    return [
        ConfusionCell(actual=str(actual), predicted=str(predicted), rows=int(rows))
        for (actual, predicted), rows in pairs.items()
    ]


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------

def _warnings(results: List[ModelResult]) -> List[Warning]:
    flags: List[Warning] = []
    for result in results:
        winner = _winner_record(result)
        seen = set()

        for message in (winner.warnings if winner else []):
            stage = "artifact" if "cannot be loaded" in message else "cross_validation"
            if message not in seen:
                seen.add(message)
                flags.append(_flag(result.model, stage, message))

        for message in result.test_warnings:
            if message not in seen:
                seen.add(message)
                flags.append(_flag(result.model, "test", message))

        if result.error and result.error not in seen:
            flags.append(_flag(result.model, "run", result.error))

        if result.eligibility != "clean" and result.eligibility_note:
            flags.append(Warning(
                model=result.model, severity="critical", stage="selection",
                message=f"{result.eligibility}: {result.eligibility_note}",
            ))
    return flags


def _flag(model: str, stage: str, message: str) -> Warning:
    severity = "critical" if any(mark in message for mark in CRITICAL_MARKS) else "warning"
    return Warning(model=model, severity=severity, stage=stage, message=message)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _rows_in(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for _ in handle) - 1
    except OSError:
        return 0


def _reason(selected: Optional[ScoreRow], scored: List[ScoreRow], context: RunContext) -> str:
    if selected is None:
        return "no model produced a usable score"

    metric = context.primary_metric
    top = _cv(selected, metric)
    parts = [f"best cross-validated {metric} of {top:.4g} among selectable models"]
    test = selected.test_scores.get(metric)
    if test is not None:
        parts.append(f"{metric} {test:.4g} on the held-out test set")

    if len(scored) > 1:
        runner_up = scored[1]
        second = _cv(runner_up, metric)
        if second:
            gap = abs(second - top) / abs(second)
            parts.append(f"{gap * 100:.1f}% ahead of {runner_up.model} in cross-validation ({second:.4g})")

    floor = context.baseline.cv_scores.get(metric)
    if floor and top:
        parts.append(f"{abs(floor) / abs(top):.1f}x better than the {context.baseline.strategy} baseline")

    if selected.best_attempt is not None:
        parts.append(f"from attempt {selected.best_attempt} of {selected.executions}")

    return "; ".join(parts)


def build_report(results: List[ModelResult], context: RunContext) -> RunReport:
    metric = context.primary_metric
    lower_is_better = context.metric_direction == "lower"

    by_model = {result.model: result for result in results}
    rows = [_row(result) for result in results]
    ranked = sorted(
        rows,
        key=lambda row: (
            result_group(by_model[row.model], _cv(row, metric)),
            (_cv(row, metric) or 0.0) * (1 if lower_is_better else -1),
            row.model,
        ),
    )
    for position, row in enumerate(ranked, start=1):
        row.rank = position

    # only a selectable result can be selected, however it scored
    scored = [
        row for row in ranked
        if _cv(row, metric) is not None and row.eligibility != "blocked"
    ]
    selected = scored[0] if scored else None
    if selected is not None:
        selected.selected = True

    winner = next((r for r in results if selected and r.model == selected.model), None)
    winner_record = _winner_record(winner) if winner else None

    floor = context.baseline.cv_scores.get(metric)
    improvement = None
    if selected is not None and floor:
        top = _score(selected, metric)
        improvement = (floor - top) / abs(floor) if lower_is_better else (top - floor) / abs(floor)

    plan = context.split_plan
    report = RunReport(
        run_id=context.run_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="complete" if scored else "no model produced a score",
        dataset=DatasetInfo(
            target=context.target,
            task_type=context.task_type,
            train_rows=_rows_in(context.train_path),
            test_rows=_rows_in(context.test_path),
            train_path=context.train_path,
            test_path=context.test_path,
            target_stats=context.target_stats,
        ),
        protocol=Protocol(
            split_method=plan.method,
            test_size=plan.test_size,
            seed=plan.random_seed,
            drop_duplicates=plan.drop_duplicates,
            cv_strategy=plan.cv_strategy,
            cv_folds=plan.cv_folds,
            split_reason=plan.reason,
            split_warnings=plan.warnings,
            baseline_strategy=context.baseline.strategy,
            baseline_applicable=context.baseline.applicable,
            baseline_scores=context.baseline.cv_scores,
            max_tries=context.max_tries,
            early_stopping_patience=context.early_stopping_patience,
            improvement_delta=context.improvement_delta,
            improvement_mode=context.improvement_mode,
            time_budget_seconds=context.time_budget_seconds,
            environment=context.environment,
            requirements=[
                {
                    "column": r.column or "",
                    "issue": r.issue,
                    "requirement": r.requirement,
                    "applies_to": ", ".join(r.applies_to),
                }
                for r in context.preprocessing_requirements
            ],
        ),
        totals=RunTotals(
            models_planned=len(results),
            models_scored=len(scored),
            executions=sum(row.executions for row in rows),
            generations=sum(row.generations for row in rows),
            repairs=sum(row.repairs for row in rows),
            failures=sum(1 for row in rows if row.status in ("failed", "unavailable")),
            wall_seconds=round(sum(row.wall_seconds for row in rows), 1),
        ),
        metrics=[
            MetricSpec(
                name=name,
                direction=METRIC_DIRECTION.get(name, "lower"),
                primary=(name == metric),
                drives_improvement=(name == context.improvement_metric),
            )
            for name in context.eval_matrics
        ],
        selected_model=selected.model if selected else None,
        selection_reason=_reason(selected, scored, context),
        improvement_over_baseline=improvement,
        comparison=ranked,
        trace={result.model: _trace(result, context) for result in results},
        importance=_importance(winner_record),
        warnings=_warnings(results),
        exclusions=[
            {"column": column, **evidence}
            for column, evidence in read_exclusions(context.run_dir).items()
        ],
        leakage_screen=list(context.leakage_screen),
    )

    if context.task_type == "regression":
        report.error_bands = _error_bands(winner_record)
    else:
        report.confusion = _confusion(winner_record)

    return report
