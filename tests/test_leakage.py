import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from automl_runtime.folds import build_folds, save_folds
from code_gen_eval import code_gen_subgraph as loop
from code_gen_eval.runner import run_candidate
from models import AttemptRecord, Baseline, Configs, Finding, RunContext, SplitPlan
from utils.reusable.baseline import run_baseline
from utils.reusable.eligibility import BLOCKED, CLEAN, UNVERIFIED, assess, ranked_eligible, replay
from utils.reusable.leakage import (
    add_exclusions,
    columns_to_ablate,
    judge_ablation,
    normalized_gain,
    read_exclusions,
    relation_screen,
)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_normalized_gain_is_scale_free():
    # an error metric: halfway from the baseline to perfect is 0.5 on any scale
    assert normalized_gain(50.0, 100.0, "lower") == pytest.approx(0.5)
    assert normalized_gain(5000.0, 10000.0, "lower") == pytest.approx(0.5)
    # a bounded metric: from a 0.6 floor, 0.8 is halfway to 1.0
    assert normalized_gain(0.8, 0.6, "higher") == pytest.approx(0.5)


def test_losing_most_of_the_gain_confirms_and_keeping_it_clears():
    # the building-materials numbers: mae 18 with units, 918 without, floor 1335
    confirmed, retained = judge_ablation(18.36, 918.1, 1334.65, "lower")
    assert confirmed is True and retained == pytest.approx(0.316, abs=0.01)
    # a strong but legitimate feature: most of the gain survives without it
    confirmed, retained = judge_ablation(20.0, 30.0, 100.0, "lower")
    assert confirmed is False and retained == pytest.approx(0.875)
    # nothing to compare against
    assert judge_ablation(None, 30.0, 100.0, "lower") == (None, None)


def test_columns_to_ablate_takes_the_fewest_carrying_most_importance(tmp_path):
    pd.DataFrame({
        "feature": ["units", "unit_price", "region", "noise"],
        "importance": [960.0, 30.0, 10.0, -5.0],
    }).to_csv(tmp_path / "feature_importance.csv", index=False)
    assert columns_to_ablate(tmp_path) == ["units"]

    pd.DataFrame({"feature": ["a", "b", "c"], "importance": [5.0, 4.0, 1.0]}).to_csv(
        tmp_path / "feature_importance.csv", index=False
    )
    assert columns_to_ablate(tmp_path) == ["a", "b"]
    assert columns_to_ablate(tmp_path / "missing") == []


def test_exclusions_accumulate_and_keep_the_first_evidence(tmp_path):
    add_exclusions(tmp_path, ["units"], {"reason": "first"})
    add_exclusions(tmp_path, ["units", "unit_price"], {"reason": "second"})
    stored = read_exclusions(tmp_path)
    assert stored["units"]["reason"] == "first"
    assert stored["unit_price"]["reason"] == "second"


# ---------------------------------------------------------------------------
# eligibility and replay
# ---------------------------------------------------------------------------

def _context(run_dir, **overrides) -> RunContext:
    fields = dict(
        run_id=1, run_dir=str(run_dir), train_path="", test_path="",
        split_plan=SplitPlan(method="random", test_size=0.2, random_seed=7, shuffle=True, drop_duplicates=False,
                             cv_strategy="kfold", cv_folds=3, reason="test"),
        summary="", target="y", task_type="regression",
        baseline={"strategy": "median", "applicable": True, "cv_scores": {"mae": 1334.65, "r2": -0.1}},
        eval_matrics=["mae", "r2"], primary_metric="mae", metric_direction="lower",
        improvement_metric="mae", improvement_delta=0.01, improvement_mode="relative",
        early_stopping_patience=1, max_tries=4, time_budget_seconds=300, n_jobs=1,
    )
    fields.update(overrides)
    return RunContext(**fields)


def _record(attempt, mae, findings=(), excluded=(), kind="generate", **extra) -> AttemptRecord:
    return AttemptRecord(
        model="mlp", attempt=attempt, kind=kind, script_path=f"attempt_{attempt}/candidate.py",
        cv_scores={"mae": mae} if mae is not None else {}, findings=list(findings),
        excluded_columns=list(excluded), **extra,
    )


SUSPECT = Finding(kind="suspect_leakage", severity="suspect", message="suspect leakage: 73x")


def test_the_leak_fix_is_no_longer_thrown_away(tmp_path):
    """The run that motivated this: the leaky attempt scored better than the fix."""
    context = _context(tmp_path)
    leaky = _record(6, 18.36, findings=[SUSPECT])
    check = _record(7, 918.1, kind="ablation", ablation_of=6, verdict="confirmed",
                    ablated_columns=["units"], excluded=["units"])
    honest = _record(8, 918.1, excluded=["units"])
    exclusions = {"units": {"reason": "leak confirmed"}}
    attempts = [leaky, check, honest]

    assert assess(leaky, attempts, exclusions)[0] == BLOCKED
    assert assess(honest, attempts, exclusions)[0] == CLEAN
    best, score, patience = replay(attempts, context, exclusions)
    assert (best, score, patience) == (8, 918.1, 0)
    assert [record.attempt for _, record in ranked_eligible(attempts, context, exclusions)] == [8]


def test_an_exclusion_found_by_another_worker_blocks_retroactively(tmp_path):
    context = _context(tmp_path)
    used_units = _record(1, 60.0)
    attempts = [used_units]
    assert assess(used_units, attempts, {})[0] == CLEAN
    assert assess(used_units, attempts, {"units": {"reason": "found by catboost"}})[0] == BLOCKED
    assert replay(attempts, context, {"units": {"reason": "x"}})[0] is None


def test_unverified_ranks_below_clean_whatever_the_score(tmp_path):
    context = _context(tmp_path)
    unsettled = _record(1, 10.0, findings=[SUSPECT])
    check = _record(1, None, kind="ablation", ablation_of=1, verdict="unresolved", status="skipped")
    clean = _record(2, 500.0)
    attempts = [unsettled, check, clean]

    assert assess(unsettled, attempts, {})[0] == UNVERIFIED
    ranked = ranked_eligible(attempts, context, {})
    assert [record.attempt for _, record in ranked] == [2, 1]
    assert replay(attempts, context, {})[0] == 2


def test_blocking_findings_and_cached_copies(tmp_path):
    constant = Finding(kind="constant_predictions", severity="blocking", message="learned nothing")
    assert assess(_record(1, 5.0, findings=[constant]), [], {})[0] == BLOCKED

    original = _record(3, 9.0, findings=[SUSPECT])
    cleared = _record(4, 20.0, kind="ablation", ablation_of=3, verdict="cleared")
    copy = _record(5, 9.0, findings=[SUSPECT], cached_from=3, status="cached")
    assert assess(copy, [original, cleared, copy], {})[0] == CLEAN


# ---------------------------------------------------------------------------
# the leak check end to end, on data with a real leak
# ---------------------------------------------------------------------------

DYNAMIC = """# model: linear_regression
# attempt: 1
# changes: initial implementation
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from automl_runtime import names


def build_pipeline(columns, task, ctx):
    numeric = names(columns, "numeric")
    prep = ColumnTransformer([("log", FunctionTransformer(np.log, feature_names_out="one-to-one"), numeric)])
    model = Pipeline([("prep", prep), ("model", LinearRegression())])
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)
"""

HARDCODED = DYNAMIC.replace('numeric = names(columns, "numeric")', 'numeric = ["a", "b", "c", "d"]')


def _leaky_run(tmp_path, target="product", available=()):
    rng = np.random.default_rng(0)
    n = 1500
    frame = pd.DataFrame({
        "a": rng.uniform(1, 100, n), "b": rng.uniform(1, 50, n),
        "c": rng.uniform(1, 10, n), "d": rng.uniform(1, 10, n),
    })
    frame["y"] = frame["a"] * frame["b"] if target == "product" else frame["a"] + rng.normal(0, 40, n)
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)

    plan = SplitPlan(method="random", test_size=0.2, random_seed=7, shuffle=True, drop_duplicates=False,
                     cv_strategy="kfold", cv_folds=3, reason="test")
    folds = build_folds("kfold", 3, 7, frame["y"])
    folds_path = save_folds(tmp_path / "folds.npz", folds)
    config = Configs(id=0, models=["linear_regression"], baseline=Baseline(strategy="median", applicable=True),
                     eval_matrics=["mae", "r2"], early_stopping_patience=1, max_tries=3,
                     improvement_delta=0.01, improvement_mode="relative", improvement_metric="mae")
    baseline = run_baseline(frame, "y", "regression", plan, config, folds)
    columns = [{"name": name, "kind": "numeric", "dtype": "float64"} for name in "abcd"]
    return _context(
        tmp_path, train_path=str(train_path), test_path=str(train_path), split_plan=plan,
        folds_path=str(folds_path), columns=columns, baseline=baseline,
        available_at_prediction=list(available),
    )


def _run_and_verify(context, code):
    record = run_candidate(code, "linear_regression", 1, context)
    assert record.status == "ok", record.traceback
    state = {"context": context, "model": "linear_regression", "attempts": [record],
             "attempt": 1, "current_code": code}
    return record, state


def test_a_real_leak_is_confirmed_and_excluded_for_the_run(tmp_path):
    context = _leaky_run(tmp_path)
    record, state = _run_and_verify(context, DYNAMIC)

    assert any(f.kind == "suspect_leakage" for f in record.findings)
    assert loop.route_after_run(state) == "verify"

    update = loop.verify(state)
    attempts = state["attempts"] + update["attempts"]
    check = update["attempts"][-1]

    assert check.verdict == "confirmed", check.decision
    assert set(check.ablated_columns) == {"a", "b"}
    assert set(read_exclusions(tmp_path)) == {"a", "b"}
    assert assess(record, attempts, read_exclusions(tmp_path))[0] == BLOCKED
    assert loop.route_after_run({**state, "attempts": attempts[:1] + [check]}) != "verify"


def test_declared_available_columns_are_never_excluded(tmp_path):
    context = _leaky_run(tmp_path, available=["a", "b"])
    record, state = _run_and_verify(context, DYNAMIC)

    update = loop.verify(state)
    check = update["attempts"][-1]

    assert check.verdict == "declared_available"
    assert read_exclusions(tmp_path) == {}
    assert assess(record, state["attempts"] + update["attempts"], {})[0] == CLEAN


def test_a_candidate_that_cannot_run_without_the_columns_is_measured_with_the_reference(tmp_path):
    context = _leaky_run(tmp_path)
    record, state = _run_and_verify(context, HARDCODED)

    update = loop.verify(state)
    measured = update["attempts"]

    assert len(measured) == 3                          # candidate failed, reference with and without
    assert measured[0].status == "error"
    assert measured[-1].verdict == "confirmed", measured[-1].decision
    assert "reference model" in measured[-1].decision
    assert update["attempt"] == 4


def test_an_honest_model_is_not_sent_to_a_leak_check(tmp_path):
    context = _leaky_run(tmp_path, target="noisy")
    code = DYNAMIC.replace("TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)", "model")
    record, state = _run_and_verify(context, code)

    assert not any(f.kind == "suspect_leakage" for f in record.findings)
    assert loop.route_after_run(state) == "judge"


# ---------------------------------------------------------------------------
# refinements from the first live run
# ---------------------------------------------------------------------------

def test_a_booster_just_under_every_threshold_is_caught_by_concentration(tmp_path):
    """The live run: catboost at 19.9x the baseline and r2 0.969, with units and
    unit_price carrying 97% of the importance, was never checked."""
    from utils.reusable.guards import check_attempt

    attempt_dir = tmp_path / "catboost" / "attempt_2"
    attempt_dir.mkdir(parents=True)
    pd.DataFrame({"feature": ["units", "unit_price", "product_category"],
                  "importance": [4143.4, 2598.1, 198.8]}).to_csv(attempt_dir / "feature_importance.csv", index=False)
    pd.DataFrame({"row_index": [0, 1], "y_true": [100.0, 200.0], "prediction": [110.0, 190.0]}).to_csv(
        attempt_dir / "oof_predictions.csv", index=False)
    record = AttemptRecord(model="catboost", attempt=2, script_path=str(attempt_dir / "candidate.py"),
                           cv_scores={"mae": 67.24}, diagnostics={"r2": 0.9688})

    findings = check_attempt(record, _context(tmp_path))
    suspect = [f for f in findings if f.kind == "suspect_leakage"]
    assert len(suspect) == 1
    assert "units, unit_price" in suspect[0].message


def test_a_strong_model_spread_over_many_columns_is_not_suspect(tmp_path):
    from utils.reusable.guards import check_attempt

    attempt_dir = tmp_path / "attempt_1"
    attempt_dir.mkdir()
    pd.DataFrame({"feature": list("abcdef"), "importance": [30.0, 25.0, 20.0, 15.0, 6.0, 4.0]}).to_csv(
        attempt_dir / "feature_importance.csv", index=False)
    pd.DataFrame({"row_index": [0, 1], "y_true": [100.0, 200.0], "prediction": [110.0, 190.0]}).to_csv(
        attempt_dir / "oof_predictions.csv", index=False)
    record = AttemptRecord(model="lightgbm", attempt=1, script_path=str(attempt_dir / "candidate.py"),
                           cv_scores={"mae": 67.0}, diagnostics={"r2": 0.97})

    assert not any(f.kind == "suspect_leakage" for f in check_attempt(record, _context(tmp_path)))


def test_models_are_selected_on_cross_validation_not_the_test_set(tmp_path):
    """The live run ranked mlp first on a test mae of 21.5 while its cv mae was 9227."""
    from code_gen_eval.base import rank_results
    from models import ModelResult
    from utils.reusable.report import build_report

    context = _context(tmp_path)
    diverged = ModelResult(model="mlp", status="no_improvement", best_attempt=2,
                           best_cv_scores={"mae": 9227.2}, test_scores={"mae": 21.5})
    steady = ModelResult(model="ridge", status="no_improvement", best_attempt=2,
                         best_cv_scores={"mae": 601.0}, test_scores={"mae": 521.3})

    assert [r.model for r in rank_results([diverged, steady], context)] == ["ridge", "mlp"]
    report = build_report([diverged, steady], context)
    assert report.selected_model == "ridge"
    assert "cross-validated" in report.selection_reason


# ---------------------------------------------------------------------------
# step 3: the screen before any model
# ---------------------------------------------------------------------------

def _screen_frame(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "units": rng.integers(1, 400, n).astype(float),
        "unit_price": rng.uniform(2, 90, n),
        "sales": rng.uniform(50, 5000, n),
        "discount": rng.uniform(0, 0.3, n),
        "region_code": rng.integers(1, 9, n),
    })
    frame["cost"] = frame["sales"] * rng.uniform(0.5, 0.9, n)
    return frame


NUMERIC = [{"name": name, "kind": "numeric"} for name in ("units", "unit_price", "sales", "discount", "cost")] + [
    {"name": "region_code", "kind": "discrete_numeric"}]


def test_screen_finds_a_product_a_difference_and_a_copy():
    frame = _screen_frame()

    frame["revenue"] = frame["units"] * frame["unit_price"]
    hits = relation_screen(frame, "revenue", "regression", NUMERIC)
    assert [set(h["columns"]) for h in hits][:1] == [{"units", "unit_price"}]
    assert hits[0]["relation"] == "a product or ratio" and hits[0]["score"] > 0.999

    frame["profit"] = frame["sales"] - frame["cost"]
    hits = relation_screen(frame.drop(columns="revenue"), "profit", "regression", NUMERIC)
    assert {"sales", "cost"} in [set(h["columns"]) for h in hits]

    frame["price_again"] = frame["unit_price"] * 100
    hits = relation_screen(frame.drop(columns=["revenue", "profit"]), "price_again", "regression", NUMERIC)
    assert hits[0]["columns"] == ["unit_price"]


def test_screen_stays_quiet_on_an_honest_target():
    frame = _screen_frame()
    rng = np.random.default_rng(3)
    frame["demand"] = 3 * frame["discount"] + 0.01 * frame["units"] + rng.normal(0, 2, len(frame))
    assert relation_screen(frame, "demand", "regression", NUMERIC) == []


def test_screen_finds_a_column_that_decides_the_class():
    frame = _screen_frame()
    frame["refunded"] = np.where(frame["discount"] > 0.2, "yes", "no")
    hits = relation_screen(frame, "refunded", "binary_classification", NUMERIC)
    assert hits and hits[0]["columns"] == ["discount"]

    rng = np.random.default_rng(4)
    frame["churned"] = np.where(rng.uniform(size=len(frame)) < 0.3, "yes", "no")
    assert relation_screen(frame, "churned", "binary_classification", NUMERIC) == []


def _screen_state(tmp_path, frame, target, declared=()):
    from models import DataColumn, DataSchema, DataType

    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    schema = DataSchema(name="t", description="t", columns=[
        DataColumn(name=name, data_type=DataType.NUMERIC, description=name,
                   available_at_prediction=True if name in declared else None, is_target=name == target)
        for name in frame.columns
    ])
    plan = SplitPlan(method="random", test_size=0.2, random_seed=7, shuffle=True, drop_duplicates=False,
                     cv_strategy="kfold", cv_folds=3, reason="test")
    return {"train_path": str(train_path), "target": target, "task_type": "regression", "columns": NUMERIC,
            "schema": schema, "split_plan": plan, "summary": "## PROFILE", "run_id": 9,
            "run_dir": str(tmp_path / "run")}


def test_screen_excludes_before_fan_out_and_tells_the_planners(tmp_path):
    from information_agent.base import prepare_run, screen_leakage

    frame = _screen_frame()
    frame["revenue"] = frame["units"] * frame["unit_price"]
    state = _screen_state(tmp_path, frame, "revenue")

    state.update(screen_leakage(state))
    assert "## LEAKAGE SCREEN" in state["summary"] and "units" in state["summary"]
    state.update(prepare_run(state))

    excluded = read_exclusions(state["run_dir"])
    assert set(excluded) == {"units", "unit_price"}
    assert excluded["units"]["found_by"] == "leakage screen"


def test_screen_keeps_columns_the_schema_declares_available(tmp_path):
    from information_agent.base import prepare_run, screen_leakage

    frame = _screen_frame()
    frame["revenue"] = frame["units"] * frame["unit_price"]
    state = _screen_state(tmp_path, frame, "revenue", declared=("units", "unit_price"))

    state.update(screen_leakage(state))
    state.update(prepare_run(state))

    assert state["leakage_screen"][0]["outcome"].startswith("kept")
    assert read_exclusions(state["run_dir"]) == {}
