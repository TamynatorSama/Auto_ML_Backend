import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from automl_runtime.contract import Contract
from automl_runtime.folds import build_folds, save_folds
from automl_runtime.metrics import score_predictions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _regression_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "size": rng.uniform(10, 100, n),
        "rooms": rng.integers(1, 6, n),
        "city": rng.choice(["north", "south", "east"], n),
        "listed": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "row_id": np.arange(n),
    })
    frame["price"] = 3 * frame["size"] + 10 * frame["rooms"] + rng.normal(0, 5, n)
    return frame


def _classification_frame(n=600, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "segment": rng.choice(["a", "b"], n)})
    logit = 2.0 * frame["x1"] - frame["x2"]
    frame["churned"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return frame


def _prepare(tmp_path, frame, target, task, metrics, candidate, strategy="kfold",
             excluded=(), test_frame=None):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.csv"
    frame.to_csv(train_path, index=False)

    folds = build_folds(strategy, 3, 7, frame[target])
    folds_path = save_folds(tmp_path / "folds.npz", folds)

    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    (attempt / "candidate.py").write_text(textwrap.dedent(candidate), encoding="utf-8")
    Contract(
        model="test", target=target, task_type=task, metrics=list(metrics), primary_metric=metrics[0],
        train_path=str(train_path), folds_path=str(folds_path), seed=7, n_jobs=1, time_budget_seconds=300,
        excluded_columns=list(excluded), ordered=strategy == "time_series_split",
    ).save(attempt / "contract.json")

    test_path = None
    if test_frame is not None:
        test_path = data_dir / "test.csv"
        test_frame.to_csv(test_path, index=False)
    return attempt, test_path


def _evaluate(attempt, final=False, test_path=None):
    command = [sys.executable, "-m", "automl_runtime.evaluate", "--attempt-dir", str(attempt)]
    if final:
        command += ["--final", "--test", str(test_path)]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    name = "result_final.json" if final else "result.json"
    return json.loads((attempt / name).read_text(encoding="utf-8"))


REGRESSION_CANDIDATE = """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from automl_runtime import DateParts, names


    def build_pipeline(columns, task, ctx):
        numeric = names(columns, "numeric", "discrete_numeric")
        categorical = names(columns, "categorical", "binary")
        dates = names(columns, "datetime")
        steps = [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
        if dates:
            steps.append(("date", DateParts(parts=("year", "month")), dates))
        return Pipeline([("prep", ColumnTransformer(steps)), ("model", Ridge(alpha=1.0))])
"""


# ---------------------------------------------------------------------------
# loop mode
# ---------------------------------------------------------------------------

def test_regression_scores_are_pooled_from_the_saved_predictions(tmp_path):
    frame = _regression_frame()
    attempt, _ = _prepare(tmp_path, frame, "price", "regression", ["mae", "rmse", "r2"], REGRESSION_CANDIDATE,
                          excluded=["row_id"])
    result = _evaluate(attempt)

    assert result["status"] == "ok", result["error"]
    oof = pd.read_csv(attempt / "oof_predictions.csv")
    recomputed, _ = score_predictions(["mae", "rmse", "r2"], "regression", oof["y_true"], oof["prediction"])
    for metric, value in recomputed.items():
        assert result["cv_scores"][metric] == pytest.approx(value)
    assert result["n_covered"] == len(frame)
    assert (attempt / "feature_importance.csv").exists()
    importance = pd.read_csv(attempt / "feature_importance.csv")
    assert importance.iloc[0]["feature"] == "size"          # raw column names, strongest first
    assert "row_id" not in set(importance["feature"])        # excluded columns never reach the model


def test_string_labels_use_the_second_sorted_class_as_positive(tmp_path):
    frame = _classification_frame()
    candidate = """
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        def build_pipeline(columns, task, ctx):
            prep = ColumnTransformer([("num", "passthrough", ["x1", "x2"]),
                                      ("cat", OneHotEncoder(handle_unknown="ignore"), ["segment"])])
            return Pipeline([("prep", prep), ("model", LogisticRegression(random_state=ctx.seed))])
    """
    metrics = ["pr_auc", "roc_auc", "f1", "balanced_accuracy"]
    attempt, _ = _prepare(tmp_path, frame, "churned", "binary_classification", metrics, candidate,
                          strategy="stratified_kfold")
    result = _evaluate(attempt)

    assert result["status"] == "ok", result["error"]
    oof = pd.read_csv(attempt / "oof_predictions.csv")
    assert {"proba_no", "proba_yes"} <= set(oof.columns)
    positive = (oof["y_true"] == "yes").astype(int)
    from sklearn.metrics import average_precision_score
    assert result["cv_scores"]["pr_auc"] == pytest.approx(average_precision_score(positive, oof["proba_yes"]))
    assert result["cv_scores"]["roc_auc"] > 0.75


def test_time_series_folds_score_only_the_rows_they_cover(tmp_path):
    frame = _regression_frame()
    attempt, _ = _prepare(tmp_path, frame, "price", "regression", ["rmse"], REGRESSION_CANDIDATE,
                          strategy="time_series_split", excluded=["row_id"])
    result = _evaluate(attempt)

    assert result["status"] == "ok", result["error"]
    assert result["n_covered"] < len(frame)
    oof = pd.read_csv(attempt / "oof_predictions.csv")
    assert len(oof) == result["n_covered"]
    assert oof["row_index"].min() > 0      # the first block is never predicted


@pytest.mark.parametrize(
    "candidate, stage, fragment",
    [
        ("import sklearn.not_a_module\n\ndef build_pipeline(columns, task, ctx):\n    return None\n",
         "import", "not_a_module"),
        ("def make_model():\n    return None\n", "build", "defines no build_pipeline"),
        ("def build_pipeline(columns, task, ctx):\n    return object()\n", "estimator", "no fit or predict"),
        (
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import FunctionTransformer\n\n"
            "def build_pipeline(columns, task, ctx):\n"
            "    keep = [c.name for c in columns if c.kind == 'numeric']\n"
            "    select = FunctionTransformer(lambda frame: frame[keep])\n"
            "    return Pipeline([('select', select), ('model', Ridge())])\n",
            "pickle", "cannot be pickled",
        ),
        (
            "from sklearn.linear_model import Ridge\n\n"
            "def build_pipeline(columns, task, ctx):\n"
            "    return Ridge()\n",
            "smoke", "could not convert string",
        ),
    ],
)
def test_each_failure_stops_at_its_own_stage(tmp_path, candidate, stage, fragment):
    frame = _regression_frame()
    attempt, _ = _prepare(tmp_path, frame, "price", "regression", ["rmse"], candidate)
    result = _evaluate(attempt)

    assert result["status"] == "error"
    assert result["stage"] == stage
    assert fragment in result["error"]
    assert not (attempt / "oof_predictions.csv").exists()


def test_margin_classifier_without_probabilities_is_rejected_before_fitting(tmp_path):
    frame = _classification_frame()
    candidate = """
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.svm import SVC

        def build_pipeline(columns, task, ctx):
            return Pipeline([("prep", ColumnTransformer([("num", "passthrough", ["x1", "x2"])])), ("model", SVC())])
    """
    attempt, _ = _prepare(tmp_path, frame, "churned", "binary_classification", ["log_loss"], candidate,
                          strategy="stratified_kfold")
    result = _evaluate(attempt)

    assert result["stage"] == "estimator"
    assert "predict_proba" in result["error"]


# ---------------------------------------------------------------------------
# final mode
# ---------------------------------------------------------------------------

def test_final_mode_scores_the_test_set_and_saves_a_loadable_model(tmp_path):
    frame = _regression_frame(n=700)
    train, test = frame.iloc[:600].reset_index(drop=True), frame.iloc[600:].reset_index(drop=True)
    candidate = textwrap.dedent(REGRESSION_CANDIDATE) + textwrap.dedent("""

    from sklearn.base import BaseEstimator, TransformerMixin

    class Clip(BaseEstimator, TransformerMixin):
        # defined in candidate.py on purpose: the saved model must still load elsewhere
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X
        def get_feature_names_out(self, input_features=None):
            return input_features

    _plain_build = build_pipeline

    def build_pipeline(columns, task, ctx):
        pipeline = _plain_build(columns, task, ctx)
        pipeline.steps.insert(1, ("clip", Clip()))
        return pipeline
    """)
    attempt, test_path = _prepare(tmp_path, train, "price", "regression", ["mae", "r2"], candidate,
                                  excluded=["row_id"], test_frame=test)
    result = _evaluate(attempt, final=True, test_path=test_path)

    assert result["status"] == "ok", result["error"]
    assert result["test_scores"]["r2"] > 0.9
    assert (attempt / "test_predictions.csv").exists()
    assert (attempt / "model.joblib").exists()
    assert not any("cannot be loaded" in warning for warning in result["warnings"])


def test_excluded_columns_are_invisible_to_the_candidate(tmp_path):
    frame = _regression_frame()
    candidate = """
        from sklearn.linear_model import Ridge

        def build_pipeline(columns, task, ctx):
            seen = {column.name for column in columns}
            if "size" in seen or "price" in seen:
                raise AssertionError(f"excluded or target column visible: {sorted(seen)}")
            return Ridge()
    """
    attempt, _ = _prepare(tmp_path, frame, "price", "regression", ["rmse"], candidate,
                          excluded=["size", "city", "listed"])
    result = _evaluate(attempt)

    assert result["stage"] != "build", result["error"]
    assert result["status"] == "ok", result["error"]
    assert result["excluded_columns"] == ["size", "city", "listed"]


def test_date_parts_include_iso_week_and_survive_bad_values():
    from automl_runtime.features import DateParts

    frame = pd.DataFrame({"when": ["2020-01-06", "not a date", "2020-12-31"]})
    parts = DateParts(parts=("year", "week")).fit(frame).transform(frame)

    assert list(parts.columns) == ["when__year", "when__week"]
    assert parts["when__week"].iloc[0] == 2
    assert np.isnan(parts["when__week"].iloc[1])
    assert parts["when__week"].iloc[2] == 53


def test_category_cleaner_merges_case_and_spacing_variants():
    from automl_runtime.features import CategoryCleaner

    frame = pd.DataFrame({"plan": ["Basic", " Basic", "BASIC  ", None, "Pro  Plan"]})
    cleaned = CategoryCleaner().fit(frame).transform(frame)

    assert cleaned["plan"].tolist()[:3] == ["basic", "basic", "basic"]
    assert pd.isna(cleaned["plan"].iloc[3])
    assert cleaned["plan"].iloc[4] == "pro plan"
