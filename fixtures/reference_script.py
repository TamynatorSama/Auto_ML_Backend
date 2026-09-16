# model: ridge
# attempt: 1
# changes: initial implementation
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN = os.environ["AUTOML_TRAIN"]
TEST = os.environ["AUTOML_TEST"]
TARGET = os.environ["AUTOML_TARGET"]
SEED = int(os.environ["AUTOML_SEED"])
N_JOBS = int(os.environ["AUTOML_N_JOBS"])
OUT = os.environ["AUTOML_OUT"]
FINAL = os.environ["AUTOML_FINAL"] == "1"
METRICS = [m for m in os.environ["AUTOML_METRICS"].split(",") if m]
CV_STRATEGY = os.environ["AUTOML_CV_STRATEGY"]
CV_FOLDS = int(os.environ["AUTOML_CV_FOLDS"])
GROUP_COLUMN = os.environ["AUTOML_GROUP_COLUMN"]
DROP_COLUMNS = [c for c in os.environ["AUTOML_DROP_COLUMNS"].split(",") if c]

HIGH_CARDINALITY = 50

train = pd.read_csv(TRAIN)
y = train[TARGET]
X = train.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in train.columns])

numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical = [c for c in X.columns if c not in numeric and X[c].nunique() <= HIGH_CARDINALITY]
# one-hot on a free-text column would be thousands of columns of noise for a
# linear model, so those are dropped rather than encoded
X = X[numeric + categorical]

pipeline = Pipeline(
    [
        (
            "prepare",
            ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        categorical,
                    ),
                ]
            ),
        ),
        ("model", Ridge(random_state=SEED)),
    ]
)


def make_cv():
    if CV_STRATEGY == "time_series_split":
        return TimeSeriesSplit(n_splits=CV_FOLDS)
    if CV_STRATEGY == "group_kfold":
        return GroupKFold(n_splits=CV_FOLDS)
    if CV_STRATEGY == "stratified_kfold":
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def score(metric, y_true, y_pred):
    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "r2":
        return float(skm.r2_score(y_true, y_pred))
    raise ValueError(f"unsupported metric: {metric}")


groups = train[GROUP_COLUMN] if GROUP_COLUMN else None

started = time.perf_counter()
oof = cross_val_predict(pipeline, X, y, cv=make_cv(), groups=groups, n_jobs=N_JOBS)
cv_scores = {metric: score(metric, y, oof) for metric in METRICS}

pipeline.fit(X, y)
fit_seconds = round(time.perf_counter() - started, 3)

joblib.dump(pipeline, os.path.join(OUT, "model.joblib"))
pd.DataFrame({"y_true": y, "oof_prediction": oof}).to_csv(
    os.path.join(OUT, "oof_predictions.csv"), index=False
)
artifacts = {"model": "model.joblib", "oof_predictions": "oof_predictions.csv"}

# the test set is read once, at the very end of the run, and never during the
# generate/judge loop that chooses what to build next
test_scores = {}
if FINAL:
    test = pd.read_csv(TEST)
    predictions = pipeline.predict(test[X.columns])
    test_scores = {metric: score(metric, test[TARGET], predictions) for metric in METRICS}
    pd.DataFrame({"y_true": test[TARGET], "prediction": predictions}).to_csv(
        os.path.join(OUT, "test_predictions.csv"), index=False
    )
    artifacts["test_predictions"] = "test_predictions.csv"

print("===AUTOML_RESULT===")
print(
    json.dumps(
        {
            "model": os.environ["AUTOML_MODEL"],
            "cv_scores": cv_scores,
            "test_scores": test_scores,
            "fit_seconds": fit_seconds,
            "artifacts": artifacts,
        }
    )
)
