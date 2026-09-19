import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from information_agent import base as information
from main import checkpoint_types
from models import DataColumn, DataSchema, DataType
from utils.prompts.split_planner import split_planner_prompt
from utils.reusable import hooks

SPLIT = {
    "method": "stratified", "test_size": 0.2, "random_seed": 7, "shuffle": True, "stratify_column": "y",
    "drop_duplicates": False, "cv_strategy": "stratified_kfold", "cv_folds": 3, "reason": "binary target",
}
CONFIG = {
    "models": ["logistic_regression", "random_forest"],
    "baseline": {"strategy": "most_frequent", "applicable": True},
    "eval_matrics": ["roc_auc", "f1"], "early_stopping_patience": 2, "max_tries": 2,
    "improvement_delta": 0.005, "improvement_mode": "absolute", "improvement_metric": "roc_auc",
    "reasons": {"models": "a linear model and a forest", "eval_matrics": "ranking quality first"},
}


class Planner:
    """Answers the split planner and the config generator, counting the calls."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        answer = SPLIT if messages[0].content == split_planner_prompt else CONFIG
        return SimpleNamespace(content=json.dumps(answer))


class PlannerHooks(hooks.Hooks):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner

    def llm_for(self, run_id, role):
        return self.planner


@pytest.fixture
def planner(monkeypatch):
    planner = Planner()
    monkeypatch.setattr(hooks, "_hooks", PlannerHooks(planner))
    return planner


@pytest.fixture
def start(tmp_path):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=300)
    frame = pd.DataFrame({
        "id": range(300), "x1": x1, "x2": rng.normal(size=300),
        "color": rng.choice(["red", "green", "blue"], size=300),
        "y": (x1 + rng.normal(scale=0.5, size=300) > 0).astype(int),
    })
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    schema = DataSchema(name="toy", description="a toy classification set", columns=[
        DataColumn(name="id", data_type=DataType.NUMERIC, description="row id", role="identifier"),
        DataColumn(name="x1", data_type=DataType.NUMERIC, description="signal"),
        DataColumn(name="x2", data_type=DataType.NUMERIC, description="noise"),
        DataColumn(name="color", data_type=DataType.CATEGORICAL, description="a colour"),
        DataColumn(name="y", data_type=DataType.CATEGORICAL, description="the label", is_target=True),
    ])
    return {"data_path": str(data_path), "schema": schema, "run_id": 1, "run_dir": str(tmp_path / "run")}


def saver():
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types()))


def test_without_review_planning_runs_straight_through(planner, start):
    final = information.graph.compile(checkpointer=saver()).invoke(start, {"configurable": {"thread_id": "a"}})
    assert "__interrupt__" not in final
    assert final["context"].primary_metric == "roc_auc"
    assert "id" in final["context"].drop_columns


def test_review_pauses_and_resumes_with_an_edited_metric_after_a_restart(planner, start, caplog):
    checkpoints, thread = saver(), {"configurable": {"thread_id": "b"}}
    paused = information.graph.compile(checkpointer=checkpoints).invoke({**start, "review": True}, thread)

    (pause,) = paused["__interrupt__"]
    assert pause.value["config"]["reasons"]["models"] == "a linear model and a forest"
    assert pause.value["split_plan"]["test_size"] == 0.2
    assert not (Path(start["run_dir"]) / "context.json").exists()

    # a freshly compiled graph on the same checkpoints stands in for a restarted worker
    with caplog.at_level(logging.WARNING):
        restarted = information.graph.compile(checkpointer=checkpoints)
        final = restarted.invoke(Command(resume={"config": {"eval_matrics": ["f1", "roc_auc"]}}), thread)

    assert final["context"].primary_metric == "f1"
    assert planner.calls == 2
    assert "allowed_msgpack_modules" not in caplog.text


def test_an_edited_split_goes_back_and_splits_again(planner, start):
    checkpoints, thread = saver(), {"configurable": {"thread_id": "c"}}
    app = information.graph.compile(checkpointer=checkpoints)
    app.invoke({**start, "review": True}, thread)
    final = app.invoke(Command(resume={"split_plan": {"test_size": 0.5}}), thread)

    test_rows = len(pd.read_csv(final["context"].test_path))
    assert test_rows == 150
    assert final["context"].split_plan.test_size == 0.5
    assert planner.calls == 2


def test_roles_and_is_target_stay_in_step():
    assert DataColumn(name="y", data_type=DataType.NUMERIC, description="", is_target=True).role == "target"
    assert DataColumn(name="y", data_type=DataType.NUMERIC, description="", role="target").is_target
    assert DataColumn(name="a", data_type=DataType.NUMERIC, description="").role == "feature"
