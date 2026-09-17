import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from code_gen_eval import base as fanout
from models import AttemptRecord, ModelResult
from tests.test_leakage import _context
from utils.reusable import resources


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

def _fake_memory(monkeypatch, available_mb):
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=available_mb * 1e6))


def test_concurrency_is_limited_by_memory_on_large_data(tmp_path, monkeypatch):
    big = tmp_path / "train.csv"
    big.write_bytes(b"x" * 70_000_000)                  # 70 MB of CSV
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 12)

    _fake_memory(monkeypatch, 1_700)
    tight = resources.plan_resources(str(big), n_models=4)
    assert tight.max_concurrency == 1 and "memory" in tight.reason

    _fake_memory(monkeypatch, 32_000)
    roomy = resources.plan_resources(str(big), n_models=4)
    assert roomy.max_concurrency == 4
    assert roomy.n_jobs == 3                             # 12 cores shared by 4 workers


def test_waiting_for_memory_gives_up_rather_than_stalling(monkeypatch):
    _fake_memory(monkeypatch, 100)
    monkeypatch.setattr(resources, "MEMORY_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(resources, "MEMORY_POLL_SECONDS", 0.05)
    assert resources.wait_for_memory(50) == pytest.approx(0, abs=0.05)
    waited = resources.wait_for_memory(10_000)
    assert 0.3 <= waited < 1.5


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_a_finished_model_is_read_back_not_retrained(tmp_path, monkeypatch):
    context = _context(tmp_path)
    saved = ModelResult(model="ridge", status="no_improvement", best_attempt=2, best_cv_scores={"mae": 10.0})
    (tmp_path / "ridge").mkdir()
    (tmp_path / "ridge" / "result.json").write_text(saved.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(fanout.code_gen_subgraph, "invoke", lambda *a, **k: pytest.fail("should not retrain"))

    update = fanout.code_gen_worker({"context": context, "model": "ridge"})
    assert update["completed_sections"][0].best_cv_scores == {"mae": 10.0}


def test_a_half_finished_model_is_set_aside_and_trained_again(tmp_path, monkeypatch):
    context = _context(tmp_path)
    (tmp_path / "knn" / "attempt_1").mkdir(parents=True)
    fresh = ModelResult(model="knn", status="max_tries", best_attempt=1, best_cv_scores={"mae": 20.0})
    monkeypatch.setattr(fanout.code_gen_subgraph, "invoke", lambda *a, **k: {"result": fresh})

    update = fanout.code_gen_worker({"context": context, "model": "knn"})

    assert update["completed_sections"][0] is fresh
    assert (tmp_path / "knn" / "result.json").exists()
    assert any(path.name.startswith("knn.interrupted-") for path in tmp_path.iterdir())


def test_load_run_reads_the_plan_and_honours_a_concurrency_override(tmp_path, monkeypatch):
    from code_gen_eval.resume import load_run
    from models import Baseline, Configs

    context = _context(tmp_path, max_concurrency=3)
    (tmp_path / "context.json").write_text(context.model_dump_json(), encoding="utf-8")
    config = Configs(models=["ridge"], baseline=Baseline(strategy="median", applicable=True), eval_matrics=["mae"],
                     early_stopping_patience=1, max_tries=2, improvement_delta=0.01, improvement_mode="relative",
                     improvement_metric="mae")
    (tmp_path / "config.json").write_text(config.model_dump_json(), encoding="utf-8")

    import code_gen_eval.resume as resume
    planned = resources.ResourcePlan(max_concurrency=1, n_jobs=4, worker_memory_mb=625.0,
                                     available_memory_mb=2000.0, cpus=12, reason="tight")
    monkeypatch.setattr(resume, "plan_resources", lambda *a, **k: planned)

    reloaded = load_run(tmp_path)[0]
    assert reloaded.max_concurrency == 1                  # re-planned, not the 3 it started with
    assert reloaded.n_jobs == min(context.n_jobs, 4)       # only ever lowered
    monkeypatch.setenv("AUTOML_CONCURRENCY", "2")
    assert load_run(tmp_path)[0].max_concurrency == 2
    assert load_run(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# late exclusions and the concurrency limit
# ---------------------------------------------------------------------------

def test_a_model_blocked_by_a_later_exclusion_is_rerun_once(tmp_path):
    from utils.reusable.leakage import add_exclusions

    context = _context(tmp_path)
    used_units = AttemptRecord(model="ridge", attempt=1, script_path="x", cv_scores={"mae": 60.0}, excluded_columns=[])
    finished = ModelResult(model="ridge", status="no_improvement", attempts=[used_units], best_attempt=1,
                           best_cv_scores={"mae": 60.0})
    add_exclusions(tmp_path, ["units"], {"reason": "leak confirmed by catboost"})

    state = {"context": context, "completed_sections": [finished], "reruns": []}
    state.update(fanout.review_results(state))
    assert state["rerun_due"] == ["ridge"]
    sends = fanout.route_review(state)
    assert [send.arg["model"] for send in sends] == ["ridge"] and sends[0].arg["rerun"] is True

    # once re-run, it is not re-run again whatever its result
    state = {"context": context, "completed_sections": [finished, finished], "reruns": ["ridge"]}
    state.update(fanout.review_results(state))
    assert fanout.route_review(state) == "summarize_results"


def test_workers_never_exceed_the_planned_concurrency(tmp_path, monkeypatch):
    context = _context(tmp_path, max_concurrency=2)
    running, peak, lock = 0, 0, threading.Lock()

    def slow_invoke(payload, config):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.2)
        with lock:
            running -= 1
        return {"result": ModelResult(model=payload["model"], status="no_improvement")}

    monkeypatch.setattr(fanout.code_gen_subgraph, "invoke", slow_invoke)
    threads = [threading.Thread(target=fanout.code_gen_worker, args=({"context": context, "model": f"m{i}"},))
               for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 2


# ---------------------------------------------------------------------------
# placement, plan cache, rendering
# ---------------------------------------------------------------------------

def test_the_run_is_placed_first_and_the_split_lives_inside_it(tmp_path):
    from information_agent import base as info
    from models import DataColumn, DataSchema, DataType, SplitPlan

    data = tmp_path / "data.csv"
    pd.DataFrame({"x": range(40), "y": range(40)}).to_csv(data, index=False)
    schema = DataSchema(name="t", description="t", columns=[
        DataColumn(name="x", data_type=DataType.NUMERIC, description="x"),
        DataColumn(name="y", data_type=DataType.NUMERIC, description="y", is_target=True)])
    state = {"data_path": str(data), "schema": schema, "run_id": 7, "run_dir": str(tmp_path / "runs" / "r7")}

    placed = info.place_run(state)
    assert placed["plan_fingerprint"] == info.place_run(state)["plan_fingerprint"]
    state.update(placed)
    state["split_plan"] = SplitPlan(method="random", test_size=0.25, random_seed=1, shuffle=True,
                                    drop_duplicates=False, cv_strategy="kfold", cv_folds=3, reason="t")
    split = info.apply_split(state)
    assert Path(split["train_path"]).parent == tmp_path / "runs" / "r7" / "splits"

    info._remember(state, "split_plan", state["split_plan"].model_dump())
    assert info._cached(state, "split_plan") is None           # reuse is opt-in
    state["reuse_plan"] = True
    assert info._cached(state, "split_plan")["test_size"] == 0.25


def test_the_html_report_renders_the_current_report_shape(tmp_path):
    from utils.reusable.report import build_report
    from utils.reusable.report_html import render_html

    context = _context(tmp_path)
    result = ModelResult(model="ridge", status="no_improvement", best_attempt=1,
                         best_cv_scores={"mae": 500.0, "r2": 0.4}, test_scores={"mae": 480.0, "r2": 0.45})
    html = render_html(build_report([result], context))
    assert "ridge" in html and "<html" in html


# ---------------------------------------------------------------------------
# the memory ceiling on each evaluator
# ---------------------------------------------------------------------------

def test_a_process_that_outgrows_its_ceiling_is_stopped_quickly(tmp_path):
    from code_gen_eval.runner import _execute

    hog = [sys.executable, "-c", "import time\ndata = b'x' * 400_000_000\ntime.sleep(60)"]
    started = time.time()
    *_, peak_mb, timed_out, died_after_error, over_memory = _execute(hog, tmp_path, dict(__import__("os").environ), 120, 150)

    assert over_memory and not timed_out and not died_after_error
    assert peak_mb > 150 and time.time() - started < 30


def test_a_candidate_that_outgrows_its_ceiling_comes_back_as_out_of_memory(tmp_path):
    from code_gen_eval.runner import run_candidate
    from tests.test_leakage import DYNAMIC, _leaky_run

    context = _leaky_run(tmp_path, target="noisy").model_copy(update={"worker_memory_mb": 150, "max_concurrency": 1000})
    hog = DYNAMIC.replace("from automl_runtime import names", "from automl_runtime import names\n\nBALLAST = b'x' * 500_000_000")

    record = run_candidate(hog, "extra_trees", 1, context)

    assert record.status == "out_of_memory"
    assert "Make the model smaller" in record.traceback
    # stopped from outside, it wrote no result; the stage it reached comes from its own announcements
    assert record.traceback.startswith("[stage: import]\n")


def test_the_final_run_keeps_the_loop_record_of_its_attempt(tmp_path):
    """The final run reuses the attempt's directory, and its record, with no
    cross-validation scores, used to replace the loop's."""
    import json

    from code_gen_eval.runner import run_candidate
    from tests.test_leakage import DYNAMIC, _leaky_run

    context = _leaky_run(tmp_path, target="noisy")
    code = DYNAMIC.replace("TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)", "model")

    loop_record = run_candidate(code, "linear_regression", 1, context)
    final_record = run_candidate(code, "linear_regression", 1, context, final=True)

    attempt = tmp_path / "linear_regression" / "attempt_1"
    assert loop_record.cv_scores and final_record.test_scores
    assert json.loads((attempt / "record.json").read_text(encoding="utf-8"))["cv_scores"] == loop_record.cv_scores
    assert json.loads((attempt / "record_final.json").read_text(encoding="utf-8"))["test_scores"]
    assert not final_record.warnings and (attempt / "load_check.json").exists()


def test_a_load_check_stopped_for_memory_keeps_the_test_scores(tmp_path):
    """The live run: a 131 MB random forest fitted and scored within 383 MB, then
    the load check, run inside the evaluator, was stopped and took the scores with it."""
    from code_gen_eval.runner import _check_model, run_candidate
    from tests.test_leakage import DYNAMIC, _leaky_run

    context = _leaky_run(tmp_path, target="noisy")
    code = DYNAMIC.replace("TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)", "model")
    scored = run_candidate(code, "linear_regression", 1, context, final=True)
    attempt = tmp_path / "linear_regression" / "attempt_1"

    warnings = _check_model(context, attempt, "subprocess", ceiling=20)

    assert scored.test_scores
    assert len(warnings) == 1 and "could not be verified" in warnings[0] and "test scores stand" in warnings[0]


def test_the_ceiling_is_never_below_the_plan(monkeypatch):
    _fake_memory(monkeypatch, 1_100)                     # 100 MB above the reserve
    assert resources.memory_limit(600, 1) == 600
    _fake_memory(monkeypatch, 5_000)
    assert resources.memory_limit(600, 2) == pytest.approx((5_000 - 1_000) * 0.8 / 2)


def test_n_jobs_is_granted_only_where_memory_pays_for_it(tmp_path, monkeypatch):
    big = tmp_path / "train.csv"
    big.write_bytes(b"x" * 70_000_000)                  # 70 MB of CSV, about 910 MB per worker
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 12)

    _fake_memory(monkeypatch, 2_400)                     # 1,120 MB usable: one worker, one spare 170
    plan = resources.plan_resources(str(big), n_models=4)
    assert (plan.max_concurrency, plan.n_jobs) == (1, 2)
    assert plan.worker_memory_mb == pytest.approx(910 + 170)

    _fake_memory(monkeypatch, 1_900)                     # nothing spare: no pool of processes
    assert resources.plan_resources(str(big), n_models=4).n_jobs == 1


def test_resume_lowers_n_jobs_when_memory_is_tight(tmp_path, monkeypatch):
    import code_gen_eval.resume as resume
    from models import Baseline, Configs

    context = _context(tmp_path, n_jobs=4)
    (tmp_path / "context.json").write_text(context.model_dump_json(), encoding="utf-8")
    config = Configs(models=["ridge"], baseline=Baseline(strategy="median", applicable=True), eval_matrics=["mae"],
                     early_stopping_patience=1, max_tries=2, improvement_delta=0.01, improvement_mode="relative",
                     improvement_metric="mae")
    (tmp_path / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
    tight = resources.ResourcePlan(max_concurrency=1, n_jobs=1, worker_memory_mb=625.0,
                                   available_memory_mb=1500.0, cpus=12, reason="tight")
    monkeypatch.setattr(resume, "plan_resources", lambda *a, **k: tight)

    assert resume.load_run(tmp_path)[0].n_jobs == 1


def test_the_ceiling_is_held_against_unique_memory(tmp_path):
    from code_gen_eval.runner import _execute, _tree_unique_mb

    assert _tree_unique_mb(__import__("psutil").Process()) > 0
    idle = [sys.executable, "-c", "import time; time.sleep(3)"]
    *_, over_memory = _execute(idle, tmp_path, dict(__import__("os").environ), 60, 5_000)
    assert not over_memory
