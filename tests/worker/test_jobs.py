"""A job through the worker on Postgres, with planning real and training faked.

Planning runs the real information graph with a scripted planner; training is a
stand-in node that records a sandbox run and writes a report, so these tests
cover the thread's checkpoints, the queue and the job's status, not the models.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph

import main
from information_agent import base as information
from utils.prompts.split_planner import split_planner_prompt
from utils.reusable import hooks
from worker import cli, db, jobs, queue
from worker import hooks as worker_hooks
from worker.__main__ import run_task
from worker.hooks import WorkerHooks
from worker.sweep import sweep

SPLIT = {
    "method": "stratified", "test_size": 0.2, "random_seed": 7, "shuffle": True, "stratify_column": "y",
    "drop_duplicates": False, "cv_strategy": "stratified_kfold", "cv_folds": 3, "reason": "binary target",
}
CONFIG = {
    "models": ["logistic_regression", "random_forest"],
    "baseline": {"strategy": "most_frequent", "applicable": True},
    "eval_matrics": ["roc_auc", "f1"], "early_stopping_patience": 2, "max_tries": 2,
    "improvement_delta": 0.005, "improvement_mode": "absolute", "improvement_metric": "roc_auc",
}


class Planner:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(SPLIT if messages[0].content == split_planner_prompt else CONFIG))


class PlannerHooks(WorkerHooks):
    def __init__(self, pool, key, planner):
        super().__init__(pool, key)
        self.planner = planner

    def llm_for(self, run_id, role):
        return self.planner


def write_report(run_dir, status, metric):
    Path(run_dir, "report.json").write_text(json.dumps({"status": status, "primary_metric": metric}), encoding="utf-8")


class Training:
    """Stands in for code_gen_eval: one sandbox run's usage, then the report."""

    def __init__(self, pool):
        self.pool, self.calls, self.crash, self.stop = pool, 0, False, False
        self.together = None   # a barrier: every job must be training at once to pass it

    def __call__(self, state):
        self.calls += 1
        context = state["context"]
        hooks.emit(context.run_id, "sandbox_usage", model="random_forest", attempt="1", step="loop",
                   usage={"cpu_seconds": 2.0, "reserved_gb_seconds": 8.0})
        if self.crash:
            self.crash = False
            raise RuntimeError("the worker died")
        if self.together:
            self.together.wait()
        if self.stop:   # a person presses stop while the job runs
            with self.pool.connection() as conn:
                cli.stop(conn, context.run_id)
        status = "stopped" if hooks.should_stop(context.run_id) else "complete"
        write_report(context.run_dir, status, context.primary_metric)
        return {"report": status}


@pytest.fixture
def world(pool, key, tmp_path, monkeypatch):
    training = Training(pool)
    graph = StateGraph(main.AutoMLState)
    graph.add_node("information", information.app)
    graph.add_node("code_gen_eval", training)
    graph.add_edge(START, "information")
    graph.add_edge("information", "code_gen_eval")
    graph.add_edge("code_gen_eval", END)
    monkeypatch.setattr(main, "graph", graph)
    monkeypatch.setattr(jobs, "BACKEND", "subprocess")
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(worker_hooks, "STOP_CHECK_SECONDS", 0)
    planner = Planner()
    restart = lambda: monkeypatch.setattr(hooks, "_hooks", PlannerHooks(pool, key, planner))   # noqa: E731
    restart()
    return SimpleNamespace(training=training, planner=planner, restart=restart)


def run_next(pool):
    """What the worker loop does with the next task; a new checkpointer each time, as after a restart."""
    with pool.connection() as conn:
        task = queue.claim(conn)
    run_task(pool, db.checkpointer(pool), hooks._hooks, task)
    return task


def fetch(pool, job_id):
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()


def approve(pool, job_id, edits=None):
    with pool.connection() as conn:
        cli.approve(conn, job_id, edits)


def test_review_survives_a_restart_and_resumes_with_an_edited_metric(pool, world, new_job):
    job_id = new_job()
    run_next(pool)
    job = fetch(pool, job_id)
    assert job["status"] == "review"
    assert job["plan"]["config"]["eval_matrics"] == ["roc_auc", "f1"]
    assert Path(job["run_dir"]) == jobs.RUNS_ROOT / str(job["workspace_id"]) / str(job_id)
    state = main.graph.compile(checkpointer=db.checkpointer(pool)).get_state(jobs.thread(job_id)).values
    assert state["backend"] == "subprocess"   # reaches planning through main's state

    world.restart()
    approve(pool, job_id, {"config": {"eval_matrics": ["f1", "roc_auc"]}})
    assert fetch(pool, job_id)["status"] == "queued"
    run_next(pool)

    job = fetch(pool, job_id)
    assert job["status"] == "succeeded" and job["report"]["primary_metric"] == "f1"
    assert job["usage_totals"]["cpu_seconds"] == 2.0 and job["files_expire_at"] is not None
    assert world.planner.calls == 2   # planned once


def test_a_training_whose_worker_died_is_requeued_and_continued(pool, world, new_job):
    job_id = new_job()
    run_next(pool)
    approve(pool, job_id)

    world.training.crash = True
    with pool.connection() as conn:
        task = queue.claim(conn)
    with pytest.raises(RuntimeError, match="the worker died"):
        jobs.train(pool, db.checkpointer(pool), task)   # dies without finishing its task
    with pool.connection() as conn:
        conn.execute("UPDATE tasks SET heartbeat_at = now() - interval '2 minutes' WHERE id = %s", (task["id"],))
        assert queue.requeue_stale(conn) == [(task["id"], "queued")]

    world.restart()
    assert run_next(pool)["tries"] == 2
    job = fetch(pool, job_id)
    assert job["status"] == "succeeded" and job["report"]["primary_metric"] == "roc_auc"
    assert job["usage_totals"]["cpu_seconds"] == 4.0   # both runs used the sandbox
    assert world.training.calls == 2 and world.planner.calls == 2


def test_stop_ends_a_run_and_resume_finishes_it(pool, world, new_job, monkeypatch):
    job_id = new_job()
    run_next(pool)
    approve(pool, job_id)

    world.training.stop = True
    run_next(pool)
    job = fetch(pool, job_id)
    assert (job["status"], job["cancel_requested"]) == ("stopped", False)

    resumed = []
    monkeypatch.setattr(jobs, "resume_run", lambda run_dir, topic: (
        resumed.append(run_dir), write_report(run_dir, "complete", "roc_auc")))
    with pool.connection() as conn:
        assert cli.resume(conn, job_id) == "queued"
    run_next(pool)
    assert fetch(pool, job_id)["status"] == "succeeded"
    assert resumed == [job["run_dir"]]


def test_a_task_that_raises_fails_its_job(pool, world, new_job, data):
    job_id = new_job()
    Path(data[0]).unlink()
    task = run_next(pool)
    job = fetch(pool, job_id)
    assert job["status"] == "failed" and "plan failed" in job["error"]
    with pool.connection() as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = %s", (task["id"],)).fetchone()["status"] == "failed"


def test_the_sweep_removes_files_and_checkpoints_but_keeps_rows(pool, world, new_job, tmp_path):
    job_id, outside_job = new_job(), new_job()
    run_next(pool)
    run_dir = Path(fetch(pool, job_id)["run_dir"])
    hooks.emit(job_id, "sandbox_usage", model="ridge", attempt="1", step="loop", usage={"cpu_seconds": 1.0})
    outside = tmp_path / "not_a_run"
    outside.mkdir()
    with pool.connection() as conn:
        conn.execute("UPDATE jobs SET run_dir = %s WHERE id = %s", (str(outside), outside_job))
        conn.execute("UPDATE jobs SET files_expire_at = now() - interval '1 day'")

    saver = db.checkpointer(pool)
    assert run_dir.exists() and saver.get_tuple(jobs.thread(job_id)) is not None
    assert sweep(pool, saver) == [job_id, outside_job]
    assert sweep(pool, saver) == []

    assert not run_dir.exists() and outside.exists()
    assert saver.get_tuple(jobs.thread(job_id)) is None
    job = fetch(pool, job_id)
    assert job["files_expired"] and job["plan"] is not None
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM usage_records WHERE job_id = %s", (job_id,)).fetchone()["n"] == 1


def test_two_peoples_jobs_train_at_the_same_time(pool, world, new_job):
    jobs = [new_job(), new_job("other")]
    run_next(pool)
    run_next(pool)
    for job_id in jobs:
        approve(pool, job_id)
    with pool.connection() as conn:
        conn.execute("UPDATE sandbox_hosts SET max_running_jobs = 2")

    world.training.together = threading.Barrier(2, timeout=60)
    threads = [threading.Thread(target=run_next, args=(pool,)) for _ in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [fetch(pool, job_id)["status"] for job_id in jobs] == ["succeeded", "succeeded"]
