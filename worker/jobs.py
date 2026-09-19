"""
jobs.py
-------
The worker's two task handlers (docs/PHASE2.md §4). A job's graph runs on
main.graph with a Postgres checkpointer, thread id = job id, so wherever a
task left the thread, the next task carries on from there:

    plan   runs planning with review on; the graph pauses at review_plan, and
           the drafted plan goes into jobs.plan
    train  continues the thread from wherever its checkpoint is:
               paused at review   -> resumed with the approved edits
               mid-run            -> invoke(None); finished models are read back
               finished, stopped  -> the pipeline's resume_run(run_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

from langgraph.types import Command

import main
from code_gen_eval.resume import resume_run
from db import FILES_KEPT_DAYS, jsonb
from models import DataSchema
from utils.reusable import hooks

RUNS_ROOT = Path("runs")
BACKEND = "sandbox"   # tests switch it to subprocess
# the report's status -> the job's
FINAL_STATUS = {"complete": "succeeded", "stopped": "stopped"}


def thread(job_id: int) -> dict:
    return {"configurable": {"thread_id": str(job_id)}}


def load(conn, job_id: int) -> dict:
    return conn.execute(
        """
        SELECT j.*, s.path AS data_path, sc.columns AS schema
        FROM jobs j JOIN sources s ON s.id = j.source_id JOIN schemas sc ON sc.id = j.schema_id
        WHERE j.id = %s
        """,
        (job_id,),
    ).fetchone()


def total_usage(conn, job_id: int) -> None:
    conn.execute(
        """
        UPDATE jobs SET usage_totals = (
            SELECT coalesce(jsonb_object_agg(meter, total), '{}') FROM (
                SELECT meter, sum(quantity) AS total FROM usage_records WHERE job_id = %s GROUP BY meter
            ) t
        ) WHERE id = %s
        """,
        (job_id, job_id),
    )


def plan(pool, saver, task: dict) -> None:
    job_id = task["job_id"]
    with pool.connection() as conn:
        job = load(conn, job_id)
    run_dir = RUNS_ROOT / str(job["workspace_id"]) / str(job_id)   # P10
    hooks.emit(job_id, "stage", stage="planning")

    graph, config = main.graph.compile(checkpointer=saver), thread(job_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        graph.invoke({
            "topic": job["name"], "data_path": job["data_path"], "schema": DataSchema.model_validate(job["schema"]),
            "run_id": job_id, "run_dir": str(run_dir), "review": True, "backend": BACKEND,
        }, config)
    elif not snapshot.interrupts:
        graph.invoke(None, config)   # a worker died mid-planning

    pause = graph.get_state(config).interrupts
    if not pause:
        raise RuntimeError("planning ended without pausing for review")
    with pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'review', plan = %s, run_dir = %s WHERE id = %s",
            (jsonb(pause[0].value), str(run_dir), job_id),
        )
        total_usage(conn, job_id)
    hooks.emit(job_id, "plan_ready")


def train(pool, saver, task: dict) -> None:
    job_id = task["job_id"]
    with pool.connection() as conn:
        job = load(conn, job_id)
        conn.execute(
            "UPDATE jobs SET status = CASE WHEN cancel_requested THEN 'stopping' ELSE 'running' END, "
            "started_at = coalesce(started_at, now()) WHERE id = %s",
            (job_id,),
        )
    hooks.emit(job_id, "stage", stage="training")
    if BACKEND == "sandbox":
        clear_sandboxes(job_id)

    graph, config = main.graph.compile(checkpointer=saver), thread(job_id)
    snapshot = graph.get_state(config)
    if snapshot.interrupts:
        # never an empty dict: LangGraph reads {} as "resume no interrupts" and nothing runs
        edits = {"split_plan": {}, "config": {}, **(job["edits"] or {})}
        graph.invoke(Command(resume=edits), config)
    elif snapshot.next:
        graph.invoke(None, config)
    elif snapshot.values:
        resume_run(job["run_dir"], job["name"])
    else:
        raise RuntimeError("the job has no plan to train")

    report = json.loads((Path(job["run_dir"]) / "report.json").read_text(encoding="utf-8"))
    status = FINAL_STATUS.get(report["status"], "failed")
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE jobs SET status = %s, report = %s, error = %s, cancel_requested = false, finished_at = now(),
                files_expire_at = now() + make_interval(days => %s)
            WHERE id = %s
            """,
            (status, jsonb(report), None if status != "failed" else report["status"], FILES_KEPT_DAYS, job_id),
        )
        total_usage(conn, job_id)


def clear_sandboxes(job_id: int) -> None:
    """Delete sandboxes a dead worker left running, so they stop holding the host's memory.

    What they used still goes into the ledger.
    """
    client = hooks.sandbox_for(job_id)
    for sandbox in client.list_sandboxes({"run": str(job_id)}):
        usage = client.delete_sandbox(sandbox["id"])
        labels = sandbox.get("labels") or {}
        print(f"job {job_id}: removed leftover sandbox {sandbox['id']} ({labels.get('model')})", flush=True)
        hooks.emit(job_id, "sandbox_usage", model=labels.get("model"), attempt=labels.get("attempt"),
                   step=labels.get("step"), usage=usage)


def fail(pool, job_id: int, error: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE jobs SET status = 'failed', error = %s, cancel_requested = false, finished_at = now(),
                files_expire_at = now() + make_interval(days => %s)
            WHERE id = %s
            """,
            (error[:2000], FILES_KEPT_DAYS, job_id),
        )
        total_usage(conn, job_id)


HANDLERS = {"plan": plan, "train": train}
