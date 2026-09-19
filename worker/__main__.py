"""
The worker: python -m worker

Claims tasks from Postgres and runs up to AUTOML_WORKER_TASKS of them at once,
each in its own thread, with at most AUTOML_WORKER_PLANS plans among them
(docs/PHASE2.md, docs/PHASE2B.md). While a task runs, a thread refreshes its
heartbeat; the main loop puts back tasks whose heartbeat went stale, and once a
day runs the sweep. Jobs training on one host share it through the fair pool.

Needs DATABASE_URL and AUTOML_SECRET_KEY in .env, and the tunnel to the server.
"""

from __future__ import annotations

import os
import threading
import time
import traceback

import psycopg
from dotenv import load_dotenv

import db
from utils.reusable import hooks
from worker import crypto, jobs, queue
from worker.hooks import WorkerHooks
from worker.sweep import sweep

POLL_SECONDS = 2
HEARTBEAT_SECONDS = 10
REQUEUE_SECONDS = 30
SWEEP_SECONDS = 24 * 3600
TASKS = int(os.environ.get("AUTOML_WORKER_TASKS") or 8)
PLANS = int(os.environ.get("AUTOML_WORKER_PLANS") or 2)   # planning holds a whole CSV in memory


def _heartbeat(pool, task_id: int, done: threading.Event) -> None:
    while not done.wait(HEARTBEAT_SECONDS):
        try:
            with pool.connection() as conn:
                queue.heartbeat(conn, task_id)
        except Exception as error:  # a missed beat is retried; a minute of them requeues the task
            print(f"heartbeat for task {task_id} failed: {error!r}", flush=True)


def run_task(pool, saver, worker_hooks: WorkerHooks, task: dict) -> None:
    print(f"task {task['id']}: {task['kind']} job {task['job_id']} (try {task['tries']})", flush=True)
    done = threading.Event()
    threading.Thread(target=_heartbeat, args=(pool, task["id"], done), daemon=True).start()
    error = None
    try:
        jobs.HANDLERS[task["kind"]](pool, saver, task)
    except Exception as exc:
        traceback.print_exc()
        error = f"{task['kind']} failed: {exc!r}"
        jobs.fail(pool, task["job_id"], error)
    finally:
        done.set()
        worker_hooks.forget(task["job_id"])
    with pool.connection() as conn:
        queue.finish(conn, task["id"], error)
    print(f"task {task['id']}: {'failed' if error else 'done'}", flush=True)


def main() -> None:
    load_dotenv()
    pool = db.pool(size=4 + 3 * TASKS)   # every running model emits events from its own thread
    applied = db.setup(pool)
    if applied:
        print(f"migrations applied: {', '.join(applied)}", flush=True)
    saver = db.checkpointer(pool)
    worker_hooks = WorkerHooks(pool, crypto.secret_key())
    hooks.install(worker_hooks)
    print(f"worker ready: up to {TASKS} tasks at once, {PLANS} of them plans", flush=True)

    running: dict = {}   # thread -> task
    last_requeue = last_sweep = float("-inf")
    while True:
        running = {thread: task for thread, task in running.items() if thread.is_alive()}
        task = None
        try:
            now = time.monotonic()
            if now - last_requeue >= REQUEUE_SECONDS:
                with pool.connection() as conn:
                    queue.requeue_stale(conn)
                last_requeue = now
            if now - last_sweep >= SWEEP_SECONDS:
                sweep(pool, saver)
                last_sweep = now
            if len(running) < TASKS:
                plans = sum(task["kind"] == "plan" for task in running.values())
                with pool.connection() as conn:
                    task = queue.claim(conn, None if plans < PLANS else ["train"])
        except psycopg.OperationalError as error:   # the tunnel or the database is down: wait
            print(f"database unavailable: {error}", flush=True)
        if task is None:
            time.sleep(POLL_SECONDS)
        else:
            thread = threading.Thread(target=run_task, args=(pool, saver, worker_hooks, task),
                                      name=f"task-{task['id']}", daemon=True)
            thread.start()
            running[thread] = task


if __name__ == "__main__":
    main()
