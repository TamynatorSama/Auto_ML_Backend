"""
sweep.py
--------
The seven-day sweep (A4): a finished job's run directory and checkpoints are
deleted once files_expire_at passes. The job row, its report, events and usage
stay.

It also collects what a deleted run leaves behind. The API deletes a job's rows
but never LangGraph's tables or the runs folder — neither is the API's to touch
(docs/PHASE5.md §11) — so it writes a deleted_runs row saying what it left, and
that row, not a walk of the runs folder, is what this deletes. runs/ also holds
the command line's own history, which is nobody's to guess about.

An uploaded source goes the same way seven days after it was last used, not
after it was uploaded, so a dataset still being run against doesn't vanish
underneath anyone (docs/PHASE4.md §8.7). Its row, profile and schemas stay, so
the history still reads; only the CSV goes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from api import files   # only for where uploads live: one rule, so the two can't drift
from worker import jobs


def sweep(pool, saver) -> list:
    """Expire what is due; returns the job ids."""
    sweep_sources(pool)
    sweep_deleted(pool, saver)
    with pool.connection() as conn:
        due = conn.execute(
            "SELECT id, run_dir FROM jobs WHERE files_expire_at < now() AND NOT files_expired ORDER BY id"
        ).fetchall()
    root = jobs.RUNS_ROOT.resolve()
    for job in due:
        run_dir = Path(job["run_dir"] or "").resolve()
        # the path comes from the database: never delete outside the runs folder
        if job["run_dir"] and run_dir.is_relative_to(root) and run_dir != root:
            shutil.rmtree(run_dir, ignore_errors=True)
        saver.delete_thread(str(job["id"]))
        with pool.connection() as conn:
            conn.execute("UPDATE jobs SET files_expired = true WHERE id = %s", (job["id"],))
        print(f"job {job['id']}: files expired", flush=True)
    return [job["id"] for job in due]


def sweep_deleted(pool, saver) -> list:
    """The thread and run directory of each run the console deleted; returns the job ids."""
    with pool.connection() as conn:
        due = conn.execute("SELECT job_id, run_dir FROM deleted_runs ORDER BY job_id").fetchall()
    root = jobs.RUNS_ROOT.resolve()
    for run in due:
        run_dir = Path(run["run_dir"] or "").resolve()
        # the path comes from the database: never delete outside the runs folder
        if run["run_dir"] and run_dir.is_relative_to(root) and run_dir != root:
            shutil.rmtree(run_dir, ignore_errors=True)
        saver.delete_thread(str(run["job_id"]))
        with pool.connection() as conn:
            conn.execute("DELETE FROM deleted_runs WHERE job_id = %s", (run["job_id"],))
        print(f"job {run['job_id']}: deleted, its files and thread swept", flush=True)
    return [run["job_id"] for run in due]


def sweep_sources(pool) -> list:
    """Delete the CSVs of sources past their day; the rows and their profiles stay."""
    with pool.connection() as conn:
        due = conn.execute(
            "SELECT id, path FROM sources WHERE files_expire_at < now() AND status <> 'expired' "
            "AND path IS NOT NULL ORDER BY id"
        ).fetchall()
    for source in due:
        # the path comes from the database, so files.remove decides whether it is ours to delete
        files.remove(source["path"])
        with pool.connection() as conn:
            conn.execute("UPDATE sources SET status = 'expired', path = NULL WHERE id = %s", (source["id"],))
        print(f"source {source['id']}: file expired", flush=True)
    return [source["id"] for source in due]
